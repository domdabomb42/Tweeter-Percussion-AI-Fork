"""Causal next-token LoRA training; validation chooses checkpoints, test is untouched."""
import json
import hashlib
import math
from pathlib import Path
import random
import shutil
import time
import torch
from collections import Counter
from transformers import AutoModelForCausalLM
from . import MODEL_ID, MODEL_REVISION
from .prepare import load_prepared


def device_precision():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.device("cpu"), torch.float32


def load_model(adapter=None, cpu=False):
    device, dtype = device_precision()
    if cpu:
        device, dtype = torch.device("cpu"), torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                  torch_dtype=dtype, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != MODEL_ID:
            raise ValueError("feature adapter belongs to a different base model")
        metadata = Path(adapter) / "experiment_metadata.json"
        if metadata.is_file():
            info = json.loads(metadata.read_text(encoding="utf-8"))
            if (info.get("model_id"), info.get("revision")) != (MODEL_ID, MODEL_REVISION):
                raise ValueError("feature adapter revision does not match this experiment")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.to(device)
    model.config.use_cache = False
    return model, device, dtype


def score(model, rows, device, tokenizer=None):
    model.eval()
    total_loss, count, correct = 0., 0, 0
    category_counts, category_correct = Counter(), Counter()
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            output = model(input_ids=ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += float(output.loss) * n
            count += n
            matches = (output.logits[:, :-1].argmax(-1) == ids[:, 1:]).squeeze(0).cpu().tolist()
            correct += sum(matches)
            if tokenizer is not None:
                from .serializer import leaf_spans, value_category
                _, spans = leaf_spans(json.loads(row["text"]))
                offsets = tokenizer(row["text"], add_special_tokens=False,
                                    return_offsets_mapping=True)["offset_mapping"]
                j = 0
                for position, hit in enumerate(matches):
                    if position + 1 >= len(offsets):
                        category = "structure_or_missing"
                    else:
                        start, end = offsets[position + 1]
                        while j < len(spans) and spans[j]["end"] <= start:
                            j += 1
                        category = (value_category(spans[j]["path"])
                                    if j < len(spans) and spans[j]["start"] < end
                                    and spans[j]["value"] is not None else "structure_or_missing")
                    category_counts[category] += 1
                    category_correct[category] += hit
            del output, ids
    loss = total_loss / count
    result = {"loss": loss, "perplexity": math.exp(min(loss, 100)),
            "token_accuracy": correct / count, "predicted_tokens": count,
            "segments": len(rows)}
    if tokenizer is not None:
        result["token_accuracy_by_category"] = {category: {"correct": category_correct[category],
                                                        "total": size,
                                                        "accuracy": category_correct[category] / size}
                                                for category, size in category_counts.items()}
    return result


def evaluate_original(dataset_dir, output_path, cpu=False):
    from .evaluate import feature_metrics, select_feature_queries, _structure
    from .baselines import train_statistics
    model, device, dtype = load_model(cpu=cpu)
    from .prepare import tokenizer_for_model
    tokenizer = tokenizer_for_model(local_only=True)
    rows = load_prepared(dataset_dir, "validation")
    stats = train_statistics(dataset_dir)
    results = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": str(device),
               "dtype": str(dtype),
               "validation": {"causal": score(model, rows, device, tokenizer),
                              "feature_values": feature_metrics(model, tokenizer, device,
                                  select_feature_queries(rows, cap=6), stats),
                              "structure": _structure(model, tokenizer, device, rows)},
               "test": "reserved until training/checkpoint selection finishes"}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def train(dataset_dir, output_dir, epochs=5, learning_rate=1e-4,
          gradient_accumulation=8, patience=2, seed=415, cpu=False, dry_run=False,
          resume=False):
    if epochs < 1 or epochs > 20 or gradient_accumulation < 1 or not 0 < learning_rate <= .001:
        raise ValueError("epochs 1..20, gradient accumulation >=1, learning rate 0..0.001 required")
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    history_path = output_dir / "history.json"
    adapter_path = output_dir / "best_adapter"
    previous = None
    if resume:
        if not history_path.is_file() or not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"no completed feature checkpoint and history in {output_dir}")
        previous = json.loads(history_path.read_text(encoding="utf-8"))
        completed = len(previous.get("epochs", []))
        if previous.get("best_epoch") != completed:
            raise ValueError("last completed epoch is not the saved best adapter; exact continuation unavailable")
        if epochs <= completed:
            raise ValueError(f"--epochs is the total desired count; must exceed completed {completed}")
        if previous.get("plan", {}).get("model") != MODEL_ID or previous["plan"].get("revision") != MODEL_REVISION:
            raise ValueError("saved checkpoint model/revision does not match this trainer")
    else:
        completed = 0
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    token_info = manifest.get("tokenizer", {})
    if (token_info.get("model_id"), token_info.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("prepared feature dataset belongs to another model or revision")
    tokenizer_path = dataset_dir / "tokenizer/tokenizer.json"
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    training = load_prepared(dataset_dir, "train")
    validation = load_prepared(dataset_dir, "validation")
    if not training or not validation:
        raise ValueError("train and validation splits must be nonempty")
    if resume and (previous["plan"].get("train_segments") != len(training)
                   or previous["plan"].get("validation_segments") != len(validation)
                   or previous["plan"].get("max_sequence_length") != manifest["max_training_length"]):
        raise ValueError("prepared dataset differs from the saved checkpoint plan")
    device, dtype = device_precision()
    if cpu:
        device, dtype = torch.device("cpu"), torch.float32
    plan = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": str(device),
            "tokenizer_sha256": tokenizer_hash,
            "dtype": str(dtype), "lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05,
            "optimizer": "AdamW", "learning_rate": learning_rate, "epochs_limit": epochs,
            "batch_size": 1, "gradient_accumulation": gradient_accumulation,
            "max_sequence_length": manifest["max_training_length"],
            "train_segments": len(training), "validation_segments": len(validation),
            "train_tokens": sum(len(r["token_ids"]) for r in training),
            "validation_tokens": sum(len(r["token_ids"]) for r in validation),
            "checkpoint_selection": "minimum validation causal loss; held-out test never loaded here",
            "resume": resume, "completed_epochs": completed,
            "additional_epochs": epochs - completed,
            "starting_adapter": str(adapter_path) if resume else None,
            "optimizer_state": "reset on continuation; earlier optimizer/scheduler state was not saved"}
    if dry_run:
        return plan
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if resume:
        backup = output_dir / f"best_adapter_epoch{completed}_backup"
        if not backup.exists():
            shutil.copytree(adapter_path, backup)
        history_backup = output_dir / f"history_epoch{completed}_backup.json"
        if not history_backup.exists():
            shutil.copy2(history_path, history_backup)
    model, _, _ = load_model(cpu=cpu)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                      lora_dropout=.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = (PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
             if resume else get_peft_model(model, lora))
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    plan["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    previous_updates = previous["epochs"][-1]["updates"] if resume else 0
    if resume:
        # The former process may have written step logs for an unfinished epoch.
        # Those weights were not checkpointed, so its step logs must be discarded.
        previous["steps"] = [step for step in previous["steps"] if step["update"] <= previous_updates]
    additional_updates = math.ceil(len(training) / gradient_accumulation) * (epochs - completed)
    total_updates = previous_updates + additional_updates
    warmup = max(1, round(additional_updates * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: (step + 1) / warmup if step < warmup else max(.1, .5 * (1 + math.cos(math.pi * (step-warmup) / max(1, additional_updates-warmup)))))
    output_dir.mkdir(parents=True, exist_ok=True)
    history = previous if resume else {"plan": plan, "steps": [], "epochs": [], "best_checkpoint": str(adapter_path)}
    if resume:
        history.setdefault("continuations", []).append(plan)
    best, stale, updates = (previous["best_validation_loss"] if resume else float("inf")), 0, previous_updates
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(completed, epochs):
        rng = random.Random(seed + epoch)
        order = list(range(len(training)))
        rng.shuffle(order)
        epoch_losses, epoch_tokens, epoch_weighted_loss = [], 0, 0.
        model.train()
        for number, index in enumerate(order, 1):
            row = training[index]
            ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            try:
                result = model(input_ids=ids, labels=ids)
                loss = result.loss
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss in epoch {epoch+1}, segment {index}")
                (loss / gradient_accumulation).backward()
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"GPU out of memory on untruncated {len(row['token_ids'])}-token segment from {row['source']}; prepare a shorter length and rerun") from error
            epoch_losses.append(float(loss.detach()))
            epoch_tokens += len(row["token_ids"]) - 1
            epoch_weighted_loss += float(loss.detach()) * (len(row["token_ids"]) - 1)
            del result, loss, ids
            if number % gradient_accumulation == 0 or number == len(order):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                history["steps"].append({"update": updates, "epoch": epoch + 1,
                                         "latest_segment_loss": epoch_losses[-1],
                                         "learning_rate": scheduler.get_last_lr()[0],
                                         "elapsed_seconds": time.perf_counter() - started})
                if updates % 10 == 0:
                    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
                    print(f"epoch {epoch+1} step {updates}/{total_updates} train segment loss {epoch_losses[-1]:.4f}", flush=True)
        val = score(model, validation, device)
        entry = {"epoch": epoch + 1, "train_mean_segment_loss": sum(epoch_losses) / len(epoch_losses),
                 "train_token_weighted_loss": epoch_weighted_loss / epoch_tokens,
                 "train_tokens": epoch_tokens, "validation": val,
                 "elapsed_seconds": time.perf_counter() - started,
                 "updates": updates}
        entry["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        history["epochs"].append(entry)
        if val["loss"] < best - 1e-4:
            best, stale = val["loss"], 0
            model.save_pretrained(adapter_path)
            (adapter_path / "experiment_metadata.json").write_text(json.dumps({
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "tokenizer_sha256": tokenizer_hash, "training_plan": plan}, indent=2) + "\n",
                encoding="utf-8")
            history["best_epoch"] = epoch + 1
            history["best_validation_loss"] = best
        else:
            stale += 1
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"epoch {epoch+1} train {entry['train_mean_segment_loss']:.4f} validation {val['loss']:.4f} best {best:.4f}", flush=True)
        if stale >= patience:
            history["early_stopped"] = True
            break
    run_elapsed = time.perf_counter() - started
    history["elapsed_seconds"] = (previous.get("elapsed_seconds", 0) if resume else 0) + run_elapsed
    if resume:
        plan["elapsed_seconds"] = run_elapsed
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return history


def cpu_smoke(dataset_dir, token_limit=64):
    """One explicit short CPU optimizer step; no dataset artifact is truncated."""
    from peft import LoraConfig, TaskType, get_peft_model
    rows = load_prepared(dataset_dir, "train")
    if not rows:
        raise ValueError("train split is empty")
    model, device, dtype = load_model(cpu=True)
    adapter = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    adapter.train()
    ids = torch.tensor([rows[0]["token_ids"][:token_limit]], dtype=torch.long)
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=1e-4)
    result = adapter(input_ids=ids, labels=ids)
    result.loss.backward()
    optimizer.step()
    return {"status": "passed", "device": "cpu", "source": rows[0]["source"],
            "smoke_tokens": ids.numel(), "full_segment_tokens": len(rows[0]["token_ids"]),
            "loss": float(result.loss.detach()),
            "scope": "one short optimizer step only; not a training or evaluation result"}
