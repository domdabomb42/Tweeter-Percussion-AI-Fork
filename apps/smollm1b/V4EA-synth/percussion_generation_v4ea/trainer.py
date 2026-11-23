"""Full-gradient musical training with exact step checkpoints."""
import json
import math
from pathlib import Path
import random
import time

from . import MODEL_ID, MODEL_REVISION, OUTPUT, VERSION
from .model import load_model
from .pipeline import verify_dataset
from percussion_generation_v3.checkpoints import (atomic_json, fingerprints,
    load_checkpoint_state, resolve_checkpoint, save_best_adapter, save_checkpoint)
from percussion_generation_v3.model import consume_loss_group, move_adam_statistics
from percussion_generation_v3.rotation import select_epoch_indices, validate_pool
from percussion_generation_v3.data import AUGMENTED_SESSIONS


def full_sequence_loss(model, row, device):
    """Transformers shifts labels internally; the final prefix token predicts target[0]."""
    import torch
    prefix, target = row["prefix_ids"], row["target_ids"]
    if not prefix or len(target) < 2 or len(prefix) + len(target) > 8192:
        raise ValueError("invalid or over-context musical example")
    ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prefix) + target], dtype=torch.long, device=device)
    return model(input_ids=ids, labels=labels, use_cache=False).loss


def optimizer_groups(model, lora_lr, embedding_lr):
    import torch
    underlying = getattr(model, "peft_model", model)
    embedding = underlying.get_input_embeddings().weight
    if not embedding.requires_grad:
        raise ValueError("remapped embeddings are frozen")
    lora = [parameter for name, parameter in underlying.named_parameters()
            if "lora_" in name and parameter.requires_grad]
    other = [name for name, parameter in underlying.named_parameters()
             if parameter.requires_grad and id(parameter) != id(embedding) and "lora_" not in name]
    if not lora or other:
        raise ValueError(f"unexpected trainable parameter groups: {other[:5]}")
    return torch.optim.AdamW([
        {"params": lora, "lr": lora_lr, "initial_lr": lora_lr, "weight_decay": 0.01,
         "name": "lora"},
        {"params": [embedding], "lr": embedding_lr, "initial_lr": embedding_lr,
         "weight_decay": 0.0, "name": "embedding"}], foreach=False)


def warmup_cosine_scheduler(optimizer, total_updates, warmup_fraction=0.1):
    import torch
    warmup = max(1, math.ceil(total_updates * warmup_fraction))
    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        span = max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(step - warmup, span) / span))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale), warmup


def validation_loss(model, rows, device):
    import torch
    model.eval()
    weighted = tokens = 0
    with torch.inference_mode():
        for row in rows:
            value = float(full_sequence_loss(model, row, device))
            count = len(row["target_ids"])
            weighted += value * count
            tokens += count
    return {"loss": weighted / tokens, "target_tokens": tokens, "pairs": len(rows)}


def _validate_pool(rows, manifest):
    augmentation = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
    generated = {row["augmented_filename"]: row for row in augmentation["generated"]}
    parents = {source: split for split, sources in manifest["source_partitions"].items()
               for source in sources}
    validate_pool(rows, {"source_partitions": parents, "generated_lookup": generated},
                  expected_originals=67)


def train(output_dir=OUTPUT, resume_checkpoint=None, epochs=12,
          examples_per_epoch=268, lora_learning_rate=None, embedding_learning_rate=None,
          warmup_fraction=0.1, accumulation=8,
          save_steps=10, patience=5, seed=415, cpu=False, dry_run=False,
          progress_seconds=180, stop_after_steps=None, model_loader=None,
          loss_fn=None, validation_fn=None):
    import torch
    output = Path(output_dir)
    manifest, groups = verify_dataset(output)
    rows = groups["train"]
    if lora_learning_rate is None:
        lora_learning_rate = 5e-5 if "1.7B" in MODEL_ID else 1e-4
    if embedding_learning_rate is None:
        embedding_learning_rate = 1e-4 if "135M" in MODEL_ID else 5e-5
    validation20 = groups["validation20"]
    validation10 = groups["validation"]
    _validate_pool(rows, manifest)
    checkpoint_root = output / "checkpoints/rhythm"
    checksums = fingerprints(output)
    current_device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    if resume_checkpoint:
        checkpoint = resolve_checkpoint(checkpoint_root, resume_checkpoint)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        expected = {"epochs": epochs, "examples_per_epoch": examples_per_epoch,
                    "lora_learning_rate": lora_learning_rate,
                    "embedding_learning_rate": embedding_learning_rate,
                    "warmup_fraction": warmup_fraction, "accumulation": accumulation,
                    "save_steps": save_steps, "patience": patience, "seed": seed,
                    "device_type": current_device,
                    "cuda_device_count": torch.cuda.device_count() if current_device == "cuda" else 0,
                    "torch_version": torch.__version__}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("resume options, device, or PyTorch version differ from checkpoint")
        saved = load_checkpoint_state(checkpoint, config, checksums)
        adapter = checkpoint / "adapter"
    else:
        if (checkpoint_root / "latest.json").exists():
            raise FileExistsError("new V4EA run already has checkpoints; use --resume-checkpoint latest")
        config = {"version": VERSION, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "tokenizer_sha256": manifest["tokenizer_sha256"],
                  "prepared_dataset_sha256": manifest["prepared_dataset_sha256"],
                  "codec_schema_sha256": manifest["codec_schema_sha256"],
                  "preset_catalog_sha256": manifest["preset_catalog_sha256"],
                  "initialization": "pretrained transformer, remapped fresh vocabulary, new LoRA",
                  "epochs": epochs, "examples_per_epoch": examples_per_epoch,
                  "lora_learning_rate": lora_learning_rate,
                  "embedding_learning_rate": embedding_learning_rate,
                  "warmup_fraction": warmup_fraction, "accumulation": accumulation,
                  "save_steps": save_steps, "patience": patience, "seed": seed,
                  "device_type": current_device,
                  "cuda_device_count": torch.cuda.device_count() if current_device == "cuda" else 0,
                  "torch_version": torch.__version__}
        saved = None
        adapter = None
    if (config["version"] != VERSION or config["model_id"] != MODEL_ID or
            config["revision"] != MODEL_REVISION or
            config["tokenizer_sha256"] != manifest["tokenizer_sha256"] or
            config["prepared_dataset_sha256"] != manifest["prepared_dataset_sha256"] or
            config["codec_schema_sha256"] != manifest["codec_schema_sha256"] or
            config["preset_catalog_sha256"] != manifest["preset_catalog_sha256"]):
        raise ValueError("checkpoint belongs to another tokenizer, dataset, or model")
    if (not 1 <= epochs <= 30 or not 67 <= examples_per_epoch <= len(rows) or
            accumulation != 8 or save_steps < 1 or patience < 1 or
            not 0 < lora_learning_rate <= 0.001 or
            not 0 < embedding_learning_rate <= 0.001 or
            not 0 < warmup_fraction < 1):
        raise ValueError("invalid V4EA training configuration")
    updates_per_epoch = math.ceil(examples_per_epoch / accumulation)
    plan = {**config, "train_pool": len(rows), "validation20_pairs": len(validation20),
            "original_training_pairs": 67,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": epochs * updates_per_epoch,
            "warmup_updates": math.ceil(epochs * updates_per_epoch * warmup_fraction),
            "resume_from": str(checkpoint) if resume_checkpoint else None}
    if dry_run:
        return plan
    model_loader = model_loader or load_model
    loss_fn = loss_fn or full_sequence_loss
    validation_fn = validation_fn or validation_loss
    if saved is None:
        torch.manual_seed(seed)
        random.seed(seed)
        if current_device == "cuda":
            torch.cuda.manual_seed_all(seed)
    model, device = model_loader(output, adapter=adapter, cpu=cpu, trainable=True)
    model.config.use_cache = False
    underlying = getattr(model, "peft_model", model)
    if hasattr(underlying, "gradient_checkpointing_enable"):
        underlying.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = optimizer_groups(model, lora_learning_rate, embedding_learning_rate)
    scheduler, _ = warmup_cosine_scheduler(optimizer, plan["total_optimizer_updates"], warmup_fraction)
    embedding_dtype = underlying.get_input_embeddings().weight.dtype
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and embedding_dtype == torch.float16)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler") is not None:
            scaler.load_state_dict(saved["scaler"])
        move_adam_statistics(optimizer, torch.device("cpu"))
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        epoch_index, position = saved["epoch_index"], saved["next_position"]
        order, global_step = saved["selected_order"], saved["global_step"]
        history = saved["history"]
        weighted, target_tokens = saved["weighted_loss"], saved["target_tokens"]
        processed = saved["examples_processed"]
        elapsed = saved["cumulative_elapsed_seconds"]
        epoch_elapsed = saved["epoch_elapsed_seconds"]
        if (not 0 <= epoch_index <= epochs or not 0 <= position <= len(order) or
                (order and (len(order) != examples_per_epoch or len(set(order)) != len(order))) or
                (position and not order) or
                (position % accumulation and position != len(order))):
            raise ValueError("checkpoint example order or position is invalid")
    else:
        epoch_index = position = global_step = 0
        order = []
        history = {"plan": plan, "epochs": [], "stale_epochs": 0}
        weighted = target_tokens = processed = 0
        elapsed = epoch_elapsed = 0.0
    history["plan"]["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters()
                                                  if parameter.requires_grad)
    run_started = epoch_started = last_report = time.perf_counter()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def state_for(next_epoch, next_position, current_order):
        now = time.perf_counter()
        return {"config": config, "fingerprints": checksums,
                "epoch_index": next_epoch, "next_position": next_position,
                "selected_order": current_order, "global_step": global_step,
                "history": history, "weighted_loss": weighted,
                "target_tokens": target_tokens, "examples_processed": processed,
                "cumulative_elapsed_seconds": elapsed + now - run_started,
                "epoch_elapsed_seconds": epoch_elapsed + now - epoch_started,
                "scaler": scaler.state_dict()}

    if saved and (epoch_index >= epochs or history.get("early_stopped")):
        return history
    while epoch_index < epochs:
        if not order:
            order = select_epoch_indices(rows, epoch_index, seed, examples_per_epoch)
            atomic_json(output / "selection" / f"epoch_{epoch_index+1:03d}.json", {
                "epoch": epoch_index+1,
                "examples": [{"index": index, "source": rows[index]["source"],
                              "target_track_id": rows[index]["target_track_id"],
                              "augmentation": rows[index].get("augmentation")}
                             for index in order]})
            print(f"V4EA epoch {epoch_index+1}: selected {len(order)} examples "
                  "(67 original, 201 augmented)", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = []
        for index in range(position, len(order)):
            row = rows[order[index]]
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(order) - group_start)
            try:
                loss = loss_fn(model, row, device)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"full-gradient example needs more GPU memory: {row['source']} "
                    f"track {row['target_track_id']}, {row['tokens']} tokens; "
                    "use a larger-memory GPU or --cpu (slow), without truncating context") from exc
            scaler.scale(loss / group_size).backward()
            pending.append((loss.detach(), len(row["target_ids"]), row))
            processed += 1
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                group_weighted, group_tokens = consume_loss_group(pending)
                weighted += group_weighted
                target_tokens += group_tokens
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               1.0, foreach=False)
                if device.type == "cuda":
                    move_adam_statistics(optimizer, device)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError("FP16 gradient overflow; optimizer step skipped")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    move_adam_statistics(optimizer, torch.device("cpu"))
                global_step += 1
                position = index + 1
                if global_step % save_steps == 0:
                    path = save_checkpoint(checkpoint_root, "periodic",
                        f"step_{global_step:08d}", model, optimizer, scheduler,
                        state_for(epoch_index, position, order))
                    print(f"saved V4EA checkpoint: {path}", flush=True)
                if stop_after_steps is not None and global_step >= stop_after_steps:
                    if global_step % save_steps:
                        save_checkpoint(checkpoint_root, "periodic",
                            f"step_{global_step:08d}", model, optimizer, scheduler,
                            state_for(epoch_index, position, order))
                    raise InterruptedError("V4EA smoke-test stop")
            now = time.perf_counter()
            if progress_seconds and now - last_report >= progress_seconds:
                print(f"V4EA epoch {epoch_index+1}: {index+1}/{len(order)} examples, "
                      f"step {global_step}, target loss "
                      f"{weighted/target_tokens if target_tokens else float('nan'):.4f}", flush=True)
                last_report = now
        val20 = validation_fn(model, validation20, device)
        val10 = validation_fn(model, validation10, device)
        now = time.perf_counter()
        entry = {"epoch": epoch_index+1, "training_loss": weighted/target_tokens,
                 "validation20": val20, "validation_original10": val10,
                 "optimizer_updates": global_step, "training_examples": processed,
                 "target_tokens": target_tokens,
                 "epoch_seconds": epoch_elapsed + now - epoch_started,
                 "cumulative_seconds": elapsed + now - run_started,
                 "lora_learning_rate": optimizer.param_groups[0]["lr"],
                 "embedding_learning_rate": optimizer.param_groups[1]["lr"],
                 "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        history["epochs"].append(entry)
        if val20["loss"] < history.get("best_validation_loss", float("inf")):
            history["best_epoch"] = epoch_index + 1
            history["best_validation_loss"] = val20["loss"]
            history["stale_epochs"] = 0
            save_best_adapter(checkpoint_root, model, {
                "version": VERSION, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "prepared_dataset_sha256": manifest["prepared_dataset_sha256"],
                "codec_schema_sha256": manifest["codec_schema_sha256"],
                "preset_catalog_sha256": manifest["preset_catalog_sha256"],
                "best_epoch": epoch_index+1, "validation20_loss": val20["loss"]})
        else:
            history["stale_epochs"] += 1
        print(f"V4EA epoch {epoch_index+1}: train {entry['training_loss']:.4f}, "
              f"validation20 {val20['loss']:.4f}, original10 {val10['loss']:.4f}; "
              f"step {global_step}", flush=True)
        epoch_index += 1
        position = 0
        order = []
        weighted = target_tokens = processed = 0
        epoch_elapsed = 0.0
        epoch_started = time.perf_counter()
        stopping = history["stale_epochs"] >= patience
        if stopping:
            history["early_stopped"] = True
        atomic_json(checkpoint_root / "history.json", history)
        save_checkpoint(checkpoint_root, "epoch_boundaries",
                        f"epoch_{epoch_index:03d}_step_{global_step:08d}",
                        model, optimizer, scheduler, state_for(epoch_index, 0, []))
        if stopping:
            break
    atomic_json(checkpoint_root / "history.json", history)
    save_checkpoint(checkpoint_root, "final", f"step_{global_step:08d}",
                    model, optimizer, scheduler, state_for(epoch_index, position, order))
    return history
