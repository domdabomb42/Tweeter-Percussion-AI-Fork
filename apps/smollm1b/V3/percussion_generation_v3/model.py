"""Target-only LoRA continuation from the feature-pretrained adapter."""
import json
import math
from pathlib import Path
import random
import shutil
import time
from datetime import datetime, timezone

from .data import FEATURE_ADAPTER, load_rows, load_tokenizer


def masked_example(row, device):
    import torch
    prefix, target = row["prefix_ids"], row["target_ids"]
    ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prefix) + target], dtype=torch.long, device=device)
    if not prefix or len(target) < 2:
        raise ValueError("invalid target-only sequence")
    return ids, labels


def cached_target_loss(model, row, device):
    """Backpropagate target tokens against a detached full-prefix KV cache.

    The final prefix token is included in the trainable segment so the first
    target token receives loss. The entire composition still conditions every
    target token; only gradient flow through the long prefix is omitted.
    """
    import torch
    prefix_ids, target_ids = row["prefix_ids"], row["target_ids"]
    if len(prefix_ids) < 2 or len(target_ids) < 2:
        raise ValueError("invalid cached target-only sequence")
    prefix = torch.tensor([prefix_ids[:-1]], dtype=torch.long, device=device)
    segment = torch.tensor([[prefix_ids[-1], *target_ids]], dtype=torch.long, device=device)
    labels = torch.tensor([[-100, *target_ids]], dtype=torch.long, device=device)
    with torch.no_grad():
        context = model(input_ids=prefix, use_cache=True)
        cache = context.past_key_values
    del context
    output = model(input_ids=segment, past_key_values=cache,
                   attention_mask=torch.ones((1, len(prefix_ids) + len(target_ids)),
                                             dtype=torch.long, device=device),
                   labels=labels, use_cache=False)
    loss = output.loss
    del output, cache
    return loss


def load_feature_model(adapter=FEATURE_ADAPTER, cpu=False, trainable=False):
    import torch
    from peft import PeftModel
    from custom_training import MODEL_ID
    from custom_training.train import load_model
    if not Path(adapter).is_dir():
        raise FileNotFoundError(f"feature-pretrained adapter missing: {adapter}")
    config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"adapter belongs to {config.get('base_model_name_or_path')}, expected {MODEL_ID}")
    metadata = Path(adapter) / "experiment_metadata.json"
    if metadata.is_file():
        from custom_training import MODEL_REVISION
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if (info.get("model_id"), info.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("adapter revision does not match this experiment")
    base, device, _ = load_model(cpu=cpu)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    model.to(device)
    model.config.use_cache = not trainable
    return model, device


def heldout_loss(model, rows, device, progress=None):
    import torch
    if not rows:
        return {"loss": None, "target_tokens": 0, "pairs": 0}
    total = count = 0
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            ids, labels = masked_example(row, device)
            result = model(input_ids=ids, labels=labels, use_cache=False)
            n = len(row["target_ids"])
            total += float(result.loss) * n
            count += n
            if progress is not None:
                progress(index, len(rows), total / count)
    loss = total / count
    return {"loss": loss, "perplexity": math.exp(min(100, loss)),
            "target_tokens": count, "pairs": len(rows)}


def move_adam_statistics(optimizer, device):
    """Keep Adam moments off the small GPU except during optimizer.step()."""
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if value is not None and value.device != device:
                state[name] = value.to(device, non_blocking=False)


def train(output_dir, mode, epochs=5, learning_rate=1e-4,
          accumulation=8, patience=2, seed=415, cpu=False, dry_run=False,
          resume=False, restart_from_feature=False, min_delta=1e-4,
          progress_seconds=0):
    if mode != "rhythm":
        raise ValueError("V3 trains only rhythm")
    if (epochs < 1 or epochs > 30 or accumulation < 1 or patience < 1
            or not 0 < learning_rate <= .001 or not 0 <= min_delta <= 1
            or progress_seconds < 0):
        raise ValueError("invalid optimizer settings")
    if resume and restart_from_feature:
        raise ValueError("--resume and --restart-from-feature cannot be combined")
    root = Path(output_dir)
    train_rows = load_rows(root, mode, "train")
    val_rows = load_rows(root, mode, "validation")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    from custom_training import MODEL_ID, MODEL_REVISION
    if manifest.get("model_id", MODEL_ID) != MODEL_ID or manifest.get("revision", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError("V3 prepared data belongs to another model")
    checkpoint = root / "checkpoints" / mode
    history_path = checkpoint / "history.json"
    adapter_path = checkpoint / "best_adapter"
    if not resume and not restart_from_feature and (history_path.exists() or adapter_path.exists()):
        raise ValueError("percussion checkpoint already exists; use --resume or --restart-from-feature")
    if restart_from_feature and not (FEATURE_ADAPTER / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"feature-pretrained adapter missing: {FEATURE_ADAPTER}")
    previous = None
    if resume:
        if not history_path.is_file() or not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"no completed {mode} checkpoint and history to continue")
        previous = json.loads(history_path.read_text(encoding="utf-8"))
        completed = len(previous.get("epochs", []))
        saved_epoch = previous.get("best_epoch")
        if type(saved_epoch) is not int or not 1 <= saved_epoch <= completed:
            raise ValueError("history has no valid best adapter epoch to resume")
        if previous.get("plan", {}).get("mode") != mode:
            raise ValueError("checkpoint mode does not match requested mode")
        if epochs <= completed:
            raise ValueError(f"--epochs is the total desired count; must exceed completed {completed}")
    else:
        completed = 0
        saved_epoch = None
    plan = {"mode": mode, "starting_adapter": str(adapter_path if resume else FEATURE_ADAPTER),
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "tokenizer_sha256": manifest.get("tokenizer_sha256"),
            "train_pairs": len(train_rows), "validation_pairs": len(val_rows),
            "max_length": manifest["max_length"], "epochs": epochs,
            "completed_epochs": completed, "additional_epochs": epochs - completed,
            "starting_checkpoint_epoch": saved_epoch,
            "resume_from_prior_best": bool(resume and saved_epoch != completed),
            "prior_stale_epochs": completed - saved_epoch if resume else 0,
            "resume": resume, "restart_from_feature": restart_from_feature,
            "patience": patience, "min_delta": min_delta,
            "progress_seconds": progress_seconds,
            "optimizer_state": ("reset on continuation; the interrupted run saved adapter weights only"
                                if resume else "new optimizer for a fresh percussion run"),
            "loss_scope": "target tokens and EOS only; conditioning labels are -100",
            "training_memory": "full conditioning prefix cached without gradients; gradients through last prefix token and rhythm target",
            "optimizer_state_memory": "Adam moments on CPU between optimizer updates",
            "checkpoint": str(adapter_path)}
    if dry_run:
        return plan
    load_tokenizer(root)
    if not train_rows or not val_rows:
        raise ValueError(f"no usable {mode} train/validation pairs fit {manifest['max_length']} tokens; "
                         "training cannot start without changing the context strategy/model")
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    if resume:
        # Keep the selected adapter and the complete pre-continuation history.
        backup = checkpoint / f"best_adapter_epoch{saved_epoch}_backup"
        if not backup.exists():
            shutil.copytree(adapter_path, backup)
            shutil.copy2(history_path, checkpoint / f"history_epoch{completed}_backup.json")
        if saved_epoch != completed:
            print(f"{mode}: continuing after epoch {completed} from saved best epoch {saved_epoch}; "
                  "later epoch weights were not checkpointed", flush=True)
    if restart_from_feature and (adapter_path.exists() or history_path.exists()):
        archive = checkpoint / "archives" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        if adapter_path.exists():
            shutil.copytree(adapter_path, archive / "best_adapter")
        if history_path.exists():
            shutil.copy2(history_path, archive / "history.json")
        plan["previous_checkpoint_archive"] = str(archive)
    if progress_seconds:
        print(f"{mode}: loading model; progress reports about every {progress_seconds} seconds", flush=True)
    model, device = load_feature_model(adapter=adapter_path if resume else FEATURE_ADAPTER,
                                       trainable=True, cpu=cpu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.config.use_cache = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate, foreach=False)
    checkpoint.mkdir(parents=True, exist_ok=True)
    history = previous if resume else {"plan": plan, "epochs": []}
    if resume:
        history.setdefault("continuations", []).append(plan)
    best = previous["best_validation_loss"] if resume else float("inf")
    stale = completed - saved_epoch if resume else 0
    started = time.perf_counter()
    last_report = started

    def report_progress(phase, epoch_number, done, total, running_loss):
        nonlocal last_report
        now = time.perf_counter()
        if progress_seconds and now - last_report >= progress_seconds:
            print(f"{mode} epoch {epoch_number} {phase} {done}/{total} "
                  f"running target loss {running_loss:.4f} "
                  f"elapsed {(now-started)/60:.1f} min", flush=True)
            last_report = now

    for epoch in range(completed, epochs):
        order = list(train_rows)
        random.Random(seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted, token_count = 0., 0
        for index, row in enumerate(order):
            if row["tokens"] > manifest["max_length"]:
                raise ValueError("prepared sequence exceeds model context")
            try:
                loss = cached_target_loss(model, row, device)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss: {row['source']} track {row['target_track_id']}")
                group_start = (index // accumulation) * accumulation
                group_size = min(accumulation, len(order) - group_start)
                (loss / group_size).backward()
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"GPU OOM on {row['tokens']} tokens from {row['source']}") from error
            n = len(row["target_ids"])
            weighted += float(loss.detach()) * n
            token_count += n
            del loss
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               1., foreach=False)
                if device.type == "cuda":
                    move_adam_statistics(optimizer, device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    move_adam_statistics(optimizer, torch.device("cpu"))
            report_progress("training pair", epoch + 1, index + 1, len(order),
                            weighted / token_count)
        validation = heldout_loss(
            model, val_rows, device,
            progress=lambda done, total, loss: report_progress(
                "validation pair", epoch + 1, done, total, loss))
        entry = {"epoch": epoch + 1, "train_target_loss": weighted / token_count,
                 "validation": validation, "elapsed_seconds": time.perf_counter() - started}
        entry["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        history["epochs"].append(entry)
        if validation["loss"] < best - min_delta:
            best, stale = validation["loss"], 0
            model.save_pretrained(adapter_path)
            (adapter_path / "experiment_metadata.json").write_text(json.dumps({
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "tokenizer_sha256": manifest.get("tokenizer_sha256"),
                "training_plan": plan}, indent=2) + "\n", encoding="utf-8")
            history["best_epoch"] = epoch + 1
            history["best_validation_loss"] = best
        else:
            stale += 1
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"{mode} epoch {epoch+1}: train {entry['train_target_loss']:.4f} "
              f"validation {validation['loss']:.4f}", flush=True)
        if stale >= patience:
            history["early_stopped"] = True
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            break
    return history
