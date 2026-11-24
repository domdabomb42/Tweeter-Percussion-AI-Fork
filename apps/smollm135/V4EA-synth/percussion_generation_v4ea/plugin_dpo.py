"""LoRA-only, rating-weighted DPO with exact safe-boundary continuation."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from . import MODEL_ID, MODEL_REVISION
from .model import load_model
from .pipeline import sha256, verify_dataset
from .plugin_checkpoints import recover, save
from .plugin_preference import (cached_reference, completion_logprob,
                                load_snapshot, weighted_dpo_loss)
from .plugin_store import canonical, digest


def frozen_parameter_hash(model):
    """Hash actual loaded non-LoRA tensors, including the tied embedding/head."""
    import torch
    underlying = model.peft_model
    value = hashlib.sha256()
    for name, parameter in underlying.named_parameters():
        if "lora_" in name:
            continue
        tensor = parameter.detach().cpu().contiguous()
        value.update(name.encode("utf-8") + b"\0")
        value.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        value.update(tensor.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def lora_optimizer(model, learning_rate=5e-6):
    import torch
    underlying = model.peft_model
    for name, parameter in underlying.named_parameters():
        parameter.requires_grad_("lora_" in name)
    parameters = [parameter for name, parameter in underlying.named_parameters()
                  if parameter.requires_grad]
    if not parameters or any("lora_" not in name for name, p in underlying.named_parameters() if p.requires_grad):
        raise ValueError("DPO optimizer would update a non-LoRA parameter")
    embedding = underlying.get_input_embeddings().weight
    head = underlying.get_output_embeddings().weight
    if embedding.requires_grad or head.requires_grad or embedding.data_ptr() != head.data_ptr():
        raise ValueError("trained tied embeddings/head are not frozen")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01, foreach=False)


def _atomic_progress(path, value):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".progress_", delete=False) as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_inputs(run_dir, output_dir, snapshot, snapshot_hash, policy_adapter,
                     epochs, accumulation, learning_rate, beta, save_steps,
                     save_total_limit, seed, patience):
    manifest, _ = verify_dataset(output_dir)
    for key, expected in (("tokenizer_sha256", snapshot["tokenizer_sha256"]),
                          ("prepared_dataset_sha256", snapshot["dataset_sha256"]),
                          ("codec_schema_sha256", snapshot["codec_schema_sha256"]),
                          ("fixed_preset_sha256", snapshot["fixed_preset_sha256"])):
        if manifest[key] != expected:
            raise ValueError(f"DPO snapshot incompatible with {key}")
    supervised = Path(snapshot["reference_adapter"])
    if digest((supervised / "adapter_model.safetensors").read_bytes()) != snapshot["reference_adapter_sha256"]:
        raise ValueError("original supervised LoRA weights changed")
    if digest((supervised / "embedding.safetensors").read_bytes()) != snapshot["embedding_sha256"]:
        raise ValueError("original trained embeddings changed")
    if sha256(Path(output_dir) / "tokenizer/tokenizer.json") != snapshot["tokenizer_sha256"]:
        raise ValueError("trained tokenizer changed")
    if (epochs < 1 or accumulation < 1 or learning_rate <= 0 or beta <= 0 or
            save_steps < 1 or save_total_limit < 3 or patience < 1):
        raise ValueError("invalid DPO hyperparameters")
    training = snapshot["pairs"]["train"]
    if not training:
        raise ValueError("DPO snapshot has no training pairs")
    if set(snapshot["train_families"]) & set(snapshot["validation_families"]):
        raise ValueError("DPO source-family leakage")
    return {"format": "TweeterDPO-run-v1", "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "snapshot_sha256": snapshot_hash,
            "reference_adapter_sha256": snapshot["reference_adapter_sha256"],
            "embedding_sha256": snapshot["embedding_sha256"],
            "tokenizer_sha256": snapshot["tokenizer_sha256"],
            "policy_initial_adapter": str(Path(policy_adapter).resolve()),
            "policy_initial_adapter_sha256": digest((Path(policy_adapter) / "adapter_model.safetensors").read_bytes()),
            "epochs": epochs, "accumulation": accumulation, "learning_rate": learning_rate,
            "beta": beta, "save_steps": save_steps, "save_total_limit": save_total_limit,
            "seed": seed, "patience": patience, "gradient_clip": 1.0,
            "train_pairs": len(training), "validation_pairs": len(snapshot["pairs"]["validation"]),
            "total_updates": epochs * math.ceil(len(training) / accumulation)}


def _loss(model, pair, snapshot, run_dir, device, beta, scale):
    chosen = completion_logprob(model, pair["prompt_ids"], pair["chosen_ids"], device)
    rejected = completion_logprob(model, pair["prompt_ids"], pair["rejected_ids"], device)
    reference_chosen = cached_reference(run_dir, snapshot, pair, "chosen")
    reference_rejected = cached_reference(run_dir, snapshot, pair, "rejected")
    return weighted_dpo_loss(chosen, rejected, reference_chosen, reference_rejected,
                             beta, pair["group_weight"] * scale)


def _validation(model, pairs, snapshot, run_dir, device, beta):
    import torch
    if not pairs:
        return None
    model.eval()
    groups = len({pair["group_id"] for pair in pairs})
    total = 0.0
    with torch.inference_mode():
        for pair in pairs:
            total += float(_loss(model, pair, snapshot, run_dir, device, beta, 1))
    return total / groups


def train_dpo(run_dir, output_dir, *, policy_adapter=None, resume_checkpoint=None,
              epochs=1, accumulation=4, learning_rate=5e-6, beta=.1,
              save_steps=10, save_total_limit=3, seed=415, patience=3,
              cpu=False, dry_run=False, max_optimizer_steps=None,
              save_on_interrupt=True, model_loader=None):
    import torch
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    snapshot, snapshot_hash = load_snapshot(run_dir)
    reference = Path(snapshot["reference_adapter"])
    policy_adapter = Path(policy_adapter or reference)
    config = _validate_inputs(run_dir, output_dir, snapshot, snapshot_hash, policy_adapter,
                              epochs, accumulation, learning_rate, beta, save_steps,
                              save_total_limit, seed, patience)
    checkpoint_root = run_dir / "checkpoints"
    if dry_run:
        return {**config, "run_dir": str(run_dir), "reference_scores_required": True,
                "optimizer_updates_per_epoch": math.ceil(config["train_pairs"] / accumulation)}
    if resume_checkpoint is None and (checkpoint_root / "latest.json").exists():
        raise FileExistsError("DPO run already has checkpoints; use --resume-checkpoint latest")
    saved = None
    if resume_checkpoint:
        checkpoint, saved, failures = recover(checkpoint_root, resume_checkpoint,
                                              snapshot_hash, config)
        print(f"DPO restoring {checkpoint} at step {saved['global_step']}"
              + (f"; rolled back after {len(failures)} invalid newer checkpoints" if failures else ""), flush=True)
        adapter = checkpoint / "adapter"
    else:
        checkpoint = None
        adapter = policy_adapter
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    loader = model_loader or load_model
    model, device = loader(output_dir, adapter=adapter, cpu=cpu, trainable=True)
    model.config.use_cache = False
    underlying = model.peft_model
    if hasattr(underlying, "gradient_checkpointing_enable"):
        underlying.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = lora_optimizer(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / max(1, config["total_updates"])))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and
                                  next(p for p in underlying.parameters() if p.requires_grad).dtype == torch.float16)
    frozen_before = frozen_parameter_hash(model)
    if saved:
        if frozen_before != saved["frozen_parameter_sha256"]:
            raise ValueError("frozen base or embeddings changed on checkpoint reload")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"])
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
        try:
            import numpy as np
            if saved.get("numpy_rng") is not None:
                np.random.set_state(saved["numpy_rng"])
        except ImportError:
            pass
        epoch, cursor, order = saved["epoch_index"], saved["cursor"], saved["order"]
        step, history = saved["global_step"], saved["history"]
        if order and sorted(order) != list(range(len(snapshot["pairs"]["train"]))):
            raise ValueError("checkpoint shuffled order is not a permutation")
    else:
        epoch, cursor, order, step = 0, 0, [], 0
        history = {"epochs": [], "checkpoint_saves": [], "best_validation_loss": None,
                   "best_checkpoint": None, "stale_epochs": 0,
                   "frozen_parameter_sha256": frozen_before}
        if policy_adapter.resolve() == reference.resolve():
            pair = snapshot["pairs"]["train"][0]
            model.eval()
            with torch.inference_mode():
                policy_score = float(completion_logprob(model, pair["prompt_ids"], pair["chosen_ids"], device))
            reference_score = cached_reference(run_dir, snapshot, pair, "chosen")
            if abs(policy_score - reference_score) > .05:
                raise ValueError("initial policy and frozen supervised reference disagree")
    if len([p for p in underlying.parameters() if p.requires_grad]) != len(optimizer.param_groups[0]["params"]):
        raise ValueError("optimizer does not contain exactly the trainable LoRA tensors")
    metadata = {"version": json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["version"],
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "tokenizer_sha256": snapshot["tokenizer_sha256"],
                "prepared_dataset_sha256": snapshot["dataset_sha256"],
                "codec_schema_sha256": snapshot["codec_schema_sha256"],
                "fixed_preset_sha256": snapshot["fixed_preset_sha256"],
                "dpo_snapshot_sha256": snapshot_hash,
                "frozen_parameter_sha256": frozen_before}
    try:
        import numpy as np
    except ImportError:
        np = None

    def state():
        return {"config": config, "snapshot_sha256": snapshot_hash,
                "frozen_parameter_sha256": frozen_before,
                "epoch_index": epoch, "cursor": cursor, "order": list(order),
                "global_step": step, "history": history,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state() if np is not None else None,
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "scaler": scaler.state_dict()}

    def checkpoint_save(category, label):
        expected = checkpoint_root / category / label
        item = {"path": str(expected), "step": step, "epoch_index": epoch, "cursor": cursor}
        history["checkpoint_saves"].append(item)
        try:
            path = save(checkpoint_root, category, label, model, optimizer, scheduler,
                        state(), metadata, save_total_limit)
        except Exception:
            history["checkpoint_saves"].pop()
            raise
        return path

    interrupted = False
    def request_interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_interrupt)
    started = time.perf_counter()
    training = snapshot["pairs"]["train"]
    validation = snapshot["pairs"]["validation"]
    group_scale = len(training) / len({pair["group_id"] for pair in training})
    try:
        while epoch < epochs:
            if not order:
                order = list(range(len(training)))
                random.shuffle(order)
                cursor = 0
            model.train()
            while cursor < len(order):
                group_size = min(accumulation, len(order)-cursor)
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for offset in range(group_size):
                    pair = training[order[cursor+offset]]
                    # Mean over pair microbatches, while the normalized group
                    # weights make every rated generation group equal per epoch.
                    loss = _loss(model, pair, snapshot, run_dir, device, beta, group_scale)
                    group_loss += float(loss.detach())
                    scaler.scale(loss / group_size).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("DPO optimizer update skipped after gradient overflow")
                scheduler.step()
                cursor += group_size
                step += 1
                progress = {"status": "training", "epoch": epoch+1, "examples": cursor,
                            "epoch_examples": len(order), "global_optimizer_step": step,
                            "last_group_loss": group_loss/group_size,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "elapsed_seconds": time.perf_counter()-started}
                _atomic_progress(run_dir / "progress.json", progress)
                if step % save_steps == 0:
                    checkpoint_save("periodic", f"step_{step:08d}")
                if (run_dir / "pause.request").exists() or interrupted or (
                        max_optimizer_steps is not None and step >= max_optimizer_steps):
                    if save_on_interrupt:
                        checkpoint_save("pause", f"step_{step:08d}_{uuid_suffix()}")
                    if frozen_parameter_hash(model) != frozen_before:
                        raise ValueError("frozen parameters changed before DPO pause")
                    (run_dir / "pause.request").unlink(missing_ok=True)
                    progress["status"] = "paused"
                    _atomic_progress(run_dir / "progress.json", progress)
                    return {"status": "paused", "global_step": step, "epoch": epoch+1,
                            "cursor": cursor, "checkpoint": str(checkpoint_root / "latest.json")}
            train_loss = _validation(model, training, snapshot, run_dir, device, beta)
            validation_loss = _validation(model, validation, snapshot, run_dir, device, beta)
            record = {"epoch": epoch+1, "train_dpo_loss": train_loss,
                      "validation_dpo_loss": validation_loss, "optimizer_steps": step,
                      "elapsed_seconds": time.perf_counter()-started}
            history["epochs"].append(record)
            improved = validation_loss is not None and (history["best_validation_loss"] is None or
                        validation_loss < history["best_validation_loss"])
            if improved:
                history["best_validation_loss"] = validation_loss
                history["stale_epochs"] = 0
            elif validation_loss is not None:
                history["stale_epochs"] += 1
            epoch += 1
            cursor, order = 0, []
            epoch_path = checkpoint_save("epochs", f"epoch_{epoch:03d}_step_{step:08d}")
            if improved:
                best_path = checkpoint_save("best", f"epoch_{epoch:03d}_step_{step:08d}")
                history["best_checkpoint"] = str(best_path)
            _atomic_progress(run_dir / "progress.json", {"status": "epoch_complete", **record})
            if validation_loss is not None and epochs > 1 and history["stale_epochs"] >= patience:
                history["early_stopped"] = True
                break
            if interrupted:
                checkpoint_save("pause", f"step_{step:08d}_{uuid_suffix()}")
                _atomic_progress(run_dir / "progress.json", {"status": "paused", "epoch": epoch,
                    "global_optimizer_step": step, "examples": 0, "epoch_examples": len(training)})
                return {"status": "paused", "global_step": step, "epoch": epoch,
                        "cursor": 0, "checkpoint": str(checkpoint_root / "latest.json")}
        final = checkpoint_save("final", f"step_{step:08d}")
        final_frozen = frozen_parameter_hash(model)
        if final_frozen != frozen_before:
            raise ValueError("frozen base or embedding parameters changed during DPO")
        _atomic_progress(run_dir / "progress.json", {"status": "complete", "epochs": epoch,
            "global_optimizer_step": step, "final_checkpoint": str(final),
            "frozen_parameter_sha256": final_frozen})
        return {"status": "complete", "epochs": epoch, "global_step": step,
                "final_checkpoint": str(final), "history": history}
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def uuid_suffix():
    import uuid
    return uuid.uuid4().hex[:8]
