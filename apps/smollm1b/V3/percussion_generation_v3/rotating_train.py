"""Epoch-7 warm start with rotating examples and exact step-boundary resume."""
import json
import math
from pathlib import Path
import random
import time
from contextlib import nullcontext

from .checkpoints import (atomic_json, fingerprints, load_checkpoint_state,
                          resolve_checkpoint, save_best_adapter, save_checkpoint,
                          sha256_file)
from .data import AUGMENTED_SESSIONS, ROOT, SMOLLM, load_rows, load_tokenizer, verify_v2_dataset
from .rotation import example_id, select_epoch_indices, validate_pool
from .profiling import StageProfiler


HISTORICAL_ADAPTER = ROOT / "output/checkpoints/rhythm/best_adapter"
HISTORICAL_HISTORY = ROOT / "output/checkpoints/rhythm/history.json"
BASELINE_DATASET_SHA256 = "f03eab45f79d31d462bee74c30755e3c5cb9ebca7bb0eb96eff2d4508078476a"
BASELINE_ADAPTER_SHA256 = "4fb56263372ed8f74e5ad6b516edb60ecfc7b1ae33086d30a3d5fad6a1306a32"
BASELINE_VALIDATION20_LOSS = 0.7691895557198547


def overall_epoch(config, experiment_epoch):
    return experiment_epoch + (config.get("historical_best_epoch") or 0)


def historical_validation20_baseline(config):
    """Measured compact-format epoch-7 score on the same 20 target pairs."""
    if (SMOLLM.name != "smollm135" or
            config.get("historical_best_epoch") != 7 or
            config.get("prepared_dataset_sha256") != BASELINE_DATASET_SHA256 or
            config.get("init_adapter_sha256") != BASELINE_ADAPTER_SHA256):
        return None
    return {"epoch": 7, "validation20_loss": BASELINE_VALIDATION20_LOSS,
            "target_tokens": 414, "codec_version": "compact",
            "adapter_sha256": BASELINE_ADAPTER_SHA256,
            "method": "saved original-10 loss plus measured added-10 loss, target-token weighted"}


def normalize_history(history, config):
    """Label prior new-run epochs after epoch 7 without changing training state."""
    offset = config.get("historical_best_epoch") or 0
    baseline = historical_validation20_baseline(config)
    if offset and history.get("epoch_numbering") != "historical-continuation":
        for entry in history.get("epochs", []):
            entry["experiment_epoch"] = entry["epoch"]
            entry["epoch"] = overall_epoch(config, entry["epoch"])
        if history.get("best_epoch") is not None:
            history["best_experiment_epoch"] = history["best_epoch"]
            history["best_epoch"] = overall_epoch(config, history["best_epoch"])
        history["epoch_numbering"] = "historical-continuation"
    if baseline:
        existing = history.get("historical_validation20_baseline")
        if existing is not None and existing != baseline:
            raise ValueError("historical validation baseline changed")
        history["historical_validation20_baseline"] = baseline
        if history.get("best_validation_loss", float("inf")) >= baseline["validation20_loss"]:
            history["best_epoch"] = 7
            history["best_experiment_epoch"] = None
            history["best_validation_loss"] = baseline["validation20_loss"]
    return history


def verify_epoch7_adapter(path):
    from custom_training import MODEL_ID, MODEL_REVISION
    path = Path(path).resolve()
    if path != HISTORICAL_ADAPTER.resolve():
        raise ValueError(f"this experiment requires the original epoch-7 adapter: {HISTORICAL_ADAPTER}")
    history = json.loads(HISTORICAL_HISTORY.read_text(encoding="utf-8"))
    if history.get("best_epoch") != 7 or not math.isclose(history.get("best_validation_loss", -1),
                                                          0.6110754831902032, abs_tol=1e-5):
        raise ValueError("historical checkpoint is no longer the verified epoch-7 best")
    metadata = json.loads((path / "experiment_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("model_id"), metadata.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("historical adapter model identity changed")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return sha256_file(weights)


def verify_initial_adapter(path):
    """Keep 135M's epoch-7 provenance; allow model-local warm starts elsewhere."""
    if SMOLLM.name == "smollm135":
        return verify_epoch7_adapter(path)
    from custom_training import MODEL_ID, MODEL_REVISION
    path = Path(path).resolve()
    if not path.is_relative_to(SMOLLM.resolve()):
        raise ValueError("initial adapter must belong to this model variant")
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("initial adapter belongs to another pretrained model")
    metadata_path = path / "experiment_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("model_id"), metadata.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("initial adapter revision differs from pinned model")
    return sha256_file(path / "adapter_model.safetensors")


def cosine_scheduler(optimizer, total_updates):
    from torch.optim.lr_scheduler import LambdaLR
    def factor(step):
        progress = min(step / max(1, total_updates), 1.0)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    return LambdaLR(optimizer, factor)


def _validate_data(root, rows, original_validation, fixed_validation, manifest):
    from custom_training import MODEL_ID, MODEL_REVISION
    if (manifest.get("model_id"), manifest.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("prepared data belongs to another model")
    if manifest.get("encoding") == "lossless-v2":
        from .codec_v2 import SCHEMA_SHA256
        if manifest.get("codec_schema_sha256") != SCHEMA_SHA256:
            raise ValueError("prepared codec schema changed")
    experiment = manifest.get("epoch7_experiment", {})
    if len(original_validation) != 10 or len(fixed_validation) != 20 or fixed_validation[:10] != original_validation:
        raise ValueError("fixed validation must start with the unchanged original 10 pairs")
    if experiment.get("validation_fixed_pairs") != 20:
        raise ValueError("epoch-7 experiment manifest missing fixed validation provenance")
    source_sets = {"train": {r["source"] for r in rows},
                   "validation": {r["source"] for r in fixed_validation},
                   "test": {r["source"] for r in load_rows(root, "rhythm", "test")}}
    if any(source_sets[a] & source_sets[b] for a,b in
           (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("original source sessions cross train/validation/test")
    augmentation = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(AUGMENTED_SESSIONS / "manifest.json") != experiment["augmentation_manifest_sha256"]:
        raise ValueError("augmentation manifest fingerprint changed")
    lookup = {"source_partitions": experiment["source_partitions"],
              "generated_lookup": {r["augmented_filename"]: r for r in augmentation["generated"]}}
    validate_pool(rows, lookup, expected_originals=67)
    for row in fixed_validation[10:]:
        shift = row.get("augmentation", {})
        record = lookup["generated_lookup"].get(shift.get("file"))
        if not record or record["original_session"] != row["source"] or row["source"] not in source_sets["validation"]:
            raise ValueError("fixed validation shift has wrong parent")


def train_rotating(output_dir, init_adapter=None, resume_checkpoint=None,
                   examples_per_epoch=None, learning_rate=None, save_steps=None,
                   epochs=None, accumulation=8, patience=None, seed=415,
                   cpu=False, dry_run=False, progress_seconds=180,
                   model_loader=None, loss_fn=None, validation_fn=None,
                   stop_after_steps=None, profile=False):
    """Train a new experiment; test hooks use a tiny model without a full run."""
    import torch
    from .model import (cached_target_loss, consume_loss_group, heldout_loss, load_feature_model,
                        move_adam_statistics)
    from custom_training import MODEL_ID, MODEL_REVISION

    root = Path(output_dir)
    rows = load_rows(root, "rhythm", "train")
    original_validation = load_rows(root, "rhythm", "validation")
    fixed_validation = [json.loads(line) for line in
                        (root / "rhythm/validation20.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    verify_v2_dataset(root, manifest)
    _validate_data(root, rows, original_validation, fixed_validation, manifest)
    checksums = fingerprints(root)
    checkpoint_root = root / "checkpoints/rhythm"
    if resume_checkpoint:
        if init_adapter is not None:
            raise ValueError("--init-adapter and --resume-checkpoint are separate modes")
        checkpoint_path = resolve_checkpoint(checkpoint_root, resume_checkpoint)
        metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        current_device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
        if (config["device_type"] != current_device
                or config["cuda_device_count"] != (torch.cuda.device_count() if current_device == "cuda" else 0)
                or config["torch_version"] != torch.__version__):
            raise ValueError("exact resume requires the saved device type, GPU count, and PyTorch version")
        for provided, key in ((examples_per_epoch,"examples_per_epoch"),
                              (learning_rate,"learning_rate"),(save_steps,"save_steps"),(epochs,"epochs")):
            if provided is not None and provided != config[key]:
                raise ValueError(f"resume must retain the saved {key}={config[key]}")
        if (accumulation, seed) != (config["accumulation"], config["seed"]):
            raise ValueError("resume accumulation or seed differs from checkpoint")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive")
        saved = load_checkpoint_state(checkpoint_path, config, checksums)
        previous_patience = config["patience"]
        if patience is not None:
            # Early stopping controls only future epochs. Restore the exact
            # optimizer/scheduler/RNG state first, then record this explicit
            # policy change in newly written checkpoints.
            config = {**config, "patience": patience}
        starting_adapter = checkpoint_path / "adapter"
    else:
        if init_adapter is None:
            raise ValueError("new rotating experiment requires --init-adapter")
        historical_hash = verify_initial_adapter(init_adapter)
        device_type = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
        config = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "warm_start": ("original V3 best adapter at epoch 7; new optimizer and schedule"
                                 if SMOLLM.name == "smollm135" else
                                 "model-local adapter; new optimizer and schedule"),
                  "historical_best_epoch": 7 if SMOLLM.name == "smollm135" else None,
                  "historical_validation_loss": (0.6110754831902032
                                                 if SMOLLM.name == "smollm135" else None),
                  "init_adapter": str(Path(init_adapter).resolve()), "init_adapter_sha256": historical_hash,
                  "device_type": device_type,
                  "cuda_device_count": torch.cuda.device_count() if device_type == "cuda" else 0,
                  "torch_version": torch.__version__,
                  "epochs": 12 if epochs is None else epochs,
                  "examples_per_epoch": 268 if examples_per_epoch is None else examples_per_epoch,
                  "learning_rate": 2e-5 if learning_rate is None else learning_rate,
                  "optimizer": "AdamW", "weight_decay": 0.01,
                  "codec_version": manifest.get("encoding", "compact"),
                  "codec_schema_sha256": manifest.get("codec_schema_sha256"),
                  "tokenizer_sha256": manifest.get("tokenizer_sha256"),
                  "prepared_dataset_sha256": manifest.get("prepared_dataset_sha256"),
                  "accumulation": accumulation,
                  "patience": 3 if patience is None else patience, "seed": seed,
                  "save_steps": 10 if save_steps is None else save_steps,
                  "scheduler": "cosine decay to 10 percent of initial learning rate",
                  "validation_selection": "lowest fixed 20-pair validation loss"}
        starting_adapter = Path(init_adapter)
        saved = None
        if (checkpoint_root / "latest.json").exists() or (checkpoint_root / "history.json").exists():
            raise FileExistsError("new experiment already has checkpoints; use --resume-checkpoint latest")
    if (not 1 <= config["epochs"] <= 30 or not 67 <= config["examples_per_epoch"] <= len(rows)
            or config["accumulation"] != 8 or config["save_steps"] < 1
            or config["patience"] < 1 or not 0 < config["learning_rate"] <= 0.001):
        raise ValueError("invalid rotating experiment configuration")
    updates_per_epoch = math.ceil(config["examples_per_epoch"] / config["accumulation"])
    plan = {**config, "train_pool": len(rows), "original_pairs": 67,
            "fixed_validation_pairs": 20, "original_validation_pairs": 10,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "total_optimizer_updates": updates_per_epoch * config["epochs"],
            "dataset_fingerprints": checksums,
            "starting_adapter": str(starting_adapter),
            "resume_checkpoint": str(checkpoint_path) if resume_checkpoint else None}
    if dry_run:
        return plan
    load_tokenizer(root)
    model_loader = model_loader or load_feature_model
    loss_fn = loss_fn or cached_target_loss
    validation_fn = validation_fn or heldout_loss
    model, device = model_loader(adapter=starting_adapter, cpu=cpu, trainable=True)
    profiler = StageProfiler(device) if profile else None
    model.config.use_cache = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], foreach=False)
    scheduler = cosine_scheduler(optimizer, plan["total_optimizer_updates"])
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        move_adam_statistics(optimizer, torch.device("cpu"))
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        epoch_index = saved["epoch_index"]
        position = saved["next_position"]
        order = saved["selected_order"]
        global_step = saved["global_step"]
        history = normalize_history(saved["history"], config)
        if config["patience"] != previous_patience:
            history.setdefault("patience_changes", []).append({
                "at_global_step": saved["global_step"],
                "from": previous_patience, "to": config["patience"]})
        weighted = saved["weighted_loss"]
        target_tokens = saved["target_tokens"]
        processed = saved["examples_processed"]
        base_elapsed = saved["cumulative_elapsed_seconds"]
        epoch_base_elapsed = saved["epoch_elapsed_seconds"]
        if (not 0 <= epoch_index <= config["epochs"] or not 0 <= position <= len(order)
                or (order and (len(order) != config["examples_per_epoch"]
                               or len(set(order)) != len(order)
                               or any(not 0 <= i < len(rows) for i in order)))
                or (position and not order)
                or (position % config["accumulation"] and position != len(order))):
            raise ValueError("checkpoint position or selected example order is invalid")
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        epoch_index = position = global_step = 0
        order = []
        history = normalize_history({"plan": plan, "epochs": []}, config)
        weighted = 0.0
        target_tokens = processed = 0
        base_elapsed = epoch_base_elapsed = 0.0
    run_started = time.perf_counter()
    epoch_started = run_started
    last_report = run_started
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def state_for(next_epoch, next_position, current_order):
        now = time.perf_counter()
        return {"config": config, "fingerprints": checksums,
                "epoch_index": next_epoch, "next_position": next_position,
                "selected_order": current_order, "global_step": global_step,
                "history": history, "weighted_loss": weighted,
                "target_tokens": target_tokens, "examples_processed": processed,
                "cumulative_elapsed_seconds": base_elapsed + now - run_started,
                "epoch_elapsed_seconds": epoch_base_elapsed + now - epoch_started}

    if saved and (epoch_index >= config["epochs"] or history.get("early_stopped")):
        final_path = checkpoint_root / "final" / f"step_{global_step:08d}"
        if not final_path.exists():
            save_checkpoint(checkpoint_root, "final", final_path.name,
                            model, optimizer, scheduler,
                            state_for(epoch_index, position, order))
        return history

    while epoch_index < config["epochs"]:
        if not order:
            order = select_epoch_indices(rows, epoch_index, seed, config["examples_per_epoch"])
            atomic_json(root / "selection" / f"epoch_{epoch_index+1:03d}.json", {
                "epoch": epoch_index + 1,
                "overall_epoch": overall_epoch(config, epoch_index + 1), "seed": seed,
                "order": [{"index": i, "source": rows[i]["source"],
                           "target_track_id": rows[i]["target_track_id"],
                           "augmentation": rows[i].get("augmentation")}
                          for i in order]})
            print(f"rhythm epoch {overall_epoch(config, epoch_index+1)}: "
                  f"selected {len(order)} distinct examples "
                  f"(67 original, {len(order)-67} shifted)", flush=True)
        if len(order) != config["examples_per_epoch"] or len(set(order)) != len(order):
            raise ValueError("saved epoch order is incomplete or duplicated")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = []
        for index in range(position, len(order)):
            row = rows[order[index]]
            if row["tokens"] > manifest["max_length"]:
                raise ValueError("prepared sequence exceeds model context")
            loss = (loss_fn(model, row, device, profiler=profiler)
                    if profiler and loss_fn is cached_target_loss else loss_fn(model, row, device))
            group_start = (index // config["accumulation"]) * config["accumulation"]
            group_size = min(config["accumulation"], len(order) - group_start)
            with (profiler.stage("backward") if profiler else nullcontext()):
                (loss / group_size).backward()
            n = len(row["target_ids"])
            pending.append((loss.detach(), n, row))
            processed += 1
            if (index + 1) % config["accumulation"] == 0 or index + 1 == len(order):
                group_weighted, group_tokens = consume_loss_group(pending)
                weighted += group_weighted
                target_tokens += group_tokens
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               1.0, foreach=False)
                with (profiler.stage("optimizer_transfer_to_device") if profiler else nullcontext()):
                    if device.type == "cuda":
                        move_adam_statistics(optimizer, device)
                with (profiler.stage("optimizer_step") if profiler else nullcontext()):
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with (profiler.stage("optimizer_transfer_to_cpu") if profiler else nullcontext()):
                    if device.type == "cuda":
                        move_adam_statistics(optimizer, torch.device("cpu"))
                global_step += 1
                position = index + 1
                if global_step % config["save_steps"] == 0:
                    saved_path = save_checkpoint(checkpoint_root, "periodic",
                        f"step_{global_step:08d}", model, optimizer, scheduler,
                        state_for(epoch_index, position, order))
                    print(f"saved step checkpoint: {saved_path}", flush=True)
                if stop_after_steps is not None and global_step >= stop_after_steps:
                    if global_step % config["save_steps"]:
                        save_checkpoint(checkpoint_root, "periodic",
                            f"step_{global_step:08d}", model, optimizer, scheduler,
                            state_for(epoch_index, position, order))
                    raise InterruptedError("smoke-test stop after optimizer step")
            now = time.perf_counter()
            if progress_seconds and now - last_report >= progress_seconds:
                print(f"rhythm epoch {overall_epoch(config, epoch_index+1)}: "
                      f"{index+1}/{len(order)} examples, "
                      f"step {global_step}, target loss "
                      f"{weighted/target_tokens if target_tokens else float('nan'):.4f}", flush=True)
                last_report = now
        validation20 = (validation_fn(model, fixed_validation, device, profiler=profiler)
                        if profiler and validation_fn is heldout_loss else
                        validation_fn(model, fixed_validation, device))
        validation10 = (validation_fn(model, original_validation, device, profiler=profiler)
                        if profiler and validation_fn is heldout_loss else
                        validation_fn(model, original_validation, device))
        now = time.perf_counter()
        entry = {"epoch": overall_epoch(config, epoch_index + 1),
                 "experiment_epoch": epoch_index + 1,
                 "training_loss": weighted / target_tokens,
                 "validation20": validation20, "validation_original10": validation10,
                 "optimizer_updates": global_step, "epoch_optimizer_updates": updates_per_epoch,
                 "training_examples": processed, "target_tokens": target_tokens,
                 "epoch_seconds": epoch_base_elapsed + now - epoch_started,
                 "cumulative_seconds": base_elapsed + now - run_started,
                 "learning_rate": optimizer.param_groups[0]["lr"]}
        history["epochs"].append(entry)
        if profiler:
            profiler.save(checkpoint_root / f"profile_epoch_{epoch_index+1:03d}.json",
                          model, examples=len(order), extra={"epoch": epoch_index + 1,
                                                          "model_id": MODEL_ID,
                                                          "revision": MODEL_REVISION})
        previous_best = history.get("best_validation_loss", float("inf"))
        if validation20["loss"] < previous_best:
            history["best_epoch"] = overall_epoch(config, epoch_index + 1)
            history["best_experiment_epoch"] = epoch_index + 1
            history["best_validation_loss"] = validation20["loss"]
            save_best_adapter(checkpoint_root, model, {
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "codec_version": config["codec_version"],
                "codec_schema_sha256": config["codec_schema_sha256"],
                "tokenizer_sha256": config["tokenizer_sha256"],
                "prepared_dataset_sha256": config["prepared_dataset_sha256"],
                "warm_start": config["warm_start"],
                "best_epoch": overall_epoch(config, epoch_index + 1),
                "best_experiment_epoch": epoch_index + 1,
                "validation20_loss": validation20["loss"], "config": config,
                "fingerprints": checksums})
            history["stale_epochs"] = 0
        else:
            history["stale_epochs"] = history.get("stale_epochs", 0) + 1
        print(f"rhythm epoch {overall_epoch(config, epoch_index+1)}: "
              f"train {entry['training_loss']:.4f}, "
              f"validation20 {validation20['loss']:.4f}, original10 {validation10['loss']:.4f}; "
              f"step {global_step}", flush=True)
        epoch_index += 1
        position = 0
        order = []
        weighted = 0.0
        target_tokens = processed = 0
        epoch_base_elapsed = 0.0
        epoch_started = time.perf_counter()
        stopping = history["stale_epochs"] >= config["patience"]
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
