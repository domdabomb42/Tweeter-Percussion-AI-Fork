"""Atomic, complete checkpoints for the rotating V3 training experiment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def retry_windows_file_io(action, attempts=12, delay=0.5):
    """Retry transient Windows sharing violations from scanners/sync clients."""
    for attempt in range(attempts):
        try:
            return action()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt + 1 == attempts:
                raise
            time.sleep(delay)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(root):
    root = Path(root)
    names = ("manifest.json", "tokenizer/tokenizer.json", "rhythm/train.jsonl",
             "rhythm/validation.jsonl", "rhythm/validation20.jsonl")
    return {name: sha256_file(root / name) for name in names}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        retry_windows_file_io(lambda: os.replace(temporary, path))
    finally:
        if temporary.exists():
            try:
                retry_windows_file_io(lambda: temporary.unlink(missing_ok=True))
            except OSError:
                pass


def save_checkpoint(root, category, label, model, optimizer, scheduler, state,
                    keep_periodic=3):
    """Publish a directory only after adapter and complete state are written."""
    import torch
    root = Path(root)
    if category not in ("periodic", "epoch_boundaries", "final"):
        raise ValueError("unknown checkpoint category")
    parent = root / category
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / label
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    temporary = parent / f".tmp_{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary / "adapter")
        payload = dict(state)
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
        payload["python_rng"] = __import__("random").getstate()
        payload["torch_rng"] = torch.get_rng_state()
        payload["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        retry_windows_file_io(lambda: torch.save(payload, temporary / "training_state.pt"))
        atomic_json(temporary / "metadata.json", {
            "global_step": state["global_step"], "epoch_index": state["epoch_index"],
            "next_position": state["next_position"], "config": state["config"],
            "fingerprints": state["fingerprints"]})
        retry_windows_file_io(lambda: temporary.rename(final))
        atomic_json(root / "latest.json", {"checkpoint": final.relative_to(root).as_posix()})
        if category in ("periodic", "epoch_boundaries"):
            prefix = "step_" if category == "periodic" else "epoch_"
            keep = keep_periodic if category == "periodic" else 1
            completed = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))
            for old in completed[:-keep]:
                if old.resolve().parent != parent.resolve():
                    raise RuntimeError("periodic checkpoint escaped its directory")
                try:
                    retry_windows_file_io(lambda: shutil.rmtree(old))
                except OSError as exc:
                    if getattr(exc, "winerror", None) not in (32, 33):
                        raise
                    # The new checkpoint is already published. A later save
                    # can prune this old directory once the lock is released.
        return final
    finally:
        if temporary.exists():
            try:
                retry_windows_file_io(lambda: shutil.rmtree(temporary))
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (32, 33):
                    raise
                # Leave an unpublished temp directory rather than mask the
                # original save failure or compromise a prior checkpoint.


def resolve_checkpoint(root, requested):
    root = Path(root)
    if str(requested) == "latest":
        record = json.loads((root / "latest.json").read_text(encoding="utf-8"))
        path = (root / record["checkpoint"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("latest checkpoint pointer escapes the output directory")
        return path
    path = Path(requested).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("resume checkpoint must be inside this experiment's checkpoint directory")
    return path


def load_checkpoint_state(path, config, expected_fingerprints):
    import torch
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["config"] != config or metadata["fingerprints"] != expected_fingerprints:
        raise ValueError("checkpoint configuration or dataset/tokenizer fingerprint changed")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state["config"] != config or state["fingerprints"] != expected_fingerprints:
        raise ValueError("checkpoint state differs from its metadata")
    if not (path / "adapter/adapter_model.safetensors").is_file():
        raise FileNotFoundError("checkpoint adapter weights missing")
    return state


def save_best_adapter(root, model, metadata):
    """Replace the new experiment's best adapter, retaining a rollback copy."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".best_{uuid.uuid4().hex}"
    backup = root / "best_adapter.previous"
    model.save_pretrained(temporary)
    atomic_json(temporary / "experiment_metadata.json", metadata)
    best = root / "best_adapter"
    if backup.exists():
        shutil.rmtree(backup)
    if best.exists():
        best.rename(backup)
    try:
        temporary.rename(best)
    except Exception:
        if backup.exists() and not best.exists():
            backup.rename(best)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return best
