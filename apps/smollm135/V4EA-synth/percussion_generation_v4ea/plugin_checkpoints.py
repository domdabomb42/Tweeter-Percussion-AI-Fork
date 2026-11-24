"""Integrity-checked, atomic DPO checkpoints with verified fallback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .plugin_store import canonical, digest


REQUIRED = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
            "adapter/embedding.safetensors", "adapter/experiment_metadata.json",
            "state.pt", "metadata.json")


def _hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _replace(source, destination):
    for attempt in range(12):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == 11:
                raise
            time.sleep(.25)


def _pointer(root, path):
    pointer = root / "latest.json"
    temporary = root / f".latest_{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical({"checkpoint": path.relative_to(root).as_posix()}) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)


def save(root, category, label, model, optimizer, scheduler, state,
         adapter_metadata, save_total_limit=3):
    import torch
    if category not in ("periodic", "epochs", "final", "pause", "best"):
        raise ValueError("unsupported DPO checkpoint category")
    if save_total_limit < 3:
        raise ValueError("retain at least three periodic checkpoints")
    root = Path(root)
    parent = root / category
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / label
    if destination.exists():
        raise FileExistsError(destination)
    temporary = parent / f".pending_{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary / "adapter")
        (temporary / "adapter/experiment_metadata.json").write_bytes(canonical(adapter_metadata) + b"\n")
        payload = {**state, "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict()}
        with (temporary / "state.pt").open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {"category": category, "label": label, "global_step": state["global_step"],
                    "epoch_index": state["epoch_index"], "cursor": state["cursor"],
                    "snapshot_sha256": state["snapshot_sha256"],
                    "config_sha256": digest(state["config"]),
                    "saved_at": time.time()}
        (temporary / "metadata.json").write_bytes(canonical(metadata) + b"\n")
        hashes = {name: _hash_file(temporary / name) for name in REQUIRED}
        (temporary / "integrity.json").write_bytes(canonical(hashes) + b"\n")
        verify(temporary, state["snapshot_sha256"], state["config"])
        _replace(temporary, destination)
        _pointer(root, destination)
        if category == "periodic":
            periodic = sorted((item for item in parent.iterdir() if item.is_dir() and item.name.startswith("step_")),
                              key=lambda item: item.stat().st_mtime)
            for old in periodic[:-save_total_limit]:
                if old == destination or old.resolve().parent != parent.resolve():
                    raise RuntimeError("unsafe periodic checkpoint cleanup")
                try:
                    shutil.rmtree(old)
                except OSError:
                    # Retention is best effort; preserving a valid checkpoint wins.
                    pass
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def verify(path, snapshot_sha256=None, config=None):
    import torch
    path = Path(path)
    hashes = json.loads((path / "integrity.json").read_text(encoding="utf-8"))
    if set(hashes) != set(REQUIRED):
        raise ValueError("checkpoint integrity manifest is incomplete")
    for name, expected in hashes.items():
        if not (path / name).is_file() or _hash_file(path / name) != expected:
            raise ValueError(f"checkpoint file missing or corrupt: {name}")
    adapter_config = json.loads((path / "adapter/adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("peft_type"):
        from safetensors import safe_open
        for name in ("adapter/adapter_model.safetensors", "adapter/embedding.safetensors"):
            with safe_open(str(path / name), framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise ValueError(f"checkpoint safetensors has no weights: {name}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if snapshot_sha256 and metadata["snapshot_sha256"] != snapshot_sha256:
        raise ValueError("checkpoint preference snapshot differs")
    if config is not None and metadata["config_sha256"] != digest(config):
        raise ValueError("checkpoint training configuration differs")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if (state["global_step"] != metadata["global_step"] or
            state["epoch_index"] != metadata["epoch_index"] or
            state["cursor"] != metadata["cursor"] or
            state["snapshot_sha256"] != metadata["snapshot_sha256"] or
            digest(state["config"]) != metadata["config_sha256"]):
        raise ValueError("checkpoint state disagrees with integrity metadata")
    if not isinstance(state["order"], list) or not 0 <= state["cursor"] <= len(state["order"]):
        raise ValueError("checkpoint dataset cursor is invalid")
    return state


def candidates(root):
    root = Path(root)
    files = [item for folder in ("periodic", "epochs", "final", "pause", "best")
             for item in (root / folder).glob("*") if item.is_dir() and not item.name.startswith(".pending_")]
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)


def recover(root, requested="latest", snapshot_sha256=None, config=None):
    root = Path(root)
    if requested != "latest":
        path = Path(requested).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("checkpoint must be inside this DPO run")
        return path, verify(path, snapshot_sha256, config), []
    preferred = None
    pointer = root / "latest.json"
    if pointer.is_file():
        try:
            relative = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
            preferred = (root / relative).resolve()
            if not preferred.is_relative_to(root.resolve()):
                preferred = None
        except (OSError, ValueError, KeyError):
            pass
    ordered = ([preferred] if preferred else []) + [p for p in candidates(root) if p != preferred]
    failures = []
    for path in ordered:
        try:
            return path, verify(path, snapshot_sha256, config), failures
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failures.append({"checkpoint": str(path), "reason": str(exc)})
    raise FileNotFoundError(f"no valid compatible DPO checkpoint; checked {len(failures)}: {failures[:3]}")


def inspect(root, snapshot_sha256=None):
    rows = []
    for path in candidates(root):
        try:
            state = verify(path, snapshot_sha256)
            rows.append({"path": str(path), "valid": True, "global_step": state["global_step"],
                         "epoch_index": state["epoch_index"], "cursor": state["cursor"]})
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            rows.append({"path": str(path), "valid": False, "error": str(exc)})
    return rows
