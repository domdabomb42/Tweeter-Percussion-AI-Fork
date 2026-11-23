"""Evaluate and export the complete predicted percussion track."""
from copy import deepcopy
import json
from pathlib import Path

from . import END_MARKER, MODEL_ID, MODEL_REVISION, OUTPUT, VERSION
from .model import load_model
from .pipeline import load_new_tokenizer, prompt_for_composition, verify_dataset
from .trainer import validation_loss
from percussion_generation_v3.codec_v2 import decode_v2
from percussion_generation_v3.roles import role
from custom_extractor import extract_tweeter, export_tweeter
from custom_extractor.validator import validate


def _best(output_dir):
    root = Path(output_dir)
    manifest, groups = verify_dataset(root)
    adapter = root / "checkpoints/rhythm/best_adapter"
    metadata = json.loads((adapter / "experiment_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("version"), metadata.get("model_id"), metadata.get("revision"),
            metadata.get("tokenizer_sha256"), metadata.get("prepared_dataset_sha256"),
            metadata.get("codec_schema_sha256")) != (
            VERSION, MODEL_ID, MODEL_REVISION, manifest["tokenizer_sha256"],
            manifest["prepared_dataset_sha256"], manifest["codec_schema_sha256"]):
        raise ValueError("best adapter is incompatible with V4EA dataset or tokenizer")
    return adapter, groups


def evaluate(output_dir=OUTPUT, partition="validation20", cpu=False):
    adapter, groups = _best(output_dir)
    model, device = load_model(output_dir, adapter=adapter, cpu=cpu, trainable=False)
    result = validation_loss(model, groups[partition], device)
    return {"partition": partition, **result,
            "adapter": str(adapter), "model_id": MODEL_ID}


def generate(output_dir, input_path, output_path, target_note_count,
             max_new_tokens=3072, cpu=False):
    import torch
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    adapter, _ = _best(output_dir)
    tokenizer, _ = load_new_tokenizer(output_dir)
    composition = extract_tweeter(input_path)
    prompt = prompt_for_composition(composition, target_note_count)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    limit = min(max_new_tokens, 8192 - len(ids))
    if limit < 2:
        raise ValueError("conditioning input leaves no context for a full track")
    model, device = load_model(output_dir, adapter=adapter, cpu=cpu, trainable=False)
    model.eval()
    with torch.inference_mode():
        generated = model.peft_model.generate(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=limit, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    text = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=False)
    if "\n" + END_MARKER not in text:
        raise ValueError("model did not complete a full-track target")
    packed = text.split("\n" + END_MARKER, 1)[0]
    result = decode_v2(packed)
    if result.get("global") != {} or len(result.get("tracks", [])) != 1:
        raise ValueError("model output is not one percussion track")
    entry = result["tracks"][0]
    track = entry["track"]
    if entry["role"] != "percussion" or role(track) != "percussion":
        raise ValueError("generated track is not classified as percussion")
    if len(track.get("notes", [])) != target_note_count:
        raise ValueError("generated note count differs from requested count")
    used = {item["id"] for item in composition.tracks}
    if track["id"] in used:
        free = next((i for i in range(1, 15) if i not in used), None)
        if free is None:
            raise ValueError("no free native track slot")
        track["id"] = free
    candidate = deepcopy(composition)
    candidate.tracks.append(track)
    validate(candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_tweeter(candidate, output_path)
    restored = extract_tweeter(output_path)
    if restored.tracks[:-1] != composition.tracks or restored.tracks[-1] != track:
        output_path.unlink(missing_ok=True)
        raise ValueError("native export changed generated track")
    return {"output": str(output_path), "track_id": track["id"],
            "notes": len(track["notes"]), "target_format": VERSION}
