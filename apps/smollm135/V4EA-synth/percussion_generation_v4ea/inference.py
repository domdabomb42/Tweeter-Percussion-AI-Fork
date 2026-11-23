"""Strict musical generation, native reconstruction, and comparison reports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

from . import END_MARKER, MODEL_ID, MODEL_REVISION, OUTPUT, VERSION
from .model import load_model
from .music import TARGET_TAG, compact, decode_target, reconstruct
from .pipeline import load_new_tokenizer, prompt_for_composition, verify_dataset
from .trainer import validation_loss
from custom_extractor import extract_tweeter, export_tweeter
from custom_extractor.models import Composition
from percussion_generation_v3.lossless_dataset import _source_path
from percussion_generation_v3.data import AUGMENTED_SESSIONS


def _best(output_dir, checkpoint_run="auto"):
    root = Path(output_dir)
    manifest, groups = verify_dataset(root)
    if checkpoint_run not in ("auto", "coverage", "legacy"):
        raise ValueError("unknown checkpoint run")
    coverage = root / "checkpoints/rhythm_coverage_v1/best_adapter"
    legacy = root / "checkpoints/rhythm/best_adapter"
    adapter = (coverage if checkpoint_run == "coverage" or
               (checkpoint_run == "auto" and coverage.is_dir()) else legacy)
    metadata = json.loads((adapter / "experiment_metadata.json").read_text(encoding="utf-8"))
    for key, expected in (("version", VERSION), ("model_id", MODEL_ID),
                          ("revision", MODEL_REVISION),
                          ("tokenizer_sha256", manifest["tokenizer_sha256"]),
                          ("prepared_dataset_sha256", manifest["prepared_dataset_sha256"]),
                          ("codec_schema_sha256", manifest["codec_schema_sha256"]),
                          ("preset_catalog_sha256", manifest["preset_catalog_sha256"])):
        if metadata.get(key) != expected:
            raise ValueError(f"best adapter has incompatible {key}")
    return adapter, groups


def _decode_generated(model, tokenizer, device, composition, count, catalog,
                      max_new_tokens, enforce_count=True):
    import torch
    ids = tokenizer.encode(prompt_for_composition(composition, count), add_special_tokens=False)
    limit = min(max_new_tokens, 8192 - len(ids))
    if limit < 2:
        raise ValueError("musical input leaves no target context")
    started = time.perf_counter()
    # Constrain the opening schema tag; strict parsing validates the remaining
    # variable-length musical fields and refuses malformed completions.
    opening = compact([TARGET_TAG])[:-1] + ","
    vocabulary = list(range(len(tokenizer)))
    pieces = [tokenizer.decode([token_id], skip_special_tokens=False)
              for token_id in vocabulary]
    allowed_cache = {}
    def allowed_tokens(_batch, sequence):
        current_ids = tuple(sequence[len(ids):].tolist())
        if current_ids in allowed_cache:
            return allowed_cache[current_ids]
        current = tokenizer.decode(list(current_ids), skip_special_tokens=False)
        if current.startswith(opening):
            return vocabulary
        allowed = []
        for token_id, piece in enumerate(pieces):
            candidate = current + piece
            if candidate != current and (opening.startswith(candidate) or candidate.startswith(opening)):
                allowed.append(token_id)
        if not allowed:
            raise ValueError("tokenizer cannot emit musical schema opening")
        allowed_cache[current_ids] = allowed
        return allowed
    with torch.inference_mode():
        output = model.peft_model.generate(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=limit, do_sample=False, use_cache=True,
            prefix_allowed_tokens_fn=allowed_tokens,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, len(ids):], skip_special_tokens=False)
    duration = time.perf_counter() - started
    if "\n" + END_MARKER not in text:
        raise ValueError("model did not complete musical target")
    packed = text.split("\n" + END_MARKER, 1)[0]
    return decode_target(packed, count if enforce_count else None, catalog), duration


def _export_checked(composition, musical, catalog, output_path):
    candidate, track = reconstruct(composition, musical, catalog)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_tweeter(candidate, output_path)
    restored = extract_tweeter(output_path)
    if restored.tracks[:-1] != composition.tracks or restored.tracks[-1] != track:
        output_path.unlink(missing_ok=True)
        raise ValueError("native export/reimport changed source or generated track")
    return track


def generate(output_dir=OUTPUT, input_path=None, output_path=None,
             target_note_count=None, max_new_tokens=512, cpu=False,
             checkpoint_run="auto"):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    adapter, _ = _best(output_dir, checkpoint_run)
    tokenizer, _ = load_new_tokenizer(output_dir)
    catalog = json.loads((Path(output_dir) / "preset_catalog.json").read_text(encoding="utf-8"))
    composition = extract_tweeter(input_path)
    model, device = load_model(output_dir, adapter=adapter, cpu=cpu, trainable=False)
    model.eval()
    musical, duration = _decode_generated(model, tokenizer, device, composition,
                                          target_note_count, catalog, max_new_tokens)
    track = _export_checked(composition, musical, catalog, output_path)
    return {"output": str(output_path), "track_id": track["id"],
            "instrument": musical["instrument"], "notes": len(track["notes"]),
            "generation_seconds": duration, "target_format": VERSION}


def evaluate(output_dir=OUTPUT, partition="validation20", cpu=False,
             max_new_tokens=512, checkpoint_run="auto"):
    adapter, groups = _best(output_dir, checkpoint_run)
    if partition not in ("validation20", "validation", "test"):
        raise ValueError("evaluation partition must be validation20, validation, or test")
    tokenizer, _ = load_new_tokenizer(output_dir)
    catalog = json.loads((Path(output_dir) / "preset_catalog.json").read_text(encoding="utf-8"))
    augmentation = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
    generated_lookup = {item["augmented_filename"]: item for item in augmentation["generated"]}
    model, device = load_model(output_dir, adapter=adapter, cpu=cpu, trainable=False)
    model.eval()
    rows = groups[partition]
    teacher = validation_loss(model, rows, device)
    counts = {"structurally_valid": 0, "exact_note_count": 0,
              "instrument_correct": 0, "native_export_success": 0}
    overlaps, patterns, durations, errors = [], set(), [], []
    with tempfile.TemporaryDirectory() as temporary:
        for index, row in enumerate(rows):
            original = extract_tweeter(_source_path(row, generated_lookup))
            expected = original.track(row["target_track_id"])
            conditioning = Composition(deepcopy(original.global_settings),
                [deepcopy(track) for track in original.tracks if track["id"] != row["target_track_id"]])
            attempt_started = time.perf_counter()
            duration_recorded = False
            try:
                musical, seconds = _decode_generated(
                    model, tokenizer, device, conditioning, len(expected["notes"]),
                    catalog, max_new_tokens, enforce_count=False)
                counts["structurally_valid"] += 1
                counts["exact_note_count"] += len(musical["notes"]) == len(expected["notes"])
                counts["instrument_correct"] += musical["instrument"] == expected["instrument"]["name"]
                actual_steps = {item["step"] for item in musical["notes"]}
                expected_steps = {item["step"] for item in expected["notes"]}
                overlaps.append(len(actual_steps & expected_steps) / len(actual_steps | expected_steps))
                patterns.add(tuple(sorted(actual_steps)))
                durations.append(seconds)
                duration_recorded = True
                _export_checked(conditioning, musical, catalog, Path(temporary) / f"sample_{index}.txt")
                counts["native_export_success"] += 1
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                if not duration_recorded:
                    durations.append(time.perf_counter() - attempt_started)
                errors.append({"index": index, "source": row["source"], "error": str(exc)})
    history_path = adapter.parent / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    epochs = history.get("epochs", [])
    result = {"model_id": MODEL_ID, "partition": partition, "examples": len(rows),
              "checkpoint_run": adapter.parent.name,
              "token_cross_entropy": teacher["loss"],
              "token_cross_entropy_note": "Tokenizer-specific; do not compare directly across models",
              "training_loss": epochs[-1]["training_loss"] if epochs else None,
              "validation_loss": epochs[-1]["validation20"]["loss"] if epochs else None,
              "test_loss": teacher["loss"] if partition == "test" else None,
              **{name + "_rate": value / len(rows) for name, value in counts.items()},
              "rhythmic_jaccard": sum(overlaps) / len(overlaps) if overlaps else None,
              "rhythmic_diversity": len(patterns) / len(overlaps) if overlaps else None,
              "mean_generation_seconds": sum(durations) / len(durations) if durations else None,
              "training_duration_seconds": epochs[-1]["cumulative_seconds"] if epochs else None,
              "peak_gpu_memory_bytes": max((entry.get("peak_gpu_memory_bytes") or 0 for entry in epochs), default=0),
              "trainable_parameters": history.get("plan", {}).get("trainable_parameters"),
              "errors": errors}
    report = Path(output_dir) / f"evaluation_{partition}.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare_reports(paths, output_path):
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len({report["model_id"] for report in reports}) != 3:
        raise ValueError("comparison requires one report from each model")
    columns = ("training_loss", "validation_loss", "test_loss",
               "structurally_valid_rate", "exact_note_count_rate",
               "instrument_correct_rate", "rhythmic_jaccard", "rhythmic_diversity",
               "native_export_success_rate", "mean_generation_seconds",
               "training_duration_seconds", "peak_gpu_memory_bytes", "trainable_parameters")
    result = {"warning": "Token cross-entropy is tokenizer-specific and is not ranked across models",
              "models": [{"model_id": report["model_id"],
                          **{key: report.get(key) for key in columns}} for report in reports]}
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
