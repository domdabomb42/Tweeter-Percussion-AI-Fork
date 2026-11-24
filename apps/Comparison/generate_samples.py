"""Generate and verify native PdL2Ork coll files for the five held-out pairs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
APPS = HERE.parent
MODELS = {
    "135m": ("smollm135", "V3"),
    "360m": ("smollm360", "V3"),
    "135mea": ("smollm135", "V4EA-synth"),
    "360mea": ("smollm360", "V4EA-synth"),
    "17bea": ("smollm1b", "V4EA-synth"),
}
SEED = 415
CPU_MODELS = {"360mea", "17bea"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heldout_plan():
    path = APPS / "smollm135/V3/output/rhythm/test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 5 or len({(row["source"], row["target_track_id"]) for row in rows}) != 5:
        raise ValueError("expected five distinct held-out target pairs")
    for row in rows:
        if row.get("augmentation"):
            raise ValueError("held-out sample is augmented")
        source = APPS / "L2Ork-Tweeter/sessions" / row["source"]
        if sha256(source) != row["source_sha256"]:
            raise ValueError(f"held-out source changed: {source}")
    random.Random(SEED).shuffle(rows)
    return [{"source": row["source"], "source_sha256": row["source_sha256"],
             "target_track_id": row["target_track_id"],
             "target_positions": row["target_positions"]} for row in rows]


def setup_model(model_name):
    model_root, experiment = MODELS[model_name]
    root = APPS / model_root / experiment
    sys.path.insert(0, str(root))
    if experiment == "V4EA-synth":
        sys.path.insert(0, str(root.parent / "V3"))
    sys.path.insert(0, str(APPS / "CustomExtractor"))
    return root


def chunked_v3_rhythm(model, tokenizer, device, text, note_count, chunk_size=256):
    """Use a KV cache during prefill to keep long V3 inputs within 6 GB VRAM."""
    import torch
    from percussion_generation_v3.inference import PositionConstraint

    started = time.perf_counter()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) >= 8192:
        raise ValueError(f"V3 prompt length {len(ids)} exceeds model context")
    constraint = PositionConstraint(tokenizer, note_count)
    cache = None
    generated = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), chunk_size):
            end = min(start + chunk_size, len(ids))
            part = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
            mask = torch.ones((1, end), dtype=torch.long, device=device)
            output = model(input_ids=part, attention_mask=mask,
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
        for index in range(min(128, 8192 - len(ids))):
            allowed = constraint.allowed_incremental()
            logits = output.logits[0, -1]
            choice = allowed[int(torch.argmax(logits[allowed]).item())]
            generated.append(choice)
            constraint.advance(choice)
            if choice == tokenizer.eos_token_id:
                break
            if index < 127:
                next_id = torch.tensor([[choice]], dtype=torch.long, device=device)
                output = model(input_ids=next_id, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
    ended = bool(generated and generated[-1] == tokenizer.eos_token_id)
    result = tokenizer.decode(generated[:-1] if ended else generated,
                              skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result, time.perf_counter() - started


def worker(model_name, plan_path, staging):
    root = setup_model(model_name)
    from custom_extractor import extract_tweeter
    from percussion_generation_v3.evaluation import remove_target
    from custom_extractor.validator import validate

    rows = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    result = []
    if model_name.endswith("ea"):
        from percussion_generation_v4ea.inference import _best, _decode_generated, _export_checked
        from percussion_generation_v4ea.model import load_model
        from percussion_generation_v4ea.pipeline import load_new_tokenizer, verify_dataset
        from percussion_generation_v4ea.target_format import format_for_version

        output_dir = root.parent / "MostRecentOutput/V4EA-positions-v1"
        adapter, _ = _best(output_dir)
        manifest, _ = verify_dataset(output_dir)
        target_format = format_for_version(manifest["version"])
        if target_format != "positions-v1":
            raise ValueError("sample generation requires the trained positions-v1 format")
        catalog = json.loads((output_dir / "fixed_preset.json").read_text(encoding="utf-8"))
        catalog["track"]["instrument"]["username"] = "AI"
        tokenizer, _ = load_new_tokenizer(output_dir)
        model, device = load_model(output_dir, adapter=adapter, cpu=model_name in CPU_MODELS,
                                   trainable=False)
        model.eval()
    else:
        from percussion_generation_v3.data import load_tokenizer, parse_positions, verify_adapter_codec
        from percussion_generation_v3.inference import choose_loop_ms, export_rhythm, prompt
        from percussion_generation_v3.model import load_feature_model

        output_dir = root / "output"
        adapter = output_dir / "checkpoints/constrained/best_adapter"
        verify_adapter_codec(output_dir, adapter)
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        catalog = json.loads((output_dir / "fixed_preset.json").read_text(encoding="utf-8"))
        catalog["track"]["instrument"]["username"] = "AI"
        tokenizer = load_tokenizer(output_dir)
        model, device = load_feature_model(adapter, cpu=model_name in CPU_MODELS,
                                           trainable=False)
        model.eval()
    for row in rows:
        source_path = APPS / "L2Ork-Tweeter/sessions" / row["source"]
        if sha256(source_path) != row["source_sha256"]:
            raise ValueError(f"source changed while generating: {source_path}")
        original = extract_tweeter(source_path)
        target = original.track(row["target_track_id"])
        if [note["step"] for note in target["notes"]] != row["target_positions"]:
            raise ValueError("held-out target differs from source")
        condition = remove_target(original, row["target_track_id"])
        legacy = [warning for warning in validate(condition) if warning["code"] == "legacy_range"]
        name = f"{Path(row['source']).stem}_track{row['target_track_id']}_{model_name}.txt"
        destination = Path(staging) / name
        if model_name.endswith("ea"):
            positions, seconds = _decode_generated(model, tokenizer, device, condition,
                                                   len(target["notes"]), catalog, 128,
                                                   target_format=target_format, constrained=True)
            track = _export_checked(condition, positions, catalog, destination)
        else:
            raw, seconds = chunked_v3_rhythm(
                model, tokenizer, device,
                prompt(condition, len(target["notes"]), manifest["encoding"]),
                len(target["notes"]))
            positions = parse_positions(raw, requested_count=len(target["notes"]))
            track = export_rhythm(condition, positions, catalog, choose_loop_ms(condition),
                                  destination, target_track_id=row["target_track_id"])
        restored = extract_tweeter(destination)
        if (restored.global_settings != condition.global_settings or
                restored.tracks[:-1] != condition.tracks or restored.tracks[-1] != track):
            raise ValueError(f"native reimport integrity failed: {destination}")
        native_warnings = validate(restored)
        if [w for w in native_warnings if w["code"] == "legacy_range" and
                w.get("track") == track["id"]]:
            raise ValueError(f"generated track inherited an invalid parameter: {destination}")
        result.append({"source": row["source"], "target_track_id": row["target_track_id"],
                       "model": model_name, "file": name, "sha256": sha256(destination),
                       "positions": positions, "expected_note_count": len(target["notes"]),
                       "generated_instrument": track["instrument"]["name"],
                       "generated_username": track["instrument"]["username"],
                       "preexisting_native_violations": legacy,
                       "native_warnings": native_warnings,
                       "generation_seconds": seconds})
        print(f"{model_name}: exported and reimported {name}", flush=True)
    (Path(staging) / f".{model_name}.json").write_text(
        json.dumps({"model": model_name, "adapter": str(adapter),
                    "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                    "samples": result}, indent=2) + "\n", encoding="utf-8")


def mark_existing_usernames_ai():
    """Retag saved generated tracks, retaining every other native byte."""
    setup_model("135m")
    from custom_extractor import extract_tweeter
    from custom_extractor.exporter import atomic_write, render_tweeter
    from custom_extractor.validator import validate
    from percussion_generation_v3.evaluation import remove_target

    root = HERE / "SampleOutputs"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [sample for model in manifest["models"] for sample in model["samples"]]
    if len(samples) != 25 or {path.name for path in root.glob("*.txt")} != \
            {sample["file"] for sample in samples}:
        raise ValueError("expected exactly 25 saved sample files")
    if (manifest.get("generated_username") == "AI" and
            all(sample.get("generated_username") == "AI" and
                sha256(root / sample["file"]) == sample["sha256"] for sample in samples)):
        print("All 25 generated tracks already have username AI")
        return
    source_hashes = {(row["source"], row["target_track_id"]): row["source_sha256"]
                     for row in manifest["pairs"]}
    pending = []
    for sample in samples:
        path = (root / sample["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != sample["sha256"]:
            raise ValueError(f"saved sample changed: {path}")
        source = APPS / "L2Ork-Tweeter/sessions" / sample["source"]
        if sha256(source) != source_hashes[(sample["source"], sample["target_track_id"])]:
            raise ValueError(f"held-out source changed: {source}")
        condition = remove_target(extract_tweeter(source), sample["target_track_id"])
        saved = extract_tweeter(path)
        if (saved.global_settings != condition.global_settings or
                saved.tracks[:-1] != condition.tracks or
                [note["step"] for note in saved.tracks[-1]["notes"]] != sample["positions"]):
            raise ValueError(f"saved native composition differs from held-out source: {path}")
        track = deepcopy(saved.tracks[-1])
        track["instrument"]["username"] = "AI"
        strict = deepcopy(condition)
        strict.tracks.append(track)
        validate(strict)
        saved.tracks[-1] = track
        original_text = path.read_bytes().decode("utf-8", errors="surrogateescape")
        updated_text = render_tweeter(saved)
        before, after = original_text.splitlines(keepends=True), updated_text.splitlines(keepends=True)
        changes = [(old, new) for old, new in zip(before, after) if old != new]
        if len(before) != len(after) or len(changes) != 1 or \
                f", {track['id']} username " not in changes[0][0] or \
                f", {track['id']} username AI;" not in changes[0][1]:
            raise ValueError(f"username update would change another native record: {path}")
        pending.append((path, updated_text, condition, track, sample))
    for path, text, condition, track, sample in pending:
        atomic_write(path, text.encode("utf-8", errors="surrogateescape"))
        restored = extract_tweeter(path)
        if (restored.global_settings != condition.global_settings or
                restored.tracks[:-1] != condition.tracks or restored.tracks[-1] != track):
            raise ValueError(f"retagged native file did not reimport exactly: {path}")
        sample["sha256"] = sha256(path)
        sample["generated_username"] = "AI"
    manifest["generated_username"] = "AI"
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"Marked all {len(samples)} generated tracks with username AI")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--mark-ai-username", action="store_true",
                        help="set AI on generated tracks in the existing sample set")
    args = parser.parse_args()
    if args.mark_ai_username:
        mark_existing_usernames_ai()
        return
    if args.worker_model:
        if args.plan is None or args.staging is None:
            parser.error("workers require --plan and --staging")
        worker(args.worker_model, args.plan, args.staging)
        return
    output = HERE / "SampleOutputs"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"sample outputs already exist: {output}")
    rows = heldout_plan()
    staging = (HERE / ".SampleOutputs.staging").resolve()
    if not staging.is_relative_to(HERE.resolve()):
        raise ValueError("staging directory escaped Comparison")
    staging.mkdir(exist_ok=True)
    plan = staging / ".plan.json"
    if plan.exists():
        if json.loads(plan.read_text(encoding="utf-8")) != rows:
            raise ValueError("staged held-out selection differs from current source")
    else:
        plan.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for name, (model, experiment) in MODELS.items():
        existing = staging / f".{name}.json"
        if existing.exists():
            saved = json.loads(existing.read_text(encoding="utf-8"))
            samples = saved.get("samples", [])
            if (saved.get("model") != name or len(samples) != 5 or
                    {(r["source"], r["target_track_id"]) for r in samples} !=
                    {(r["source"], r["target_track_id"]) for r in rows} or
                    sha256(Path(saved["adapter"]) / "adapter_model.safetensors") !=
                    saved["adapter_sha256"] or
                    any(sha256(staging / sample["file"]) != sample["sha256"]
                        for sample in samples)):
                raise ValueError(f"staged outputs are incomplete or changed: {name}")
            print(f"Reusing five verified {name} outputs", flush=True)
        else:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--worker-model", name, "--plan", str(plan),
                            "--staging", str(staging)],
                           cwd=APPS / model / experiment, check=True)
    models = [json.loads((staging / f".{name}.json").read_text(encoding="utf-8"))
              for name in MODELS]
    for item in models:
        item["adapter"] = str(Path(item["adapter"]).relative_to(HERE.parent.parent)).replace("\\", "/")
    samples = [sample for item in models for sample in item["samples"]]
    if (len(samples) != 25 or
            {path.name for path in staging.glob("*.txt")} !=
            {sample["file"] for sample in samples}):
        raise ValueError("expected 25 verified native outputs")
    manifest = {"seed": SEED, "selection": "seeded shuffle of all five held-out target pairs",
                "decoder": "ordered constrained positions",
                "generated_username": "AI",
                "v3_prefill": "256-token KV-cache chunks; outputs can differ from full prefill",
                "pairs": rows, "models": models}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    for metadata in staging.glob(".*.json"):
        metadata.unlink()
    if output.exists():
        output.rmdir()  # Only an empty, explicitly named destination can be replaced.
    staging.replace(output)
    print(f"Saved 25 native coll files in {output}")


if __name__ == "__main__":
    main()
