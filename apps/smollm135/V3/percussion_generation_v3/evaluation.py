"""Held-out rhythm measurements and read-only V1/V2/V3 comparison."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from .data import APPS, DEFAULT_SESSIONS, ROOT, V1_OUTPUT, V2_OUTPUT, load_rows, load_tokenizer, parse_positions, recount_prefix, verify_adapter_codec

from custom_extractor import extract_tweeter
from custom_extractor.exporter import render_tweeter
from custom_extractor.extractor import extract_text, identify
from custom_extractor.parser import parse
from custom_extractor.validator import validate

from .inference import SharedPrefixSession, build_track, choose_loop_ms, generate_rhythm
from .model import heldout_loss, load_feature_model
from .roles import role


def remove_target(source, track_id):
    text = source.native_text
    records = parse(text)
    filtered = "".join(text[record.start:record.end] for record in records
                       if identify(record)[0] != track_id)
    result = extract_text(filtered)
    if result.tracks != [track for track in source.tracks if track["id"] != track_id]:
        raise ValueError("target removal changed conditioning tracks")
    return result


def score_positions(condition, reference, positions, requested, loop_ms, preset):
    track = build_track(condition, positions, preset, loop_ms,
                        target_track_id=reference["id"])
    candidate = deepcopy(condition)
    candidate.tracks.append(track)
    validate(candidate)
    text = render_tweeter(candidate)
    restored = extract_text(text)
    export_valid = restored.tracks == candidate.tracks and restored.tracks[:-1] == condition.tracks
    melody_steps = {note["step"] for other in condition.tracks if role(other) == "non_percussion"
                    for note in other["notes"]}
    step_set = set(positions)
    reference_steps = {note["step"] for note in reference["notes"]}
    overlap = len(step_set & reference_steps)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    wrap = positions[0] + 64 - positions[-1] if len(positions) > 1 else None
    return {"parse_valid": True, "export_valid": export_valid,
            "generated_note_count": len(positions),
            "absolute_count_error": abs(len(positions) - requested),
            "exact_count": len(positions) == requested,
            "within_one": abs(len(positions) - requested) <= 1,
            "positions": positions,
            "melody_onset_alignment": len(step_set & melody_steps) / len(step_set) if step_set else None,
            "reference_onset_f1": 2 * overlap / (len(step_set) + len(reference_steps))
                                  if step_set or reference_steps else 1.0,
            "loop_ms": loop_ms, "reference_loop_ms": reference["timing"]["loop_ms"],
            "loop_ms_match": loop_ms == reference["timing"]["loop_ms"],
            "wrap_gap_ratio": wrap / statistics.median(gaps) if gaps else None,
            "loop_valid": bool(positions and all(0 <= p < 64 for p in positions) and loop_ms > 0)}


def summarize(entries):
    valid = [entry for entry in entries if entry.get("parse_valid") and entry.get("export_valid")]
    count_entries = [entry for entry in entries if entry.get("generated_note_count") is not None]
    patterns = {tuple(entry["positions"]) for entry in valid}
    return {"attempts": len(entries), "parse_valid": sum(bool(e.get("parse_valid")) for e in entries),
            "export_valid": len(valid),
            "median_generated_note_count": statistics.median(
                [e["generated_note_count"] for e in count_entries]) if count_entries else None,
            "mean_absolute_count_error": statistics.mean(
                [e["absolute_count_error"] for e in count_entries]) if count_entries else None,
            "exact_count_fraction": sum(e.get("exact_count", False) for e in entries) / len(entries) if entries else None,
            "within_one_fraction": sum(e.get("within_one", False) for e in entries) / len(entries) if entries else None,
            "unique_patterns": len(patterns),
            "diversity_fraction": len(patterns) / len(valid) if valid else None,
            "mean_melody_alignment": statistics.mean(
                [e["melody_onset_alignment"] for e in valid if e["melody_onset_alignment"] is not None])
                if valid else None,
            "mean_reference_onset_f1": statistics.mean(
                [e["reference_onset_f1"] for e in valid]) if valid else None,
            "loop_valid_fraction": sum(bool(e.get("loop_valid")) for e in entries) / len(entries)
                                   if entries else None,
            "loop_ms_match_fraction": sum(bool(e.get("loop_ms_match")) for e in entries) / len(entries)
                                      if entries else None,
            "mean_wrap_gap_ratio": statistics.mean(
                [e["wrap_gap_ratio"] for e in valid if e["wrap_gap_ratio"] is not None])
                if any(e["wrap_gap_ratio"] is not None for e in valid) else None,
            "mean_generated_tokens": statistics.mean(
                [e["generated_tokens"] for e in entries if e.get("generated_tokens") is not None])
                if any(e.get("generated_tokens") is not None for e in entries) else None,
            "mean_input_tokens": statistics.mean(
                [e["input_tokens"] for e in entries if e.get("input_tokens") is not None])
                if any(e.get("input_tokens") is not None for e in entries) else None,
            "mean_prefill_seconds": statistics.mean(
                [e["prefill_seconds"] for e in entries if e.get("prefill_seconds") is not None])
                if any(e.get("prefill_seconds") is not None for e in entries) else None,
            "mean_decode_seconds": statistics.mean(
                [e["decode_seconds"] for e in entries if e.get("decode_seconds") is not None])
                if any(e.get("decode_seconds") is not None for e in entries) else None,
            "mean_total_seconds": statistics.mean(
                [e["total_seconds"] for e in entries if e.get("total_seconds") is not None])
                if any(e.get("total_seconds") is not None for e in entries) else None}


def source_for_row(row):
    path = DEFAULT_SESSIONS / row["source"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
        raise ValueError(f"source changed: {row['source']}")
    source = extract_tweeter(path)
    return remove_target(source, row["target_track_id"]), source.track(row["target_track_id"])


def evaluate_v3(output_dir=ROOT / "output", constrained=False, partition="test",
                cap=5, counts=(5, 8, 10, 12, 16, 20), max_new_tokens=512,
                cpu=False, cd2=False, shared_prefix_cache=True, profile=False):
    if constrained and cd2:
        raise ValueError("choose ordered constraints or CD2")
    root = Path(output_dir)
    rows = load_rows(root, "rhythm", partition)[:cap]
    if not rows:
        raise ValueError("no held-out V3 examples")
    adapter = root / "checkpoints" / ("constrained" if constrained or cd2 else "rhythm") / "best_adapter"
    verify_adapter_codec(root, adapter)
    model, device = load_feature_model(adapter, cpu=cpu)
    import torch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = load_tokenizer(root)
    preset = json.loads((root / "fixed_preset.json").read_text(encoding="utf-8"))
    examples = []
    shared_prefills = 0
    for row in rows:
        condition, reference = source_for_row(row)
        requests = [("reference", row["target_note_count"]),
                    *(("sweep", count) for count in counts)]
        session = None
        for request_type, requested in requests:
            started = time.perf_counter()
            entry = {"source": row["source"], "target_track_id": row["target_track_id"],
                     "requested_note_count": requested,
                     "request_type": request_type}
            try:
                prefix = recount_prefix(row["prefix"], row["target_note_count"], requested)
                if shared_prefix_cache and session is None:
                    prefixes = [recount_prefix(row["prefix"], row["target_note_count"], count)
                                for _, count in requests]
                    session = SharedPrefixSession(model, tokenizer, device, prefixes)
                raw, timing = generate_rhythm(model, tokenizer, device, prefix,
                                              max_new_tokens, requested if constrained or cd2 else None,
                                              cd2=cd2, shared_session=session)
                entry.update(timing)
                entry["raw_output"] = raw
                parse_start = time.perf_counter()
                positions = parse_positions(raw, allow_unordered=cd2)
                entry["parse_seconds"] = time.perf_counter() - parse_start
                loop_ms = choose_loop_ms(condition)
                export_start = time.perf_counter()
                entry.update(score_positions(condition, reference, sorted(positions) if cd2 else positions,
                                             requested, loop_ms, preset))
                entry["export_seconds"] = time.perf_counter() - export_start
            except (ValueError, RuntimeError, TypeError) as error:
                entry.setdefault("parse_valid", False)
                entry["export_valid"] = False
                entry["error"] = str(error)
            entry["total_seconds"] = time.perf_counter() - started
            examples.append(entry)
            print(f"V3 {'cd2' if cd2 else 'constrained' if constrained else 'greedy'} {row['source']} "
                  f"requested {requested}: generated {entry.get('generated_note_count', 'invalid')}", flush=True)
        if session is not None:
            shared_prefills += session.prefill_count
    reference_entries = [e for e in examples if e["request_type"] == "reference"]
    sweep_entries = [e for e in examples if e["request_type"] == "sweep"]
    result = {"model": "v3_cd2" if cd2 else "v3_constrained" if constrained else "v3_greedy",
              "partition": partition, "count_requests": list(counts),
              "max_new_tokens": max_new_tokens,
              "shared_prefix_cache": shared_prefix_cache,
              "heldout_loss": heldout_loss(model, rows, device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "reference_summary": summarize(reference_entries),
              "sweep_summary": summarize(sweep_entries), "examples": examples,
              "audio_listening_test": "not performed"}
    if profile:
        successful = [e for e in examples if "prefill_seconds" in e]
        prefill = sum(e["prefill_seconds"] for e in successful)
        decode = sum(e["decode_seconds"] for e in successful)
        elapsed = sum(e["total_seconds"] for e in examples)
        generated = sum(e["generated_tokens"] for e in successful)
        result["profiling"] = {
            "model_id": manifest_model_id(root),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "device": str(device), "dtype": str(next(model.parameters()).dtype),
            "torch_version": torch.__version__,
            "attention_backend": getattr(model.config, "_attn_implementation", None),
            "peak_gpu_bytes": result["peak_gpu_bytes"],
            "requests": len(examples), "shared_prefills": shared_prefills,
            "reused_prefix_tokens": sum(e.get("reused_prefix_tokens", 0) for e in successful),
            "prefill_seconds": prefill, "decode_seconds": decode,
            "total_request_seconds": elapsed,
            "generated_tokens_per_second": generated / decode if decode else None}
    destination = root / "evaluation" / f"v3_{'cd2' if cd2 else 'constrained' if constrained else 'greedy'}_{partition}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if profile:
        profile_path = destination.with_name("profiling_" + destination.name)
        profile_path.write_text(json.dumps(result["profiling"], indent=2) + "\n", encoding="utf-8")
    return {"report": str(destination), "reference_summary": result["reference_summary"],
            "sweep_summary": result["sweep_summary"]}


def manifest_model_id(root):
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8")).get("model_id")


def compare(output_dir=ROOT / "output", partition="test", cap=5,
            counts=(5, 8, 10, 12, 16, 20), max_new_tokens=512, cpu=False):
    """Run V1/V2 on V3's held-out rows, then include V3's own results."""
    v2_path = str(APPS / "PercussionGenerationV2")
    if v2_path not in sys.path:
        sys.path.insert(0, v2_path)
    from percussion_generation_v2.data import load_v2_tokenizer, with_note_count
    from percussion_generation_v2.inference import generate_object, make_track
    import torch

    root = Path(output_dir)
    rows = load_rows(root, "rhythm", partition)[:cap]
    v2_rows = load_rows(V2_OUTPUT, "rhythm", partition)[:cap]
    v1_rows = load_rows(V1_OUTPUT, "rhythm", partition)[:cap]
    if [(r["source"], r["target_track_id"]) for r in rows] != [
            (r["source"], r["target_track_id"]) for r in v2_rows] or [
            (r["source"], r["target_track_id"]) for r in rows] != [
            (r["source"], r["target_track_id"]) for r in v1_rows]:
        raise ValueError("V1/V2/V3 held-out rows are not aligned")
    tokenizer = load_v2_tokenizer(V2_OUTPUT)
    preset = json.loads((root / "fixed_preset.json").read_text(encoding="utf-8"))
    report = {"partition": partition, "counts": list(counts), "max_new_tokens": max_new_tokens,
              "models": {}, "comparison_note": "same source sessions, greedy decoding and token cap; "
              "V1 has no count input; V2/V3 receive requested counts"}
    for name, data_root in (("v1", V1_OUTPUT), ("v2", V2_OUTPUT)):
        adapter = data_root / "checkpoints/rhythm/best_adapter"
        model, device = load_feature_model(adapter, cpu=cpu)
        examples = []
        for row, old1, old2 in zip(rows, v1_rows, v2_rows):
            condition, reference = source_for_row(row)
            requests = [row["target_note_count"]] if name == "v1" else [row["target_note_count"], *counts]
            for requested in requests:
                entry = {"source": row["source"], "requested_note_count": requested,
                         "target_track_id": row["target_track_id"]}
                prefix = old1["prefix"] if name == "v1" else with_note_count(
                    old1["prefix"], requested)
                try:
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    prediction, timing = generate_object(model, tokenizer, device, prefix,
                                                         max_new_tokens)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    entry["total_seconds"] = time.perf_counter() - started
                    entry.update(timing)
                    track = make_track(prediction, "rhythm", row["target_track_id"], preset)
                    entry.update(score_positions(condition, reference,
                                                 [note["step"] for note in track["notes"]],
                                                 requested, track["timing"]["loop_ms"], preset))
                except (ValueError, RuntimeError, TypeError) as error:
                    entry.update(parse_valid=False, export_valid=False, error=str(error))
                examples.append(entry)
                print(f"{name} {row['source']} requested {requested}: "
                      f"{entry.get('generated_note_count', 'invalid')}", flush=True)
        history_file = data_root / "checkpoints/rhythm/history.json"
        history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.is_file() else {}
        report["models"][name] = {"summary": summarize(examples), "examples": examples,
                                  "reported_training_seconds": history.get("epochs", [{}])[-1].get("elapsed_seconds")}
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, constrained in (("v3", False), ("v3_constrained", True)):
        path = root / "evaluation" / f"v3_{'constrained' if constrained else 'greedy'}_{partition}.json"
        if not path.is_file():
            evaluate_v3(root, constrained, partition, cap, counts, max_new_tokens, cpu)
        v3_report = json.loads(path.read_text(encoding="utf-8"))
        history_file = root / "checkpoints/rhythm/history.json"
        history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.is_file() else {}
        report["models"][name] = {"summary": summarize(v3_report["examples"]),
                                  "examples": v3_report["examples"],
                                  "reported_training_seconds": history.get("epochs", [{}])[-1].get("elapsed_seconds"),
                                  "same_weights_as_v3": constrained}
    destination = root / "evaluation" / f"comparison_{partition}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report": str(destination),
            "summaries": {name: model["summary"] for name, model in report["models"].items()}}
