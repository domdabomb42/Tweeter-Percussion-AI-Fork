"""Reproducible corpus audit, real demo, native checks and runtime report.

Run from apps/CustomExtractor: python tools/verify.py [--skip-native]
Only writes inside this project's examples/ and reports/ directories.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custom_extractor import (extract_tweeter, export_tweeter, save_json, load_json,
                              dumps, loads, set_parameters, add_note, remove_note, add_track)
from custom_extractor.native_check import check_native, find_pd
from custom_extractor.reporting import coverage, canonical, musical_features
from custom_extractor.parameter_registry import PARAMETERS


def timed(function, *args):
    start = perf_counter()
    value = function(*args)
    return value, (perf_counter() - start) * 1000


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def demonstration():
    folder = ROOT / "examples"
    folder.mkdir(exist_ok=True)
    original = ROOT.parent / "L2Ork-Tweeter" / "sessions" / "slowbeat-session-coll.txt"
    (folder / "original.txt").write_bytes(original.read_bytes())
    composition = extract_tweeter(original)
    save_json(composition, folder / "composition.json")
    export_tweeter(load_json(folder / "composition.json"), folder / "reconstructed.txt")
    modified = copy.deepcopy(composition)
    tid = modified.tracks[0]["id"]
    carrier_size = len(modified.track(tid)["synth"]["carrier_envelope"])
    modulator_size = len(modified.track(tid)["synth"]["modulator_envelope"])
    edits = {"carrier": "square", "modulator": "pink_noise",
             "ae": [round(max(0, 1 - i / 80), 12) for i in range(carrier_size)],
             "me": [round(max(0, 1 - i / 60), 12) for i in range(modulator_size)], "harmonicity": 3.25, "mod-amp": 850,
             "note-dur": 380, "octave": -7, "overdrive": 2,
             "overdrive-override": 2.5, "inst-name": "CustomExtractor Demo Synth",
             "reverb-amp": .55, "reverb-type": 3, "echo-amp": .25,
             "echo-fb": .6, "echo-ms": 375, "flanger-amp": .35,
             "flanger-hz": .75, "flanger-mod-amp": 22.5, "flanger-ms": 4,
             "dry-amp": .9, "out-amp": 85, "out-pan": .7,
             "loop-length": 7200, "loop": 1, "solo": 0,
             "tracker-time": {"next_step": 0, "remaining_ms": 0, "running": 0}}
    set_parameters(modified, tid, edits)
    removed = copy.deepcopy(modified.track(tid)["notes"][0])
    remove_note(modified, tid, **removed)
    occupied = {n["step"] for n in modified.track(tid)["notes"]}
    new_step = next(step for step in range(64) if step not in occupied and step != removed["step"])
    add_note(modified, tid, new_step, 66)
    add_track(modified, 13, {"inst-name": "CustomExtractor Added Part", "carrier": "triangle", "modulator": "sine",
                            "harmonicity": 2, "mod-amp": 200, "loop-length": 3600, "note-dur": 200,
                            "octave": 0, "loop": 1, "out-amp": 80, "out-pan": .7854},
              [{"step": 0, "pitch": 60}, {"step": 16, "pitch": 64}, {"step": 32, "pitch": 67}])
    save_json(modified, folder / "modified.json")
    export_tweeter(load_json(folder / "modified.json"), folder / "modified.txt")
    reimported = extract_tweeter(folder / "modified.txt")
    assert canonical(modified) == canonical(reimported)
    assert composition.tracks[1:] == reimported.tracks[1:-1]
    write_json(folder / "features.json", {"features": musical_features(composition), "coverage": coverage(composition)})
    with tempfile.TemporaryDirectory(prefix="custom-extractor-demo-timing-") as temporary:
        timed_output = Path(temporary) / "reconstructed.txt"
        extraction_samples = [timed(extract_tweeter, original)[1] for _ in range(5)]
        export_samples = [timed(export_tweeter, composition, timed_output)[1] for _ in range(5)]
    return {"source": str(original.relative_to(ROOT.parent)), "edited_track": tid,
            "edits": edits, "removed_note": removed, "added_note": {"step": new_step, "pitch": 66},
            "added_track": 13, "unmodified_tracks_equal": True, "edited_roundtrip_equal": True,
            "runtime_method": "Five perf_counter wall-time runs: extract the source file, then validate and atomically export the original composition to a temporary file; JSON import and native Pd checks excluded.",
            "runtime_ms": {"extraction_median": statistics.median(extraction_samples),
                           "export_median": statistics.median(export_samples),
                           "samples": 5, "extraction_samples": extraction_samples,
                           "export_samples": export_samples},
            "coverage": coverage(composition), "original_bytes": original.stat().st_size,
            "json_bytes": len(dumps(composition).encode("utf-8"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(),
              "corpus_scope": "apps/L2Ork-Tweeter/sessions",
              "timing_method": "perf_counter wall time; validated extraction and atomic file export, milliseconds; native subprocess startup excluded",
              "demo": demonstration(), "files": [], "empty_inputs_rejected": [], "failures": []}
    native = None if args.skip_native else find_pd()
    corpus = ROOT.parent / "L2Ork-Tweeter"
    paths = sorted((corpus / "sessions").glob("*session*.txt"))
    print(f"Auditing {len(paths)} real files; native Pd checks: {bool(native)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="custom-extractor-audit-") as temporary:
        output = Path(temporary) / "reconstructed.txt"
        for index, path in enumerate(paths):
            name = path.relative_to(corpus).as_posix()
            if path.stat().st_size == 0:
                try:
                    extract_tweeter(path)
                except ValueError as error:
                    report["empty_inputs_rejected"].append({"file": name, "error": str(error)})
                    continue
                raise AssertionError("empty input was unexpectedly accepted")
            try:
                composition, extract_ms = timed(extract_tweeter, path)
                serialized = dumps(composition)
                restored = loads(serialized)
                _, export_ms = timed(export_tweeter, restored, output)
                reimported = extract_tweeter(output)
                byte_equal = path.read_bytes() == output.read_bytes()
                semantic_equal = canonical(composition) == canonical(reimported)
                if not byte_equal or not semantic_equal:
                    raise AssertionError("round-trip inequality")
                row = {"file": name, "original_bytes": path.stat().st_size, "json_bytes": len(serialized.encode()),
                       "extraction_ms": extract_ms, "export_ms": export_ms,
                       "byte_equal": byte_equal, "semantic_equal": semantic_equal, **coverage(composition)}
                if native:
                    row["native"] = check_native(output, native)
                report["files"].append(row)
            except Exception as error:
                report["failures"].append({"file": name, "error": str(error)})
            if (index + 1) % 10 == 0:
                print(f"Checked {index + 1}/{len(paths)}; failures {len(report['failures'])}", flush=True)
    if native:
        report["demo"]["native_reconstructed"] = check_native(ROOT / "examples" / "reconstructed.txt", native)
        report["demo"]["native_modified"] = check_native(ROOT / "examples" / "modified.txt", native)
    else:
        report["demo"]["native_modified"] = {"status": "not run", "playback_verified": False}
    rows = report["files"]
    report["summary"] = {"files_found": len(paths), "files_passed": len(rows),
                         "empty_files_rejected": len(report["empty_inputs_rejected"]), "failures": len(report["failures"]),
                         "byte_exact_roundtrips": sum(r["byte_equal"] for r in rows),
                         "semantic_exact_roundtrips": sum(r["semantic_equal"] for r in rows),
                         "native_checks_passed": sum(r.get("native", {}).get("status") == "passed" for r in rows),
                         "tracks": sum(r["tracks"] for r in rows), "notes": sum(r["notes"] for r in rows),
                         "extracted_parameters": sum(r["extracted_parameters"] for r in rows)}
    if rows:
        report["summary"]["runtime_ms"] = {kind: {"median": statistics.median(r[kind] for r in rows),
                                                       "mean": statistics.mean(r[kind] for r in rows),
                                                       "min": min(r[kind] for r in rows), "max": max(r[kind] for r in rows)}
                                               for kind in ("extraction_ms", "export_ms")}
    report["registry"] = [vars(p) for p in PARAMETERS]
    write_json(ROOT / "reports" / "demo.json", report["demo"])
    write_json(ROOT / "reports" / "verification.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
