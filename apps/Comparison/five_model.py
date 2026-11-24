"""Compare the two historical V3 models with three V4EA position models.

Only the five shared held-out target pairs and metrics with matching definitions
appear in the comparison table. EA evaluations are opt-in and snapshotted by
decoding mode; ordinary comparison never loads or trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
APPS = HERE.parent
DECODINGS = ("greedy", "constrained")
EA_MODELS = {
    "135mea": ("smollm135", "HuggingFaceTB/SmolLM2-135M"),
    "360mea": ("smollm360", "HuggingFaceTB/SmolLM2-360M"),
    "17bea": ("smollm1b", "HuggingFaceTB/SmolLM2-1.7B"),
}
MODEL_ORDER = ("135m", "360m", *EA_MODELS)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets_from_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    targets = [{"source": row["source"], "target_track_id": row["target_track_id"],
                "positions": row["target_positions"]} for row in rows]
    if len(targets) != 5 or len({(row["source"], row["target_track_id"]) for row in targets}) != 5:
        raise ValueError(f"expected five distinct held-out target pairs: {path}")
    return targets


def valid_positions(entry: dict) -> bool:
    positions = entry.get("positions")
    return (entry.get("parse_valid") is True and isinstance(positions, list)
            and bool(positions) and positions == sorted(set(positions))
            and all(type(step) is int and 0 <= step < 64 for step in positions))


def jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def summarize_v3(report: dict, targets: list[dict], decoding: str) -> dict:
    if report.get("partition") != "test" or report.get("model") != f"v3_{decoding}":
        raise ValueError("V3 report has the wrong partition or decoder")
    if report.get("heldout_loss", {}).get("pairs") != 5:
        raise ValueError("V3 report loss does not cover five test pairs")
    expected = {(row["source"], row["target_track_id"]): row["positions"] for row in targets}
    references = [row for row in report["examples"] if row.get("request_type") == "reference"]
    if len(references) != 5 or {(row["source"], row["target_track_id"]) for row in references} != set(expected):
        raise ValueError("V3 report reference requests differ from held-out pairs")
    valid = [row for row in references if valid_positions(row)]
    return {"pairs": 5,
            "structurally_valid_rate": len(valid) / 5,
            "exact_note_count_rate": sum(row.get("exact_count") is True for row in references) / 5,
            "native_export_success_rate": sum(row.get("export_valid") is True for row in references) / 5,
            "rhythmic_jaccard_valid": (sum(jaccard(row["positions"], expected[(row["source"], row["target_track_id"])])
                                            for row in valid) / len(valid) if valid else None),
            "valid_jaccard_denominator": len(valid),
            "target_loss": report["heldout_loss"]["loss"],
            "target_loss_comparable_across_models": False,
            "generation_token_budget": report.get("max_new_tokens")}


def summarize_ea(report: dict, model_id: str) -> dict:
    if (report.get("model_id") != model_id or report.get("partition") != "test"
            or report.get("examples") != 5):
        raise ValueError("V4EA report has the wrong model, partition, or pair count")
    fields = ("structurally_valid_rate", "exact_note_count_rate",
              "native_export_success_rate")
    if any(type(report.get(field)) not in (int, float) or not 0 <= report[field] <= 1
           for field in fields):
        raise ValueError("V4EA report has invalid rate fields")
    j = report.get("rhythmic_jaccard")
    if j is not None and (type(j) not in (int, float) or not 0 <= j <= 1):
        raise ValueError("V4EA report has invalid rhythmic Jaccard")
    valid_count = round(report["structurally_valid_rate"] * 5)
    if abs(valid_count / 5 - report["structurally_valid_rate"]) > 1e-9:
        raise ValueError("V4EA structural rate is incompatible with five examples")
    if ((valid_count == 0) != (j is None) or
            report["exact_note_count_rate"] > report["structurally_valid_rate"] or
            report["native_export_success_rate"] > report["structurally_valid_rate"]):
        raise ValueError("V4EA report rates and rhythmic Jaccard disagree")
    return {"pairs": 5, **{field: report[field] for field in fields},
            "rhythmic_jaccard_valid": j, "valid_jaccard_denominator": valid_count,
            "target_loss": report.get("test_loss"),
            "target_loss_comparable_across_models": False,
            "generation_token_budget": 512}


def ea_output(slug: str) -> Path:
    return APPS / EA_MODELS[slug][0] / "MostRecentOutput/V4EA-positions-v1"


def best_adapter(output: Path) -> Path:
    coverage = output / "checkpoints/rhythm_coverage_v1/best_adapter"
    return coverage if coverage.is_dir() else output / "checkpoints/rhythm/best_adapter"


def evaluate_ea(slug: str, cpu: bool = False,
                decodings: tuple[str, ...] = DECODINGS) -> list[str]:
    """Run selected test decoders and save distinct, fingerprinted snapshots."""
    output = ea_output(slug)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return [f"{slug}: dataset is not prepared"]
    manifest = read_json(manifest_path)
    if (manifest.get("encoding") != "positions-v1" or
            manifest.get("model_id") != EA_MODELS[slug][1]):
        raise ValueError(f"{slug}: output is not the expected V4EA positions experiment")
    adapter = best_adapter(output)
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        return [f"{slug}: best adapter is not available"]
    targets = targets_from_rows(output / "rhythm/test.jsonl")
    snapshot_dir = HERE / "reports"
    snapshot_dir.mkdir(exist_ok=True)
    messages = []
    if not decodings or any(decoding not in DECODINGS for decoding in decodings):
        raise ValueError("unknown or empty decoder selection")
    for decoding in decodings:
        command = [sys.executable, "-m", "percussion_generation_v4ea",
                   "--output-dir", str(output), "evaluate", "--partition", "test"]
        if decoding == "constrained":
            command.append("--constrained")
        if cpu:
            command.append("--cpu")
        subprocess.run(command, cwd=APPS / EA_MODELS[slug][0] / "V4EA-synth", check=True)
        report_path = output / "evaluation_test.json"
        report = read_json(report_path)
        summarize_ea(report, EA_MODELS[slug][1])
        snapshot = {"schema_version": 1, "model": slug, "decoding": decoding,
                    "target_format": "positions-v1", "report": report,
                    "report_content_sha256": hashlib.sha256(
                        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "test_targets": targets,
                    "dataset_sha256": manifest["prepared_dataset_sha256"],
                    "tokenizer_sha256": manifest["tokenizer_sha256"],
                    "manifest_sha256": sha256(manifest_path),
                    "adapter_sha256": sha256(adapter_weights),
                    "adapter_metadata_sha256": sha256(adapter / "experiment_metadata.json"),
                    "report_sha256": sha256(report_path)}
        destination = snapshot_dir / f"{slug}_{decoding}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        messages.append(f"saved {destination.relative_to(HERE)}")
    return messages


def load_ea_snapshot(slug: str, decoding: str, expected: list[dict]) -> tuple[dict | None, str | None]:
    path = HERE / "reports" / f"{slug}_{decoding}.json"
    if not path.is_file():
        return None, "evaluation snapshot is missing"
    snapshot = read_json(path)
    if (snapshot.get("schema_version") != 1 or snapshot.get("model") != slug or
            snapshot.get("decoding") != decoding or snapshot.get("target_format") != "positions-v1"):
        raise ValueError(f"{path}: incompatible snapshot identity")
    if snapshot.get("test_targets") != expected:
        raise ValueError(f"{path}: test targets differ from historical V3")
    report_hash = hashlib.sha256(json.dumps(snapshot["report"], sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
    if snapshot.get("report_content_sha256") != report_hash:
        raise ValueError(f"{path}: evaluation report changed inside snapshot")
    output = ea_output(slug)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (manifest.get("encoding") != "positions-v1" or
                manifest.get("model_id") != EA_MODELS[slug][1] or
                manifest.get("prepared_dataset_sha256") != snapshot.get("dataset_sha256") or
                manifest.get("tokenizer_sha256") != snapshot.get("tokenizer_sha256") or
                sha256(manifest_path) != snapshot.get("manifest_sha256")):
            return None, "prepared dataset changed; rerun --evaluate-ea"
        adapter = best_adapter(output)
        weights = adapter / "adapter_model.safetensors"
        metadata = adapter / "experiment_metadata.json"
        if (not weights.is_file() or not metadata.is_file() or
                sha256(weights) != snapshot.get("adapter_sha256") or
                sha256(metadata) != snapshot.get("adapter_metadata_sha256")):
            return None, "best adapter changed; rerun --evaluate-ea"
    return summarize_ea(snapshot["report"], EA_MODELS[slug][1]), None


def build_comparison() -> dict:
    reference_path = APPS / "smollm135/V3/output/rhythm/test.jsonl"
    targets = targets_from_rows(reference_path)
    other = targets_from_rows(APPS / "smollm360/V3/output/rhythm/test.jsonl")
    if other != targets:
        raise ValueError("the two V3 models use different held-out targets")
    result = {"schema_version": 1, "test_targets": targets, "models": {}, "pending": {},
              "sources": {"v3_test_rows_sha256": sha256(reference_path)},
              "metric_scope": "five original-count held-out test requests per model and decoder"}
    for slug, model in (("135m", "smollm135"), ("360m", "smollm360")):
        result["models"][slug] = {}
        for decoding in DECODINGS:
            path = (APPS / model / "V3/output/evaluation" / f"v3_{decoding}_test.json"
                    if slug == "135m" else HERE / f"smollm360_{decoding}_512_backup.json")
            report = read_json(path)
            result["models"][slug][decoding] = summarize_v3(report, targets, decoding)
            result["sources"][f"{slug}_{decoding}"] = {"path": str(path.relative_to(APPS)),
                                                          "sha256": sha256(path)}
    for slug in EA_MODELS:
        result["models"][slug] = {}
        for decoding in DECODINGS:
            item, reason = load_ea_snapshot(slug, decoding, targets)
            result["models"][slug][decoding] = item
            if reason:
                result["pending"][f"{slug}_{decoding}"] = reason
            else:
                path = HERE / "reports" / f"{slug}_{decoding}.json"
                result["sources"][f"{slug}_{decoding}"] = {"path": str(path.relative_to(APPS)),
                                                              "sha256": sha256(path)}
    return result


def percent(value: float | None) -> str:
    return "Pending" if value is None else f"{value:.0%}"


def score(value: float | None) -> str:
    return "Pending" if value is None else f"{value:.3f}"


def render(result: dict) -> str:
    lines = ["# Five-model percussion comparison", "",
             "The table uses the same five held-out target tracks for the historical",
             "135M and 360M V3 models and the 135M, 360M, and 1.7B V4EA position models.",
             "Each row covers five original-count requests. Missing V4EA evaluations",
             "are shown as Pending; no scores are estimated.", "",
             "| Model | Decoder | Structural validity | Exact count | Native export | Rhythmic Jaccard (valid outputs) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for slug in MODEL_ORDER:
        for decoding in DECODINGS:
            item = result["models"][slug][decoding]
            if item is None:
                lines.append(f"| {slug} | {decoding} | Pending | Pending | Pending | Pending |")
            else:
                lines.append(f"| {slug} | {decoding} | {percent(item['structurally_valid_rate'])} | "
                             f"{percent(item['exact_note_count_rate'])} | "
                             f"{percent(item['native_export_success_rate'])} | "
                             f"{score(item['rhythmic_jaccard_valid'])} "
                             f"({item['valid_jaccard_denominator']}/5) |")
    lines += ["", "Rhythmic Jaccard is averaged over structurally valid outputs only; its",
              "denominator appears in each cell. Invalid outputs lower the validity rate",
              "but do not enter that Jaccard mean. V3 uses the five reference requests",
              "from its saved 35-request evaluation; sweep requests are excluded.", "",
              "Token cross-entropy is omitted from the ranking because tokenizers and",
              "conditioning representations differ. Inference time is omitted because the saved V3",
              "latencies include parsing and export while V4EA records generation time.",
              "V3-135M used a 128-token generation cap; the historical V3-360M report",
              "used 512. V4EA evaluation defaults to 512. These are separate training",
              "runs and do not establish a controlled speed comparison.", ""]
    if result["pending"]:
        lines += ["## Pending evaluations", ""]
        lines += [f"- {name}: {reason}" for name, reason in result["pending"].items()]
        lines += [""]
    lines += ["## Reproduce", "", "From `apps/Comparison`:", "",
              "```powershell", "python compare.py", "# After each V4EA model finishes training:",
              "python compare.py --evaluate-ea 135mea", "python compare.py --evaluate-ea 360mea",
              "python compare.py --evaluate-ea 17bea", "# Require all ten rows to be present:",
              "python compare.py --strict", "```", "",
              "`--evaluate-ea` performs test evaluation only. It does not train models.",
              "The two V3 reports and historical comparison files remain unchanged.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-ea", nargs="*", choices=tuple(EA_MODELS),
                        help="evaluate one or more finished V4EA models, or all if no names follow")
    parser.add_argument("--cpu", action="store_true", help="run requested V4EA evaluations on CPU")
    parser.add_argument("--decoding", choices=("both", *DECODINGS), default="both",
                        help="evaluate both decoders or only one (default: both)")
    parser.add_argument("--strict", action="store_true", help="fail if any result is pending")
    args = parser.parse_args(argv)
    if args.evaluate_ea is not None:
        requested = args.evaluate_ea or list(EA_MODELS)
        for slug in requested:
            selected = DECODINGS if args.decoding == "both" else (args.decoding,)
            for message in evaluate_ea(slug, cpu=args.cpu, decodings=selected):
                print(message)
    result = build_comparison()
    if args.strict and result["pending"]:
        parser.error("pending results: " + ", ".join(result["pending"]))
    (HERE / "five_model_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (HERE / "five_model_comparison.md").write_text(render(result), encoding="utf-8")
    print(f"Wrote five-model comparison; {len(result['pending'])} pending evaluation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
