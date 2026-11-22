"""Reproduce the held-out SmolLM2-135M versus 360M V3 comparison.

Run ``python compare.py`` from this directory after the two 135M test
evaluations exist. The complete historical 360M evaluations are kept here.
This script reads saved reports; it never trains a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
APPS = HERE.parent
MODELS = ("smollm135", "smollm360")
DECODINGS = ("greedy", "constrained")
PULSE_STEPS = 4  # Descriptive grid proxy, not an asserted meter.
MAX_NEW_TOKENS = 128


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def fraction(values):
    return mean([bool(value) for value in values])


def pair_key(entry):
    return (entry["source"], entry["target_track_id"])


def request_key(entry):
    return (*pair_key(entry), entry["request_type"], entry["requested_note_count"])


def valid_entry(entry) -> bool:
    positions = entry.get("positions")
    return bool(entry.get("parse_valid") and entry.get("export_valid")
                and entry.get("loop_valid") and isinstance(positions, list)
                and positions and len(positions) == len(set(positions))
                and all(type(position) is int and 0 <= position < 64 for position in positions))


def onset_f1(generated: set[int], original: set[int]) -> float:
    return 2 * len(generated & original) / (len(generated) + len(original)) if generated or original else 1.0


def rhythm_structure(positions: list[int]) -> dict:
    steps = set(positions)
    if not steps:
        return {"pulse_alignment_4_step": None, "off_pulse_4_step": None,
                "repeat_16_step": None, "repeat_32_step": None, "circular_gap_cv": None}
    ordered = sorted(steps)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 64 - ordered[-1])
    pulse = sum(step % PULSE_STEPS == 0 for step in steps) / len(steps)
    return {"pulse_alignment_4_step": pulse,
            "off_pulse_4_step": 1 - pulse,
            "repeat_16_step": len(steps & {(step + 16) % 64 for step in steps}) / len(steps),
            "repeat_32_step": len(steps & {(step + 32) % 64 for step in steps}) / len(steps),
            "circular_gap_cv": statistics.pstdev(gaps) / statistics.mean(gaps)}


def structure_summary(patterns: list[list[int]]) -> dict:
    metrics = [rhythm_structure(pattern) for pattern in patterns]
    return {key: mean([item[key] for item in metrics if item[key] is not None])
            for key in rhythm_structure([0])}


def pairwise_jaccard_distance(patterns: list[list[int]]) -> float | None:
    if len(patterns) < 2:
        return None
    sets = [set(pattern) for pattern in patterns]
    return mean([1 - len(left & right) / len(left | right) if left or right else 0.0
                 for index, left in enumerate(sets) for right in sets[index + 1:]])


def duration_summary(history: dict) -> dict:
    epochs = history["epochs"]
    durations = []
    previous = 0.0
    for item in epochs:
        elapsed = item["elapsed_seconds"]
        duration = elapsed - previous if elapsed >= previous else elapsed
        if duration <= 0:
            raise ValueError("invalid elapsed_seconds history")
        durations.append(duration)
        previous = elapsed
    peaks = [item["peak_gpu_bytes"] for item in epochs if item.get("peak_gpu_bytes") is not None]
    return {"completed_epochs": len(epochs), "best_epoch": history.get("best_epoch"),
            "best_validation_loss": history.get("best_validation_loss"),
            "epoch_seconds": durations, "mean_epoch_seconds": mean(durations),
            "total_recorded_epoch_seconds": sum(durations),
            "peak_training_gpu_bytes_recorded": max(peaks) if peaks else None}


def summarize(report: dict, references: dict, history: dict) -> dict:
    examples = report["examples"]
    reference_entries = [entry for entry in examples if entry["request_type"] == "reference"]
    sweep_entries = [entry for entry in examples if entry["request_type"] == "sweep"]
    for entry in reference_entries:
        if pair_key(entry) not in references:
            raise ValueError(f"unknown held-out target {pair_key(entry)}")
        if valid_entry(entry):
            expected = onset_f1(set(entry["positions"]), set(references[pair_key(entry)]))
            if abs(expected - entry["reference_onset_f1"]) > 1e-12:
                raise ValueError("saved onset F1 does not match original target")
    generated_reference = [entry["positions"] for entry in reference_entries if valid_entry(entry)]
    generated_sweep = [entry["positions"] for entry in sweep_entries if valid_entry(entry)]
    valid = [entry for entry in examples if valid_entry(entry)]
    all_latencies = [entry["total_seconds"] for entry in examples if entry.get("total_seconds") is not None]
    return {
        "heldout_loss": report["heldout_loss"],
        "onset_f1_mean_reference_requests": mean([entry["reference_onset_f1"] if valid_entry(entry) else 0.0
                                                    for entry in reference_entries]),
        "onset_f1_mean_valid_reference": mean([entry["reference_onset_f1"]
                                                for entry in reference_entries if valid_entry(entry)]),
        "reference_requests": len(reference_entries), "sweep_requests": len(sweep_entries),
        "valid_reference_patterns_for_groove": len(generated_reference),
        "note_count_exact_reference": fraction([entry.get("exact_count", False) for entry in reference_entries]),
        "note_count_exact_sweep": fraction([entry.get("exact_count", False) for entry in sweep_entries]),
        "mean_absolute_note_count_error_sweep": mean([entry["absolute_count_error"]
                                                       for entry in sweep_entries if entry.get("absolute_count_error") is not None]),
        "valid_generation_rate_all": len(valid) / len(examples),
        "valid_generation_count_all": len(valid), "attempts_all": len(examples),
        "valid_generation_rate_reference": fraction([valid_entry(entry) for entry in reference_entries]),
        "valid_generation_rate_sweep": fraction([valid_entry(entry) for entry in sweep_entries]),
        "groove_proxies_reference": structure_summary(generated_reference),
        "diversity_unique_fraction_reference": len({tuple(x) for x in generated_reference}) / len(generated_reference)
                                               if generated_reference else None,
        "diversity_unique_fraction_sweep": len({tuple(x) for x in generated_sweep}) / len(generated_sweep)
                                           if generated_sweep else None,
        "diversity_pairwise_jaccard_reference": pairwise_jaccard_distance(generated_reference),
        "diversity_pairwise_jaccard_sweep": pairwise_jaccard_distance(generated_sweep),
        "human_listening_score": None,
        "latency_device": "cuda" if report.get("gpu_name") else "unrecorded",
        "mean_generation_seconds_all": mean(all_latencies),
        "median_generation_seconds_all": statistics.median(all_latencies) if all_latencies else None,
        "mean_generation_seconds_reference": mean([entry["total_seconds"] for entry in reference_entries
                                                     if entry.get("total_seconds") is not None]),
        "peak_evaluation_gpu_bytes_recorded": report.get("peak_gpu_bytes"),
        "training": duration_summary(history),
    }


def format_number(value, places=3):
    return "Not measured" if value is None else f"{value:.{places}f}"


def format_percent(value):
    return "Not measured" if value is None else f"{100 * value:.1f}%"


def render_markdown(result: dict, decoding: str) -> str:
    a, b = (result["models"][name][decoding] for name in MODELS)
    real = result["real_test_tracks"]["groove_proxies"]
    table = [
        ("Held-out target loss / perplexity", f"{format_number(a['heldout_loss']['loss'])} / {format_number(a['heldout_loss']['perplexity'])}",
         f"{format_number(b['heldout_loss']['loss'])} / {format_number(b['heldout_loss']['perplexity'])}"),
        ("Onset F1, 5 original-count requests", format_number(a["onset_f1_mean_reference_requests"]),
         format_number(b["onset_f1_mean_reference_requests"])),
        ("Exact note count, 30 count-sweep requests", format_percent(a["note_count_exact_sweep"]),
         format_percent(b["note_count_exact_sweep"])),
        ("Valid generation, all 35 requests", format_percent(a["valid_generation_rate_all"]),
         format_percent(b["valid_generation_rate_all"])),
        ("16-step repetition proxy; real " + format_number(real["repeat_16_step"]),
         format_number(a["groove_proxies_reference"]["repeat_16_step"]),
         format_number(b["groove_proxies_reference"]["repeat_16_step"])),
        ("32-step repetition proxy; real " + format_number(real["repeat_32_step"]),
         format_number(a["groove_proxies_reference"]["repeat_32_step"]),
         format_number(b["groove_proxies_reference"]["repeat_32_step"])),
        ("4-step pulse alignment proxy; real " + format_percent(real["pulse_alignment_4_step"]),
         format_percent(a["groove_proxies_reference"]["pulse_alignment_4_step"]),
         format_percent(b["groove_proxies_reference"]["pulse_alignment_4_step"])),
        ("Off-pulse positions proxy; real " + format_percent(real["off_pulse_4_step"]),
         format_percent(a["groove_proxies_reference"]["off_pulse_4_step"]),
         format_percent(b["groove_proxies_reference"]["off_pulse_4_step"])),
        ("Circular onset-gap variation; real " + format_number(real["circular_gap_cv"]),
         format_number(a["groove_proxies_reference"]["circular_gap_cv"]),
         format_number(b["groove_proxies_reference"]["circular_gap_cv"])),
        ("Unique rhythms, 5 original-count requests", format_percent(a["diversity_unique_fraction_reference"]),
         format_percent(b["diversity_unique_fraction_reference"])),
        ("Unique rhythms, 30 count-sweep requests", format_percent(a["diversity_unique_fraction_sweep"]),
         format_percent(b["diversity_unique_fraction_sweep"])),
        ("Pairwise rhythm distance; real " + format_number(result["real_test_tracks"]["pairwise_jaccard_distance"]),
         format_number(a["diversity_pairwise_jaccard_reference"]),
         format_number(b["diversity_pairwise_jaccard_reference"])),
        ("Human listening score", "Not measured", "Not measured"),
        ("Mean generation latency, 35 requests", format_number(a["mean_generation_seconds_all"]) + " s (GPU)",
         format_number(b["mean_generation_seconds_all"]) + " s (device unrecorded)"),
        ("Peak GPU allocation during evaluation",
         format_number(a["peak_evaluation_gpu_bytes_recorded"] / 1024**3, 2) + " GiB" if a["peak_evaluation_gpu_bytes_recorded"] is not None else "Not recorded",
         format_number(b["peak_evaluation_gpu_bytes_recorded"] / 1024**3, 2) + " GiB" if b["peak_evaluation_gpu_bytes_recorded"] is not None else "Not recorded"),
        ("Recorded mean training epoch", format_number(a["training"]["mean_epoch_seconds"] / 60, 1) + " min",
         format_number(b["training"]["mean_epoch_seconds"] / 60, 1) + " min"),
        ("Recorded epoch time, all completed epochs", format_number(a["training"]["total_recorded_epoch_seconds"] / 3600, 2) + " h (" + str(a["training"]["completed_epochs"]) + " epochs)",
         format_number(b["training"]["total_recorded_epoch_seconds"] / 3600, 2) + " h (" + str(b["training"]["completed_epochs"]) + " epochs)"),
        ("Recorded training GPU peak", (format_number(a["training"]["peak_training_gpu_bytes_recorded"] / 1024**3, 2) + " GiB"
                                          if a["training"]["peak_training_gpu_bytes_recorded"] is not None else "Not recorded"),
         (format_number(b["training"]["peak_training_gpu_bytes_recorded"] / 1024**3, 2) + " GiB"
          if b["training"]["peak_training_gpu_bytes_recorded"] is not None else "Not recorded")),
    ]
    rows = "\n".join(f"| {name} | {left} | {right} |" for name, left, right in table)
    winner = ("135M" if a["heldout_loss"]["loss"] < b["heldout_loss"]["loss"] else "360M")
    extra_flag = " --constrained" if decoding == "constrained" else ""
    previous_max = result["sources"]["smollm360"]["backups"][decoding]["max_generated_tokens"]
    return f"""# SmolLM2 V3 percussion comparison: {decoding}

This compares the **135M** model with the existing **360M** model (the requested
“smollm350”). Both were scored with {decoding} decoding on the same five held-out
percussion targets from three source sessions. Each target was also requested
at six note counts (5, 8, 10, 12, 16, 20), giving 35 generation attempts per
model. The 135M evaluation recorded a GPU device; the earlier 360M report did
not record its device. The saved training runs were not controlled hardware
benchmarks.
Both checkpoints were trained on the original V3 pairs; the new
`L2Ork-Tweeter/AugmentedSessions` files were not used.

The 135M generation limit was 128 new tokens per request. The 360M
evaluation used a 512-token limit, but all 35 outputs ended by token
{previous_max}, within the 135M limit. Its complete report is preserved as
`smollm360_{decoding}_512_backup.json`.

| Metric | 135M | 360M |
| --- | ---: | ---: |
{rows}

Lower held-out loss favors **{winner}** on these five examples. Onset F1 uses
the original requested count and compares generated tracker positions with
the actual withheld percussion track; invalid outputs score zero. Exact count
and validity also include failed generations in their denominators. A
generation is valid only when parsing,
0–63 step bounds, uniqueness and native Tweeter export all succeed.

The groove rows are **descriptive proxies**, computed on valid original-count
outputs (135M: {a['valid_reference_patterns_for_groove']}/5; 360M:
{b['valid_reference_patterns_for_groove']}/5) and the five real target tracks.
A 4-step pulse is a grid assumption,
not a verified time signature or a claim about musical beat strength.
“Off-pulse” is only a syncopation proxy. A high repetition score need not sound
better; compare its closeness to the real tracks. These statistics and onset F1
do not establish musical coherence. The 135M and 360M models predict tracker
positions only and use a fixed percussion preset, so they do not learn the
source track's sound design.

**Human listening was not performed.** A meaningful score requires rendered
audio from both models over the same accompaniment, randomized presentation,
blinded human ratings and a written scale. No numerical score is assigned.

Generation latency includes parsing/export. The 135M run used the GPU; the
360M report does not identify its device, so these latency numbers are not
a controlled speed comparison. The training times come from separate
historical runs. The 135M history records GPU peak allocation;
the 360M history does not. The peak value therefore cannot be ranked fairly.
The 135M recorded peak exceeds this GPU's 6 GiB of dedicated memory;
Windows can use shared GPU memory, and the historical run's memory setup
was not independently recorded.
See `comparison.json` for individual measurements, exact denominators,
reference-track proxies and source hashes.

## Reproduce

From `apps/Comparison`, run `python compare.py`. The script reads
`../smollm135/V3/output/evaluation/v3_{decoding}_test.json` and the saved
360M report in this folder, checkpoint histories and held-out row manifests. It checks
that the two reports use identical test targets and generation requests, then
regenerates `comparison.json` and this README. To regenerate the 135M raw
evaluation, run from `apps/smollm135/V3`:

```powershell
$env:USE_TF='0'
python -m percussion_generation_v3 evaluate{extra_flag} --partition test --cap 5 --counts 5,8,10,12,16,20 --max-new-tokens 128
```

The analogous command in `apps/smollm360/V3` can create a new GPU evaluation,
but it overwrites that model's original evaluation file. Neither command
trains or changes a checkpoint. Keep the hardware workload similar if
repeating latency measurements.
"""


def render_overview(result: dict) -> str:
    lines = ["# SmolLM2 V3 percussion comparison", "",
             "The requested `smollm350` is the existing `smollm360` experiment.",
             "Both checkpoints were evaluated with greedy and constrained decoding on",
             "the same five held-out percussion targets from three sessions. They also",
             "received 30 requested-note-count trials each per decoding method.",
             "The constrained adapter is a copy of the rhythm adapter's weights;",
             "constrained decoding is a rule at inference, not a second trained model.",
             "The new augmented sessions were not used to train these checkpoints.", "",
             "| Decoding | Model | Test loss | Perplexity | Onset F1 | Exact count (30) | Valid (35) | Mean latency |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for decoding in DECODINGS:
        for name in MODELS:
            item = result["models"][name][decoding]
            lines.append(f"| {decoding} | {name} | {format_number(item['heldout_loss']['loss'])} | "
                         f"{format_number(item['heldout_loss']['perplexity'])} | "
                         f"{format_number(item['onset_f1_mean_reference_requests'])} | "
                         f"{format_percent(item['note_count_exact_sweep'])} | "
                         f"{format_percent(item['valid_generation_rate_all'])} | "
                         f"{format_number(item['mean_generation_seconds_all'])} s ({item['latency_device']}) |")
    lines += ["", "Constrained decoding achieved the requested count in all 30 sweep trials",
              "for each model; greedy decoding achieved it in none. Both constrained",
              "runs exported 35/35 valid tracks. The 360M checkpoint has lower held-out",
              "loss and higher onset F1 on this small test set. Onset F1",
              "remains below 0.18 in every condition, so count compliance alone does",
              "not establish that the rhythm fits the source music.", "",
              "Detailed tables cover groove proxies, diversity, training time and peak GPU use:",
              "[greedy results](greedy.md) and [constrained results](constrained.md).",
              "[comparison.json](comparison.json) holds the calculated measurements and source hashes.",
              "The complete 360M evaluations used a 512-token cap, but all outputs",
              "ended before 128 tokens. The 135M runs used a 128-token cap.",
              "The 360M reports did not record the evaluation device, so the latency",
              "figures are not a controlled speed comparison.", "",
              "No human listening score has been measured. Groove scores are grid-based",
              "proxies and do not establish that a generated percussion line sounds good.", "",
              "## Reproduce", "", "From this directory run `python compare.py`. The raw",
              "V3 evaluation commands are given in each detailed report. They evaluate",
              "existing checkpoints and do not train a model.", ""]
    return "\n".join(lines)


def main() -> None:
    reports, histories, rows, sources = {}, {}, {}, {}
    for name in MODELS:
        root = APPS / name / "V3/output"
        history_path = root / "checkpoints/rhythm/history.json"
        rows_path = root / "rhythm/test.jsonl"
        histories[name] = read_json(history_path)
        rows[name] = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        sources[name] = {"history": str(history_path.relative_to(APPS)), "history_sha256": fingerprint(history_path),
                         "test_rows": str(rows_path.relative_to(APPS)), "test_rows_sha256": fingerprint(rows_path),
                         "evaluations": {}, "adapters": {}, "backups": {}}
        reports[name] = {}
        for decoding in DECODINGS:
            report_path = (HERE / f"smollm360_{decoding}_512_backup.json" if name == "smollm360"
                           else root / f"evaluation/v3_{decoding}_test.json")
            adapter_path = root / "checkpoints" / ("constrained" if decoding == "constrained" else "rhythm") / "best_adapter/adapter_model.safetensors"
            reports[name][decoding] = read_json(report_path)
            sources[name]["evaluations"][decoding] = {"path": str(report_path.relative_to(APPS)),
                                                     "sha256": fingerprint(report_path)}
            sources[name]["adapters"][decoding] = {"path": str(adapter_path.relative_to(APPS)),
                                                  "sha256": fingerprint(adapter_path)}
        if sources[name]["adapters"]["greedy"]["sha256"] != sources[name]["adapters"]["constrained"]["sha256"]:
            raise ValueError(f"{name}: constrained and greedy adapter weights differ")
    for decoding in DECODINGS:
        backup = HERE / f"smollm360_{decoding}_512_backup.json"
        prior = read_json(backup)
        if len(prior["examples"]) != 35 or not all(item["ended_with_eos"] and item["generated_tokens"] < 128
                                                   for item in prior["examples"]):
            raise ValueError(f"{decoding}: 360M original 512-token run did not fit within 128 tokens")
        sources["smollm360"]["backups"][decoding] = {"path": backup.name, "sha256": fingerprint(backup),
                                                    "max_generated_tokens": max(item["generated_tokens"] for item in prior["examples"])}
    targets = lambda name: {(item["source"], item["target_track_id"]): item["target_positions"] for item in rows[name]}
    if targets(MODELS[0]) != targets(MODELS[1]):
        raise ValueError("models do not use the same held-out targets")
    expected_requests = [request_key(item) for item in reports[MODELS[0]]["greedy"]["examples"]]
    for name in MODELS:
        for decoding in DECODINGS:
            report = reports[name][decoding]
            if report["partition"] != "test" or report["model"] != f"v3_{decoding}":
                raise ValueError(f"{name} {decoding}: wrong test report")
            expected_budget = 512 if name == "smollm360" else MAX_NEW_TOKENS
            if report["count_requests"] != [5, 8, 10, 12, 16, 20] or report["max_new_tokens"] != expected_budget:
                raise ValueError(f"{name} {decoding}: incompatible generation settings")
            if [request_key(item) for item in report["examples"]] != expected_requests:
                raise ValueError(f"{name} {decoding}: test requests do not match")
            if report["heldout_loss"]["pairs"] != len(rows[name]):
                raise ValueError(f"{name} {decoding}: held-out loss covers a different number of pairs")
    original_patterns = list(targets(MODELS[0]).values())
    result = {"schema_version": 2, "evaluation": "greedy and constrained; identical held-out rows and requests",
              "generation_settings": {"max_new_tokens_135m": MAX_NEW_TOKENS,
                                      "max_new_tokens_360m": 512,
                                      "all_360m_outputs_ended_before_128_tokens": True,
                                      "requested_note_counts": [5, 8, 10, 12, 16, 20],
                                      "test_pairs": len(rows[MODELS[0]])},
              "sources": sources,
              "real_test_tracks": {"pairs": len(original_patterns),
                                   "source_sessions": len({item["source"] for item in rows[MODELS[0]]}),
                                   "groove_proxies": structure_summary(original_patterns),
                                   "unique_pattern_fraction": len({tuple(x) for x in original_patterns}) / len(original_patterns),
                                   "pairwise_jaccard_distance": pairwise_jaccard_distance(original_patterns)},
              "models": {name: {decoding: summarize(reports[name][decoding], targets(name), histories[name])
                                 for decoding in DECODINGS} for name in MODELS},
              "limitations": ["No human listening study or audio coherence score was performed.",
                              "4-step pulse, off-pulse and repetition are grid proxies, not ground-truth groove ratings.",
                              "Historical training times and GPU peaks were not measured in a controlled matched run.",
                              "360M evaluation device was not recorded; latency cannot be ranked against the 135M GPU run.",
                              "The 135M recorded peak exceeds this GPU's 6 GiB of dedicated memory; shared GPU memory or a different historical setup may explain it."]}
    if any(result["models"]["smollm135"][decoding]["latency_device"] != "cuda"
           for decoding in DECODINGS):
        raise ValueError("135M latency reports must be measured on GPU")
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (HERE / "greedy.md").write_text(render_markdown(result, "greedy"), encoding="utf-8")
    (HERE / "constrained.md").write_text(render_markdown(result, "constrained"), encoding="utf-8")
    (HERE / "README.md").write_text(render_overview(result), encoding="utf-8")
    print("Wrote apps/Comparison/comparison.json, README.md, greedy.md, constrained.md")


if __name__ == "__main__":
    main()
