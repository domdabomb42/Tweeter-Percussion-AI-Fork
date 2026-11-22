"""Render measured experiment files; do not replace absent results with claims."""
from pathlib import Path
import json


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render(output_root, report_path):
    root = Path(output_root)
    manifest = _read(root / "dataset" / "manifest.json")
    if not manifest:
        raise FileNotFoundError("prepare the dataset before building the report")
    base = _read(root / "evaluation" / "original_full.json")
    early_base = _read(root / "evaluation" / "original_validation.json")
    fine = _read(root / "evaluation" / "fine_tuned.json")
    compared = _read(root / "evaluation" / "comparison.json")
    history = _read(root / "checkpoints" / "history.json")
    tokens = manifest["token_counts"]
    counts = manifest["split"]["actual_counts"]
    lines = ["# SmolLM2-1.7B CustomExtractor proof of concept", "",
             "## Dataset", "",
             f"The main sessions folder yielded **{manifest['sessions']} real native sessions**, "
             f"**{manifest['tracks']} tracks**, and **{manifest['notes']} active notes**. "
             f"The source-grouped split contains **{counts['train']} train**, "
             f"**{counts['validation']} validation**, and **{counts['test']} test** files. "
             f"There are **{tokens['full_compositions']:,} full-composition tokenizer tokens** "
             f"and **{tokens['prepared_segments_including_eos']:,} prepared tokens including repeated global context/EOS.", "",
             "Track loop lengths are cycle proxies; the files do not record total playback durations. "
             "Exact per-file counts, lengths, source hashes, and all distributions are in `training_output/dataset/manifest.json`.", "",
             f"There are **{manifest['distinct_waveform_combinations']} carrier/modulator waveform pairs**. "
             f"Whole-composition token count: mean {tokens['per_composition']['mean']:.1f}, "
             f"median {tokens['per_composition']['median']}, max {tokens['per_composition']['max']}. "
             f"Track tokens: mean {tokens['per_track']['mean']:.1f}, max {tokens['per_track']['max']}.", "",
             "| Context limit | Complete sessions fitting |", "| --- | ---: |"]
    for bound, value in tokens["fit_without_segmentation"].items():
        lines.append(f"| {bound} | {value['count']}/{manifest['sessions']} ({value['percent']:.1f}%) |")
    lines.extend(["", "The 4,096-token training sequences contain whole tracks, with no value quantization or silent truncation. "
                  "The same original composition never crosses splits; related versions and near-overlapping note patterns are grouped.", "",
                  "| Parameter | Samples | Distinct values | Min | Median | Max |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for parameter in ("synth.harmonicity", "synth.mod_amplitude", "effects.reverb.amplitude",
                      "effects.reverb.type", "effects.echo.amplitude", "effects.echo.feedback",
                      "effects.echo.delay_ms", "effects.flanger.amplitude",
                      "effects.flanger.frequency_hz", "effects.flanger.mod_amplitude",
                      "effects.flanger.delay_ms", "effects.overdrive.level"):
        value = manifest["numeric_distributions"].get(parameter)
        if value:
            lines.append(f"| {parameter} | {value['count']} | {value['distinct']} | "
                         f"{value['min']} | {value['median']} | {value['max']} |")
    lines.extend(["", f"Of {tokens['prepared_segments_including_eos']:,} prepared tokens, "
                  f"{tokens['field_token_categories'].get('synthesis', 0):,} cover synthesis values and "
                  f"{tokens['field_token_categories'].get('notes', 0):,} cover note values. "
                  f"The 51 files form only **{len(manifest['split']['groups'])} source/arrangement groups**; "
                  f"validation uses **{len({r['family'] for r in manifest['split']['partitions']['validation']})} families** "
                  f"and test uses **{len({r['family'] for r in manifest['split']['partitions']['test']})} families**.", "",
                  "## Training", ""])
    if history:
        plan = history["plan"]
        lines += [f"SmolLM2-1.7B was fine-tuned on **{plan['device']}** using **{plan['dtype']}**, "
                  f"LoRA rank {plan['lora_rank']}, alpha {plan['lora_alpha']}, dropout {plan['lora_dropout']}. "
                  f"It completed **{history['epochs'][-1]['updates']} optimizer updates** in "
                  f"{history.get('elapsed_seconds', 0)/60:.1f} minutes. Best checkpoint: "
                  f"`{history['best_checkpoint']}` (epoch {history['best_epoch']}, "
                  f"validation loss {history['best_validation_loss']:.4f}).", "",
                  "| Epoch | Train token loss | Validation loss | Updates |", "| ---: | ---: | ---: | ---: |"]
        for epoch in history["epochs"]:
            lines.append(f"| {epoch['epoch']} | {epoch['train_token_weighted_loss']:.4f} | "
                         f"{epoch['validation']['loss']:.4f} | {epoch['updates']} |")
        lines += ["", f"The last train/validation loss gap is "
                  f"{history['epochs'][-1]['validation']['loss'] - history['epochs'][-1]['train_token_weighted_loss']:+.4f} "
                  "(validation minus training)."]
    else:
        lines.append("No completed LoRA training history is available. Training results cannot be claimed.")
    lines += ["", "## Evaluation", ""]
    if compared:
        lines += ["| Partition | Original loss | Tuned loss | Improvement | Original PPL | Tuned PPL |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for partition, value in compared["partitions"].items():
            lines.append(f"| {partition} | {value['original_loss']:.4f} | {value['fine_tuned_loss']:.4f} | "
                         f"{value['loss_improvement_percent']:+.1f}% | "
                         f"{value['original_perplexity']:.2f} | {value['fine_tuned_perplexity']:.2f} |")
        test = compared["partitions"]["test"]
        lines += ["", "Feature completion compares **complete parsed values**. Categorical values use exact matches; "
                  "continuous values use MAE. Invalid or incomplete numeric outputs are excluded from MAE and counted in coverage. "
                  "The simple predictor is the training-only mode or median.", "",
                  "| Test feature | n | Original exact | Tuned exact | Statistical exact | Original MAE | Tuned MAE | Statistical MAE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        def fmt(value):
            return "â€”" if value is None else f"{value:.3f}"
        for name, value in sorted(test["features"].items()):
            lines.append(f"| {name} | {value['n']} | {fmt(value['base_exact_accuracy'])} | "
                         f"{fmt(value['fine_exact_accuracy'])} | {fmt(value['statistical_exact_accuracy'])} | "
                         f"{fmt(value['base_mae'])} | {fmt(value['fine_mae'])} | {fmt(value['statistical_mae'])} |")
        better = [name for name, value in test["features"].items()
                  if value["fine_exact_accuracy"] is not None and value["statistical_exact_accuracy"] is not None
                  and value["fine_exact_accuracy"] > value["statistical_exact_accuracy"]
                  or value["fine_mae"] is not None and value["statistical_mae"] is not None
                  and value["fine_mae"] < value["statistical_mae"]]
        lines += ["", f"The tuned model beats the simple train-only predictor on "
                  f"**{len(better)} of {len(test['features'])} scored feature types** by exact accuracy "
                  "or MAE, where those measures are defined. A feature type has at most six sampled "
                  "targets, so this is descriptive rather than a significance test."]
        lines += ["", f"CustomExtractor-valid short continuations on the test set: "
                  f"original **{test['original_structural_valid']}**, tuned **{test['fine_structural_valid']}** "
                  f"out of {fine['test']['structure']['count']} prompts. "
                  "Token-category accuracy, parse errors, and individual completions are in the evaluation JSON files."]
    elif early_base:
        lines.append(f"The original model's validation loss was **{early_base['validation']['causal']['loss']:.4f}** "
                     f"(perplexity **{early_base['validation']['causal']['perplexity']:.2f}**). "
                     "The test set remains sealed until a checkpoint is selected.")
    else:
        lines.append("Neither original-model nor tuned-model evaluation has completed.")
    lines += ["", "## Interpretation and future requirements", "",
              "This sample is small and grouped families can make the effective number of independent pieces lower than the file count. "
              "Lower token loss does not demonstrate acoustic understanding; feature-value results against the statistical baseline are the relevant check. "
              "Additional independent real sessions would make held-out conclusions more stable. "
              "Only one seed was run unless separate output directories contain further runs, so split sensitivity is unmeasured. "
              "The low note-token share suggests that future real note-rich sessions would be especially useful. "
              "No further pretraining, data generation, or serializer change is implemented here.", ""]
    path = Path(report_path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
