"""Read measured artifacts for matched SmolLM2 size experiments."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = {"smollm135": "HuggingFaceTB/SmolLM2-135M",
          "smollm360": "HuggingFaceTB/SmolLM2-360M",
          "smollm1b": "HuggingFaceTB/SmolLM2-1.7B"}
REVISIONS = {"smollm135": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
             "smollm360": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
             "smollm1b": "effd688a12921b4cc83e3312b6feb579f70f9c71"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def parameter_count(model_id, revision):
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        return model.num_parameters()
    except (ImportError, OSError, ValueError):
        return None


def collect(include_parameters=False):
    results = {}
    requests = {}
    for name, model_id in MODELS.items():
        root = ROOT / "apps" / name
        feature = root / "CustomTraining/training_output"
        v3 = root / "V3/output"
        feature_history = read(feature / "checkpoints/history.json")
        rhythm_history = read(v3 / "checkpoints/rhythm/history.json")
        item = {"model_id": model_id,
                "parameter_count": parameter_count(model_id, REVISIONS[name]) if include_parameters else None,
                "feature_pretraining": "trained" if (feature / "checkpoints/best_adapter/adapter_model.safetensors").is_file() else "untrained",
                "percussion": "trained" if (v3 / "checkpoints/rhythm/best_adapter/adapter_model.safetensors").is_file() else "untrained",
                "feature_training_loss": None,
                "feature_validation_loss": None,
                "feature_best_validation_loss": None,
                "percussion_training_loss": None,
                "percussion_validation_loss": None,
                "percussion_best_validation_loss": None,
                "decoding": {}}
        if feature_history and feature_history.get("epochs"):
            epoch = feature_history["epochs"][-1]
            item["feature_training_loss"] = epoch.get("train_token_weighted_loss")
            item["feature_validation_loss"] = epoch.get("validation", {}).get("loss")
            item["feature_best_validation_loss"] = feature_history.get("best_validation_loss")
        if rhythm_history and rhythm_history.get("epochs"):
            epoch = rhythm_history["epochs"][-1]
            item["percussion_training_loss"] = epoch.get("train_target_loss")
            item["percussion_validation_loss"] = epoch.get("validation", {}).get("loss")
            item["percussion_best_validation_loss"] = rhythm_history.get("best_validation_loss")
        for decoder in ("greedy", "constrained", "cd2"):
            evaluation = read(v3 / "evaluation" / f"v3_{decoder}_test.json")
            if evaluation is None:
                item["decoding"][decoder] = "not evaluated"
                continue
            examples = evaluation["examples"]
            requests[(name, decoder)] = {(r["source"], r["target_track_id"],
                                           r["request_type"], r["requested_note_count"])
                                          for r in examples}
            summary = evaluation["sweep_summary"]
            item["decoding"][decoder] = {
                "test_sources": sorted({r["source"] for r in examples}),
                "requested_counts": evaluation["count_requests"],
                "max_new_tokens": evaluation["max_new_tokens"],
                "requested_vs_generated_mean_absolute_error": summary.get("mean_absolute_count_error"),
                "exact_count_fraction": summary.get("exact_count_fraction"),
                "note_position_reference_f1": summary.get("mean_reference_onset_f1"),
                "rhythmic_diversity_fraction": summary.get("diversity_fraction"),
                "temporal_wrap_gap_ratio": summary.get("mean_wrap_gap_ratio"),
                "inference_seconds_mean": summary.get("mean_total_seconds"),
                "generated_tokens_mean": summary.get("mean_generated_tokens"),
                "output_valid_fraction": (summary["export_valid"] / summary["attempts"]
                                          if summary.get("attempts") else None),
                "heldout_loss": evaluation.get("heldout_loss", {}).get("loss"),
                "peak_gpu_bytes": evaluation.get("peak_gpu_bytes"),
                "gpu_name": evaluation.get("gpu_name")}
        results[name] = item
    aligned = {}
    for decoder in ("greedy", "constrained", "cd2"):
        sets = [requests[(name, decoder)] for name in MODELS if (name, decoder) in requests]
        aligned[decoder] = len(sets) >= 2 and all(s == sets[0] for s in sets[1:])
    return {"comparison_scope": "same held-out sources, target counts, decoder and token ceiling required",
            "matched_requests_by_decoder": aligned,
            "hardware_equivalence": "verify GPU name and conditions before interpreting latency",
            "models": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-parameters", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.count_parameters), indent=2))
