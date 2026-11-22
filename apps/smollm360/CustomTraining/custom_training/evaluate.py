"""Held-out next-token, complete feature-value, and native-structure checks."""
from collections import defaultdict
from pathlib import Path
import json
import math
import random
import re
import statistics
import torch
from .baselines import feature_instances, train_statistics
from .prepare import load_prepared, tokenizer_for_model
from .serializer import serialize_object
from .train import load_model, score


NUMBER = re.compile(r"\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s*[,}\]]|\s*$)")


def _value(fragment):
    fragment = fragment.lstrip()
    if fragment.startswith('"'):
        try:
            return json.JSONDecoder().raw_decode(fragment)[0]
        except ValueError:
            return None
    match = NUMBER.match(fragment)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if type(parsed) is not float or math.isfinite(parsed) else None
    except ValueError:
        return None


def select_feature_queries(rows, cap=6, seed=415):
    grouped = defaultdict(list)
    for instance in feature_instances(rows):
        grouped[instance["name"]].append(instance)
    rng = random.Random(seed)
    selected = []
    for name, items in sorted(grouped.items()):
        if len(items) > cap:
            items = rng.sample(items, cap)
        selected.extend(items)
    return selected


def complete_value(model, tokenizer, device, prefix, max_new_tokens=32):
    ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.shape[1] > 8192 - max_new_tokens:
        raise ValueError("feature context exceeds model window; segment preparation was invalid")
    with torch.inference_mode():
        generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   use_cache=True)
    fragment = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    return _value(fragment), fragment


def _native_valid(name, value):
    from custom_extractor.parameter_registry import BY_PATH
    from custom_extractor.validator import check_parameter
    if value is None:
        return False
    if name == "note.step":
        return type(value) in (int, float) and math.isfinite(value) and value == int(value) and 0 <= value <= 63
    if name == "note.pitch":
        return type(value) in (int, float) and math.isfinite(value) and 0 < abs(value) <= 3.402823466e38
    parameter = BY_PATH.get(name)
    if parameter is None:
        return False
    try:
        check_parameter(parameter, value, name)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def feature_metrics(model, tokenizer, device, queries, training_stats):
    results = defaultdict(lambda: {"targets": 0, "valid_predictions": 0, "exact_matches": 0,
                                   "absolute_errors": [], "baseline_absolute_errors": [],
                                   "baseline_exact_matches": 0, "examples": []})
    model.eval()
    for query in queries:
        name, true = query["name"], query["value"]
        prediction, fragment = complete_value(model, tokenizer, device, query["prefix"])
        baseline = training_stats.get(name, {}).get("prediction")
        row = results[name]
        row["targets"] += 1
        row["valid_predictions"] += prediction is not None
        row["native_range_valid"] = row.get("native_range_valid", 0) + _native_valid(name, prediction)
        row["exact_matches"] += prediction == true
        row["baseline_exact_matches"] += baseline == true
        if type(true) in (int, float):
            if type(prediction) in (int, float) and math.isfinite(prediction):
                row["absolute_errors"].append(abs(prediction - true))
            if type(baseline) in (int, float):
                row["baseline_absolute_errors"].append(abs(baseline - true))
        row["examples"].append({"source": query["source"], "path": query["path"],
                                "target": true, "prediction": prediction, "baseline": baseline,
                                "native_range_valid": _native_valid(name, prediction),
                                "generated_fragment": fragment[:120]})
    for row in results.values():
        row["exact_accuracy"] = row["exact_matches"] / row["targets"]
        row["baseline_exact_accuracy"] = row["baseline_exact_matches"] / row["targets"]
        row["mae"] = statistics.mean(row["absolute_errors"]) if row["absolute_errors"] else None
        row["baseline_mae"] = statistics.mean(row["baseline_absolute_errors"]) if row["baseline_absolute_errors"] else None
        row["numeric_prediction_coverage"] = len(row["absolute_errors"]) / row["targets"]
        row["native_range_valid_rate"] = row["native_range_valid"] / row["targets"]
        del row["absolute_errors"], row["baseline_absolute_errors"]
    return dict(results)


def _structure(model, tokenizer, device, rows, seed=415, count=4):
    from custom_extractor import Composition, validate, render_tweeter, extract_text
    rng = random.Random(seed)
    sample = rng.sample(rows, min(count, len(rows)))
    result = []
    model.eval()

    def same_field_order(template, candidate):
        if isinstance(template, dict):
            return isinstance(candidate, dict) and list(template) == list(candidate) and all(
                same_field_order(template[key], candidate[key]) for key in template)
        if isinstance(template, list):
            return isinstance(candidate, list) and (not template or all(
                same_field_order(template[0], item) for item in candidate))
        return True

    for row in sample:
        original_ids = row["token_ids"][:-1]
        tail = min(96, max(16, len(original_ids) // 5))
        prefix_ids = original_ids[:-tail]
        ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=tail + 32, do_sample=False,
                                       eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.eos_token_id,
                                       use_cache=True)
        text = tokenizer.decode(generated[0], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        record = {"source": row["source"], "segment": row["segment"],
                  "prompt_tokens": len(prefix_ids), "continuation_tokens": len(generated[0])-len(prefix_ids),
                  "json_parseable": False, "custom_extractor_valid": False, "correct_field_order": False}
        try:
            parsed = json.loads(text)
            record["json_parseable"] = True
            if set(parsed) != {"global", "tracks", "opaque_records"}:
                raise ValueError("canonical root keys differ")
            record["correct_field_order"] = same_field_order(json.loads(row["text"]), parsed)
            composition = Composition(global_settings=parsed["global"], tracks=parsed["tracks"])
            validate(composition)
            native = render_tweeter(composition)
            reconstructed = extract_text(native)
            if reconstructed.tracks != composition.tracks:
                raise ValueError("native reconstruction changed tracks")
            record["custom_extractor_valid"] = True
        except (ValueError, TypeError, KeyError) as error:
            record["error"] = str(error)[:250]
        result.append(record)
    return {"count": len(result), "json_parseable": sum(r["json_parseable"] for r in result),
            "custom_extractor_valid": sum(r["custom_extractor_valid"] for r in result),
            "correct_field_order": sum(r["correct_field_order"] for r in result),
            "samples": result}


def evaluate(dataset_dir, checkpoint_dir, output_path, baseline_path=None,
             cap_per_feature=6, cpu=False, seed=415):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"trained adapter missing: {checkpoint_dir}")
    model, device, dtype = load_model(adapter=checkpoint_dir, cpu=cpu)
    tokenizer = tokenizer_for_model(local_only=True)
    stats = train_statistics(dataset_dir)
    results = {"model": "SmolLM2-360M + LoRA", "checkpoint": str(checkpoint_dir),
               "device": str(device), "dtype": str(dtype),
               "feature_query_cap_per_parameter": cap_per_feature,
               "training_only_statistical_baselines": stats}
    for partition in ("validation", "test"):
        rows = load_prepared(dataset_dir, partition)
        queries = select_feature_queries(rows, cap_per_feature, seed)
        results[partition] = {"causal": score(model, rows, device, tokenizer),
                              "feature_values": feature_metrics(model, tokenizer, device, queries, stats),
                              "structure": _structure(model, tokenizer, device, rows, seed)}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return results


def evaluate_base_features(dataset_dir, output_path, cap_per_feature=6, cpu=False, seed=415, include_test=True):
    model, device, dtype = load_model(cpu=cpu)
    tokenizer = tokenizer_for_model(local_only=True)
    stats = train_statistics(dataset_dir)
    results = {"model": "original SmolLM2-360M", "device": str(device), "dtype": str(dtype)}
    for partition in (("validation", "test") if include_test else ("validation",)):
        rows = load_prepared(dataset_dir, partition)
        queries = select_feature_queries(rows, cap_per_feature, seed)
        results[partition] = {"causal": score(model, rows, device, tokenizer),
                              "feature_values": feature_metrics(model, tokenizer, device, queries, stats),
                              "structure": _structure(model, tokenizer, device, rows, seed)}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return results


def compare(base_path, tuned_path, history_path, output_path):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    tuned = json.loads(Path(tuned_path).read_text(encoding="utf-8"))
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    report = {"best_epoch": history.get("best_epoch"), "best_validation_loss": history.get("best_validation_loss"),
              "updates": history["epochs"][-1]["updates"] if history["epochs"] else 0,
              "train_validation_gap": (history["epochs"][-1]["validation"]["loss"] - history["epochs"][-1]["train_token_weighted_loss"]) if history["epochs"] else None,
              "partitions": {}}
    for partition in ("validation", "test"):
        original = base[partition]["causal"]["loss"]
        fine = tuned[partition]["causal"]["loss"]
        categories = {}
        for name, actual in tuned[partition]["feature_values"].items():
            earlier = base[partition]["feature_values"].get(name, {})
            categories[name] = {"n": actual["targets"],
                                "base_exact_accuracy": earlier.get("exact_accuracy"),
                                "fine_exact_accuracy": actual["exact_accuracy"],
                                "statistical_exact_accuracy": actual["baseline_exact_accuracy"],
                                "base_mae": earlier.get("mae"), "fine_mae": actual["mae"],
                                "statistical_mae": actual["baseline_mae"]}
        report["partitions"][partition] = {"original_loss": original, "fine_tuned_loss": fine,
                                            "loss_improvement_percent": 100*(original-fine)/original,
                                            "original_perplexity": base[partition]["causal"]["perplexity"],
                                            "fine_tuned_perplexity": tuned[partition]["causal"]["perplexity"],
                                            "original_structural_valid": base[partition]["structure"]["custom_extractor_valid"],
                                            "fine_structural_valid": tuned[partition]["structure"]["custom_extractor_valid"],
                                            "features": categories}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return report
