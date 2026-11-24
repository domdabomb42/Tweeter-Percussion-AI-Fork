"""Matched stochastic evaluation; human win rates require separate human votes."""
import json
from pathlib import Path
import statistics
import time

from . import OUTPUT
from .model import load_model
from .pipeline import load_new_tokenizer, verify_dataset
from .plugin_generation import decode_sample, prepare_feedback_pair, render_candidate
from .trainer import validation_loss
from percussion_generation_v3.data import AUGMENTED_SESSIONS


def evaluate_policies(output_dir=OUTPUT, adapter=None, partition="validation20",
                      cpu=False, seed=415, temperature=.8, top_p=.95):
    if partition not in ("validation20", "validation", "test"):
        raise ValueError("evaluation partition must be held out")
    output_dir = Path(output_dir)
    manifest, groups = verify_dataset(output_dir)
    tokenizer, _ = load_new_tokenizer(output_dir)
    preset = json.loads((output_dir / "fixed_preset.json").read_text(encoding="utf-8"))
    augment = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
    generated = {item["augmented_filename"]: item for item in augment["generated"]}
    rows = groups[partition]
    prepared = [prepare_feedback_pair(row, generated, tokenizer) for row in rows]
    supervised = output_dir / "checkpoints/rhythm_coverage_v1/best_adapter"
    adapters = {"supervised": supervised, "dpo": Path(adapter)}
    result = {"partition": partition, "examples": len(rows), "seed": seed,
              "temperature": temperature, "top_p": top_p, "models": {},
              "human_preference_win_rate": None,
              "human_preference_note": "Requires actual paired held-out human ratings; not inferred from reference-track overlap."}
    for label, path in adapters.items():
        model, device = load_model(output_dir, adapter=path, cpu=cpu, trainable=False)
        model.eval()
        cross_entropy = validation_loss(model, rows, device)["loss"]
        durations, rhythms, jaccards, failures = [], set(), [], []
        for index, (_, original, condition, removed, prompt, ids) in enumerate(prepared):
            try:
                positions, completion, seconds = decode_sample(
                    model, tokenizer, device, ids, len(removed["notes"]), seed+index,
                    temperature, top_p)
                track, native = render_candidate(condition, original, removed, positions, preset)
                durations.append(seconds)
                rhythms.add(tuple(positions))
                reference = {note["step"] for note in removed["notes"]}
                actual = set(positions)
                jaccards.append(len(actual & reference) / len(actual | reference))
            except (ValueError, RuntimeError, OSError) as exc:
                failures.append({"index": index, "source": rows[index]["source"], "error": str(exc)})
        result["models"][label] = {"adapter": str(path),
            "token_cross_entropy": cross_entropy,
            "structurally_valid_rate": len(durations) / len(rows),
            "exact_note_count_rate": len(durations) / len(rows),
            "native_export_success_rate": len(durations) / len(rows),
            "rhythmic_diversity": len(rhythms) / len(durations) if durations else None,
            "rhythmic_jaccard_to_original": statistics.mean(jaccards) if jaccards else None,
            "mean_inference_seconds": statistics.mean(durations) if durations else None,
            "median_inference_seconds": statistics.median(durations) if durations else None,
            "failures": failures}
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return result
