"""Freeze human feedback and calculate rating-weighted, completion-only DPO."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .plugin_store import FeedbackStore, atomic_bytes, canonical, digest, timestamp


def eligible_pairs(group, ratings, minimum_difference=2):
    if not 1 <= minimum_difference <= 9:
        raise ValueError("minimum rating difference must be 1..9")
    candidates = {c["candidate_id"]: c for c in group["candidates"]}
    rated = [(candidates[key], ratings[key]["rating"]) for key in sorted(ratings)]
    raw = []
    for index, (first, first_rating) in enumerate(rated):
        for second, second_rating in rated[index+1:]:
            difference = abs(first_rating - second_rating)
            if difference < minimum_difference:
                continue
            chosen, rejected = ((first, second) if first_rating > second_rating
                                else (second, first))
            raw.append({"group_id": group["group_id"], "source": group["source"],
                        "source_family": group["source_family"],
                        "chosen_id": chosen["candidate_id"], "rejected_id": rejected["candidate_id"],
                        "chosen_rating": max(first_rating, second_rating),
                        "rejected_rating": min(first_rating, second_rating),
                        "rating_difference": difference,
                        "prompt_ids": group["prompt_ids"],
                        "chosen_ids": chosen["completion_ids"],
                        "rejected_ids": rejected["completion_ids"]})
    total = sum(pair["rating_difference"] for pair in raw)
    for pair in raw:
        pair["group_weight"] = pair["rating_difference"] / total
    return raw


def make_snapshot(store, run_dir, manifest, reference_adapter, minimum_difference=2,
                  validation_fraction=.2, seed=415):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError("DPO run directory already exists")
    if not 0 < validation_fraction < .5:
        raise ValueError("validation fraction must be between 0 and .5")
    reference_adapter = Path(reference_adapter)
    reference_hash = digest((reference_adapter / "adapter_model.safetensors").read_bytes())
    embedding_hash = digest((reference_adapter / "embedding.safetensors").read_bytes())
    groups = list(store.all_groups())
    pairs, event_ids = [], []
    for group in groups:
        for key in ("tokenizer_sha256", "dataset_sha256", "codec_schema_sha256", "fixed_preset_sha256"):
            manifest_key = "prepared_dataset_sha256" if key == "dataset_sha256" else key
            if group[key] != manifest[manifest_key]:
                raise ValueError(f"feedback group {group['group_id']} has incompatible {key}")
        current = store.current_ratings(group["group_id"])
        event_ids.extend(event["submission_id"] for event in current.values())
        pairs.extend(eligible_pairs(group, current, minimum_difference))
    if not pairs:
        raise ValueError("no eligible rated preference pairs")
    families = sorted({pair["source_family"] for pair in pairs},
                      key=lambda name: digest([seed, name]))
    validation_count = max(1, round(len(families) * validation_fraction)) if len(families) >= 2 else 0
    validation_families = set(families[:validation_count])
    partitions = {"train": [], "validation": []}
    for pair in pairs:
        split = "validation" if pair["source_family"] in validation_families else "train"
        partitions[split].append(pair)
    if not partitions["train"]:
        raise ValueError("no training preference pairs after family partition")
    snapshot = {"format": "TweeterDPO-snapshot-v1", "created_at": timestamp(),
                "minimum_rating_difference": minimum_difference,
                "validation_fraction": validation_fraction, "partition_seed": seed,
                "reference_adapter": str(reference_adapter.resolve()),
                "reference_adapter_sha256": reference_hash,
                "embedding_sha256": embedding_hash,
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "dataset_sha256": manifest["prepared_dataset_sha256"],
                "codec_schema_sha256": manifest["codec_schema_sha256"],
                "fixed_preset_sha256": manifest["fixed_preset_sha256"],
                "rating_submission_ids": sorted(event_ids), "groups": len(groups),
                "raw_rating_event_count": sum(1 for _ in store.events()),
                "validation_families": sorted(validation_families),
                "train_families": sorted(set(families) - validation_families),
                "pairs": partitions}
    raw = canonical(snapshot) + b"\n"
    run_dir.mkdir(parents=True)
    atomic_bytes(run_dir / "snapshot.json", raw)
    atomic_bytes(run_dir / "snapshot.sha256", digest(raw).encode("ascii") + b"\n")
    return {"run_dir": str(run_dir), "snapshot_sha256": digest(raw),
            "train_pairs": len(partitions["train"]),
            "validation_pairs": len(partitions["validation"]),
            "rated_groups": len({pair["group_id"] for pair in pairs})}


def load_snapshot(run_dir):
    root = Path(run_dir)
    raw = (root / "snapshot.json").read_bytes()
    if digest(raw) != (root / "snapshot.sha256").read_text(encoding="ascii").strip():
        raise ValueError("frozen preference snapshot changed")
    data = json.loads(raw)
    if data.get("format") != "TweeterDPO-snapshot-v1":
        raise ValueError("unsupported preference snapshot")
    if set(data["train_families"]) & set(data["validation_families"]):
        raise ValueError("source family crosses DPO partitions")
    return data, digest(raw)


def completion_logprob(model, prompt_ids, completion_ids, device):
    """Sum unmasked causal log probabilities of completion, including EOS."""
    import torch
    import torch.nn.functional as F
    if not prompt_ids or not completion_ids or len(prompt_ids) + len(completion_ids) > 8192:
        raise ValueError("invalid DPO sequence")
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = output.logits[:, len(prompt_ids)-1:-1].float()
    labels = ids[:, len(prompt_ids):]
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).sum()


def reference_key(snapshot, prompt_ids, completion_ids):
    return digest(["unmasked-causal-summed-v1", snapshot["reference_adapter_sha256"],
                   snapshot["tokenizer_sha256"], prompt_ids, completion_ids])


def precompute_reference(run_dir, output_dir, cpu=False, model_loader=None):
    import torch
    from .model import load_model
    snapshot, snapshot_hash = load_snapshot(run_dir)
    loader = model_loader or load_model
    cache = Path(run_dir).parent / ".reference_scores"
    cache.mkdir(exist_ok=True)
    unique = {}
    for split in ("train", "validation"):
        for pair in snapshot["pairs"][split]:
            for suffix in ("chosen", "rejected"):
                ids = pair[suffix + "_ids"]
                key = reference_key(snapshot, pair["prompt_ids"], ids)
                unique[key] = (pair["prompt_ids"], ids)
    missing = {}
    for key, value in unique.items():
        file = cache / f"{key}.json"
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            if (record["key"] != key or
                    record["reference_adapter_sha256"] != snapshot["reference_adapter_sha256"] or
                    record["integrity_sha256"] != digest({k: v for k, v in record.items() if k != "integrity_sha256"}) or
                    not math.isfinite(record["log_probability"])):
                raise ValueError("invalid cached reference score")
        except (OSError, ValueError, KeyError, TypeError):
            file.unlink(missing_ok=True)  # Derived cache only; raw feedback remains immutable.
            missing[key] = value
    if missing:
        model, device = loader(output_dir, adapter=snapshot["reference_adapter"],
                               cpu=cpu, trainable=False)
        model.eval()
        with torch.inference_mode():
            for key, (prompt_ids, completion_ids) in missing.items():
                score = float(completion_logprob(model, prompt_ids, completion_ids, device))
                record = {"key": key, "log_probability": score,
                          "snapshot_sha256": snapshot_hash,
                          "reference_adapter_sha256": snapshot["reference_adapter_sha256"]}
                record["integrity_sha256"] = digest(record)
                atomic_bytes(cache / f"{key}.json", canonical(record) + b"\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"unique_candidates": len(unique), "new_scores": len(missing),
            "cache_dir": str(cache), "snapshot_sha256": snapshot_hash}


def cached_reference(run_dir, snapshot, pair, suffix):
    key = reference_key(snapshot, pair["prompt_ids"], pair[suffix + "_ids"])
    file = Path(run_dir).parent / ".reference_scores" / f"{key}.json"
    record = json.loads(file.read_text(encoding="utf-8"))
    if (record["key"] != key or
            record["reference_adapter_sha256"] != snapshot["reference_adapter_sha256"] or
            record["integrity_sha256"] != digest({k: v for k, v in record.items() if k != "integrity_sha256"}) or
            not math.isfinite(record["log_probability"])):
        raise ValueError("reference score cache identity mismatch")
    return record["log_probability"]


def weighted_dpo_loss(policy_chosen, policy_rejected, reference_chosen,
                      reference_rejected, beta, weight):
    import torch.nn.functional as F
    margin = beta * ((policy_chosen - policy_rejected) -
                     (reference_chosen - reference_rejected))
    return -weight * F.logsigmoid(margin)
