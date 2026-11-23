"""Deterministic, balanced rotation through the prepared V3 rhythm pairs."""
from collections import defaultdict
import hashlib
import json
import random


def pair_key(row):
    return (row["source"], row["target_track_id"])


def example_id(row):
    augmentation = row.get("augmentation")
    return (row["source"], row["target_track_id"],
            augmentation["file"] if augmentation else None)


def validate_pool(rows, manifest, expected_originals=None):
    """Reject duplicate rows, foreign shifts, and children without a parent."""
    originals = {}
    augmented = defaultdict(list)
    seen = set()
    source_partitions = manifest.get("source_partitions", {})
    generated = manifest.get("generated_lookup", {})
    for index, row in enumerate(rows):
        identity = example_id(row)
        if identity in seen:
            raise ValueError(f"duplicate rhythm example: {identity}")
        seen.add(identity)
        key = pair_key(row)
        if source_partitions and source_partitions.get(row["source"]) != "train":
            raise ValueError(f"training example has held-out parent: {identity}")
        shift = row.get("augmentation")
        if shift:
            record = generated.get(shift["file"])
            if generated and (not record or record["original_session"] != row["source"]
                    or record["reference_offset"] != shift["reference_offset"]
                    or record["sha256"] != shift["sha256"]
                    or record["validation"] != "passed"
                    or shift["reference_offset"] == 0):
                raise ValueError(f"shift does not match augmentation manifest: {identity}")
            if shift["reference_offset"] == 0:
                raise ValueError("offset-zero duplicate is not a training shift")
            augmented[key].append(index)
        else:
            if key in originals:
                raise ValueError(f"multiple original rows for {key}")
            originals[key] = index
    if expected_originals is not None and len(originals) != expected_originals:
        raise ValueError(f"expected {expected_originals} originals, found {len(originals)}")
    if set(augmented) - set(originals):
        raise ValueError("shifted rhythm has no original training pair")
    return originals, augmented


def _group_seed(seed, key):
    payload = json.dumps([seed, key[0], key[1]], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_epoch_indices(rows, epoch_index, seed=415, examples_per_epoch=268):
    """Select originals plus rotating, distinct shifts; epoch_index is zero based."""
    if epoch_index < 0:
        raise ValueError("epoch_index must be nonnegative")
    originals, groups = validate_pool(rows, {})
    keys = sorted(originals)
    augmented_count = examples_per_epoch - len(keys)
    if augmented_count < 0 or augmented_count > sum(len(v) for v in groups.values()):
        raise ValueError("requested epoch size cannot be filled with distinct shifts")
    pools = {}
    for key in keys:
        pool = sorted(groups[key], key=lambda i: rows[i]["augmentation"]["file"])
        random.Random(_group_seed(seed, key)).shuffle(pool)
        pools[key] = pool
    cursor = {key: 0 for key in keys}
    order = None
    for current_epoch in range(epoch_index + 1):
        selected = [originals[key] for key in keys]
        used = {key: 0 for key in keys}
        remaining = augmented_count

        def take(key):
            nonlocal remaining
            pool = pools[key]
            if not pool or used[key] >= len(pool) or remaining == 0:
                return False
            selected.append(pool[cursor[key] % len(pool)])
            cursor[key] += 1
            used[key] += 1
            remaining -= 1
            return True

        for key in keys:
            for _ in range(min(3, len(pools[key]))):
                take(key)
        extra_keys = [key for key in keys if used[key] < len(pools[key])]
        random.Random(seed + current_epoch).shuffle(extra_keys)
        while remaining:
            progressed = False
            for key in extra_keys:
                progressed |= take(key)
                if remaining == 0:
                    break
            if not progressed:
                raise ValueError("not enough distinct shifts for the epoch")
        random.Random(seed * 1000003 + current_epoch).shuffle(selected)
        if len(selected) != examples_per_epoch or len(set(selected)) != len(selected):
            raise AssertionError("epoch selection is not distinct and complete")
        order = selected
    return order
