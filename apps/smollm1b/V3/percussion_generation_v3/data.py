"""Retokenize the retained, source-verified V3 rhythm examples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SMOLLM = ROOT.parent
APPS = SMOLLM.parent
for sibling in (APPS / "CustomExtractor", SMOLLM / "CustomTraining"):
    path = str(sibling)
    if path not in sys.path:
        sys.path.insert(0, path)

import custom_training
if Path(custom_training.__file__).resolve() != (SMOLLM / "CustomTraining/custom_training/__init__.py").resolve():
    raise RuntimeError("another model's CustomTraining package is already loaded; use a separate Python process")

from custom_extractor import extract_tweeter

DEFAULT_SESSIONS = APPS / "L2Ork-Tweeter" / "sessions"
AUGMENTED_SESSIONS = APPS / "L2Ork-Tweeter" / "AugmentedSessions"
V1_OUTPUT = APPS / "PercussionGeneration" / "output"
V2_OUTPUT = APPS / "PercussionGenerationV2" / "output"
FEATURE_ADAPTER = (ROOT / "initial_feature_adapter" if SMOLLM.name == "smollm360"
                   else SMOLLM / "CustomTraining/training_output/checkpoints/best_adapter")
REFERENCE_OUTPUT = APPS / "smollm360/V3/output"
V2_MARKER = "\nPERCUSSION_TARGET="
V3_MARKER = "\nRHYTHM_TARGET="
END_MARKER = "[END_RHYTHM]"


def load_tokenizer(output_dir):
    from transformers import AutoTokenizer
    from custom_training import MODEL_ID, MODEL_REVISION
    root = Path(output_dir)
    identity = root / "tokenizer_identity.json"
    if identity.is_file():
        recorded = json.loads(identity.read_text(encoding="utf-8"))
        if (recorded.get("model_id"), recorded.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("V3 tokenizer belongs to another base model")
        actual_sha = hashlib.sha256((root / "tokenizer/tokenizer.json").read_bytes()).hexdigest()
        if actual_sha != recorded["sha256"]:
            # The checked-in original output/tokenizer.json is formatted
            # differently from its recorded identity, although its parsed
            # tokenizer JSON equals the later model-local copy. Preserve old
            # evaluation without rewriting any historical artifact.
            known_historical = (root.resolve() == (ROOT / "output").resolve()
                and actual_sha == "3930b6a89f1b21588f999a1e58ed4b18d11b31d77f43d7ad76dd81d22ef2825c"
                and recorded["sha256"] == "7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7")
            if not known_historical:
                raise ValueError("V3 tokenizer bytes changed")
    elif SMOLLM.name != "smollm360":
        raise FileNotFoundError("model-specific V3 tokenizer identity missing; run prepare")
    return AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)


def verify_adapter_codec(output_dir, adapter):
    """Reject cross-format inference while retaining historical legacy adapters."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata_path = Path(adapter) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    encoding = manifest.get("encoding", "compact")
    if encoding == "lossless-v2":
        from .codec_v2 import SCHEMA_SHA256
        if manifest.get("codec_schema_sha256") != SCHEMA_SHA256:
            raise ValueError("prepared V2 schema differs from installed codec")
        verify_v2_dataset(root, manifest)
        if (metadata.get("codec_version") != encoding or
                metadata.get("codec_schema_sha256") != SCHEMA_SHA256 or
                metadata.get("tokenizer_sha256") != manifest.get("tokenizer_sha256") or
                metadata.get("prepared_dataset_sha256") != manifest.get("prepared_dataset_sha256")):
            raise ValueError("adapter is not validated for this V2 codec, tokenizer, and dataset")
    elif metadata.get("codec_version") == "lossless-v2":
        raise ValueError("V2 adapter cannot be used with legacy inputs")
    return manifest


def verify_v2_dataset(output_dir, manifest=None):
    root = Path(output_dir)
    if manifest is None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("encoding") != "lossless-v2":
        return
    from .codec_v2 import SCHEMA_SHA256
    from custom_training import MODEL_ID, MODEL_REVISION
    if (manifest.get("model_id"), manifest.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("V2 model identity differs from pinned base model")
    if manifest.get("codec_version") != "lossless-v2":
        raise ValueError("V2 codec version differs from dataset manifest")
    if manifest.get("codec_schema_sha256") != SCHEMA_SHA256:
        raise ValueError("V2 codec schema fingerprint changed")
    if manifest.get("envelope_mode") != "zlib":
        raise ValueError("V2 envelope mode differs from the installed prompt encoder")
    if hashlib.sha256((root / "tokenizer/tokenizer.json").read_bytes()).hexdigest() != manifest.get("tokenizer_sha256"):
        raise ValueError("V2 tokenizer fingerprint differs from dataset manifest")
    digest = hashlib.sha256()
    for split in ("train", "validation", "test", "validation20"):
        path = root / "rhythm" / f"{split}.jsonl"
        if path.is_file():
            digest.update(split.encode() + b"\0" + path.read_bytes())
    if digest.hexdigest() != manifest.get("prepared_dataset_sha256"):
        raise ValueError("V2 prepared dataset fingerprint changed")


def v3_prefix(v2_prefix, count):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("target note count must be an integer from 1 to 64")
    expected = V2_MARKER + str(count) + "\n"
    if not v2_prefix.endswith(expected):
        raise ValueError("V2 prefix/count mismatch")
    return v2_prefix[:-len(expected)] + V3_MARKER + str(count) + "\n"


def recount_prefix(v3_input, old_count, new_count):
    if type(new_count) is not int or not 1 <= new_count <= 64:
        raise ValueError("target note count must be 1..64")
    suffix = V3_MARKER + str(old_count) + "\n"
    if not v3_input.endswith(suffix):
        raise ValueError("V3 prefix/count mismatch")
    return v3_input[:-len(suffix)] + V3_MARKER + str(new_count) + "\n"


def serialize_positions(positions):
    if not positions or any(type(p) is not int or not 0 <= p < 64 for p in positions):
        raise ValueError("positions must be nonempty tracker steps 0..63")
    if positions != sorted(positions):
        raise ValueError("positions must be chronological")
    # Preserve duplicates in the text. Native one-track export rejects them later.
    return " ".join(map(str, positions)) + "\n" + END_MARKER


def parse_positions(text, requested_count=None, allow_duplicates=False,
                    allow_unordered=False):
    if not isinstance(text, str) or not text.endswith("\n" + END_MARKER):
        raise ValueError("rhythm must end with newline and [END_RHYTHM]")
    line = text[:-(len(END_MARKER) + 1)]
    parts = line.split(" ")
    if not parts or any(not part or not part.isascii() or not part.isdecimal() or
                        (len(part) > 1 and part.startswith("0")) for part in parts):
        raise ValueError("rhythm must contain space-separated decimal positions")
    positions = [int(part) for part in parts]
    if any(not 0 <= p < 64 for p in positions):
        raise ValueError("positions out of range")
    if not allow_unordered and positions != sorted(positions):
        raise ValueError("positions out of order")
    if not allow_duplicates and len(set(positions)) != len(positions):
        raise ValueError("native Tweeter permits one pitch per step in one track")
    if requested_count is not None and len(positions) != requested_count:
        raise ValueError(f"requested {requested_count} positions; generated {len(positions)}")
    return positions


def load_rows(output_dir, mode, partition):
    if mode != "rhythm" or partition not in ("train", "validation", "test"):
        raise ValueError("V3 only supports rhythm and train/validation/test")
    path = Path(output_dir) / mode / f"{partition}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def augmented_training_rows(reference_rows, tokenizer, max_length, progress_seconds=180):
    """Add exact time shifts of training parents without crossing held-out splits."""
    manifest_path = AUGMENTED_SESSIONS / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    augmentation = json.loads(manifest_bytes)
    if augmentation.get("validation_failures") or augmentation.get("tracker_steps") != 64:
        raise ValueError("augmented session manifest is invalid")
    parents = {}
    for row in reference_rows:
        parents.setdefault(row["source"], []).append(row)
    generated_root = (AUGMENTED_SESSIONS / "generated").resolve()
    from .evaluation import remove_target
    from .inference import prompt
    added = []
    skipped_length = 0
    seen = set()
    candidates = [entry for entry in augmentation["generated"]
                  if entry["original_session"] in parents and entry["reference_offset"] != 0]
    started = last_report = time.monotonic()
    print(f"{SMOLLM.name} V3 preparation: checking {len(candidates)} shifted training files",
          flush=True)
    for index, entry in enumerate(candidates, 1):
        source = entry["original_session"]
        relative = Path(entry["augmented_filename"])
        path = (generated_root / relative).resolve()
        if not path.is_relative_to(generated_root) or path in seen:
            raise ValueError(f"invalid augmented path: {relative}")
        seen.add(path)
        if entry.get("validation") != "passed" or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"augmented session failed verification: {relative}")
        shifted = extract_tweeter(path)
        shifts = {item["track_id"]: item["tracker_shift"] for item in entry["tracks"]}
        for original in parents[source]:
            track_id = original["target_track_id"]
            target_track = shifted.track(track_id)
            positions = [note["step"] for note in target_track["notes"]]
            expected = sorted((step - shifts[track_id]) % 64 for step in original["target_positions"])
            if positions != expected or len(positions) != original["target_note_count"]:
                raise ValueError(f"augmented target differs from manifest: {relative}, track {track_id}")
            target = serialize_positions(positions)
            prefix = prompt(remove_target(shifted, track_id), len(positions), "legacy-v1")
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            tokens = len(prefix_ids) + len(target_ids)
            if tokens > max_length:
                skipped_length += 1
                continue
            row = dict(original)
            row.update(prefix=prefix, target=target, target_positions=positions,
                       prefix_ids=prefix_ids, target_ids=target_ids, tokens=tokens,
                       augmentation={"file": relative.as_posix(),
                                     "sha256": entry["sha256"],
                                     "reference_offset": entry["reference_offset"]})
            added.append(row)
        now = time.monotonic()
        if progress_seconds and now - last_report >= progress_seconds:
            print(f"{SMOLLM.name} V3 preparation: {index}/{len(candidates)} files checked; "
                  f"{len(added)} pairs added, {skipped_length} over context; "
                  f"elapsed {(now - started) / 60:.1f} min", flush=True)
            last_report = now
    print(f"{SMOLLM.name} V3 preparation: finished {len(candidates)} files; "
          f"{len(added)} pairs added, {skipped_length} over context; "
          f"elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return added, {"added_pairs": len(added), "skipped_over_context": skipped_length,
                   "augmented_files_checked": len(seen),
                   "augmented_files_included": len({row["augmentation"]["file"] for row in added}),
                   "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "offset_zero_excluded": True,
                   "parent_split": "train only"}


def prepare(output_dir=ROOT / "output", max_length=8192):
    output_dir = Path(output_dir)
    if max_length != 8192:
        raise ValueError("V3 uses an 8192-token context")
    if SMOLLM.name == "smollm360" and output_dir.resolve() == REFERENCE_OUTPUT.resolve():
        # The 360M artifacts are historical measurements. Never rewrite them.
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if output_dir.exists() and any((output_dir / p).exists() for p in
                                   ("checkpoints", "evaluation", "generated", "rhythm")):
        raise FileExistsError("V3 output already has data or results; prepare in a fresh output directory")
    from custom_training import MODEL_ID, MODEL_REVISION
    from custom_training.prepare import tokenizer_for_model
    tokenizer = tokenizer_for_model()
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS")
    cache = {}
    summaries = {}
    augmentation_summary = None
    groups = {}
    rows_by_split = {}
    for partition in ("train", "validation", "test"):
        reference_rows = load_rows(REFERENCE_OUTPUT, "rhythm", partition)
        converted = []
        for old in reference_rows:
            source = old["source"]
            if source not in cache:
                path = DEFAULT_SESSIONS / source
                if hashlib.sha256(path.read_bytes()).hexdigest() != old["source_sha256"]:
                    raise ValueError(f"source changed: {source}")
                cache[source] = extract_tweeter(path)
            source_track = cache[source].track(old["target_track_id"])
            positions = [note["step"] for note in source_track["notes"]]
            if positions != old["target_positions"] or serialize_positions(positions) != old["target"]:
                raise ValueError(f"retained V3 target differs from source: {source}")
            if len(set(positions)) != len(positions):
                raise ValueError("a native one-track target has duplicate steps")
            target = old["target"]
            if parse_positions(target, len(positions)) != positions:
                raise ValueError("position sequence did not roundtrip")
            prefix = old["prefix"]
            if not prefix.endswith(V3_MARKER + str(len(positions)) + "\n"):
                raise ValueError(f"retained V3 conditioning differs from count: {source}")
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            row = dict(old)
            row.update(prefix_ids=prefix_ids, target_ids=target_ids,
                       tokens=len(prefix_ids) + len(target_ids))
            if row["tokens"] > max_length:
                raise ValueError(f"V3 row exceeds context: {source} {row['tokens']}")
            converted.append(row)
        rows_by_split[partition] = converted
        if partition == "train" and SMOLLM.name != "smollm1b":
            extra, augmentation_summary = augmented_training_rows(converted, tokenizer, max_length)
            converted.extend(extra)
        groups[partition] = {row["source"] for row in converted}
        counts = [row["target_note_count"] for row in converted]
        lengths = [row["tokens"] for row in converted]
        summaries[partition] = {"pairs": len(converted), "sessions": len(groups[partition]),
                                "note_count_histogram": dict(sorted(Counter(counts).items())),
                                "note_count_min": min(counts, default=None),
                                "note_count_median": statistics.median(counts) if counts else None,
                                "note_count_max": max(counts, default=None),
                                "token_length_max": max(lengths, default=None)}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"),
                                             ("validation", "test"))):
        raise ValueError("source-session groups cross splits")
    manifest = {"source_dataset": "apps/smollm360/V3/output/rhythm",
                "starting_adapter": "apps/" + FEATURE_ADAPTER.relative_to(APPS).as_posix(),
                "max_length": max_length,
                "tracker_steps": 64, "mode": "rhythm", "encoding": "compact",
                "target": "positions only, then [END_RHYTHM]",
                "splits": summaries, "v2_split_reused": True,
                "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    if augmentation_summary is not None:
        manifest["augmentation"] = augmentation_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    from shutil import copyfile
    copyfile(ROOT / "output/fixed_preset.json", output_dir / "fixed_preset.json")
    manifest["tokenizer_sha256"] = hashlib.sha256(
        (output_dir / "tokenizer/tokenizer.json").read_bytes()).hexdigest()
    (output_dir / "tokenizer_identity.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "revision": MODEL_REVISION,
         "sha256": manifest["tokenizer_sha256"]}, indent=2) + "\n", encoding="utf-8")
    for partition, rows in rows_by_split.items():
        destination = output_dir / "rhythm" / f"{partition}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_epoch7_experiment(output_dir, source_output=ROOT / "output_augmented"):
    """Copy the verified augmented pool and add ten fixed held-out shifts."""
    from .evaluation import remove_target
    from .inference import prompt
    from .rotation import validate_pool
    from custom_training import MODEL_ID, MODEL_REVISION

    output_dir, source_output = Path(output_dir), Path(source_output)
    if output_dir.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    if source_output.resolve() == output_dir.resolve():
        raise ValueError("source and experiment output must differ")
    source_manifest = json.loads((source_output / "manifest.json").read_text(encoding="utf-8"))
    if (source_manifest.get("model_id"), source_manifest.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("prepared pool is for another model")
    augmentation_bytes = (AUGMENTED_SESSIONS / "manifest.json").read_bytes()
    if source_manifest["augmentation"]["manifest_sha256"] != hashlib.sha256(augmentation_bytes).hexdigest():
        raise ValueError("augmentation manifest changed since the pool was prepared")
    augmentation = json.loads(augmentation_bytes)
    generated = {entry["augmented_filename"]: entry for entry in augmentation["generated"]}
    if len(generated) != len(augmentation["generated"]):
        raise ValueError("duplicate augmented filenames in manifest")
    original_validation = load_rows(source_output, "rhythm", "validation")
    original_test = load_rows(source_output, "rhythm", "test")
    training = load_rows(source_output, "rhythm", "train")
    if len(original_validation) != 10 or len(original_test) != 5:
        raise ValueError("expected the fixed original 10 validation and 5 test pairs")
    source_sets = {"train": {r["source"] for r in training},
                   "validation": {r["source"] for r in original_validation},
                   "test": {r["source"] for r in original_test}}
    if any(source_sets[a] & source_sets[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("original sessions cross V3 splits")
    source_partitions = {source: partition for partition, members in source_sets.items()
                         for source in members}
    lookup = {"source_partitions": source_partitions, "generated_lookup": generated}
    validate_pool(training, lookup, expected_originals=67)
    if any("augmentation" in row for row in original_validation + original_test):
        raise ValueError("held-out original rows are already augmented")
    family_sets = {partition: {_experiment_family(source) for source in members}
                   for partition, members in source_sets.items()}
    inherited_family_overlap = sorted(family_sets["train"] & family_sets["validation"])
    if family_sets["train"] & family_sets["test"] or family_sets["validation"] & family_sets["test"]:
        raise ValueError("unexpected family overlap with test partition")

    tokenizer = load_tokenizer(source_output)
    by_source = {}
    for entry in augmentation["generated"]:
        if entry["original_session"] in source_sets["validation"] and entry["reference_offset"]:
            by_source.setdefault(entry["original_session"], []).append(entry)
    shifted_cache = {}
    augmented_validation = []
    used_files = set()
    for original in original_validation:
        candidates = sorted(by_source.get(original["source"], []),
                            key=lambda e: (e["augmented_filename"] in used_files,
                                           e["reference_offset"]))
        chosen = None
        for entry in candidates:
            relative = Path(entry["augmented_filename"])
            path = (AUGMENTED_SESSIONS / "generated" / relative).resolve()
            if not path.is_relative_to((AUGMENTED_SESSIONS / "generated").resolve()):
                raise ValueError(f"invalid augmented path: {relative}")
            if entry.get("validation") != "passed" or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"invalid validation shift: {relative}")
            if str(path) not in shifted_cache:
                shifted_cache[str(path)] = extract_tweeter(path)
            shifted = shifted_cache[str(path)]
            track_id = original["target_track_id"]
            positions = [note["step"] for note in shifted.track(track_id)["notes"]]
            shifts = {item["track_id"]: item["tracker_shift"] for item in entry["tracks"]}
            expected = sorted((step - shifts[track_id]) % 64 for step in original["target_positions"])
            if positions != expected:
                raise ValueError(f"validation shift target differs from parent: {relative}")
            target = serialize_positions(positions)
            prefix = prompt(remove_target(shifted, track_id), len(positions), "legacy-v1")
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            tokens = len(prefix_ids) + len(target_ids)
            if tokens > source_manifest["max_length"]:
                continue
            chosen = dict(original)
            chosen.update(prefix=prefix, target=target, target_positions=positions,
                          prefix_ids=prefix_ids, target_ids=target_ids, tokens=tokens,
                          augmentation={"file": relative.as_posix(), "sha256": entry["sha256"],
                                        "reference_offset": entry["reference_offset"]})
            used_files.add(entry["augmented_filename"])
            break
        if chosen is None:
            raise ValueError(f"no shifted validation pair fits context: {original['source']} track {original['target_track_id']}")
        augmented_validation.append(chosen)
    fixed_validation = original_validation + augmented_validation
    if len(fixed_validation) != 20 or len({(r["source"], r["target_track_id"],
                                            r.get("augmentation", {}).get("file"))
                                           for r in fixed_validation}) != 20:
        raise AssertionError("fixed validation must have 20 distinct pairs")
    report = {"train_original_pairs": 67, "train_augmented_pool": len(training) - 67,
              "validation_original_pairs": 10, "validation_fixed_pairs": 20,
              "test_pairs": 5, "source_partitions": source_partitions,
              "inherited_train_validation_family_overlap": inherited_family_overlap,
              "new_validation_sources": sorted({r["source"] for r in augmented_validation}),
              "augmentation_manifest_sha256": hashlib.sha256(augmentation_bytes).hexdigest()}
    temp = Path(tempfile.mkdtemp(prefix=".preparing_epoch7_", dir=output_dir.parent))
    try:
        shutil.copytree(source_output / "rhythm", temp / "rhythm")
        shutil.copytree(source_output / "tokenizer", temp / "tokenizer")
        for name in ("tokenizer_identity.json", "fixed_preset.json"):
            shutil.copy2(source_output / name, temp / name)
        with (temp / "rhythm/validation20.jsonl").open("w", encoding="utf-8") as stream:
            for row in fixed_validation:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        manifest = dict(source_manifest)
        manifest["epoch7_experiment"] = report
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp.rename(output_dir)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return report


def validate_epoch7_splits(output_dir):
    """Check parent assignments and report the accepted historical family overlap."""
    from .rotation import validate_pool
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["epoch7_experiment"]
    train_rows = load_rows(root, "rhythm", "train")
    val10 = load_rows(root, "rhythm", "validation")
    test = load_rows(root, "rhythm", "test")
    val20 = [json.loads(line) for line in (root / "rhythm/validation20.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(val10) != 10 or len(val20) != 20 or len(test) != 5 or val20[:10] != val10:
        raise ValueError("fixed validation or original test count changed")
    augmentation_path = AUGMENTED_SESSIONS / "manifest.json"
    if hashlib.sha256(augmentation_path.read_bytes()).hexdigest() != experiment["augmentation_manifest_sha256"]:
        raise ValueError("augmentation manifest changed")
    generated = {entry["augmented_filename"]: entry
                 for entry in json.loads(augmentation_path.read_text(encoding="utf-8"))["generated"]}
    source_sets = {"train": {r["source"] for r in train_rows},
                   "validation": {r["source"] for r in val20},
                   "test": {r["source"] for r in test}}
    if any(source_sets[a] & source_sets[b] for a,b in
           (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("source session crosses data partitions")
    lookup = {"source_partitions": experiment["source_partitions"],
              "generated_lookup": generated}
    validate_pool(train_rows, lookup, expected_originals=67)
    for row in val20[10:]:
        shift = row.get("augmentation", {})
        entry = generated.get(shift.get("file"))
        if (not entry or entry["original_session"] != row["source"]
                or entry["reference_offset"] != shift["reference_offset"]
                or entry["sha256"] != shift["sha256"]
                or experiment["source_partitions"].get(row["source"]) != "validation"):
            raise ValueError("validation shift has wrong parent or manifest entry")
    families = {name: {_experiment_family(source) for source in members}
                for name, members in source_sets.items()}
    overlap = sorted(families["train"] & families["validation"])
    if overlap != experiment["inherited_train_validation_family_overlap"]:
        raise ValueError("train/validation family overlap changed from historical V3")
    if families["train"] & families["test"] or families["validation"] & families["test"]:
        raise ValueError("test family overlaps another partition")
    return {"train_pairs": len(train_rows), "train_original_pairs": 67,
            "validation_fixed_pairs": len(val20), "validation_original_pairs": len(val10),
            "test_pairs": len(test), "train_sources": len(source_sets["train"]),
            "validation_sources": len(source_sets["validation"]),
            "test_sources": len(source_sets["test"]),
            "inherited_family_overlap": overlap}


def _experiment_family(source):
    from custom_training.data_loader import family_name
    family = family_name(source)
    # The bare "3rd-beat-session-coll.txt" misses the upstream prefix rule.
    return "beat-sequence" if family == "3rd-beat" else family
