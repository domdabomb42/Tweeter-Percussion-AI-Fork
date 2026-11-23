"""Train a fresh BPE vocabulary and prepare musical V4EA targets."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from . import END_MARKER, MODEL_ID, MODEL_REVISION, MODEL_ROOT, OUTPUT, TARGET_MARKER, VERSION
from percussion_generation_v3.codec_v2 import dumps
from percussion_generation_v3.data import AUGMENTED_SESSIONS
from percussion_generation_v3.lossless_dataset import _row_groups, _source_output, _source_path
from percussion_generation_v3.roles import role
from custom_extractor import extract_tweeter
from custom_extractor.models import Composition
from .music import SCHEMA_SHA256, encode_input, encode_target, decode_target


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_groups(source_output=None):
    root = Path(source_output) if source_output else _source_output()
    groups, sources = _row_groups(root)
    source_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    augmentation_path = AUGMENTED_SESSIONS / "manifest.json"
    augmentation_sha = sha256(augmentation_path)
    if source_manifest.get("augmentation", {}).get("manifest_sha256") != augmentation_sha:
        raise ValueError("source pool and augmentation manifest disagree")
    generated = {entry["augmented_filename"]: entry for entry in
                 json.loads(augmentation_path.read_text(encoding="utf-8"))["generated"]}
    if len(generated) != len(json.loads(augmentation_path.read_text(encoding="utf-8"))["generated"]):
        raise ValueError("duplicate augmentation file")
    if (len([r for r in groups["train"] if not r.get("augmentation")]),
            len(groups["validation"]), len(groups["validation20"]), len(groups["test"])) != (67, 10, 20, 5):
        raise ValueError("unexpected source split sizes")
    if groups["validation20"][:10] != groups["validation"]:
        raise ValueError("fixed validation does not start with original ten")
    return root, groups, sources, generated, augmentation_sha


def texts_for_pair(row, generated, composition=None):
    path = _source_path(row, generated)
    composition = composition if composition is not None else extract_tweeter(path)
    target = composition.track(row["target_track_id"])
    if [note["step"] for note in target["notes"]] != row["target_positions"]:
        raise ValueError("target positions differ from source pair")
    if role(target) != "percussion":
        raise ValueError("target is not a verified percussion track")
    remaining = [track for track in composition.tracks if track["id"] != row["target_track_id"]]
    if len(remaining) + 1 != len(composition.tracks):
        raise ValueError("target track ID is ambiguous")
    condition = Composition(composition.global_settings, remaining)
    target_text = encode_target(target)
    decoded = decode_target(target_text, len(target["notes"]))
    if decoded["notes"] != target["notes"] or decoded["instrument"] != target["instrument"]["name"]:
        raise ValueError("musical target round trip failed")
    prefix = prompt_for_composition(condition, len(target["notes"]))
    return prefix, target_text + "\n" + END_MARKER, target


def prompt_for_composition(composition, target_note_count):
    if type(target_note_count) is not int or not 1 <= target_note_count <= 64:
        raise ValueError("target note count must be 1..64")
    return encode_input(composition, target_note_count) + TARGET_MARKER


def build_preset_catalog(groups, generated):
    """Use original native tracks as sound templates; store no target notes."""
    from copy import deepcopy
    catalog = {}
    for split in ("train", "validation", "test"):
        for row in groups[split]:
            if row.get("augmentation"):
                continue
            track = extract_tweeter(_source_path(row, generated)).track(row["target_track_id"])
            name = track["instrument"]["name"]
            if name not in catalog:
                template = deepcopy(track)
                template["id"] = 0
                template["notes"] = []
                template["playback"]["cursor"] = None
                catalog[name] = template
    return catalog


def _unique_train_rows(rows):
    seen = set()
    for row in rows:
        identity = (row["source"], row["target_track_id"],
                    row.get("augmentation", {}).get("file"))
        if identity in seen:
            raise ValueError("duplicate tokenizer training pair")
        seen.add(identity)
        yield row


def train_tokenizer(output_dir=OUTPUT, source_output=None, vocab_size=16384,
                    progress_seconds=180, dry_run=False):
    from transformers import AutoTokenizer
    if not 4096 <= vocab_size <= 49152:
        raise ValueError("vocab size must be 4096..49152")
    output = Path(output_dir)
    if (output / "tokenizer").exists() or (output / "tokenizer_identity.json").exists():
        raise FileExistsError("tokenizer already exists in this experiment output")
    source, groups, _, generated, augmentation_sha = source_groups(source_output)
    train_rows = list(_unique_train_rows(groups["train"]))
    source_ids = {(r["source"], r.get("augmentation", {}).get("file")) for r in train_rows}
    plan = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "source_dataset": str(source.resolve()), "train_pairs": len(train_rows),
            "train_session_variants": len(source_ids), "vocab_size_requested": vocab_size,
            "augmentation_manifest_sha256": augmentation_sha,
            "heldout_sources_excluded": True}
    if dry_run:
        return plan
    old = AutoTokenizer.from_pretrained(MODEL_ROOT / "V3/output/tokenizer",
                                        use_fast=True, local_files_only=True)
    if not old.is_fast or old.eos_token_id is None:
        raise ValueError("pinned tokenizer template must be fast and have EOS")
    old_sha = sha256(MODEL_ROOT / "V3/output/tokenizer/tokenizer.json")
    last = time.monotonic()
    def corpus():
        nonlocal last
        cache = OrderedDict()
        for index, row in enumerate(train_rows, 1):
            key = row.get("augmentation", {}).get("file") or row["source"]
            if key not in cache:
                cache[key] = extract_tweeter(_source_path(row, generated))
                if len(cache) > 16:
                    cache.popitem(last=False)
            prefix, target, _ = texts_for_pair(row, generated, cache[key])
            yield [prefix, target]
            now = time.monotonic()
            if progress_seconds and now - last >= progress_seconds:
                print(f"tokenizer corpus: {index}/{len(train_rows)} training pairs", flush=True)
                last = now
    trained = old.train_new_from_iterator(corpus(), vocab_size=vocab_size,
                                          length=len(train_rows),
                                          new_special_tokens=[END_MARKER])
    if trained.eos_token_id is None or trained.eos_token != old.eos_token:
        raise ValueError("new tokenizer lost the pinned EOS token")
    output.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".tokenizer_", dir=output))
    try:
        trained.save_pretrained(temp)
        identity = {**plan, "tokenizer_kind": "fresh-bpe-from-pinned-pretokenizer",
                    "pretrained_tokenizer_sha256": old_sha,
                    "tokenizer_sha256": sha256(temp / "tokenizer.json"),
                    "vocab_size_actual": len(trained), "eos_token_id": trained.eos_token_id}
        temp.rename(output / "tokenizer")
        (output / "tokenizer_identity.json").write_text(dumps(identity) + "\n", encoding="utf-8")
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return identity


def load_new_tokenizer(output_dir=OUTPUT):
    from transformers import AutoTokenizer
    root = Path(output_dir)
    identity = json.loads((root / "tokenizer_identity.json").read_text(encoding="utf-8"))
    if (identity["model_id"], identity["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("tokenizer belongs to another SmolLM2 variant")
    if identity["tokenizer_sha256"] != sha256(root / "tokenizer/tokenizer.json"):
        raise ValueError("trained tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    if tokenizer.eos_token_id != identity["eos_token_id"]:
        raise ValueError("EOS token ID changed")
    return tokenizer, identity


def prepare(output_dir=OUTPUT, source_output=None, progress_seconds=180, dry_run=False):
    output = Path(output_dir)
    if (output / "manifest.json").exists() or (output / "rhythm").exists():
        raise FileExistsError("V4EA dataset already exists; use a new output directory")
    tokenizer, identity = load_new_tokenizer(output)
    source, groups, sources, generated, augmentation_sha = source_groups(source_output)
    plan = {"train_pairs": len(groups["train"]), "validation_pairs": len(groups["validation"]),
            "fixed_validation_pairs": len(groups["validation20"]),
            "test_pairs": len(groups["test"]), "tokenizer_sha256": identity["tokenizer_sha256"]}
    if dry_run:
        return plan
    cache = OrderedDict()
    converted = {}
    lengths = {}
    failures = []
    total = sum(map(len, groups.values()))
    done = 0
    last = time.monotonic()
    for split in ("train", "validation", "test", "validation20"):
        rows = groups[split]
        result = []
        for row in rows:
            key = row.get("augmentation", {}).get("file") or row["source"]
            if key not in cache:
                cache[key] = extract_tweeter(_source_path(row, generated))
                if len(cache) > 16:
                    cache.popitem(last=False)
            prefix, target, track = texts_for_pair(row, generated, cache[key])
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            count = len(prefix_ids) + len(target_ids)
            if count > 8192:
                failures.append((split, row["source"], row["target_track_id"], count))
            new = {**row, "encoding": VERSION, "prefix": prefix, "target": target,
                   "prefix_ids": prefix_ids, "target_ids": target_ids, "tokens": count,
                   "target_track_sha256": hashlib.sha256(dumps(track).encode()).hexdigest()}
            result.append(new)
            done += 1
            now = time.monotonic()
            if progress_seconds and now - last >= progress_seconds:
                print(f"V4EA preparation: {done}/{total} pairs", flush=True)
                last = now
        converted[split] = result
        lengths[split] = {"pairs": len(result), "max_tokens": max(r["tokens"] for r in result),
                          "mean_tokens": sum(r["tokens"] for r in result) / len(result),
                          "target_tokens": sum(len(r["target_ids"]) for r in result)}
    if failures:
        raise ValueError(f"{len(failures)} full-track examples exceed 8192 tokens; first: {failures[:3]}")
    catalog = build_preset_catalog(groups, generated)
    temp = Path(tempfile.mkdtemp(prefix=".v4ea_dataset_", dir=output))
    try:
        (temp / "rhythm").mkdir()
        digest = hashlib.sha256()
        for split, rows in converted.items():
            path = temp / "rhythm" / f"{split}.jsonl"
            path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")
            digest.update(split.encode() + b"\0" + path.read_bytes())
        manifest = {"version": VERSION, "encoding": "musical-v2",
                    "model_id": MODEL_ID, "revision": MODEL_REVISION,
                    "codec_schema_sha256": SCHEMA_SHA256,
                    "tokenizer_sha256": identity["tokenizer_sha256"],
                    "tokenizer_kind": identity["tokenizer_kind"],
                    "source_dataset": str(source.resolve()),
                    "source_augmentation_manifest_sha256": augmentation_sha,
                    "source_partitions": {name: sorted(members) for name, members in sources.items()},
                    "splits": lengths, "max_length": 8192,
                    "prepared_dataset_sha256": digest.hexdigest(),
                    "target": "percussion identity, notes, timing, then [END_MUSIC] and EOS",
                    "preset_catalog_sha256": hashlib.sha256(dumps(catalog).encode()).hexdigest()}
        (temp / "preset_catalog.json").write_text(dumps(catalog) + "\n", encoding="utf-8")
        (temp / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
        (temp / "rhythm").rename(output / "rhythm")
        (temp / "preset_catalog.json").replace(output / "preset_catalog.json")
        (temp / "manifest.json").replace(output / "manifest.json")
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {**plan, "splits": lengths, "prepared_dataset_sha256": digest.hexdigest()}


def verify_dataset(output_dir=OUTPUT):
    output = Path(output_dir)
    _, identity = load_new_tokenizer(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["version"], manifest["model_id"], manifest["revision"],
            manifest["codec_schema_sha256"], manifest["tokenizer_sha256"]) != (
            VERSION, MODEL_ID, MODEL_REVISION, SCHEMA_SHA256, identity["tokenizer_sha256"]):
        raise ValueError("V4EA dataset identity mismatch")
    if sha256(AUGMENTED_SESSIONS / "manifest.json") != manifest["source_augmentation_manifest_sha256"]:
        raise ValueError("augmentation manifest changed")
    digest = hashlib.sha256()
    groups = {}
    for split in ("train", "validation", "test", "validation20"):
        path = output / "rhythm" / f"{split}.jsonl"
        digest.update(split.encode() + b"\0" + path.read_bytes())
        groups[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if digest.hexdigest() != manifest["prepared_dataset_sha256"]:
        raise ValueError("V4EA prepared dataset changed")
    catalog = json.loads((output / "preset_catalog.json").read_text(encoding="utf-8"))
    if hashlib.sha256(dumps(catalog).encode()).hexdigest() != manifest["preset_catalog_sha256"]:
        raise ValueError("V4EA preset catalog changed")
    sources = {split: {row["source"] for row in groups[split]}
               for split in ("train", "validation", "test")}
    if any(sources[a] & sources[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("cross-split source leakage")
    if groups["validation20"][:10] != groups["validation"]:
        raise ValueError("validation20 changed")
    if any(row["source"] not in sources["validation"] for row in groups["validation20"]):
        raise ValueError("validation augmentation leaked")
    return manifest, groups
