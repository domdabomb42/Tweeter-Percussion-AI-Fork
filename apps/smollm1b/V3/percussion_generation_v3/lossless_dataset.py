"""Source-derived V2 preparation and full-pool tokenizer benchmarking."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from .codec_v2 import SCHEMA_SHA256, VERSION, decode_v2, dumps, encode_v2, envelope_variants
from .data import (APPS, AUGMENTED_SESSIONS, DEFAULT_SESSIONS, FEATURE_ADAPTER,
                   ROOT, SMOLLM, load_rows, load_tokenizer, serialize_positions)
from .inference import prompt
from .roles import role
from custom_extractor import extract_tweeter
from custom_extractor.models import Composition
from custom_training import MODEL_ID, MODEL_REVISION


def sequence_dictionary_variant(values):
    """Benchmark-only reversible dictionary of four-value numeric sequences."""
    chunks = [values[i:i + 4] for i in range(0, len(values), 4)]
    dictionary, indexes, seen = [], [], {}
    for chunk in chunks:
        key = dumps(chunk)
        if key not in seen:
            seen[key] = len(dictionary)
            dictionary.append(chunk)
        indexes.append(seen[key])
    return ["Q", dictionary, indexes]


def restore_sequence_dictionary(entry):
    if entry[0] != "Q":
        raise ValueError("not a sequence dictionary")
    return [value for index in entry[2] for value in entry[1][index]]


def _envelope_candidates(values):
    candidates = envelope_variants(values)
    candidates["sequence_dictionary"] = sequence_dictionary_variant(values)
    return candidates


def _source_output():
    for candidate in (APPS / "smollm135/V3/output_augmented_from_epoch7",
                      ROOT / "output_augmented_from_epoch7", ROOT / "output_augmented", ROOT / "output"):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("no local V3 split metadata found")


def _row_groups(root):
    groups = {split: load_rows(root, "rhythm", split)
              for split in ("train", "validation", "test")}
    fixed = root / "rhythm/validation20.jsonl"
    if fixed.is_file():
        groups["validation20"] = [json.loads(line) for line in fixed.read_text(encoding="utf-8").splitlines()]
    sources = {split: {row["source"] for row in groups[split]}
               for split in ("train", "validation", "test")}
    if any(sources[a] & sources[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("source sessions cross splits")
    for split, rows in groups.items():
        parent_split = "validation" if split == "validation20" else split
        if any(row["source"] not in sources[parent_split] for row in rows):
            raise ValueError("augmented child crosses source split")
    return groups, sources


def _statistics(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"count": len(ordered), "median": statistics.median(ordered),
            "mean": statistics.mean(ordered),
            "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
            "maximum": ordered[-1], "over_8192": sum(x > 8192 for x in ordered),
            "total": sum(ordered)}


def _source_path(row, generated):
    aug = row.get("augmentation")
    if aug:
        entry = generated.get(aug["file"])
        if (not entry or entry["original_session"] != row["source"]
                or entry["sha256"] != aug["sha256"]
                or entry["reference_offset"] != aug["reference_offset"]):
            raise ValueError("augmented file or parent differs from manifest")
        root = (AUGMENTED_SESSIONS / "generated").resolve()
        path = (root / aug["file"]).resolve()
        if not path.is_relative_to(root) or entry["validation"] != "passed":
            raise ValueError("invalid augmented path or validation")
        expected = aug["sha256"]
    else:
        path = DEFAULT_SESSIONS / row["source"]
        expected = row["source_sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"session bytes changed: {path}")
    return path


def _row_key(row):
    return (row["source"], row["target_track_id"],
            row.get("augmentation", {}).get("file"))


def _markdown(report):
    old, new = report["legacy"], report["lossless_v2"]
    lines = ["# SmolLM2-135M V3 codec benchmark", "",
             f"Model: `{report['model_id']}` at `{report['revision']}`. "
             f"Examples: {report['example_count']}. Actual tokenizer IDs; no training run.", "",
             "| Metric | Legacy compact | Lossless V2 |", "|---|---:|---:|",
             f"| Median tokens | {old['median']} | {new['median']} |",
             f"| Mean tokens | {old['mean']:.2f} | {new['mean']:.2f} |",
             f"| P95 tokens | {old['p95']} | {new['p95']} |",
             f"| Maximum tokens | {old['maximum']} | {new['maximum']} |",
             f"| Over 8192 | {old['over_8192']} | {new['over_8192']} |",
             f"| Total tokens | {old['total']} | {new['total']} |", "",
             f"Total reduction: **{report['reduction_percent']:.2f}%**. "
             "Estimated token-processing work falls by this fraction if cost per token "
             "is held constant; actual training speed and memory savings are unmeasured.", "",
             "## Envelope alternatives (tokenizer IDs for all conditioning envelopes)", "",
             "| Mode | Median | Mean | P95 | Maximum | Over 8192 | Total |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, value in report["envelope_modes"].items():
        lines.append(f"| {mode} | {value['median']} | {value['mean']:.2f} | "
                     f"{value['p95']} | {value['maximum']} | {value['over_8192']} | {value['total']} |")
    lines += ["", "Zlib/Base64 was selected for full-resolution envelopes because the "
              "actual tokenizer produces fewer IDs on this dataset; its opaque bank "
              "is tagged and exactly reversible. The positional schema exposes all "
              "non-envelope musical values directly.", "",
              "Per-example complete-prompt counts and envelope totals are in `codec_benchmark.json`.", ""]
    if "legacy_prefix" in report:
        lines += [f"Prefix-only tokens: {report['legacy_prefix']['total']} legacy vs "
                  f"{report['lossless_v2_prefix']['total']} V2 "
                  f"({report['prefix_reduction_percent']:.2f}% fewer).", ""]
    return "\n".join(lines)


def process(output_dir=None, prepare=False, source_output=None, progress_seconds=180):
    source_output = Path(source_output) if source_output else _source_output()
    source_manifest = json.loads((source_output / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("model_id") not in (MODEL_ID, "HuggingFaceTB/SmolLM2-135M"):
        raise ValueError("source split metadata uses an unknown model")
    groups, sources = _row_groups(source_output)
    augmentation_bytes = (AUGMENTED_SESSIONS / "manifest.json").read_bytes()
    aug_hash = hashlib.sha256(augmentation_bytes).hexdigest()
    if source_manifest.get("augmentation", {}).get("manifest_sha256") != aug_hash:
        raise ValueError("augmentation manifest fingerprint differs from source dataset")
    generated = {entry["augmented_filename"]: entry
                 for entry in json.loads(augmentation_bytes)["generated"]}
    if len(generated) != len(json.loads(augmentation_bytes)["generated"]):
        raise ValueError("duplicate augmentation paths")
    tokenizer_root = source_output if source_manifest.get("model_id") == MODEL_ID else ROOT / "output"
    tokenizer = load_tokenizer(tokenizer_root)
    tokenizer_sha = hashlib.sha256((tokenizer_root / "tokenizer/tokenizer.json").read_bytes()).hexdigest()
    output_dir = Path(output_dir) if output_dir else None
    if prepare and (output_dir is None or output_dir.exists()):
        raise FileExistsError("new V2 dataset requires a fresh --output-dir")
    candidate_modes = ("raw", "rle", "dictionary", "sequence_dictionary", "zlib")
    old_lengths, new_lengths, differences, envelope_lengths = [], [], [], {m: [] for m in candidate_modes}
    old_prefix_lengths, new_prefix_lengths = [], []
    prepared = {split: [] for split in groups}
    envelope_token_cache = {}
    cache = {}
    source_cache = OrderedDict()
    first = last = time.monotonic()
    total = sum(len(rows) for rows in groups.values()) - len(groups.get("validation", []))
    done = 0
    counted = set()
    for split, rows in groups.items():
        for row in rows:
            key = _row_key(row)
            if key in cache:
                result = cache[key]
            else:
                file_key = row.get("augmentation", {}).get("file") or row["source"]
                if file_key in source_cache:
                    composition = source_cache[file_key]
                    source_cache.move_to_end(file_key)
                else:
                    path = _source_path(row, generated)
                    composition = extract_tweeter(path)
                    source_cache[file_key] = composition
                    if len(source_cache) > 32:
                        source_cache.popitem(last=False)
                target = composition.track(row["target_track_id"])
                positions = [note["step"] for note in target["notes"]]
                if positions != row["target_positions"] or serialize_positions(positions) != row["target"]:
                    raise ValueError(f"original target changed: {key}")
                # The model sees global settings and tracks, not native_text.
                # Filtering the parsed track list is equivalent to the legacy
                # native-record removal for those fields and avoids reparsing.
                condition = Composition(composition.global_settings,
                                        [track for track in composition.tracks
                                         if track["id"] != row["target_track_id"]])
                original = {"global": condition.global_settings,
                            "tracks": [{"role": role(track), "track": track} for track in condition.tracks]}
                encodings = {"zlib": encode_v2(original, "zlib")}
                if decode_v2(encodings["zlib"]) != original:
                    raise ValueError(f"lossless round-trip failed: {key}")
                suffix = "\nRHYTHM_TARGET=" + str(row["target_note_count"]) + "\n"
                old_prefix = prompt(condition, row["target_note_count"], "legacy-v1")
                prefix_by_mode = {mode: value + suffix for mode, value in encodings.items()}
                mode_ids = {"zlib": tokenizer.encode(prefix_by_mode["zlib"], add_special_tokens=False)}
                old_ids = tokenizer.encode(old_prefix, add_special_tokens=False)
                seen_envelopes = set()
                for track in original["tracks"]:
                    for env_key in ("carrier_envelope", "modulator_envelope"):
                        envelope = track["track"].get("synth", {}).get(env_key)
                        if isinstance(envelope, list):
                            signature = dumps(envelope)
                            if signature in seen_envelopes:
                                continue
                            seen_envelopes.add(signature)
                            if signature not in envelope_token_cache:
                                variants = _envelope_candidates(envelope)
                                envelope_token_cache[signature] = {
                                    mode: len(tokenizer.encode(dumps(variants[mode]),
                                                               add_special_tokens=False))
                                    for mode in candidate_modes}
                            for mode in candidate_modes:
                                envelope_lengths[mode].append(envelope_token_cache[signature][mode])
                result = (prefix_by_mode, mode_ids, old_ids)
                if split == "validation":
                    cache[key] = result
            prefix_by_mode, mode_ids, old_ids = result
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if source_manifest.get("model_id") == MODEL_ID and target_ids != row["target_ids"]:
                raise ValueError(f"target tokenization changed: {key}")
            old_count = len(old_ids) + len(target_ids)
            new_count = len(mode_ids["zlib"]) + len(target_ids)
            if key not in counted:
                counted.add(key)
                old_lengths.append(old_count)
                new_lengths.append(new_count)
                old_prefix_lengths.append(len(old_ids))
                new_prefix_lengths.append(len(mode_ids["zlib"]))
                differences.append({"split": split, "source": row["source"],
                                    "target_track_id": row["target_track_id"],
                                    "augmentation_file": row.get("augmentation", {}).get("file"),
                                    "legacy_tokens": old_count, "v2_tokens": new_count,
                                    "difference": old_count - new_count,
                                    "legacy_prefix_tokens": len(old_ids),
                                    "v2_prefix_tokens": len(mode_ids["zlib"]),
                                    "reduction_percent": 100 * (old_count - new_count) / old_count})
                done += 1
            if prepare:
                converted = dict(row)
                converted.update(encoding=VERSION, prefix=prefix_by_mode["zlib"],
                                 prefix_ids=mode_ids["zlib"], target_ids=target_ids,
                                 tokens=new_count)
                prepared[split].append(converted)
            now = time.monotonic()
            if progress_seconds and now - last >= progress_seconds:
                print(f"V2 codec: {done}/{total} source pairs checked; "
                      f"elapsed {(now-first)/60:.1f} min", flush=True)
                last = now
    report = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
              "tokenizer_sha256": tokenizer_sha,
              "codec_version": VERSION, "codec_schema_sha256": SCHEMA_SHA256,
              "example_count": len(old_lengths), "selected_envelope_mode": "zlib",
              "legacy": _statistics(old_lengths), "lossless_v2": _statistics(new_lengths),
              "legacy_prefix": _statistics(old_prefix_lengths),
              "lossless_v2_prefix": _statistics(new_prefix_lengths),
              "prefix_reduction_percent": 100 * (1 - sum(new_prefix_lengths) / sum(old_prefix_lengths)),
              "reduction_percent": 100 * (1 - sum(new_lengths) / sum(old_lengths)),
              "envelope_modes": {mode: _statistics(envelope_lengths[mode]) for mode in candidate_modes},
              "envelope_only": {mode: _statistics(envelope_lengths[mode]) for mode in candidate_modes},
              "per_example": differences,
              "estimated_sequence_token_work_savings_percent": 100 * (1 - sum(new_lengths) / sum(old_lengths))}
    if not prepare:
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "codec_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (output_dir / "codec_benchmark.md").write_text(_markdown(report), encoding="utf-8")
        return report
    temp = Path(tempfile.mkdtemp(prefix=".preparing_lossless_v2_", dir=output_dir.parent))
    try:
        shutil.copytree(tokenizer_root / "tokenizer", temp / "tokenizer")
        shutil.copy2(source_output / "fixed_preset.json", temp / "fixed_preset.json")
        (temp / "tokenizer_identity.json").write_text(json.dumps(
            {"model_id": MODEL_ID, "revision": MODEL_REVISION,
             "sha256": tokenizer_sha}, indent=2) + "\n", encoding="utf-8")
        dataset_hash = hashlib.sha256()
        summaries = {}
        for split, rows in prepared.items():
            destination = temp / "rhythm" / f"{split}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(dumps(row) + "\n")
            dataset_hash.update(split.encode() + b"\0" + destination.read_bytes())
            summaries[split] = {"pairs": len(rows), "sessions": len({r["source"] for r in rows}),
                                "token_lengths": _statistics([r["tokens"] for r in rows])}
        manifest = dict(source_manifest)
        manifest.update(source_dataset=str(source_output), encoding=VERSION,
                        model_id=MODEL_ID, revision=MODEL_REVISION,
                        starting_adapter=(source_manifest.get("starting_adapter")
                                          if source_manifest.get("model_id") == MODEL_ID
                                          else str(FEATURE_ADAPTER)),
                        codec_version=VERSION, codec_schema_sha256=SCHEMA_SHA256,
                        envelope_mode="zlib", prepared_dataset_sha256=dataset_hash.hexdigest(),
                        source_augmentation_manifest_sha256=aug_hash,
                        splits=summaries, tokenizer_sha256=hashlib.sha256(
                            (temp / "tokenizer/tokenizer.json").read_bytes()).hexdigest())
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (temp / "codec_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (temp / "codec_benchmark.md").write_text(_markdown(report), encoding="utf-8")
        temp.rename(output_dir)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {"output_dir": str(output_dir), "splits": summaries,
            "codec_version": VERSION, "reduction_percent": report["reduction_percent"],
            "over_8192": report["lossless_v2"]["over_8192"],
            "benchmark": str(output_dir / "codec_benchmark.json")}


def refresh_envelope_benchmark(output_dir):
    """Recount each deduplicated bank entry once per unique prepared example."""
    root = Path(output_dir)
    report_path = root / "codec_benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(root)
    lengths = {mode: [] for mode in ("raw", "rle", "dictionary", "sequence_dictionary", "zlib")}
    cache, seen_examples, target_lengths = {}, set(), {}
    for split in ("train", "validation", "test", "validation20"):
        rows = ([json.loads(line) for line in
                 (root / "rhythm/validation20.jsonl").read_text(encoding="utf-8").splitlines()]
                if split == "validation20" else load_rows(root, "rhythm", split))
        for row in rows:
            key = _row_key(row)
            if key in seen_examples:
                continue
            seen_examples.add(key)
            target_lengths[key] = len(row["target_ids"])
            packed = json.loads(row["prefix"].split("\nRHYTHM_TARGET=", 1)[0])
            if packed[0] != "L2V2":
                raise ValueError("prepared row is not V2")
            for tag, blob in packed[2]:
                if tag != "Z":
                    raise ValueError("unexpected V2 envelope bank encoding")
                import base64, zlib
                array_text = zlib.decompress(base64.b64decode(blob)).decode()
                if array_text not in cache:
                    variants = _envelope_candidates(json.loads(array_text))
                    cache[array_text] = {mode: len(tokenizer.encode(dumps(variants[mode]),
                                                                  add_special_tokens=False))
                                         for mode in lengths}
                for mode in lengths:
                    lengths[mode].append(cache[array_text][mode])
    if len(seen_examples) != report["example_count"]:
        raise ValueError("benchmark example count differs from prepared dataset")
    report["envelope_modes"] = {mode: _statistics(values) for mode, values in lengths.items()}
    report["envelope_only"] = report["envelope_modes"]
    report["envelope_counting"] = "one tokenized entry per unique envelope in each example's bank"
    old_prefix, new_prefix = [], []
    for example in report["per_example"]:
        key = (example["source"], example["target_track_id"], example["augmentation_file"])
        target_length = target_lengths[key]
        example["legacy_prefix_tokens"] = example["legacy_tokens"] - target_length
        example["v2_prefix_tokens"] = example["v2_tokens"] - target_length
        old_prefix.append(example["legacy_prefix_tokens"])
        new_prefix.append(example["v2_prefix_tokens"])
    report["legacy_prefix"] = _statistics(old_prefix)
    report["lossless_v2_prefix"] = _statistics(new_prefix)
    report["prefix_reduction_percent"] = 100 * (1 - sum(new_prefix) / sum(old_prefix))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "codec_benchmark.md").write_text(_markdown(report), encoding="utf-8")
    return {mode: {"total": value["total"], "median": value["median"]}
            for mode, value in report["envelope_modes"].items()}
