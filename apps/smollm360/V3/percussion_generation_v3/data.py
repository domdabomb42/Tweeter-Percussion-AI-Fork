"""Retokenize the retained, source-verified V3 rhythm examples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

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
        if hashlib.sha256((root / "tokenizer/tokenizer.json").read_bytes()).hexdigest() != recorded["sha256"]:
            raise ValueError("V3 tokenizer bytes changed")
    elif SMOLLM.name != "smollm360":
        raise FileNotFoundError("model-specific V3 tokenizer identity missing; run prepare")
    return AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)


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


def prepare(output_dir=ROOT / "output", max_length=8192):
    output_dir = Path(output_dir)
    if max_length != 8192:
        raise ValueError("V3 uses an 8192-token context")
    if SMOLLM.name == "smollm360" and output_dir.resolve() == REFERENCE_OUTPUT.resolve():
        # The 360M artifacts are historical measurements. Never rewrite them.
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    from custom_training import MODEL_ID, MODEL_REVISION
    from custom_training.prepare import tokenizer_for_model
    tokenizer = tokenizer_for_model()
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS")
    cache = {}
    summaries = {}
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
    if output_dir.exists() and any((output_dir / p).exists() for p in
                                   ("checkpoints", "evaluation", "generated")):
        raise FileExistsError("V3 output already has results; prepare in a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
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
