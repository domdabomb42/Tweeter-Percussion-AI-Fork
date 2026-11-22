"""One source file per split; segments contain complete tracks and full values."""
from collections import Counter
from pathlib import Path
import json
import statistics
import sys
from transformers import AutoTokenizer
from . import MODEL_ID, MODEL_REVISION
from .data_loader import load_sessions, split_sessions, dataset_summary
from .serializer import serialize, serialize_object, whole_track_object, leaf_spans, value_category


def tokenizer_for_model(local_only=False):
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                         use_fast=True, local_files_only=local_only)


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def segment_session(session, tokenizer, max_length):
    composition = session.composition
    full = serialize(composition)
    full_ids = _encode(tokenizer, full)
    if len(full_ids) + 1 <= max_length:
        return [{"text": full, "track_ids": [t["id"] for t in composition.tracks],
                 "token_ids": full_ids + [tokenizer.eos_token_id]}]
    tracks = composition.tracks
    groups, current = [], []
    for track in tracks:
        candidate = current + [track]
        candidate_text = serialize_object(whole_track_object(composition, candidate))
        if len(_encode(tokenizer, candidate_text)) + 1 <= max_length:
            current = candidate
        else:
            if not current:
                raise ValueError(f"{session.name}: track {track['id']} alone exceeds {max_length} tokens; increase max length up to model context 8192; no truncation performed")
            groups.append(current)
            current = [track]
            one_text = serialize_object(whole_track_object(composition, current))
            if len(_encode(tokenizer, one_text)) + 1 > max_length:
                raise ValueError(f"{session.name}: track {track['id']} alone exceeds {max_length} tokens; no truncation performed")
    if current:
        groups.append(current)
    rows = []
    for group in groups:
        text = serialize_object(whole_track_object(composition, group))
        ids = _encode(tokenizer, text) + [tokenizer.eos_token_id]
        assert len(ids) <= max_length
        rows.append({"text": text, "track_ids": [t["id"] for t in group], "token_ids": ids})
    if sorted(tid for row in rows for tid in row["track_ids"]) != sorted(t["id"] for t in tracks):
        raise AssertionError("track loss during segmentation")
    return rows


def _token_categories(text, tokenizer):
    obj = json.loads(text)
    rendered, spans = leaf_spans(obj)
    assert rendered == text
    mapping = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    counts = Counter()
    j = 0
    for start, end in mapping:
        while j < len(spans) and spans[j]["end"] <= start:
            j += 1
        if j < len(spans) and spans[j]["start"] < end and spans[j]["value"] is not None:
            counts[value_category(spans[j]["path"])] += 1
        else:
            counts["structure_or_missing"] += 1
    return counts


def prepare(sessions_dir, output_dir, seed=415, max_length=4096, local_only=False):
    if max_length > 8192 or max_length < 256:
        raise ValueError("max_length must be 256..8192")
    sessions = load_sessions(sessions_dir)
    split, manifest = split_sessions(sessions, seed)
    tokenizer = tokenizer_for_model(local_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("SmolLM2 tokenizer lacks EOS token")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out / "tokenizer")
    summary = dataset_summary(sessions)
    counts, total, composition_lengths, per_track, file_rows = Counter(), 0, [], [], []
    observed = {"2048": 0, "4096": 0, "8192": 0}
    rows_by_split = {}
    completed = 0
    for name, members in split.items():
        rows = []
        for session in members:
            whole_text = serialize(session.composition)
            length = len(_encode(tokenizer, whole_text)) + 1
            composition_lengths.append(length)
            for bound in observed:
                observed[bound] += length <= int(bound)
            track_lens = [len(_encode(tokenizer, serialize_object(track))) for track in session.composition.tracks]
            per_track.extend(track_lens)
            for segment_index, segment in enumerate(segment_session(session, tokenizer, max_length)):
                counts.update(_token_categories(segment["text"], tokenizer))
                total += len(segment["token_ids"])
                rows.append({"source": session.name, "family": session.family,
                             "segment": segment_index, "track_ids": segment["track_ids"],
                             "text": segment["text"], "token_ids": segment["token_ids"]})
            file_rows.append({"file": session.name, "split": name, "whole_composition_tokens": length,
                              "track_tokens": track_lens,
                              "segments": sum(row["source"] == session.name for row in rows)})
            completed += 1
            if completed % 5 == 0:
                print(f"Prepared {completed}/{len(sessions)} sessions", file=sys.stderr, flush=True)
        rows_by_split[name] = rows
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
    summary["tokenizer"] = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                            "vocab_size": tokenizer.vocab_size,
                            "model_max_length": tokenizer.model_max_length,
                            "added_custom_tokens": 0}
    summary["token_counts"] = {"full_compositions": sum(composition_lengths),
                               "prepared_segments_including_eos": total,
                               "per_split": {name: sum(len(r["token_ids"]) for r in rows) for name, rows in rows_by_split.items()},
                               "segments_per_split": {name: len(rows) for name, rows in rows_by_split.items()},
                               "per_composition": {"mean": statistics.mean(composition_lengths),
                                                    "median": statistics.median(composition_lengths),
                                                    "max": max(composition_lengths),
                                                    "min": min(composition_lengths)},
                               "per_track": {"mean": statistics.mean(per_track),
                                             "median": statistics.median(per_track), "max": max(per_track)},
                               "field_token_categories": dict(counts),
                               "fit_without_segmentation": {bound: {"count": n, "percent": 100*n/len(sessions)} for bound, n in observed.items()},
                               "files": sorted(file_rows, key=lambda row: row["file"])}
    summary["split"] = manifest
    summary["max_training_length"] = max_length
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_prepared(output_dir, partition):
    path = Path(output_dir) / f"{partition}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
