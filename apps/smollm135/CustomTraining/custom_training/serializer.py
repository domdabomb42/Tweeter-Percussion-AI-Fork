"""Deterministic compact JSON of CustomExtractor's canonical feature state."""
import json
import sys
from pathlib import Path

_EXTRACTOR = Path(__file__).resolve().parents[3] / "CustomExtractor"
if str(_EXTRACTOR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTOR))
from custom_extractor.reporting import canonical


def serialize_object(obj):
    return json.dumps(obj, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def serialize(composition):
    return serialize_object(canonical(composition))


def whole_track_object(composition, tracks, opaque=None):
    view = canonical(composition)
    selected_ids = {t["id"] for t in tracks}
    all_ids = {t["id"] for t in composition.tracks}
    first_group = bool(composition.tracks) and composition.tracks[0]["id"] in selected_ids
    view["tracks"] = tracks
    if opaque is not None:
        view["opaque_records"] = opaque
    else:
        view["opaque_records"] = [r for r in view["opaque_records"]
                                   if r["values"] and isinstance(r["values"][0], (int, float))
                                   and r["values"][0] in selected_ids
                                   or first_group and (not r["values"] or not isinstance(r["values"][0], (int, float))
                                                       or r["values"][0] not in all_ids)]
    return view


def leaf_spans(obj):
    """Return exact source offsets for scalar values, for multi-token evaluation."""
    pieces, spans, offset = [], [], 0

    def add(text):
        nonlocal offset
        pieces.append(text)
        offset += len(text)

    def visit(value, path):
        if isinstance(value, dict):
            add("{")
            for index, (key, item) in enumerate(value.items()):
                if index:
                    add(",")
                add(serialize_object(key))
                add(":")
                visit(item, path + (key,))
            add("}")
        elif isinstance(value, list):
            add("[")
            for index, item in enumerate(value):
                if index:
                    add(",")
                visit(item, path + (index,))
            add("]")
        else:
            begin = offset
            add(serialize_object(value))
            spans.append({"path": list(path), "start": begin, "end": offset,
                          "value": value})

    visit(obj, ())
    return "".join(pieces), spans


def value_category(path):
    if not path:
        return "structure"
    if "notes" in path:
        return "notes"
    if "synth" in path:
        return "synthesis"
    if "effects" in path:
        return "effects"
    if "timing" in path or "playback" in path:
        return "timing_playback"
    return "other_values"
