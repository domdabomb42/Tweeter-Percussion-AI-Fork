"""Strict structure validation; immutable historical range exceptions are visible."""
import math
import struct
from .models import Composition
from .parameter_registry import PARAMETERS, WAVEFORMS, empty_track, get_path
from .parser import atom_text


class ValidationError(ValueError):
    pass


def _fail(where, value, message):
    raise ValidationError(f"{where}: {value!r}: {message}")


def _number(value, where):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3.402823466e38:
        _fail(where, value, "expected finite native float32-range number")
    if value and struct.unpack("f", struct.pack("f", value))[0] == 0:
        _fail(where, value, "underflows native float32")


def _integer(value, where, lo, hi):
    _number(value, where)
    if int(value) != value or not lo <= value <= hi:
        _fail(where, value, f"expected integer in {lo}..{hi}")


def check_parameter(parameter, value, where, check_bounds=True):
    if value is None:
        return
    kind = parameter.kind
    if kind == "waveform":
        if not isinstance(value, str) or value not in WAVEFORMS:
            _fail(where, value, f"expected one of {WAVEFORMS}")
    elif kind == "symbol":
        if not isinstance(value, str):
            _fail(where, value, "expected a native symbol string")
        atom_text(value)
    elif kind in ("array", "atoms"):
        if not isinstance(value, list) or not value:
            _fail(where, value, "expected a nonempty native array/list")
        for i, item in enumerate(value):
            if kind == "array":
                _number(item, f"{where}[{i}]")
            else:
                atom_text(item)
    elif kind == "cursor":
        if not isinstance(value, dict) or set(value) != {"next_step", "remaining_ms", "running"}:
            _fail(where, value, "expected next_step, remaining_ms, running")
        _integer(value["next_step"], where + ".next_step", 0, 64)
        _number(value["remaining_ms"], where + ".remaining_ms")
        if value["remaining_ms"] < 0:
            _fail(where, value, "remaining time must be nonnegative")
        _integer(value["running"], where + ".running", 0, 1)
    else:
        _number(value, where)
        if kind == "integer" and int(value) != value:
            _fail(where, value, "expected an integer")
        if check_bounds and not (parameter.native == "echo-ms" and value == 0):
            if parameter.minimum is not None and value < parameter.minimum or parameter.maximum is not None and value > parameter.maximum:
                _fail(where, value, f"native editing range is {parameter.minimum}..{parameter.maximum}")


def _shape(value, template, where):
    if not isinstance(value, dict) or set(value) != set(template):
        _fail(where, value if not isinstance(value, dict) else list(value), f"expected fields {list(template)}")
    for key, expected in template.items():
        if isinstance(expected, dict):
            _shape(value[key], expected, where + "." + key)


def validate(composition):
    """Return diagnostics; raise on unexportable values. No clamping or mutation."""
    from .extractor import extract_text
    if not isinstance(composition, Composition):
        raise ValidationError("expected a Composition")
    baseline = extract_text(composition.native_text) if composition.native_text else Composition()
    old = {t["id"]: t for t in baseline.tracks}
    _shape(composition.global_settings, Composition().global_settings, "global")
    if type(composition.global_settings["tracker_steps"]) is not int or composition.global_settings["tracker_steps"] != 64:
        raise ValidationError("global.tracker_steps: native Tweeter uses exactly 64 steps")
    title = composition.global_settings["metadata"]["session_title"]
    if title is not None:
        if not isinstance(title, list):
            _fail("global.metadata.session_title", title, "expected list of native atoms, or null")
        for atom in title:
            atom_text(atom)
    if not isinstance(composition.tracks, list):
        raise ValidationError("tracks must be a list")
    warnings, ids, precision_count = [], [], 0
    for track in composition.tracks:
        _shape(track, empty_track(1), "track")
        tid = track["id"]
        if type(tid) is not int or not 1 <= tid <= 14 or tid in ids:
            _fail("track.id", tid, "expected unique integer in 1..14 (native patch instrument slots)")
        ids.append(tid)
        for parameter in PARAMETERS:
            value = get_path(track, parameter.path)
            where = f"track {tid}.{parameter.path} ({parameter.native})"
            check_parameter(parameter, value, where, check_bounds=False)
            if parameter.kind == "array" and value is not None and len(value) < 100:
                warnings.append({"code": "partial_envelope_update", "track": tid, "parameter": parameter.native,
                                 "message": "fewer than 100 samples: native tabwrite updates only this prefix; remaining table samples depend on application state"})
            numeric_values = value if isinstance(value, list) else list(value.values()) if isinstance(value, dict) else [value]
            precision_count += sum(type(v) in (int, float) and struct.unpack("f", struct.pack("f", v))[0] != v for v in numeric_values)
            try:
                check_parameter(parameter, value, where)
            except ValidationError:
                if tid not in old or get_path(old[tid], parameter.path) != value:
                    raise
                warnings.append({"code": "legacy_range", "track": tid, "parameter": parameter.native, "value": value,
                                 "message": "unchanged source value outside current UI range retained exactly"})
        notes = track["notes"]
        if not isinstance(notes, list):
            _fail(f"track {tid}.notes", notes, "expected a list")
        last_step = -1
        for note in notes:
            if not isinstance(note, dict) or set(note) != {"step", "pitch"}:
                _fail(f"track {tid}.notes", note, "expected step and pitch; no per-note velocity or duration exists")
            _integer(note["step"], f"track {tid}.note.step", 0, 63)
            _number(note["pitch"], f"track {tid}.note.pitch")
            precision_count += struct.unpack("f", struct.pack("f", note["pitch"]))[0] != note["pitch"]
            if note["pitch"] == 0:
                _fail(f"track {tid}.note.pitch", note["pitch"], "zero is a native rest; remove the note instead")
            if note["step"] <= last_step:
                _fail(f"track {tid}.notes", note, "one native pitch per step; events must be in ascending step order")
            last_step = note["step"]
        if track["timing"]["loop_ms"] is not None and track["playback"]["cursor"] is not None:
            if track["playback"]["cursor"]["remaining_ms"] > track["timing"]["loop_ms"]:
                warnings.append({"code": "cursor_exceeds_loop", "track": tid, "message": "saved synchronization delay exceeds loop; preserved, not normalized"})
        if notes and any(track["timing"][k] is None for k in ("loop_ms", "note_duration_ms", "transposition_semitones")):
            warnings.append({"code": "missing_timing", "track": tid, "message": "partial snapshot; missing values depend on native application state; derived timing/pitch may be unavailable"})
    # Existing source record ordering can affect native message side effects.
    # Reject reorder/delete instead of silently rearranging or dropping opaque data.
    original_ids = [t["id"] for t in baseline.tracks]
    if ids[:len(original_ids)] != original_ids:
        raise ValidationError("tracks: preserve existing track IDs and order; append new tracks only")
    if not ids and title is None:
        raise ValidationError("composition requires at least one track or session-title")
    for track in composition.tracks[len(original_ids):]:
        if not any(get_path(track, p.path) is not None for p in PARAMETERS) and not track["notes"]:
            _fail("new track", track["id"], "requires at least one native parameter or note")
        if track["notes"] and any(track["timing"][k] is None for k in ("loop_ms", "note_duration_ms", "transposition_semitones")):
            _fail("new track", track["id"], "notes require loop-length, note-dur and octave")
    if precision_count:
        warnings.append({"code": "native_float32_precision", "count": precision_count,
                         "message": "JSON/TXT values are retained; Pd's 32-bit float engine approximates these decimal values"})
    return warnings
