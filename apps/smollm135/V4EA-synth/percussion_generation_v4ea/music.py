"""Versioned musical schema shared by the three V4EA model sizes.

Input: [tag, requested_count, [[role, instrument_name, username, loop_ms,
         [[step, pitch], ...]], ...]] in original track order.
Target: [tag, instrument_name, loop_ms, note_duration_ms,
         transposition_semitones, [[step, pitch], ...]].
Only musical information is predicted; native instrument settings come from
the separately stored, validated preset catalog.
"""
from copy import deepcopy
import hashlib
import json
import math

from percussion_generation_v3.roles import role
from custom_extractor.validator import validate
from custom_extractor.models import Composition

INPUT_TAG = "V4EA-MUSIC-INPUT-2"
TARGET_TAG = "V4EA-MUSIC-TARGET-2"
SCHEMA_SHA256 = hashlib.sha256(__doc__.encode("utf-8")).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode_input(composition, count):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("requested note count must be 1..64")
    tracks = []
    for track in composition.tracks:
        tracks.append([role(track), track["instrument"]["name"],
                       track["instrument"].get("username"), track["timing"]["loop_ms"],
                       [[note["step"], note["pitch"]] for note in track["notes"]]])
    return compact([INPUT_TAG, count, tracks])


def target_object(track):
    timing = track["timing"]
    return [TARGET_TAG, track["instrument"]["name"], timing["loop_ms"],
            timing["note_duration_ms"], timing["transposition_semitones"],
            [[note["step"], note["pitch"]] for note in track["notes"]]]


def encode_target(track):
    return compact(target_object(track))


def decode_target(text, requested_count=None, allowed_instruments=None):
    value = json.loads(text)
    if not isinstance(value, list) or len(value) != 6 or value[0] != TARGET_TAG:
        raise ValueError("unsupported musical target schema")
    _, instrument, loop_ms, duration, transpose, notes = value
    if not isinstance(instrument, str) or not instrument:
        raise ValueError("invalid percussion instrument")
    if allowed_instruments is not None and instrument not in allowed_instruments:
        raise ValueError("instrument has no validated preset")
    for name, number in (("loop_ms", loop_ms), ("note_duration_ms", duration)):
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"invalid {name}")
    if type(transpose) not in (int, float) or not math.isfinite(transpose):
        raise ValueError("invalid transposition")
    if not isinstance(notes, list) or not 1 <= len(notes) <= 64:
        raise ValueError("invalid note list")
    if requested_count is not None and len(notes) != requested_count:
        raise ValueError("generated note count differs from requested count")
    previous = -1
    for note in notes:
        if not isinstance(note, list) or len(note) != 2:
            raise ValueError("invalid note pair")
        step, pitch = note
        if type(step) is not int or not 0 <= step <= 63 or step <= previous:
            raise ValueError("note positions must be unique, ascending, and 0..63")
        if type(pitch) not in (int, float) or not math.isfinite(pitch) or pitch == 0:
            raise ValueError("invalid note pitch")
        previous = step
    return {"instrument": instrument, "loop_ms": loop_ms,
            "note_duration_ms": duration, "transposition_semitones": transpose,
            "notes": [{"step": step, "pitch": pitch} for step, pitch in notes]}


def reconstruct(composition, musical, presets):
    instrument = musical["instrument"]
    if instrument not in presets:
        raise ValueError("instrument has no validated preset")
    track = deepcopy(presets[instrument])
    used = {item["id"] for item in composition.tracks}
    track_id = next((item for item in range(1, 15) if item not in used), None)
    if track_id is None:
        raise ValueError("no free native track slot")
    track["id"] = track_id
    track["notes"] = deepcopy(musical["notes"])
    for key in ("loop_ms", "note_duration_ms", "transposition_semitones"):
        track["timing"][key] = musical[key]
    # Interface cursor is state, not music. The preset carries deterministic
    # sound settings; clear a stale cursor that may exceed a generated loop.
    track["playback"]["cursor"] = None
    candidate = Composition(deepcopy(composition.global_settings),
                            deepcopy(composition.tracks) + [track], composition.native_text)
    validate(candidate)
    return candidate, track
