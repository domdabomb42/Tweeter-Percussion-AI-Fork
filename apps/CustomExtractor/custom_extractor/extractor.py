"""Interpret verified Tweeter fields while retaining the entire native document."""
from pathlib import Path
from .models import Composition
from .parser import parse, FormatError
from .parameter_registry import BY_NATIVE, empty_track, decode, put_path


def identify(record):
    values = record.values
    if values[0] == "session-title":
        return None, "session-title", values[1:]
    if (len(values) >= 2 and type(values[0]) in (int, float)
            and values[0] == int(values[0]) and 1 <= values[0] <= 14
            and isinstance(values[1], str)):
        return int(values[0]), values[1], values[2:]
    return None, None, values


def extract_text(text):
    composition = Composition(native_text=text)
    tracks, slots = {}, {}
    recognized = 0
    for index, record in enumerate(parse(text)):
        track_id, name, values = identify(record)
        if name == "session-title":
            composition.global_settings["metadata"]["session_title"] = values
            recognized += 1
        elif track_id is not None:
            if track_id not in tracks:
                tracks[track_id] = empty_track(track_id)
                slots[track_id] = {}
            track = tracks[track_id]
            try:
                if name in BY_NATIVE:
                    put_path(track, BY_NATIVE[name].path, decode(BY_NATIVE[name], values))
                    recognized += 1
                elif name == "tracker-data":
                    if len(values) != 2 or any(type(x) not in (int, float) for x in values) or values[0] != int(values[0]):
                        raise ValueError("tracker-data requires integer step and numeric native pitch")
                    step, pitch = int(values[0]), values[1]
                    if not 0 <= step <= 64:
                        raise ValueError("tracker-data requires step 0..64")
                    if step < 64:
                        slots[track_id][step] = pitch
                    recognized += 1
                elif name == "tracker-char":
                    # Display data is deliberately preserved independently of notes.
                    if len(values) not in (1, 2) or type(values[0]) not in (int, float) or values[0] != int(values[0]) or not 0 <= values[0] <= 64:
                        raise ValueError("tracker-char requires step 0..64 and at most one display atom")
                    recognized += 1
            except (ValueError, TypeError) as error:
                raise FormatError(f"record {index}, track {track_id}, {name}: {error}") from error
    if not recognized:
        raise FormatError("coll file has no recognized Tweeter session fields")
    for track_id, track in tracks.items():
        track["notes"] = [{"step": step, "pitch": pitch} for step, pitch in sorted(slots[track_id].items()) if pitch]
    composition.tracks = list(tracks.values())
    return composition


def extract_tweeter(input_path):
    from .validator import validate
    text = Path(input_path).read_bytes().decode("utf-8", errors="surrogateescape")
    composition = extract_text(text)
    validate(composition)
    return composition
