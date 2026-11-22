"""Lossless prompt packing of CustomExtractor tracks for the 8k context.

All saved feature values survive. Names, notes, timing, synth scalar values,
effects, and mixing are visible; envelope sample arrays are compressed and
deduplicated within the prompt. The fixed alias table is part of the dataset
format and does not change CustomExtractor's native import/export.
"""
import base64
import json
import zlib

ALIASES = dict(zip(
    ("amplitude", "carrier", "carrier_envelope", "cursor", "delay_ms",
     "delete", "dry_amplitude", "echo", "effects", "feedback", "flanger",
     "frequency_hz", "harmonicity", "hover", "id", "insert", "instrument",
     "keyboard_enabled", "level", "loop_enabled", "loop_ms", "mixing",
     "mod_amplitude", "modulator", "modulator_envelope", "name",
     "next_step", "note_duration_ms", "notes", "output_amplitude",
     "overdrive", "pan", "pitch", "playback", "remaining_ms", "reverb",
     "running", "solo", "step", "synth", "timing",
     "transposition_semitones", "type", "username", "metadata",
     "session_title", "tracker_steps"),
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"))
REVERSE = {v: k for k, v in ALIASES.items()}
assert len(ALIASES) == len(REVERSE)


def _pack_keys(value):
    if isinstance(value, dict):
        return {ALIASES[key]: _pack_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_keys(item) for item in value]
    return value


def _unpack_keys(value):
    if isinstance(value, dict):
        return {REVERSE[key]: _unpack_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_keys(item) for item in value]
    return value


def pack_condition(condition):
    bank, indexes, tracks = [], {}, []
    for entry in condition["tracks"]:
        track = json.loads(json.dumps(entry["track"], ensure_ascii=True, allow_nan=False))
        track["notes"] = [[n["step"], n["pitch"]] for n in track["notes"]]
        for key in ("carrier_envelope", "modulator_envelope"):
            if key in track.get("synth", {}):
                array_text = json.dumps(track["synth"][key], ensure_ascii=True,
                                        allow_nan=False, separators=(",", ":"))
                if array_text not in indexes:
                    indexes[array_text] = len(bank)
                    bank.append(base64.b64encode(zlib.compress(array_text.encode(), 9)).decode())
                track["synth"][key] = indexes[array_text]
        tracks.append({"r": entry["role"], "t": _pack_keys(track)})
    return {"g": _pack_keys(condition["global"]), "b": bank, "x": tracks}


def unpack_condition(packed):
    arrays = [json.loads(zlib.decompress(base64.b64decode(value)).decode()) for value in packed["b"]]
    tracks = []
    for entry in packed["x"]:
        track = _unpack_keys(entry["t"])
        track["notes"] = [{"step": step, "pitch": pitch} for step, pitch in track["notes"]]
        for key in ("carrier_envelope", "modulator_envelope"):
            if key in track.get("synth", {}):
                track["synth"][key] = arrays[track["synth"][key]]
        tracks.append({"role": entry["r"], "track": track})
    return {"global": _unpack_keys(packed["g"]), "tracks": tracks}
