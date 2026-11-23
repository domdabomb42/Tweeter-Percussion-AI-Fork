"""Versioned, exact positional serialization of V3 conditioning data.

Each object is [presence_bitset, values in SCHEMAS order, optional [-1,
[[unknown_key, unmodified_value], ...]]]. A zero bit means absent; a set bit
with a null value means explicitly null. Envelope bank entries are tagged
R (raw), L (run length), D (dictionary), or Z (zlib/Base64). No numbers are
rounded or normalized. The schema hash is stored with prepared datasets.
"""
import base64
import hashlib
import json
import zlib
from functools import lru_cache

VERSION = "lossless-v2"
MAGIC = "L2V2"
SCHEMAS = {
    "global": ("tracker_steps", "metadata"),
    "metadata": ("session_title",),
    "track": ("id", "synth", "timing", "effects", "instrument", "mixing", "playback", "notes"),
    "synth": ("carrier", "modulator", "carrier_envelope", "modulator_envelope", "harmonicity", "mod_amplitude"),
    "timing": ("note_duration_ms", "transposition_semitones", "loop_ms"),
    "effects": ("overdrive", "reverb", "echo", "flanger"),
    "overdrive": ("level", "override"),
    "reverb": ("amplitude", "type"),
    "echo": ("amplitude", "feedback", "delay_ms"),
    "flanger": ("amplitude", "frequency_hz", "mod_amplitude", "delay_ms"),
    "instrument": ("name", "username"),
    "mixing": ("dry_amplitude", "output_amplitude", "pan"),
    "playback": ("loop_enabled", "solo", "tracker_enabled", "keyboard_enabled", "insert", "delete", "hover", "cursor"),
    "cursor": ("next_step", "remaining_ms", "running"),
    "note": ("step", "pitch"),
}
CHILD = {"global": {"metadata": "metadata"}, "track": {
    "synth": "synth", "timing": "timing", "effects": "effects",
    "instrument": "instrument", "mixing": "mixing", "playback": "playback",
    "notes": "note"}, "effects": {k: k for k in SCHEMAS["effects"]},
    "playback": {"cursor": "cursor"}}
SCHEMA_SHA256 = hashlib.sha256(json.dumps({"fields": SCHEMAS, "children": CHILD,
                                           "envelope_tags": ["R", "L", "D", "Z"],
                                           "magic": MAGIC}, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def envelope_variants(values):
    """Return fully reversible candidate bank entries, including numeric types."""
    return _envelope_variants_text(dumps(values))


@lru_cache(maxsize=4096)
def _envelope_variants_text(array_text):
    values = json.loads(array_text)
    raw = ["R", values]
    runs = []
    for item in values:
        if runs and dumps(runs[-1][0]) == dumps(item):
            runs[-1][1] += 1
        else:
            runs.append([item, 1])
    dictionary, indexes, seen = [], [], {}
    for item in values:
        key = dumps(item)
        if key not in seen:
            seen[key] = len(dictionary)
            dictionary.append(item)
        indexes.append(seen[key])
    zipped = base64.b64encode(zlib.compress(array_text.encode(), 9)).decode("ascii")
    return {"raw": raw, "rle": ["L", runs],
            "dictionary": ["D", dictionary, indexes], "zlib": ["Z", zipped]}


def _restore_envelope(entry):
    tag = entry[0]
    if tag == "R":
        return entry[1]
    if tag == "L":
        return [value for value, count in entry[1] for _ in range(count)]
    if tag == "D":
        return [entry[1][index] for index in entry[2]]
    if tag == "Z":
        return json.loads(zlib.decompress(base64.b64decode(entry[1])).decode())
    raise ValueError("unknown envelope encoding tag")


def _encode_object(obj, schema, bank, indexes, envelope_mode):
    if not isinstance(obj, dict):
        return obj
    fields = SCHEMAS[schema]
    mask = sum(1 << i for i, key in enumerate(fields) if key in obj)
    result = [mask]
    for key in fields:
        if key not in obj:
            continue
        value = obj[key]
        child = CHILD.get(schema, {}).get(key)
        if schema == "synth" and key in ("carrier_envelope", "modulator_envelope") and isinstance(value, list):
            serialized = dumps(value)
            if serialized not in indexes:
                indexes[serialized] = len(bank)
                bank.append(envelope_variants(value)[envelope_mode])
            value = ["E", indexes[serialized]]
        elif child and isinstance(value, list) and key == "notes":
            value = [_encode_object(item, child, bank, indexes, envelope_mode) for item in value]
        elif child:
            value = _encode_object(value, child, bank, indexes, envelope_mode)
        result.append(value)
    extras = [[key, value] for key, value in obj.items() if key not in fields]
    if extras:
        result.append([-1, extras])
    return result


def _decode_object(packed, schema, bank):
    if not isinstance(packed, list):
        return packed
    fields = SCHEMAS[schema]
    mask = packed[0]
    if type(mask) is not int or mask < 0 or mask >= (1 << len(fields)):
        raise ValueError(f"invalid {schema} presence mask")
    result, position = {}, 1
    for i, key in enumerate(fields):
        if not mask & (1 << i):
            continue
        value = packed[position]
        position += 1
        child = CHILD.get(schema, {}).get(key)
        if schema == "synth" and key in ("carrier_envelope", "modulator_envelope") and isinstance(value, list) and value[:1] == ["E"]:
            value = bank[value[1]]
        elif child and isinstance(value, list) and key == "notes":
            value = [_decode_object(item, child, bank) for item in value]
        elif child:
            value = _decode_object(value, child, bank)
        result[key] = value
    if position < len(packed):
        extension = packed[position]
        if len(packed) != position + 1 or extension[0] != -1:
            raise ValueError(f"invalid {schema} extension")
        for key, value in extension[1]:
            if key in result or key in fields:
                raise ValueError(f"duplicate {schema} key")
            result[key] = value
    return result


def encode_v2(condition, envelope_mode="zlib"):
    if envelope_mode not in ("raw", "rle", "dictionary", "zlib"):
        raise ValueError("unknown envelope mode")
    if not isinstance(condition, dict) or "global" not in condition or "tracks" not in condition:
        raise ValueError("condition requires global and tracks")
    bank, indexes = [], {}
    tracks = [[entry["role"], _encode_object(entry["track"], "track", bank, indexes, envelope_mode),
               *([[-1, [[k, v] for k, v in entry.items() if k not in ("role", "track")]]]
                 if any(k not in ("role", "track") for k in entry) else [])]
              for entry in condition["tracks"]]
    extras = [[key, value] for key, value in condition.items() if key not in ("global", "tracks")]
    packed = [MAGIC, _encode_object(condition["global"], "global", bank, indexes, envelope_mode), bank, tracks]
    if extras:
        packed.append([-1, extras])
    return dumps(packed)


def decode_v2(text):
    packed = json.loads(text) if isinstance(text, str) else text
    if not isinstance(packed, list) or len(packed) not in (4, 5) or packed[0] != MAGIC:
        raise ValueError("unknown lossless codec schema version")
    bank = [_restore_envelope(entry) for entry in packed[2]]
    result = {"global": _decode_object(packed[1], "global", bank), "tracks": []}
    for entry in packed[3]:
        if len(entry) not in (2, 3):
            raise ValueError("invalid track entry")
        item = {"role": entry[0], "track": _decode_object(entry[1], "track", bank)}
        if len(entry) == 3:
            if entry[2][0] != -1:
                raise ValueError("invalid track extension")
            item.update(entry[2][1])
        result["tracks"].append(item)
    if len(packed) == 5:
        if packed[4][0] != -1:
            raise ValueError("invalid condition extension")
        result.update(packed[4][1])
    return result
