"""Versioned V4EA targets; the full musical target remains readable."""
import hashlib

from . import (END_MARKER, POSITIONS_END_MARKER, POSITIONS_MARKER,
               POSITIONS_VERSION, TARGET_MARKER, VERSION)
from .music import SCHEMA_SHA256, encode_input, encode_target, decode_target
from percussion_generation_v3.data import parse_positions, serialize_positions

POSITIONS_SCHEMA_SHA256 = hashlib.sha256(
    (SCHEMA_SHA256 + ":v4ea-positions-v1:V4EA-MUSIC-INPUT-3+"
     "RHYTHM_TARGET=count+ascending decimal steps 0..63 separated by spaces+"
     "END_RHYTHM").encode("utf-8")
).hexdigest()


def version_for(target_format):
    if target_format == "positions-v1":
        return POSITIONS_VERSION
    if target_format == "musical-v2":
        return VERSION
    raise ValueError("unsupported V4EA target format")


def format_for_version(version):
    if version == POSITIONS_VERSION:
        return "positions-v1"
    if version == VERSION:
        return "musical-v2"
    raise ValueError("unsupported V4EA target version")


def schema_for(target_format):
    if target_format == "positions-v1":
        return POSITIONS_SCHEMA_SHA256
    if target_format == "musical-v2":
        return SCHEMA_SHA256
    raise ValueError("unsupported V4EA target format")


def end_marker_for(target_format):
    if target_format == "positions-v1":
        return POSITIONS_END_MARKER
    if target_format == "musical-v2":
        return END_MARKER
    raise ValueError("unsupported V4EA target format")


def prompt(composition, count, target_format="positions-v1"):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("target note count must be 1..64")
    prefix = encode_input(composition, count)
    if target_format == "positions-v1":
        return prefix + POSITIONS_MARKER + str(count) + "\n"
    if target_format == "musical-v2":
        return prefix + TARGET_MARKER
    raise ValueError("unsupported V4EA target format")


def encode(track, target_format):
    if target_format == "positions-v1":
        return serialize_positions([note["step"] for note in track["notes"]])
    if target_format == "musical-v2":
        return encode_target(track) + "\n" + END_MARKER
    raise ValueError("unsupported V4EA target format")


def decode(text, count, target_format, allowed_instruments=None):
    if target_format == "positions-v1":
        return parse_positions(text, requested_count=count)
    if target_format == "musical-v2":
        return decode_target(text.removesuffix("\n" + END_MARKER), count,
                             allowed_instruments)
    raise ValueError("unsupported V4EA target format")
