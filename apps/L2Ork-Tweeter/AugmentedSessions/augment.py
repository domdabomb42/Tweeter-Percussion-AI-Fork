"""Exact, time-synchronized rotations of native Tweeter tracker sessions.

Run from this directory with ``python augment.py`` or ``python augment.py --dry-run``.
Only the third payload atom (the tracker slot) in native tracker-data and
tracker-char records is rewritten. CustomExtractor parses and validates both
the source and every exported result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
APPS = HERE.parent.parent
sys.path.insert(0, str(APPS / "CustomExtractor"))

from custom_extractor import extract_tweeter, validate  # noqa: E402
from custom_extractor.exporter import atomic_write  # noqa: E402
from custom_extractor.extractor import identify  # noqa: E402
from custom_extractor.parser import parse, SPACE  # noqa: E402

SOURCE_DIR = HERE.parent / "sessions"
OUTPUT_DIR = HERE / "generated"
MANIFEST = HERE / "manifest.json"
SLOTS = 64
POSITIONAL = frozenset(("tracker-data", "tracker-char"))


def exact_number(value: int | float) -> Fraction:
    """Treat a native decimal as decimal, never as a binary float ratio."""
    return Fraction(Decimal(str(value)))


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def native_loop_lengths(records, tracks) -> dict[int, Fraction | None]:
    """Use the last saved loop-length lexeme, before float decoding can round it."""
    loops = {track["id"]: None for track in tracks}
    for record in records:
        track_id, name, _ = identify(record)
        if track_id in loops and name == "loop-length":
            loops[track_id] = Fraction(Decimal(record.atoms[2].raw))
    return loops


def payload_atom_span(record_text: str, number: int) -> tuple[int, int]:
    """Find an atom after the coll comma, respecting Pd backslash escapes."""
    comma = None
    index = 0
    while index < len(record_text):
        char = record_text[index]
        if char == "\\":
            index += 2
        elif char == ",":
            comma = index
            break
        else:
            index += 1
    if comma is None:
        raise ValueError("coll record lacks a comma")
    index = comma + 1
    for atom_number in range(number + 1):
        while index < len(record_text) and record_text[index] in SPACE:
            index += 1
        start = index
        while index < len(record_text):
            char = record_text[index]
            if char == "\\":
                index += 2
            elif char in SPACE + ",;":
                break
            else:
                index += 1
        if start == index:
            raise ValueError(f"coll payload lacks atom {number}")
        if atom_number == number:
            return start, index
    raise AssertionError("unreachable")


def source_identifier(relative_path: Path, repeated_stems: set[str]) -> str:
    stem = relative_path.stem
    if stem not in repeated_stems:
        return stem
    suffix = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{stem}--{suffix}"


def plan_offsets(composition, loops=None) -> tuple[Fraction | None, list[dict], list[dict]]:
    """Return the longest loop and exact per-track shifts for 64 candidates."""
    tracks = composition.tracks
    if loops is None:
        loops = {t["id"]: (None if t["timing"]["loop_ms"] is None
                           else exact_number(t["timing"]["loop_ms"])) for t in tracks}
    if any(loop is not None and loop <= 0 for loop in loops.values()):
        raise ValueError("nonpositive loop length")
    reference = max((loop for loop in loops.values() if loop is not None), default=None)
    valid, skipped = [], []
    for offset in range(SLOTS):
        time_shift = Fraction(offset * reference, SLOTS) if reference is not None else Fraction(0)
        shifts: dict[int, int] = {}
        reasons = []
        if offset and reference is None:
            reasons.append({"track_id": None, "reason": "no valid reference loop-length"})
        for track in tracks:
            track_id = track["id"]
            loop = loops[track_id]
            if offset == 0:
                shifts[track_id] = 0
            elif loop is None:
                reasons.append({"track_id": track_id, "reason": "missing loop-length"})
            else:
                shift = Fraction(offset * reference, loop)
                if shift.denominator != 1:
                    reasons.append({"track_id": track_id, "reason": "fractional tracker shift",
                                    "required_steps": fraction_text(shift)})
                else:
                    shifts[track_id] = int(shift)
        if reasons:
            skipped.append({"reference_offset": offset, "time_shift_ms_exact": fraction_text(time_shift),
                            "reasons": reasons})
        else:
            valid.append({"reference_offset": offset, "time_shift_ms": float(time_shift),
                          "time_shift_ms_exact": fraction_text(time_shift), "shifts": shifts})
    return reference, valid, skipped


def rotate_native(text: str, records, shifts: dict[int, int]) -> str:
    """Rotate every explicit data/display slot, including zero-valued rests.

    Native coll keys are record identifiers, not tracker positions. Step 64 is
    historical out-of-range data and is preserved unchanged.
    """
    pieces = [text[:records[0].start]]
    for record in records:
        original = text[record.start:record.end]
        track_id, name, values = identify(record)
        if track_id in shifts and name in POSITIONAL and values and 0 <= values[0] < SLOTS:
            if type(values[0]) is not int:
                raise ValueError("tracker slot must be an integer")
            shift = shifts[track_id] % SLOTS
            if shift:
                start, end = payload_atom_span(original, 2)
                if original[start:end] != record.atoms[2].raw:
                    raise ValueError("tracker slot atom location mismatch")
                original = original[:start] + str((values[0] - shift) % SLOTS) + original[end:]
        pieces.append(original)
    pieces.append(text[records[-1].end:])
    return "".join(pieces)


def validate_rotation(source, result, original_records, output_records,
                      shifts: dict[int, int], time_shift: Fraction, loops=None) -> None:
    """Check native records, structured state, phase, and inverse rotation."""
    validate(result)
    if source.global_settings != result.global_settings:
        raise ValueError("global settings changed")
    if len(source.tracks) != len(result.tracks):
        raise ValueError("track count changed")
    if len(original_records) != len(output_records):
        raise ValueError("native record count changed")
    for before, after in zip(source.tracks, result.tracks):
        track_id = before["id"]
        if track_id != after["id"]:
            raise ValueError("track assignment/order changed")
        old_other = {key: value for key, value in before.items() if key != "notes"}
        new_other = {key: value for key, value in after.items() if key != "notes"}
        if old_other != new_other:
            raise ValueError(f"track {track_id}: non-note setting changed")
        shift = shifts[track_id]
        expected = sorted(((note["step"] - shift) % SLOTS, note["pitch"]) for note in before["notes"])
        actual = sorted((note["step"], note["pitch"]) for note in after["notes"])
        if actual != expected:
            raise ValueError(f"track {track_id}: note rotation, pitch, or count changed")
        if Counter(p for _, p in actual) != Counter(p for _, p in expected):
            raise ValueError(f"track {track_id}: pitch multiset changed")
        if sorted(((step + shift) % SLOTS, pitch) for step, pitch in actual) != sorted(
                (note["step"], note["pitch"]) for note in before["notes"]):
            raise ValueError(f"track {track_id}: inverse rotation failed")
        loop_ms = (loops[track_id] if loops is not None else
                   None if before["timing"]["loop_ms"] is None else exact_number(before["timing"]["loop_ms"]))
        if loop_ms is not None and Fraction(shift * loop_ms, SLOTS) % loop_ms != time_shift % loop_ms:
            raise ValueError(f"track {track_id}: actual time shift differs")
        if any(not 0 <= note["step"] < SLOTS for note in after["notes"]):
            raise ValueError(f"track {track_id}: tracker slot out of range")
    for old_record, new_record in zip(original_records, output_records):
        old_id, old_name, old_values = identify(old_record)
        new_id, new_name, new_values = identify(new_record)
        if old_record.keys != new_record.keys or (old_id, old_name) != (new_id, new_name):
            raise ValueError("native record identity changed")
        if old_id in shifts and old_name in POSITIONAL and old_values and 0 <= old_values[0] < SLOTS:
            if new_values[0] != (old_values[0] - shifts[old_id]) % SLOTS or new_values[1:] != old_values[1:]:
                raise ValueError("native note/rest/display slot changed incorrectly")
        elif old_record != new_record:
            # Dataclass offsets change when earlier record lengths change.
            if old_record.keys != new_record.keys or old_record.atoms != new_record.atoms:
                raise ValueError("unrelated native record changed")


def generate(sessions_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR,
             manifest_path: Path = MANIFEST, dry_run: bool = False) -> dict:
    sources = sorted(sessions_dir.rglob("*.txt"), key=lambda p: p.relative_to(sessions_dir).as_posix())
    if not sources:
        raise ValueError(f"no source .txt sessions under {sessions_dir}")
    stems = Counter(path.stem for path in sources)
    repeated = {stem for stem, count in stems.items() if count > 1}
    manifest = {"schema_version": 1, "source_directory": str(sessions_dir.resolve()),
                "algorithm": "exact common-time shift; per-track steps = offset * longest_loop_ms / track_loop_ms",
                "tracker_steps": SLOTS, "dry_run": dry_run, "sources": [], "generated": [],
                "skipped": [], "validation_failures": []}
    original_hashes = {}
    for source_path in sources:
        relative = source_path.relative_to(sessions_dir)
        identifier = source_identifier(relative, repeated)
        original_bytes = source_path.read_bytes()
        original_hashes[source_path] = hashlib.sha256(original_bytes).hexdigest()
        source = extract_tweeter(source_path)
        original_records = parse(source.native_text)
        loops = native_loop_lengths(original_records, source.tracks)
        reference, candidates, skipped = plan_offsets(source, loops)
        loop_details = [{"track_id": track["id"], "loop_ms": track["timing"]["loop_ms"],
                         "loop_ms_exact": None if loops[track["id"]] is None else fraction_text(loops[track["id"]])}
                        for track in source.tracks]
        manifest["sources"].append({"original_session": source_path.name,
                                    "source_relative_path": relative.as_posix(),
                                    "source_identifier": identifier,
                                    "source_sha256": original_hashes[source_path],
                                    "reference_loop_ms": None if reference is None else float(reference),
                                    "reference_loop_ms_exact": None if reference is None else fraction_text(reference),
                                    "track_loops": loop_details,
                                    "valid_offsets": len(candidates), "skipped_offsets": len(skipped)})
        for item in skipped:
            manifest["skipped"].append({"original_session": source_path.name,
                                        "source_identifier": identifier, **item})
        for candidate in candidates:
            offset = candidate["reference_offset"]
            destination = output_dir / identifier / f"{identifier}_shift_{offset:02d}.txt"
            shifts = candidate["shifts"]
            track_details = [{"track_id": track["id"], "loop_ms": track["timing"]["loop_ms"],
                              "loop_ms_exact": None if loops[track["id"]] is None else fraction_text(loops[track["id"]]),
                              "tracker_shift": shifts[track["id"]]}
                             for track in source.tracks]
            entry = {"original_session": source_path.name, "source_relative_path": relative.as_posix(),
                     "source_identifier": identifier,
                     "augmented_filename": destination.relative_to(output_dir).as_posix(),
                     "reference_loop_ms": None if reference is None else float(reference),
                     "reference_loop_ms_exact": None if reference is None else fraction_text(reference),
                     "reference_offset": offset, "time_shift_ms": candidate["time_shift_ms"],
                     "time_shift_ms_exact": candidate["time_shift_ms_exact"],
                     "tracks": track_details}
            if dry_run:
                entry["validation"] = "not run (dry run)"
                manifest["generated"].append(entry)
                continue
            temporary = destination.with_name("." + destination.name + ".validating")
            try:
                output_text = (source.native_text if offset == 0 else
                               rotate_native(source.native_text, original_records, shifts))
                output_bytes = original_bytes if offset == 0 else output_text.encode("utf-8", errors="surrogateescape")
                # Stage and reimport before replacing a previously valid output.
                atomic_write(temporary, output_bytes)
                imported = extract_tweeter(temporary)
                output_records = parse(imported.native_text)
                validate_rotation(source, imported, original_records, output_records,
                                  shifts, Fraction(candidate["time_shift_ms_exact"]), loops)
                if temporary.read_bytes() != output_bytes:
                    raise ValueError("export/reimport changed information")
                if offset == 0 and output_bytes != original_bytes:
                    raise ValueError("offset zero is not byte-identical")
                os.replace(temporary, destination)
                entry["validation"] = "passed"
                entry["sha256"] = hashlib.sha256(output_bytes).hexdigest()
                manifest["generated"].append(entry)
            except Exception as error:
                temporary.unlink(missing_ok=True)
                manifest["validation_failures"].append({"original_session": source_path.name,
                                                        "reference_offset": offset, "error": str(error)})
        print(f"{source_path.name}: {len(candidates)} valid offsets, {len(skipped)} skipped", flush=True)
    for source_path, initial_hash in original_hashes.items():
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != initial_hash:
            raise RuntimeError(f"original source changed: {source_path}")
    manifest["summary"] = {"original_sessions": len(sources),
                           "expected_candidates": len(sources) * SLOTS,
                           "valid_candidates": len(manifest["generated"]) + len(manifest["validation_failures"]),
                           "successfully_generated": 0 if dry_run else len(manifest["generated"]),
                           "skipped_offsets": len(manifest["skipped"]),
                           "validation_failures": len(manifest["validation_failures"])}
    if not dry_run:
        atomic_write(manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8"))
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count exact offsets without writing files")
    args = parser.parse_args()
    result = generate(dry_run=args.dry_run)
    return 0 if not result["validation_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
