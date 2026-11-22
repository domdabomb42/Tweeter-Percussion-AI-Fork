"""Feature coverage, canonical comparison and derived musical measurements."""
from copy import deepcopy
from collections import Counter
from .extractor import identify, extract_text
from .parser import parse
from .parameter_registry import BY_NATIVE, PARAMETERS, get_path
from .validator import validate


def native_diagnostics(composition):
    rows = []
    if not composition.native_text:
        return rows
    for i, record in enumerate(parse(composition.native_text)):
        tid, name, values = identify(record)
        status = BY_NATIVE[name].status if name in BY_NATIVE else "supported" if name in ("session-title", "tracker-data") else "partial" if name == "tracker-char" else "unknown_preserved"
        reason = None
        if name in ("tracker-data", "tracker-char") and values[0] == 64:
            status, reason = "partial", "stale step 64 retained as raw record; active tracker is 0..63"
        elif name == "tracker-char":
            reason = "display atoms retained; updated only when the associated note is edited"
        elif name == "overdrive-override":
            reason = "ordered live command, not an independent persistent gain stage"
        elif name in ("hover", "delete", "preset-id"):
            reason = "native control/preset metadata retained; not interpreted as automation or audio state"
        rows.append({"record": i, "track": tid, "parameter": name,
                     "status": status, "reason": reason})
    return rows


def canonical(composition):
    """Supported state plus opaque records; ignores lexical formatting/display."""
    opaque = []
    if composition.native_text:
        for record in parse(composition.native_text):
            tid, name, values = identify(record)
            if name is None or name not in BY_NATIVE and name not in ("session-title", "tracker-data", "tracker-char") or name == "tracker-data" and values[0] == 64:
                opaque.append({"keys": [a.value for a in record.keys], "values": record.values})
    return deepcopy({"global": composition.global_settings, "tracks": composition.tracks, "opaque_records": opaque})


def coverage(composition, include_records=False):
    # Render edits before reporting record coverage so newly added fields count.
    from .exporter import render_tweeter
    current = extract_text(render_tweeter(composition))
    diagnostics = native_diagnostics(current)
    counts = Counter(row["status"] for row in diagnostics)
    result = {
        "tracks": len(composition.tracks),
        "notes": sum(len(t["notes"]) for t in composition.tracks),
        "extracted_parameters": sum(get_path(t, p.path) is not None for t in composition.tracks for p in PARAMETERS),
        "native_records": len(diagnostics),
        "status_counts": dict(counts),
        "unknown_parameter_names": sorted({row["parameter"] or "<unrecognized record>" for row in diagnostics if row["status"] == "unknown_preserved"}),
        "unsupported_parameters": counts["unknown_preserved"],
        "warnings": validate(composition),
    }
    if include_records:
        result["records"] = diagnostics
    return result


def musical_features(composition):
    """Only derivations supported by this snapshot, never guessed tempo/gates."""
    results = []
    for track in composition.tracks:
        timing = track["timing"]
        loop, transpose = timing["loop_ms"], timing["transposition_semitones"]
        results.append({"id": track["id"], "name": track["instrument"]["name"],
                        "loop_ms": loop, "steps": 64,
                        "track_note_duration_ms": timing["note_duration_ms"],
                        "notes": [{**n, "onset_ms": n["step"] * loop / 64 if loop is not None else None,
                                   "sounding_pitch": n["pitch"] - 12 + transpose if transpose is not None else None}
                                  for n in track["notes"]]})
    loops = {t["timing"]["loop_ms"] for t in composition.tracks}
    return {"tempo": None, "meter": None, "global_loop_ms": None,
            "common_track_loop_ms": next(iter(loops)) if len(loops) == 1 else None,
            "velocity": None, "per_note_gate_duration": None, "automation": None,
            "tracks": results}
