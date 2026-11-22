"""Reconstruct native coll records, updating only the requested native fields."""
from pathlib import Path
import os
import tempfile
from .models import Composition
from .parser import parse, record_text, SPACE
from .extractor import extract_text, identify
from .parameter_registry import PARAMETERS, empty_track, get_path, encode

# key-diatonic-coll.txt: ASCII keyboard key -> native stored pitch 48..84.
KEYS = "zsxdcvgbhnjm,l.;/q2w3e4rt6y7ui9o0p-[]"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _display(pitch):
    # In the native keyboard map, a digit key is a symbolic character too.
    return KEYS[int(pitch) - 48] if int(pitch) == pitch and 48 <= pitch <= 84 else "\u00a0"


def render_tweeter(composition):
    from .validator import validate
    validate(composition)
    text = composition.native_text
    records = parse(text) if text else []
    baseline = extract_text(text) if text else Composition()
    old_tracks = {t["id"]: t for t in baseline.tracks}
    positions, replacements, additions = {}, {}, []
    keys = [a.value for record in records for a in record.keys if isinstance(a.value, (int, float))]
    next_key = int(max(keys, default=-1)) + 1
    for index, record in enumerate(records):
        tid, name, values = identify(record)
        key = (tid, name)
        if name in ("tracker-data", "tracker-char"):
            key += (values[0],)
        positions.setdefault(key, []).append(index)

    def replace(key, values):
        nonlocal next_key
        indexes = positions.get(key, [])
        if values is None:
            for index in indexes:
                replacements[index] = ""
        elif indexes:
            index = indexes[-1]
            record = records[index]
            original = text[record.start:record.end]
            # Retain the record's preceding whitespace and original typed keys.
            prefix = original[:len(original) - len(original.lstrip(SPACE))]
            replacements[index] = prefix + record_text([a.value for a in record.keys], values)
        else:
            if next_key > 16777216:
                raise ValueError("cannot allocate an exactly representable native float32 coll key")
            additions.append(record_text([next_key], values))
            next_key += 1

    title = composition.global_settings["metadata"]["session_title"]
    if title != baseline.global_settings["metadata"]["session_title"]:
        replace((None, "session-title"), None if title is None else ["session-title", *title])
    for track in composition.tracks:
        tid = track["id"]
        old = old_tracks.get(tid, empty_track(tid))
        for parameter in PARAMETERS:
            value = get_path(track, parameter.path)
            if value != get_path(old, parameter.path):
                replace((tid, parameter.native), None if value is None else [tid, parameter.native, *encode(parameter, value)])
        old_notes = {note["step"]: note["pitch"] for note in old["notes"]}
        new_notes = {note["step"]: note["pitch"] for note in track["notes"]}
        for step in sorted(old_notes.keys() | new_notes.keys()):
            pitch = new_notes.get(step, 0)
            if pitch != old_notes.get(step, 0):
                # A native zero clears a note. Removing a record could resurrect
                # an earlier duplicate or leave a note in the current Pd session.
                replace((tid, "tracker-data", step), [tid, "tracker-data", step, pitch])
                replace((tid, "tracker-char", step), [tid, "tracker-char", step, _display(pitch)])
        if tid not in old_tracks:
            # Explicit rests clear any previous instrument's tracker state when
            # importing a newly constructed track into an existing application.
            for step in range(64):
                if step not in new_notes:
                    replace((tid, "tracker-data", step), [tid, "tracker-data", step, 0])
                    replace((tid, "tracker-char", step), [tid, "tracker-char", step, _display(0)])
    if not records:
        output = ""
    else:
        output = text[:records[0].start] + "".join(replacements.get(i, text[r.start:r.end]) for i, r in enumerate(records)) + text[records[-1].end:]
    if additions:
        newline = "\r\n" if "\r\n" in text else "\n"
        if output and not output.endswith(("\r", "\n")):
            output += newline
        output += newline.join(additions) + newline
    if not output.strip("\ufeff" + SPACE):
        raise ValueError("export would contain no native records")
    restored = extract_text(output)
    if restored.tracks != composition.tracks or restored.global_settings != composition.global_settings:
        raise ValueError("native export cannot represent the requested structure; a track may have lost its last native field")
    return output


def export_tweeter(composition, output_path):
    text = render_tweeter(composition)
    atomic_write(output_path, text.encode("utf-8", errors="surrogateescape"))
    return Path(output_path)
