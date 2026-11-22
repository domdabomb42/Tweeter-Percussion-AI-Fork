"""All editing helpers validate a deep copy before committing any change."""
from copy import deepcopy
from .parameter_registry import BY_NATIVE, BY_PATH, put_path, empty_track
from .validator import validate, check_parameter, ValidationError


def _commit(composition, edit):
    candidate = deepcopy(composition)
    edit(candidate)
    validate(candidate)
    composition.tracks = candidate.tracks
    composition.global_settings = candidate.global_settings
    return composition


def _set(composition, track_id, parameter, value):
    definition = BY_NATIVE.get(parameter) or BY_PATH.get(parameter)
    if definition is None:
        raise ValidationError(f"track {track_id}: unsupported parameter {parameter!r}")
    check_parameter(definition, value, f"track {track_id}.{definition.path}")
    put_path(composition.track(track_id), definition.path, value)


def set_parameters(composition, track_id, parameters):
    """Apply a mapping of native names or dotted JSON paths atomically."""
    def edit(candidate):
        for name, value in parameters.items():
            _set(candidate, track_id, name, value)
    return _commit(composition, edit)


def set_parameter(composition, track_id, parameter, value):
    return set_parameters(composition, track_id, {parameter: value})


def add_note(composition, track_id, step, pitch):
    def edit(candidate):
        track = candidate.track(track_id)
        if any(n["step"] == step for n in track["notes"]):
            raise ValidationError(f"track {track_id}, step {step}: already occupied; remove the note first")
        track["notes"].append({"step": step, "pitch": pitch})
        track["notes"].sort(key=lambda n: n["step"] if type(n["step"]) in (int, float) else -1)
    return _commit(composition, edit)


def remove_note(composition, track_id, step, pitch=None):
    def edit(candidate):
        track = candidate.track(track_id)
        matches = [n for n in track["notes"] if n["step"] == step and (pitch is None or n["pitch"] == pitch)]
        if not matches:
            raise ValidationError(f"track {track_id}, step {step}, pitch {pitch}: no matching note")
        track["notes"].remove(matches[0])
    return _commit(composition, edit)


def set_loop_length(composition, track_id, loop_ms):
    return set_parameter(composition, track_id, "loop-length", loop_ms)


def add_track(composition, track_id, parameters, notes=()):
    """Create a native part without manufacturing missing instrument defaults."""
    def edit(candidate):
        if any(t["id"] == track_id for t in candidate.tracks):
            raise ValidationError(f"track {track_id}: already exists")
        candidate.tracks.append(empty_track(track_id))
        for parameter, value in parameters.items():
            _set(candidate, track_id, parameter, value)
        candidate.tracks[-1]["notes"] = deepcopy(list(notes))
        # A new note-playing instrument needs explicit timing, not guessed defaults.
        if notes and any(candidate.tracks[-1]["timing"][k] is None for k in ("loop_ms", "note_duration_ms", "transposition_semitones")):
            raise ValidationError(f"new track {track_id}: notes require loop-length, note-dur and octave")
    return _commit(composition, edit)
