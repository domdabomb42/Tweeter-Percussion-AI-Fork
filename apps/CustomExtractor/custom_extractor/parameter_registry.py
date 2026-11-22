"""Native names and bounds from L2OrkTweet.pd, not MIDI ranges.

UI bounds govern new edits. Historical out-of-UI values remain lossless and
are reported; envelopes are native sample arrays with no enforced amplitude
clipping. Unspecified controls are absent, never filled with patch defaults.
"""
from dataclasses import dataclass
from copy import deepcopy

WAVEFORMS = ("sine", "sawtooth", "triangle", "square", "pink_noise")


@dataclass(frozen=True)
class Parameter:
    native: str
    path: str
    kind: str = "number"
    minimum: float | None = None
    maximum: float | None = None
    unit: str = "native"
    status: str = "supported"


PARAMETERS = [
    Parameter("carrier", "synth.carrier", "waveform", 0, 4),
    Parameter("modulator", "synth.modulator", "waveform", 0, 4),
    Parameter("ae", "synth.carrier_envelope", "array", unit="native sample amplitude"),
    Parameter("me", "synth.modulator_envelope", "array", unit="native sample amplitude"),
    Parameter("harmonicity", "synth.harmonicity", minimum=0, maximum=99999, unit="ratio"),
    Parameter("mod-amp", "synth.mod_amplitude", minimum=1, maximum=9999),
    Parameter("note-dur", "timing.note_duration_ms", minimum=1, maximum=1e37, unit="ms; track-wide envelope scan"),
    Parameter("octave", "timing.transposition_semitones", minimum=-192, maximum=192, unit="semitones"),
    Parameter("overdrive", "effects.overdrive.level", minimum=0.1, maximum=10),
    Parameter("overdrive-override", "effects.overdrive.override", minimum=0.1, maximum=10, status="partial"),
    Parameter("inst-name", "instrument.name", "symbol"),
    Parameter("reverb-amp", "effects.reverb.amplitude", minimum=0, maximum=1),
    Parameter("reverb-type", "effects.reverb.type", "integer", 0, 3),
    Parameter("echo-amp", "effects.echo.amplitude", minimum=0, maximum=1),
    Parameter("echo-fb", "effects.echo.feedback", minimum=0, maximum=0.999),
    Parameter("echo-ms", "effects.echo.delay_ms", minimum=0.001, maximum=9999, unit="ms; zero also accepted (native default)"),
    Parameter("flanger-amp", "effects.flanger.amplitude", minimum=0, maximum=1),
    Parameter("flanger-hz", "effects.flanger.frequency_hz", minimum=0.001, maximum=100, unit="Hz"),
    Parameter("flanger-mod-amp", "effects.flanger.mod_amplitude", minimum=0, maximum=100),
    Parameter("flanger-ms", "effects.flanger.delay_ms", minimum=0.001, maximum=50, unit="ms"),
    Parameter("dry-amp", "mixing.dry_amplitude", minimum=0, maximum=1),
    Parameter("out-amp", "mixing.output_amplitude", minimum=0, maximum=100, unit="native Pd dB scale"),
    Parameter("out-pan", "mixing.pan", minimum=0, maximum=1.5708, unit="radians; 0=left, 1.5708=right"),
    Parameter("loop-length", "timing.loop_ms", minimum=500, maximum=1e7, unit="ms"),
    Parameter("loop", "playback.loop_enabled", "integer", 0, 1),
    Parameter("solo", "playback.solo", "integer", 0, 1),
    Parameter("tracker", "playback.tracker_enabled", "integer", 0, 1),
    Parameter("keyboard", "playback.keyboard_enabled", "integer", 0, 1),
    Parameter("insert", "playback.insert", "integer", 0, 1),
    Parameter("delete", "playback.delete", "integer", 0, 3, status="partial"),
    Parameter("hover", "playback.hover", "number", status="partial"),
    Parameter("tracker-time", "playback.cursor", "cursor", unit="[next step, remaining ms, running]"),
    Parameter("username", "instrument.username", "symbol"),
]
BY_NATIVE = {p.native: p for p in PARAMETERS}
BY_PATH = {p.path: p for p in PARAMETERS}


def get_path(obj, path):
    for key in path.split("."):
        obj = obj[key]
    return obj


def put_path(obj, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = deepcopy(value)


def empty_track(track_id):
    result = {"id": track_id}
    for parameter in PARAMETERS:
        put_path(result, parameter.path, None)
    result["notes"] = []
    return result


def decode(parameter, values):
    kind = parameter.kind
    if kind == "symbol":
        # net-io-sym feeds [tosymbol], supporting historical unescaped words.
        # Preserve exact source atom boundaries in the native document.
        if not values:
            raise ValueError(f"{parameter.native} requires a name")
        return " ".join(str(v) for v in values)
    if kind in ("array", "atoms"):
        return list(values)
    if kind == "cursor":
        if len(values) != 3:
            raise ValueError("tracker-time requires three values")
        return dict(zip(("next_step", "remaining_ms", "running"), values))
    if len(values) != 1:
        raise ValueError(f"{parameter.native} requires one value")
    value = values[0]
    if kind == "waveform":
        if not isinstance(value, (int, float)) or value != int(value) or not 0 <= value <= 4:
            raise ValueError(f"invalid waveform code {value!r}")
        return WAVEFORMS[int(value)]
    return value


def encode(parameter, value):
    if parameter.kind == "waveform":
        return [WAVEFORMS.index(value)]
    if parameter.kind in ("array", "atoms"):
        return list(value)
    if parameter.kind == "cursor":
        return [value[k] for k in ("next_step", "remaining_ms", "running")]
    return [value]
