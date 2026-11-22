# Native parameter registry

Bounds below apply to new edits. Missing values remain null. Unchanged historical values outside current UI bounds are retained with diagnostics. Source: `../L2Ork-Tweeter/core/L2OrkTweet.pd` and its net-io/array abstractions.

| Native name | Structured path | Type / edit range | Units / interpretation | Coverage |
| --- | --- | --- | --- | --- |
| `carrier` | `synth.carrier` | sine, sawtooth, triangle, square, pink_noise | native | supported |
| `modulator` | `synth.modulator` | sine, sawtooth, triangle, square, pink_noise | native | supported |
| `ae` | `synth.carrier_envelope` | nonempty finite numeric sample array; no clipping | native sample amplitude | supported |
| `me` | `synth.modulator_envelope` | nonempty finite numeric sample array; no clipping | native sample amplitude | supported |
| `harmonicity` | `synth.harmonicity` | number: 0..99999 | ratio | supported |
| `mod-amp` | `synth.mod_amplitude` | number: 1..9999 | native | supported |
| `note-dur` | `timing.note_duration_ms` | number: 1..1e+37 | ms; track-wide envelope scan | supported |
| `octave` | `timing.transposition_semitones` | number: -192..192 | semitones | supported |
| `overdrive` | `effects.overdrive.level` | number: 0.1..10 | native | supported |
| `overdrive-override` | `effects.overdrive.override` | number: 0.1..10 | native | partial |
| `inst-name` | `instrument.name` | 1..999 UTF-8 bytes; no NUL | native | supported |
| `reverb-amp` | `effects.reverb.amplitude` | number: 0..1 | native | supported |
| `reverb-type` | `effects.reverb.type` | integer: 0..3 | native | supported |
| `echo-amp` | `effects.echo.amplitude` | number: 0..1 | native | supported |
| `echo-fb` | `effects.echo.feedback` | number: 0..0.999 | native | supported |
| `echo-ms` | `effects.echo.delay_ms` | number: 0.001..9999 | ms; zero also accepted (native default) | supported |
| `flanger-amp` | `effects.flanger.amplitude` | number: 0..1 | native | supported |
| `flanger-hz` | `effects.flanger.frequency_hz` | number: 0.001..100 | Hz | supported |
| `flanger-mod-amp` | `effects.flanger.mod_amplitude` | number: 0..100 | native | supported |
| `flanger-ms` | `effects.flanger.delay_ms` | number: 0.001..50 | ms | supported |
| `dry-amp` | `mixing.dry_amplitude` | number: 0..1 | native | supported |
| `out-amp` | `mixing.output_amplitude` | number: 0..100 | native Pd dB scale | supported |
| `out-pan` | `mixing.pan` | number: 0..1.5708 | radians; 0=left, 1.5708=right | supported |
| `loop-length` | `timing.loop_ms` | number: 500..10000000.0 | ms | supported |
| `loop` | `playback.loop_enabled` | integer: 0..1 | native | supported |
| `solo` | `playback.solo` | integer: 0..1 | native | supported |
| `tracker` | `playback.tracker_enabled` | integer: 0..1 | native | supported |
| `keyboard` | `playback.keyboard_enabled` | integer: 0..1 | native | supported |
| `insert` | `playback.insert` | integer: 0..1 | native | supported |
| `delete` | `playback.delete` | integer: 0..3 | native | partial |
| `hover` | `playback.hover` | finite native float32-range number; no verified narrower bound | native | partial |
| `tracker-time` | `playback.cursor` | next_step 0..64, remaining_ms >= 0, running 0/1 | [next step, remaining ms, running] | supported |
| `username` | `instrument.username` | 1..999 UTF-8 bytes; no NUL | native | supported |

Waveform native codes are 0=sine, 1=sawtooth, 2=triangle, 3=square, 4=pink noise. Both oscillators use this mapping.

`echo-ms` additionally accepts 0, the explicit native net-io default, even though the UI minimum is 0.001. Overdrive UI range is 0.1..10; historical 0 and 11 values are preserved and flagged. `delete` encodes native key/editor modes (0..3), not a boolean. `hover` is retained as a partially interpreted native numeric control.

Special records: `session-title` is a global ordered atom list; `tracker-data` is [step 0..63, finite numeric native stored pitch, without a MIDI range limit] (0=rest); `tracker-char` is display metadata (including legacy empty cells); `tracker-time` is a saved synchronization cursor. Stale step 64 is preserved with a warning and excluded from active notes.

There is no verified `preset-id` field in the inspected saved sessions/receivers. Such future or third-party fields are unknown-preserved rather than invented editable synthesis controls. No native tempo, meter, per-note velocity or timestamped automation is asserted.

Array lengths are preserved exactly. Short arrays are partial native table updates, with a diagnostic; they are not interpreted as ADSR. Native table shape and extra reporting samples are explained in README.md. Numeric source values retain textual precision; the actual Pd engine operates at float32 precision.
