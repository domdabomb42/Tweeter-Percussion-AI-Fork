# CustomExtractor

## Research problem and approach

Native Pd-L2Ork Tweeter sessions store notes, synthesis settings and effects as
ordered `coll` records. A useful feature representation must expose those values
for analysis and editing while keeping records that it cannot interpret.
CustomExtractor converts a native `.txt` session to editable JSON and exports it
back to a native session. It is intended for lossless session inspection, feature
research and controlled edits; it does not synthesize audio or infer missing
musical information. No MIDI conversion, machine learning or external Python
packages are used.

**Recorded results on the 51 published sessions:** 51/51 byte-identical
TXT-to-JSON-to-TXT round trips, 51/51 equal extracted structures and 51/51
headless Pd loader checks. These files contain 629 tracks and 6,408 active notes.
See [RESULTS.md](reports/RESULTS.md) and [verification.json](reports/verification.json)
for the measurements and their limits.

## Workflow

`Tweeter .txt` → `extract_tweeter()` → editable `Composition` / JSON →
`set_parameters()` and note edits → `validate()` → `export_tweeter()` →
native Tweeter `.txt`.

The first 20 parameters in [PARAMETERS.md](PARAMETERS.md) cover the requested carrier/modulator waveforms, both full envelope arrays, harmonicity, modulation amplitude, note duration, transposition, overdrive and override, original Pd instrument name, reverb, echo and flanger. Additional fields cover dry/output levels, pan, independent loop lengths, synchronization cursors, usernames and playback/editor controls.

## Run

Requires Python 3.10 or newer. Run inside `apps/CustomExtractor`:

```powershell
python -m custom_extractor inspect examples/original.txt
python -m custom_extractor extract examples/original.txt --output composition.json
python -m custom_extractor validate composition.json
python -m custom_extractor export-tweeter composition.json --output reconstructed.txt
python -m custom_extractor roundtrip examples/original.txt
python -m custom_extractor coverage examples/original.txt --records
python -m unittest discover -s tests -v
python tools/verify.py
```

Optional installation: `python -m pip install -e .`. This adds the `custom-extractor` command. Without installation, `python -m custom_extractor` works from this folder.

CLI failures return exit code 2 with an explanatory message. The round-trip command returns 1 on a mismatch. Output files are written atomically; an existing explicitly named output is replaced only after validation and reconstruction succeed.

## Python API

```python
from custom_extractor import (
    extract_tweeter, export_tweeter, save_json, load_json,
    set_parameter, set_parameters, add_note, remove_note,
    set_loop_length, add_track, validate,
)

composition = extract_tweeter("examples/original.txt")
track_id = composition.tracks[0]["id"]
set_parameters(composition, track_id, {
    "carrier": "square",
    "synth.harmonicity": 3.0,
    "echo-fb": 0.6,
    "inst-name": "My original Pd instrument name",
})
set_loop_length(composition, track_id, 7200)
# Remove an existing note before replacing an occupied monophonic slot.
first = composition.track(track_id)["notes"][0]
remove_note(composition, track_id, first["step"], first["pitch"])
add_note(composition, track_id, first["step"], 66)
save_json(composition, "edited.json")
export_tweeter(load_json("edited.json"), "edited.txt")
print(validate(composition))  # diagnostic warnings; invalid data raises ValueError
```

`Composition` exposes `tracks`, `global_settings`, `track(id)` and `to_dict()`. Tracks use ordinary nested dictionaries. Direct dictionary edits are validated on JSON serialization and native export. Editing helpers work on a copy and commit only after the entire edit passes validation. `set_parameters` makes a group of edits atomic. Parameter names can be native names or the dotted JSON paths in the registry.

`add_track(composition, id, parameters, notes)` appends a supported part, using an unused ID from 1 through 14. Supply `loop-length`, `note-dur` and `octave` when adding notes. A new track's export writes all 64 note slots, including explicit rests, to clear stale notes in that part. Other omitted parameters remain unspecified and depend on the receiving application's state. No instrument preset is invented. The inspected full and Lite patches instantiate exactly 14 instrument slots. Other part IDs remain opaque native records rather than supported instruments.

`None` removes an optional parameter's records from the exported snapshot; it does **not** send a reset/default to a running synth. Existing track deletion or reordering is rejected because it could silently discard or rearrange opaque native commands. Notes and optional parameters can be removed independently. Removing a track's last representable record is rejected.

## Verified native format

Pd-L2Ork sources used to verify the native format and parameter mappings, pinned
to [upstream commit `c9edfcc`](https://github.com/pd-l2ork/pd-l2ork/tree/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537):

- [L2Ork-Tweeter.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/L2Ork-Tweeter.pd): session loader/saver uses `coll`, `clear, read`, `write`, `dump`, and routes `session-title` separately from part messages.
- [L2OrkTweet.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/L2OrkTweet.pd): waveforms, synthesis, note arithmetic, 64-step tracker, UI bounds, effects, mixing, keyboard map and synchronization.
- [L2OrkTweet-Lite.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/L2OrkTweet-Lite.pd): same native message family; not a second invented JSON/text format.
- [net-io.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/net-io.pd), [net-io-sym.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/net-io-sym.pd), [net-io-coll.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/net-io-coll.pd): part/parameter routing, scalar values, multiword symbols and tracker records.
- [preset_node_array_custom.pd](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/preset_node_array_custom.pd): full native array lists, ordered `tabwrite`, table size plus three samples on reporting.
- [key-diatonic-coll.txt](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/apps/L2Ork-Tweeter/core/key-diatonic-coll.txt): display keys for stored notes 48..84.
- [m_binbuf.c](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/pd/src/m_binbuf.c) and [coll.c](https://github.com/pd-l2ork/pd-l2ork/blob/c9edfccd5c63986ff9c6229d8a306e7a2dfd2537/externals/miXed/cyclone/hammer/coll.c): native token escaping, numeric grammar, coll record loading and duplicate key behavior.

Example native records:

```text
0, session-title My\ Composition;
1, 12 inst-name Laser\ Snare\ -\ Caleb;
2, 12 carrier 0;
3, 12 modulator 3;
4, 12 loop-length 6000;
5, 12 tracker-data 0 48;
6, 12 tracker-char 0 z;
7, 12 tracker-time 21 36.0516 1;
```

The number before the comma is a coll key, **not a timestamp**. Records end with a semicolon and can span multiple lines. There is no additional required header or track delimiter. Backslashes escape symbols, spaces, commas and semicolons; quotes are literal symbol characters, not string delimiters. Standalone `#` comments are not part of coll syntax. Valid coll metadata/unknown records are retained instead of discarded. Historic names such as `inst-name Long Lead` are interpreted through the native `tosymbol` behavior while their original atom boundaries remain preserved.

The corpus contains legacy byte `0xA0` display cells and non-UTF-8 symbol bytes. Files are decoded with UTF-8 plus `surrogateescape`; the original bytes survive JSON via explicit escapes and are restored on export. No replacement characters or encoding guesses are introduced. CRLF/LF, multiline arrays, duplicate records, original number spellings and record order survive unchanged exports byte for byte. BOM preservation is supported by the text round trip but is not a recommended native export format.

Loop-only/preset-only coll files with different payload layouts are outside this session adapter. An unrecognized coll document or empty file is rejected with a format error. Settings-only and empty-note **sessions** are supported.

## Representation and musical meaning

JSON has `format: "CustomExtractor"`, `version: 1`, `global`, `tracks`, and `native`.

- `global.tracker_steps` is the verified native constant 64. `global.metadata.session_title` retains the original title atom list; `null` means absent, while `[]` means an explicitly empty title.
- Each track has `id`, `instrument`, `synth`, `effects`, `mixing`, `timing`, `playback`, and sparse `notes`. Original part order is preserved.
- Each note has `step` and `pitch`. `pitch` is the **stored Tweeter numeric value**, including fractional/negative/out-of-MIDI-range pitches; 0 is a rest and is omitted from the sparse list. No MIDI pitch bound is imposed: the native receiver forwards nonzero numeric notes through `- 12`, transposition and `mtof`. Numbers must be finite and representable within the engine's float32 range. Steps are 0..63. Stale step-64 records are preserved as raw records with diagnostics, not made into active notes.
- Sounding pitch is `stored_pitch - 12 + transposition_semitones`. It may lie outside MIDI's 0..127 range; no clipping or transposition correction is performed.
- Cycle-relative onset is `step * track_loop_ms / 64`. Every track retains its own loop length. `tracker-time` stores `next_step`, `remaining_ms`, and `running`; it is synchronization state, not note microtiming.
- `note-dur` is a **track-wide envelope scan duration**, not a per-note gate. Retriggering and overlapping parts depend on native synthesis. One tracker slot holds one pitch; notes in different parts can occur simultaneously.
- No per-note velocities, recorded tempo, meter or timestamped automation were found in these session files. They are not fabricated. Unknown future fields, including explicit preset IDs or global timing extensions, are retained verbatim with coverage diagnostics, without inventing semantics.
- Envelopes are ordered sample arrays, not ADSR parameters or individually timestamped control points. All saved samples—including legacy lengths and extra/guard samples—are preserved. Native `tabwrite` does not resize the receiving table: fewer than 100 samples update only a prefix of the current table, and trailing samples beyond its capacity do not establish a larger audible envelope. Short lists generate a warning. For full replacement in the current patch, supply the complete saved array or at least the native 100 samples. No silent padding, truncation or interpolation is applied.
- `overdrive-override` is an ordered live command that updates overdrive, not a second independent persistent gain stage. Its literal value/order are retained; effective audible state depends on its order relative to `overdrive`. `delete` and `hover` are native editor/control messages, not a guessed track-enabled flag.
- Output amplitude is the native Pd 0..100 level scale; pan is 0..1.5708 radians, not an assumed -1..1 MIDI pan.

`custom_extractor.reporting.musical_features(composition)` provides derived onsets and sounding pitches separately. Missing timing stays `None`. A common loop duration is reported only when every track has the same saved loop duration, and is labeled as derived rather than a native global tempo.

The `native.text` preservation layer retains all original records, including unknown fields and historical spellings. It is not a substitute for the writer: a new `Composition()` can be built and exported without any source text. During export, structured edits replace the last applicable native parameter record; earlier duplicates and unrelated records remain in order. Removing an optional parameter removes all of its duplicates. Deleting a note emits an explicit native rest so an earlier duplicate cannot revive it. Associated tracker display cells are updated using the verified keyboard map. Treat the preservation layer as source data; edit structured fields rather than rewriting it manually.

## Validation, fidelity and limits

See [PARAMETERS.md](PARAMETERS.md) for every supported field and range. New numeric edits are checked against native UI ranges; values outside those ranges are never silently clamped. Historical out-of-range values already present in a source file are retained with `legacy_range` diagnostics. Missing values are distinct from zero. Invalid structures, duplicate JSON keys, non-finite numbers, invalid note positions/pitches and unsupported versions fail clearly.

JSON/TXT decimal values are retained without extra rounding. The installed Pd engine uses **32-bit floats** and its own text writer uses about six significant digits. Validation reports how many values will be approximated by that engine; this is separate from CustomExtractor's exact text/JSON preservation. The native checker also verifies live float-versus-symbol atom types because Pd's own writer can lose numeric-symbol escaping. CustomExtractor preserves that escaping.

Coverage distinguishes supported, partially interpreted and unknown-preserved records. Preservation is checked by the separately reported byte-identical native round trips and extracted-structure comparisons, rather than an assumed per-field count. Malformed files fail instead of yielding a misleading partial success. Opaque records may have side effects the extractor cannot evaluate. Application state, library versions, audio buffer limits (for example the echo delay buffer versus its larger UI range), live transport and parameter order can affect the audible result.

## Tests and example artifacts

- [original.txt](examples/original.txt): real `slowbeat-session-coll.txt`, copied without changes.
- [composition.json](examples/composition.json): original features, all envelopes and native preservation data.
- [reconstructed.txt](examples/reconstructed.txt): JSON-to-native round trip.
- [modified.json](examples/modified.json), [modified.txt](examples/modified.txt): edits to all 20 requested controls, additional mixing/timing edits, note removal/addition and a new part.
- [features.json](examples/features.json): derived musical features and coverage diagnostics.
- [verification.json](reports/verification.json): per-file extraction/export runtimes, sizes, counts, diagnostics and native results.
- [RESULTS.md](reports/RESULTS.md): concise measured results.

`tools/verify.py` discovers the published `*session*.txt` files directly in `../L2Ork-Tweeter/sessions`. It tests TXT → JSON → native TXT byte/semantic equality, generates the demo, measures validated file extraction/atomic file export, and checks native compatibility if Pd-L2Ork is installed. Use `--skip-native` on systems without Pd. No original sessions or MIDIAI outputs are modified.

Native checks launch a separate, headless Pd process with audio and MIDI disabled, preferences disabled, and `-unique`. The real cyclone `coll` reads, dumps/routes and rewrites each export; record/atom counts, values and live atom types are compared. These establish **native file-loader and message compatibility**, not full Tweeter GUI behavior or audible playback. No audible playback verification is claimed.

Tests additionally cover all registered parameter edits, every waveform, precision, escaping, raw legacy bytes, unknown fields, duplicates, independent loops, note boundaries, atomic failure, CLI commands, generated tracks and 50 deterministic randomized compositions.

To reproduce the tests and per-file report from `apps/CustomExtractor`:

```powershell
python -m unittest discover -s tests -v
python tools/verify.py
```

The second command regenerates `reports/verification.json` and `reports/demo.json`.
Native Pd checks require a local Pd-L2Ork installation; `--skip-native` runs the
remaining checks when it is unavailable.

## Code layout

`parser.py` handles native syntax; `extractor.py` interprets records; `parameter_registry.py` supplies the shared mapping; `models.py` and `json_adapter.py` define the versioned representation; `validator.py` checks types/ranges; `exporter.py` reconstructs native records; `edits.py` provides atomic operations; `reporting.py` computes diagnostics/derived features; `cli.py` exposes commands; `native_check.py` invokes the external Pd oracle.
