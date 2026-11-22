# CustomExtractor verification results

Verified locally on Windows with Python 3.10.0 and the installed Pd-L2Ork native `coll` loader.

## Automated tests

**49 tests passed** in 84.534 seconds. The corpus results below cover the 51 published sessions. The suite also covered 50 deterministic randomized compositions, every registered parameter edit, all waveforms, independent track loops, sparse notes, fractional and non-MIDI-range pitches, duplicate records, escapes, numeric symbols, legacy bytes, missing fields, unknown fields, atomic validation failures, CLI operations and native Pd integration tests.

The per-file audit evidence is in [verification.json](verification.json).

## Real session corpus

| Check | Result |
| --- | ---: |
| Session files discovered | 51 |
| Nonempty sessions tested | 51 |
| Byte-identical TXT → JSON → TXT round trips | 51 / 51 |
| Equal extracted canonical structures | 51 / 51 |
| Native Pd loader checks passed | 51 / 51 |
| Empty files in published corpus | 0 |
| Unexpected failures | 0 |
| Tracks across accepted sessions | 629 |
| Active notes across accepted sessions | 6,408 |
| Extracted parameter values | 18,977 |

The corpus is the 51 files in `apps/L2Ork-Tweeter/sessions` that are included in this repository. These figures come from a fresh `python tools/verify.py` run on those files. Work-specific collections outside `sessions/` are outside this report's scope.

## Runtimes

Measured with `perf_counter`, including validation for extraction and validation/reconstruction/atomic disk writing for export. The figures below are from the fresh 51-session run. JSON import and native subprocess checks are measured separately from these two operations. These are observed wall times, not isolated hardware benchmarks.

| Operation | Median | Mean | Range |
| --- | ---: | ---: | ---: |
| Extract a real session | 269.20 ms | 267.82 ms | 128.70–366.03 ms |
| Export a real session | 517.43 ms | 522.79 ms | 274.41–706.79 ms |

`tools/verify.py` also makes five `perf_counter` runs on the `slowbeat` example:
it extracts the original file each time and validates/atomically exports the
original composition to a temporary file each time. JSON import and native Pd
checks are excluded from these times. The recorded medians are **230.42 ms
extraction** and **454.03 ms export**; all ten samples are in [demo.json](demo.json).

## Real example and edited export

The example source is `slowbeat-session-coll.txt`: **12 tracks, 97 active notes, 336 parameter values, 1,873 native records**, 64,859 original bytes and 137,247 JSON bytes. The JSON includes the raw preservation layer as well as structured features.

The edited example changes all 20 requested controls, additional mixing/timing/synchronization settings, deletes one note, adds another, and adds three notes on native part **13**. Complete carrier/modulator sample arrays retain their original lengths. Existing unmodified tracks compare equal after export/reimport.

The final modified native file passed Pd's loader and routing checks with **2,013 records**, **12,186 compared key/payload atoms**, and **10,173 live payload atom types verified**. Both reconstructed and modified example exports passed native checks.

See [demo.json](demo.json), [composition.json](../examples/composition.json), [modified.json](../examples/modified.json) and [modified.txt](../examples/modified.txt).

## Feature coverage and limits

All 20 requested controls are extracted/reconstructed when present: independent carrier and modulator waveforms, both full envelope arrays, harmonicity, modulation amplitude, track-wide note duration, transposition, overdrive and override, original Pd instrument name, two reverb controls, three echo controls and four flanger controls. The registry covers **33 named native controls** in total, plus title and tracker records. Missing controls remain unavailable rather than receiving fabricated defaults.

Unknown parameters are preserved and tested with synthetic records. Unchanged historical values outside current UI bounds are retained with diagnostics. Short native envelope lists update only a prefix of the receiving table; their values and lengths are preserved and this limitation is reported. `overdrive-override`, editor controls and tracker display metadata have explicitly documented partial interpretation.

These snapshots do not provide verified tempo, meter, per-note velocity, per-note gate length, explicit preset identifiers, or timestamped automation. The utility does not invent them. The native patch has 14 supported part slots; stored nonzero numeric pitches are not constrained to MIDI integers.

**Native loader compatibility is verified; audible playback and full Tweeter GUI state are not.** The checks use the actual Pd-L2Ork cyclone `coll` loader, native message routing and live atom-type checks, with audio/MIDI disabled. Pd's own writer rounds numeric values and can remove numeric-symbol escaping; the external comparison accounts for that documented behavior. CustomExtractor's own unchanged round trips remain byte exact.

See [verification.json](verification.json) for all per-file measurements, diagnostics and external checks, and [PARAMETERS.md](../PARAMETERS.md) for mappings and validated ranges.
