# V3: position-only Tweeter percussion

**Historical experiment:** the saved V3 dataset, adapter, tokenizer, and
evaluation results document their original training and decoding. They have
not been rewritten for the native plugin. Current generation code applies
Tweeter factory #64's ElectricKick output volume (`85.9649`), pan
(`0.765262`), and zero echo amplitude when constructing a **new** native
track. Thus a newly exported file can differ in those three settings from a
historical generated file even with identical predicted positions. Use saved
historical outputs for byte-for-byte reproduction of the old export.

This repository no longer includes the V1 and V2 folders. The prepared V3
data, tokenizer, adapters, and examples remain available for inference and
training. `prepare` now validates retained V3 examples against native source
sessions; it does not rewrite this model's historical output. The old V1/V2
`compare` and preservation checks still require the removed folders.

V3 is independent of `apps/PercussionGeneration` (V1) and
`apps/PercussionGenerationV2` (V2). The original preparation read V2's rhythm split without
re-splitting sessions and copied V2's frozen starting feature adapter and tokenizer;
and writes all V3 data, checkpoints, generated files, and evaluations under this
folder. Run `verify_preservation.py` to check V1/V2 against the snapshot taken
before V3 work.

The actual V2 *rhythm* target contains `timing.loop_ms` and notes with both
`step` and `pitch`; V2's separate full mode additionally includes synthesis
settings. V3 predicts only ascending tracker positions, for example:

```
0 4 8 12 16
[END_RHYTHM]
```

Its input is V2's compact CustomExtractor composition, with the count marker
changed to `RHYTHM_TARGET=N`. The exact underlying source/target rows are the
same 67 train, 10 validation, and 5 test rhythm pairs.
V2's condition includes all tracks other than the held-out target; this can
include other verified percussion, silent, and ambiguous tracks alongside the
non-percussion tracks. V3 retains that condition so the target format is the
main change.

Native Tweeter is a
64-step tracker and holds one pitch at each step in a single track. V3 retains
duplicate positions while serializing, but rejects them at inference because a
single exported track cannot represent simultaneous notes at one step. V3
generates one percussion track, using V2's verified `electrikick` preset and
fixed pitch. It does not produce synthesizer parameters or edit existing tracks.

## One command for overnight baseline and constrained experiment

Run this in PowerShell from `apps/smollm360/V3`:

```powershell
python run_overnight.py
```

This trains a target-only LoRA adapter from V3's frozen copy of V2's *starting*
feature adapter. It uses V2's latest rhythm settings: 20-epoch ceiling,
learning rate `0.00005`, accumulation 8, patience 3, minimum validation
improvement `0.005`, progress every 180 seconds, and greedy decoding. It saves
the best baseline adapter under `output/checkpoints/rhythm/best_adapter`.
Immediately afterward it copies the best adapter to
`output/checkpoints/constrained/best_adapter`, evaluates baseline and
grammar-constrained decoding on held-out rows, then runs the V1/V2/V3
comparison. No intervention is needed between stages.
The constrained copy has **identical model weights**; constrained decoding is
an inference method, not a second fine-tuning run. Its config is at
`output/checkpoints/constrained/decoding_config.json`.

On the 6 GB RTX 3060, ordinary backward ran out of GPU memory. Offloading all
activations then exhausted system RAM. V3 now reads the full composition into
a detached attention cache, then backpropagates through the last prompt token
and rhythm target. All conditioning tracks and target labels remain present,
but gradients no longer flow through the long prompt. This is a training-method
change from V2, recorded in the training plan. Adam's running statistics stay
on the CPU between updates. Held-out validation still uses the full
target-only loss.

Progress and failures are printed and appended to `output/overnight.log`.
Per-epoch history is saved in `output/checkpoints/rhythm/history.json`.
The final summary is `output/overnight_result.json`. Evaluation JSONs are in
`output/evaluation/`, including raw model output for each test. The default
comparison requests each held-out original count plus 5, 8, 10, 12, 16, and
20 notes. Evaluation uses a shared 512-token ceiling to bound repetitive
outputs; the decoder remains greedy.

Inspect the full run without training:

```powershell
python run_overnight.py --dry-run
```

## Inference after training

From `apps/smollm360/V3`, generate with the learned baseline:

```powershell
python -m percussion_generation_v3 generate --input 'input.txt' --output 'output\generated\example.txt' --target-note-count 16
```

Add `--constrained` to use the separately saved constrained decoder. The
`--cd2` option uses the same adapter with a distinct decoder: it requires the
requested count and forbids duplicate steps, while allowing steps to be
generated in any order. It sorts steps only when exporting a Tweeter track.
The raw generation order is retained in diagnostics. CD2's configuration is
`output/checkpoints/cd2/decoding_config.json`; no additional training is needed.
For example:

```powershell
python -m percussion_generation_v3 generate --input 'input.txt' --output 'output\generated\cd2.txt' --target-note-count 16 --cd2
```

The default loop length is the unique most common loop among existing verified
percussion tracks, or among verified non-percussion tracks if no percussion
track is present. If that is ambiguous, supply `--loop-ms 2700` or
`--reference-track-id 3`. You may specify `--target-track-id` for a free slot
or `--preset` for a compatible verified preset JSON. Invalid positions,
duplicates, or count mismatches are never silently fixed. A valid mismatch is
exported unchanged and reported. Raw output and timings are written to
`output/generated/<output-stem>.diagnostics.json`; malformed output yields no
Tweeter file.

## Evaluation and tests

```powershell
python -m percussion_generation_v3 compare --cap 5 --counts 5,8,10,12,16,20 --max-new-tokens 512
python -m unittest discover -s tests -v
python verify_preservation.py
```

The comparison runs V1, V2, and V3 on aligned held-out rows. It scores note
count, validity, output tokens, latency, onset overlap with melody and the
reference rhythm, repeated patterns, and loop checks. The onset overlap is a
limited proxy: syncopated rhythms may be musically good despite low overlap.
Listening quality requires a separate human evaluation. V1 has no requested
count input, while V2 and V3 receive the same requested counts.
