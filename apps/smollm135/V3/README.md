# SmolLM2-135M V3 percussion

**Historical experiment:** the saved V3 dataset, adapter, tokenizer, and
evaluation results document their original training and decoding. They have
not been rewritten for the native plugin. Current generation code applies
Tweeter factory #64's ElectricKick output volume (`85.9649`), pan
(`0.765262`), and zero echo amplitude when constructing a **new** native
track. Thus a newly exported file can differ in those three settings from a
historical generated file even with identical predicted positions. Use saved
historical outputs for byte-for-byte reproduction of the old export.

Three-model execution consistency and efficiency measurements are documented
in [../../SMOLLM_V3_EFFICIENCY.md](../../SMOLLM_V3_EFFICIENCY.md).

V3 predicts tracker positions followed by `[END_RHYTHM]`. The pretrained
`HuggingFaceTB/SmolLM2-135M` tokenizer, LoRA model, target-only loss, 8192-token
context, rhythm grammar, and native Tweeter export remain unchanged.

## Input formats

Historical `compact`/`legacy-v1` input remains in `codec.py`: aliased JSON
objects plus a deduplicated zlib/Base64 envelope bank. Historical output,
checkpoints, tokenizer files, and results remain available. New preparation
defaults to `lossless-v2` in `codec_v2.py`. V2 extracts the original session,
removes the target track, retains all fields in the remaining global/track
structure (including explicit nulls), and serializes them directly. Training
and generation call the same V2 encoder. Unknown fields go into an extension
list, and an absent field has a clear bit in the presence mask. Numeric JSON
values are not rounded. The envelope bank is still compressed losslessly;
each entry has an explicit `Z` tag. Other reversible benchmark candidates are
tagged `R` (raw), `L` (run length), and `D` (dictionary).
The benchmark also tests a `Q` dictionary of repeated four-value sequences;
`Q` is a benchmark-only candidate, since the selected codec uses `Z`.

The V2 top-level array is `["L2V2", global, envelope_bank, tracks, extensions?]`.
Each `tracks` entry is `[role, track, extensions?]`. Each schema object is
`[presence_mask, present_values_in_field_order..., extensions?]` where an
extension is `[-1, [[key, value], ...]]`. The bit for a field follows the
zero-based order below. `null` is a present value; an unset bit means missing.
Envelope references use `["E", bank_index]`.

| Structure | Fixed field order |
|---|---|
| global | tracker_steps, metadata |
| metadata | session_title |
| track | id, synth, timing, effects, instrument, mixing, playback, notes |
| synth | carrier, modulator, carrier_envelope, modulator_envelope, harmonicity, mod_amplitude |
| timing | note_duration_ms, transposition_semitones, loop_ms |
| effects | overdrive, reverb, echo, flanger |
| overdrive | level, override |
| reverb | amplitude, type |
| echo | amplitude, feedback, delay_ms |
| flanger | amplitude, frequency_hz, mod_amplitude, delay_ms |
| instrument | name, username |
| mixing | dry_amplitude, output_amplitude, pan |
| playback | loop_enabled, solo, tracker_enabled, keyboard_enabled, insert, delete, hover, cursor |
| cursor | next_step, remaining_ms, running |
| note | step, pitch |

The schema hash covers the field order, nested object mapping, envelope tags,
and version magic. The dataset manifest also records the tokenizer SHA256,
model ID/revision, prepared dataset hash, split statistics, and augmentation
manifest identity. A V2 adapter is rejected with mismatched codec, tokenizer,
or dataset metadata. The epoch-7 compact-format adapter is permitted only with
the explicit `--init-adapter` warm start; it has not been validated directly
on V2 prompts. Exact resume restores optimizer, scheduler, RNG, and the
selected example order from a step checkpoint.

## PowerShell commands

Run these from `apps/smollm135/V3` after installing the repository's Python
requirements. The first command reads the original Tweeter sessions and the
existing local split/augmentation metadata; it re-tokenizes every pair with
the pinned tokenizer, validates exact round trips and source splits, and
creates a separate output directory. It refuses an existing output directory.

```powershell
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 prepare --encoding lossless-v2
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 validate-splits
python -m percussion_generation_v3 --output-dir .\output_lossless_v2_benchmark compare-codecs --from-output .\output_augmented_from_epoch7
```

The preparation writes `codec_benchmark.json` and `codec_benchmark.md`,
including every example's old/new token lengths, envelope alternatives,
context-limit counts, and an **estimated** input-token work reduction.
On 2,713 distinct local examples, complete sequence tokens fell from
20,216,900 to 18,775,656, a **7.13% reduction**. Median length fell from
7,533 to 7,057 tokens; maximum fell from 8,149 to 7,671. No example exceeded
8,192 tokens under either codec. All 2,713 examples used fewer tokens (249 to
810 fewer per example). Prefix-only tokens fell from 20,107,415 to
18,666,171, a **7.17% reduction**. Full results are in
`output_lossless_v2/codec_benchmark.md`. Zlib/Base64 was selected for the
full-resolution envelopes because it used 9,772,419 tokenizer IDs across all
deduplicated conditioning envelope bank entries, versus 22,866,285 for run
length, 28,887,707 for sequence dictionaries, 30,698,324 for raw arrays,
and 31,001,170 for value dictionaries. The positional schema keeps all other
values visible.

```powershell
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 train --init-adapter .\output\checkpoints\rhythm\best_adapter --examples-per-epoch 268 --learning-rate 2e-5 --save-steps 10 --epochs 12
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 train --resume-checkpoint latest --patience 5
Get-Content .\output_lossless_v2\checkpoints\rhythm\history.json
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 evaluate --partition test --cap 5
```

The inherited split has disjoint source sessions across train, validation,
and test. The original train and validation sets both contain a beat-sequence
family; this known family overlap is retained with the 67 original training
pairs and original 10 validation pairs. The 20-pair fixed validation set adds
ten shifts of those held-out validation parents. Augmented children remain in
their parent's split.

The resume command explicitly raises early-stopping patience to five
non-improving epochs. This changes only the future stopping policy: the
optimizer, schedule, RNG state, and example position are restored from the
checkpoint. Subsequent resumes may omit `--patience` to keep the saved value.
The run labels its first new epoch as overall epoch 8, following the original
best epoch 7. The epoch-7 adapter's compact-format loss on the same fixed
20 target pairs was measured as 0.7691896 and is retained as the initial
best-loss threshold. New epoch 8 scored 0.7566006, so it is the current best;
the resumed second new epoch is displayed as overall epoch 9. This comparison
uses the same targets but each adapter's own input codec. Checkpoint example
positions and optimizer steps remain local to the new experiment.
Keep sufficient free disk space for adapter and optimizer-state checkpoint
writes; a disk-full error leaves the last published checkpoint intact.

Historical evaluation uses `--output-dir .\output` with the same `evaluate`
command. Historical preparation remains available explicitly with
`prepare --encoding legacy-v1` in a fresh output directory.
