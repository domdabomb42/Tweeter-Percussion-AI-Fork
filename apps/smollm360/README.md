# Existing 360M experiment

The separate [V4EA-synth](V4EA-synth/README.md) experiment trains a fresh
tokenizer and predicts a complete percussion track. Its new files go to
`MostRecentOutput` and do not replace V3.

The three-model V3 consistency, optimization, and V2 preparation notes are in
[../SMOLLM_V3_EFFICIENCY.md](../SMOLLM_V3_EFFICIENCY.md).

The original [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) base-model experiment was moved from `apps/Smollm` to this directory. Its feature adapter, V3 rhythm and constrained adapters, tokenizers, prepared datasets, evaluations, and generated Tweeter files remain unchanged. The V3 code is now mirrored across model variants; historical inputs still use their saved encoding.

Run feature commands in [`CustomTraining`](CustomTraining) and percussion commands in [`V3`](V3). `python -m custom_training train --dry-run` inspects the feature plan; `python run_overnight.py --dry-run` inspects the V3 plan. Existing checkpoint protection prevents accidental retraining over saved V3 adapters. For generation, from `V3`:

```powershell
python -m percussion_generation_v3 generate --input output/POC_inputs/simple-melody-bass.txt --output output/generated/new-example.txt --target-note-count 16 --loop-ms 10800 --cd2 --max-new-tokens 512
```

Use `--constrained` for ordered constraints or omit both decoding flags for greedy output. Run `python tests/regression_360.py` from the repository root to compare exact exported bytes against the original greedy and CD2 examples. Historical manifests retain their original absolute paths as records of the earlier run; current code resolves paths within this repository.

## Augmented percussion training

V3 preparation now adds exact shifted sessions from `../L2Ork-Tweeter/AugmentedSessions` to the original rhythm training pairs. It takes children of training sessions only and excludes duplicate offset-zero copies. Validation and test remain the original held-out pairs. Prepare and train in a fresh output directory to preserve the historical 360M results:

```powershell
cd apps/smollm360/V3
python -m percussion_generation_v3 --output-dir output_augmented prepare --encoding legacy-v1
python -m percussion_generation_v3 --output-dir output_augmented train
```

The new dataset manifest reports included pairs and shifts skipped because they exceed the 8,192-token context.
Preparation prints shifted-file and pair counts about every three minutes; training uses the same default progress interval.
