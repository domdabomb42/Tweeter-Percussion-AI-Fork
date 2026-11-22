# Existing 360M experiment

The original [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) base-model experiment was moved from `apps/Smollm` to this directory. Its feature adapter, V3 rhythm and constrained adapters, tokenizers, prepared datasets, evaluations, and generated Tweeter files remain unchanged. The V3 source, input encoding, target-only loss, fixed preset, and three decoding algorithms are retained.

Run feature commands in [`CustomTraining`](CustomTraining) and percussion commands in [`V3`](V3). `python -m custom_training train --dry-run` inspects the feature plan; `python run_overnight.py --dry-run` inspects the V3 plan. Existing checkpoint protection prevents accidental retraining over saved V3 adapters. For generation, from `V3`:

```powershell
python -m percussion_generation_v3 generate --input output/POC_inputs/simple-melody-bass.txt --output output/generated/new-example.txt --target-note-count 16 --loop-ms 10800 --cd2 --max-new-tokens 512
```

Use `--constrained` for ordered constraints or omit both decoding flags for greedy output. Run `python tests/regression_360.py` from the repository root to compare exact exported bytes against the original greedy and CD2 examples. Historical manifests retain their original absolute paths as records of the earlier run; current code resolves paths within this repository.
