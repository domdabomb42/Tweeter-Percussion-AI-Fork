# 1.7B experiment

The completed [V4EA-synth](V4EA-synth/README.md) positions-v1 experiment
trains a fresh tokenizer and predicts percussion tracker positions (0–63).
Native export supplies the rest of the track from a fixed ElectricKick preset.
This 1.7B model is a comparison candidate and is not selected by the native
feedback plugin. Its local artifacts live in `MostRecentOutput` and do not
replace historical V3 output.

The three-model V3 consistency, optimization, and V2 preparation notes are in
[../SMOLLM_V3_EFFICIENCY.md](../SMOLLM_V3_EFFICIENCY.md).

Despite the folder name, `smollm1b` uses the **1.7B** parameter [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/tree/effd688a12921b4cc83e3312b6feb579f70f9c71) base model, revision `effd688a12921b4cc83e3312b6feb579f70f9c71`. A feature adapter is present; no V3 percussion adapter is present. Prepared feature data and 67/10/5 V3 examples retain the same source-session assignments and rhythm targets as 360M; its tokenizer was independently loaded from the pinned 1.7B model.

From the repository root, `python run_smollm1b.py` runs missing preparation, feature pretraining, and then V3 percussion training in order without intervention. `python run_smollm1b.py --dry-run` previews the remaining stages. The pretrained tokenizer is used as-is; the feature model learns from its tokenized Tweeter compositions.

From `CustomTraining`:

```powershell
python -m custom_training prepare
python -m custom_training train --dry-run
python -m custom_training train
python -m custom_training evaluate
```

After feature training completes, from `V3`:

```powershell
python -m percussion_generation_v3 prepare --encoding legacy-v1
python run_overnight.py --dry-run
python run_overnight.py
python -m percussion_generation_v3 generate --input input.txt --output output/generated/example.txt --target-note-count 16 --cd2 --max-new-tokens 512
python -m percussion_generation_v3 evaluate --partition test --cd2
```

Omit `--cd2` for greedy decoding, or choose `--constrained` for ordered constraints. The rhythm target, fixed preset, loss masking, and detached-prefix cache match the 360M pipeline. No 360M weights or evaluation results are copied here.

The preceding commands describe the historical V3 pipeline. They do not
describe the completed V4EA positions-v1 run. Its best adapter, final
checkpoint, evaluation, tokenizer, and dataset are present locally under
`MostRecentOutput/V4EA-positions-v1`; the selected adapter and inference
artifacts are excluded from this branch. A local V4EA
4,096-token optimizer-step smoke test peaked at 5,338 MiB reserved on a 6 GiB
RTX 3060; that measurement does not guarantee every run or concurrent GPU
workload fits. See [V4EA-synth](V4EA-synth/README.md) for current commands.
