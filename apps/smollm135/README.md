# 135M experiment

The completed [V4EA-synth](V4EA-synth/README.md) positions-v1 experiment
trains a fresh tokenizer and predicts percussion tracker positions (0–63).
Native export supplies the rest of the track from a fixed ElectricKick preset.
This trained 135M V4EA model is the one selected by the experimental native
Tweeter feedback plugin. Its local artifacts live in `MostRecentOutput` and
do not replace historical V3 output.

The three-model V3 consistency, optimization, and benchmark notes are in
[../SMOLLM_V3_EFFICIENCY.md](../SMOLLM_V3_EFFICIENCY.md).

For the new lossless V3 input format and its preparation, benchmark, training,
resume, and evaluation commands, see [V3/README.md](V3/README.md).

Base model: [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The checkout includes saved feature and V3 adapters. Prepared feature segments and the original 67/10/5 V3 rhythm pairs are model-local and verified against the same native source sessions as 360M.

From the repository root, run `python run_smollm135.py` for sequential feature and percussion training without an intermediate prompt. Run `python run_smollm135.py --dry-run` to preview it. Dependencies are in `CustomTraining/requirements.txt`.

Individual commands from `CustomTraining`:

```powershell
python -m custom_training prepare
python -m custom_training train --dry-run
python -m custom_training train
python -m custom_training evaluate
```

Individual commands from `V3`:

```powershell
python -m percussion_generation_v3 prepare --encoding legacy-v1
python run_overnight.py --dry-run
python run_overnight.py
python -m percussion_generation_v3 generate --input input.txt --output output/generated/example.txt --target-note-count 16 --cd2 --max-new-tokens 512
python -m percussion_generation_v3 evaluate --partition test --cd2
```

Omit `--cd2` for greedy generation or use `--constrained` for ascending ordered constraints. V3 loads this folder's feature adapter, predicts tracker positions only, and exports with this folder's fixed verified percussion preset. No 360M adapter is copied into this experiment.

## Augmented percussion training

V3 preparation now adds the exact shifted sessions in `../L2Ork-Tweeter/AugmentedSessions` to the original rhythm training pairs. It includes only children of training sessions and excludes offset zero, which duplicates each original. Validation and test keep the original held-out pairs. The manifest records how many shifted pairs fit the 8,192-token context. Prepare a fresh output directory so existing checkpoints and evaluations remain available:

```powershell
cd apps/smollm135/V3
python -m percussion_generation_v3 --output-dir output_augmented prepare --encoding legacy-v1
python -m percussion_generation_v3 --output-dir output_augmented train
```

The `train` command starts from the 135M feature adapter. Use `--dry-run` after `train` to inspect its plan without training.
Preparation prints shifted-file and pair counts about every three minutes; training uses the same default progress interval.

## Epoch-7 warm-start experiment

The separate `V3/output_augmented_from_epoch7` experiment starts from the original V3 best adapter at epoch 7 (validation loss 0.6111). It starts a new AdamW optimizer and cosine learning-rate schedule; it does not restore the historical optimizer. The prepared pool has 67 original and 2,621 shifted training pairs. Each new epoch uses all 67 originals plus 201 distinct shifted pairs selected by deterministic rotation, for 268 examples and 34 optimizer updates. The fixed validation set contains the original 10 pairs and 10 shifts of those validation parents. Best-adapter selection uses all 20; the original 10 are reported separately.

The original V3 source split keeps `3rd-beat` in training and other beat sessions in validation. These distinct sessions share the broader `beat-sequence` family. This inherited family overlap is recorded in the new manifest and validation report; no shifted validation parent enters training.

From `apps/smollm135/V3` in PowerShell, prepare the separate experiment from an existing verified `output_augmented` pool and inspect its split:

```powershell
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 prepare --encoding legacy-v1 --from-output output_augmented
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 validate-splits
```

If `output_augmented` is absent, first prepare it with `python -m percussion_generation_v3 --output-dir output_augmented prepare --encoding legacy-v1`. Preparation creates a new directory and refuses to overwrite an existing one.

Preview and start the new 12-epoch maximum run:

```powershell
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 train --init-adapter output/checkpoints/rhythm/best_adapter --examples-per-epoch 268 --learning-rate 2e-5 --save-steps 10 --epochs 12 --dry-run
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 train --init-adapter output/checkpoints/rhythm/best_adapter --examples-per-epoch 268 --learning-rate 2e-5 --save-steps 10 --epochs 12
```

Resume exactly from the latest complete optimizer-step checkpoint:

```powershell
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 train --resume-checkpoint latest
```

`checkpoints/rhythm/latest.json` points to the latest complete checkpoint. `checkpoints/rhythm/history.json` records completed epochs, and `selection/epoch_###.json` records the exact example order. The trainer keeps the three latest periodic checkpoints, the latest complete epoch-boundary checkpoint, the best validation adapter, and a final checkpoint. Inspect progress with:

```powershell
Get-Content output_augmented_from_epoch7/checkpoints/rhythm/latest.json
Get-Content output_augmented_from_epoch7/checkpoints/rhythm/history.json
Get-ChildItem output_augmented_from_epoch7/checkpoints/rhythm/periodic
```

Evaluate the selected best adapter with:

```powershell
python -m percussion_generation_v3 --output-dir output_augmented_from_epoch7 evaluate --partition test
```
