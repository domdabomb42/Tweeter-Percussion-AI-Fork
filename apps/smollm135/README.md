# 135M experiment

Base model: [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. This is the base model. Its feature and V3 adapters have **not** been trained. Prepared feature segments and 67/10/5 V3 rhythm pairs are model-local and verified against the same native source sessions as 360M.

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
python -m percussion_generation_v3 prepare
python run_overnight.py --dry-run
python run_overnight.py
python -m percussion_generation_v3 generate --input input.txt --output output/generated/example.txt --target-note-count 16 --cd2 --max-new-tokens 512
python -m percussion_generation_v3 evaluate --partition test --cd2
```

Omit `--cd2` for greedy generation or use `--constrained` for ascending ordered constraints. V3 loads this folder's feature adapter, predicts tracker positions only, and exports with this folder's fixed verified percussion preset. No 360M adapter is copied into this experiment.
