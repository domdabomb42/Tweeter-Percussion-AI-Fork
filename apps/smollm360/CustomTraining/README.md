# CustomExtractor → SmolLM2-360M experiment

This project tests causal next-token learning on **real native Tweeter sessions**. It uses the existing CustomExtractor Python package in `../../CustomExtractor`, its canonical `global`/`tracks`/`opaque_records` feature representation, and the unmodified pretrained tokenizer for [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M). The exact model revision is pinned in `custom_training/__init__.py`. No English instruction examples, invented musical fields or synthetic compositions are used in the real experiment.

The input is `../../L2Ork-Tweeter/sessions`. There are **51** parseable sessions in that main folder. Additional copies and work variants elsewhere in the repository are not included. The split is by source file families and note arrangement similarity, with a deterministic seed. It targets about 41 training, 5 validation and 5 test sessions; the exact grouped counts are recorded in `training_output/dataset/manifest.json`. The test partition is reserved until checkpoint selection has ended.

## Environment

Python dependencies are listed in `requirements.txt`. Install them into your Python environment before running the project. The model's ordinary Hugging Face cache is used. A CUDA GPU is preferred for training; `smoke` is a small, explicitly bounded CPU optimizer check. The default run uses LoRA rank 16, alpha 32, dropout 0.05, AdamW, learning rate 0.0001, batch size 1, gradient accumulation 8, at most five epochs and validation-loss early stopping. BF16 is selected when CUDA supports it. The code raises a clear error for GPU out-of-memory instead of truncating a segment.

## Commands

Run inside this directory:

```powershell
python -m custom_training inspect --sessions ../../L2Ork-Tweeter/sessions
python -m custom_training prepare --sessions ../../L2Ork-Tweeter/sessions --seed 415 --max-length 4096
python -m custom_training train --dry-run
python -m custom_training smoke
python -m custom_training baseline
python -m custom_training train --epochs 5 --patience 2
python -m custom_training train --resume --epochs 20 --patience 2 --learning-rate 0.00003
python -m custom_training evaluate
python -m custom_training compare
python -m custom_training complete --partition test --feature synth.carrier --index 0
python -m unittest discover -s tests -v
```

Use `--cpu` with `baseline`, `train`, `evaluate` or `complete` for explicit CPU execution. `smoke` always uses CPU. The top-level `--output PATH` argument, placed before the subcommand, changes the output root. `prepare` supports `--local-only` once the tokenizer is cached. Change `--seed` on `inspect` and `prepare` to repeat the grouped split; keep output roots separate when comparing seeds.

`inspect` counts tracks, notes, distinct waveform pairs, parameter distributions, per-session maximum track loop lengths and token counts. A track loop length is **not** a known total composition duration; native sessions contain no recorded stop time. The source-level manifest stores SHA-256 hashes, family groups, similarity edges, train/validation/test assignments, token counts and model/tokenizer identity. `prepare` writes JSONL records with full compact JSON and pretrained token IDs; no new vocabulary tokens are added. Repeating the run with the same seed yields the same split and serialized text.

Whole compositions frequently exceed 8,192 tokens. Preparation groups complete tracks into sequences up to the selected maximum length, repeating only the global context. Every track appears in exactly one training segment per source. Numeric values and native envelope arrays remain intact. If even one track cannot fit, preparation fails and names the offending session and track rather than silently truncating it. The manifest reports fit percentages for 2,048, 4,096 and 8,192 tokens, plus per-track, note, synthesis and effects token counts.

`baseline` measures the original model on **validation only**, before fine-tuning. `train` uses training segments for causal LM optimization and validation loss for early stopping and saving `training_output/checkpoints/best_adapter`. The trainer never opens test data. `evaluate` loads the selected adapter, scores validation and test, then separately scores the original model on the same held-out queries. `compare` collects matched loss, perplexity, structural, feature-value and train-only statistical baseline results.

`train --resume` continues from the saved best feature adapter. `--epochs` is the total ceiling including completed epochs; `--patience 2` stops after two consecutive validation epochs without at least 0.0001 loss improvement. The prior run saved adapter weights but not optimizer or scheduler state, so both reset on continuation. The existing adapter and history are backed up before continuation starts. Further feature training does not automatically update an already trained percussion adapter.

Feature completion starts immediately before each value's JSON text and predicts the **complete parsed value**, including multi-token numbers. The same sampled target queries are used for the original and fine-tuned models, and the per-feature sample count is reported. The train-only baseline uses the mode for categorical values and median for numeric values. Syntax accuracy is reported separately from note, synthesis and effects value accuracy. A generated continuation counts as structurally valid only if it parses and can be validated, exported and reimported by CustomExtractor. These probes measure serialized-data prediction, not audible understanding.

Output artifacts:

- `training_output/dataset/manifest.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, and tokenizer configuration.
- `training_output/evaluation/original_validation.json`, `original_full.json`, `fine_tuned.json`, and `comparison.json`.
- `training_output/checkpoints/best_adapter/` and `history.json`.
- `REPORT.md` records the observed result and its limits.

This is a small proof of concept. Similar compositions are grouped, and the held-out counts are small. Validation improvements alone cannot establish musical or acoustic understanding. The report should be read with the feature-wise statistical baselines and seed sensitivity in mind.
