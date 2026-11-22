# 1.7B experiment

Despite the folder name, `smollm1b` uses the **1.7B** parameter [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/tree/effd688a12921b4cc83e3312b6feb579f70f9c71) base model, revision `effd688a12921b4cc83e3312b6feb579f70f9c71`. Its feature and V3 adapters have **not** been trained. Prepared feature data and 67/10/5 V3 examples retain the same source-session assignments and rhythm targets as 360M; its tokenizer was independently loaded from the pinned 1.7B model.

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
python -m percussion_generation_v3 prepare
python run_overnight.py --dry-run
python run_overnight.py
python -m percussion_generation_v3 generate --input input.txt --output output/generated/example.txt --target-note-count 16 --cd2 --max-new-tokens 512
python -m percussion_generation_v3 evaluate --partition test --cd2
```

Omit `--cd2` for greedy decoding, or choose `--constrained` for ordered constraints. The rhythm target, fixed preset, loss masking, and detached-prefix cache match the 360M pipeline. No 360M weights or evaluation results are copied here.

The approximately 6 GB RTX 3060 has not been shown to fit a 1.7B LoRA run on the current untruncated sequences. The 1.7B base weights alone require several GB in 16-bit precision, with activations and optimizer memory on top. This pipeline is configured for more capable hardware; it fails on out-of-memory rather than silently shortening or changing examples. No 1.7B training result is claimed.
