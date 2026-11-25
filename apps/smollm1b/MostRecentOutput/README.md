# Local V4EA output (smollm1b)

The default `V4EA-positions-v1` experiment stores its trained tokenizer,
prepared dataset, manifest, fixed export preset, evaluations, selection logs,
and LoRA checkpoints here. The model predicts **percussion tracker positions**;
the fixed preset supplies the complete native track at export. The separate
`V4EA-music-v3` directory, when present, is the historical full-musical-target
experiment. Neither directory replaces V3 output.

The trained `V4EA-positions-v1` tokenizer, dataset, selected adapter,
embeddings, and evaluations are excluded from this branch by `.gitignore`.
GitHub rejected the attempted Git LFS publication to this public fork, so a
fresh clone contains this README and the ignore rule, not usable V4EA weights.
There is currently no repository download for the trained artifacts. To use a
saved model, obtain the entire `V4EA-positions-v1` directory from the
training machine, including `tokenizer`, `tokenizer_identity.json`,
`manifest.json`, `fixed_preset.json`, `rhythm`, and
`checkpoints/rhythm_coverage_v1/best_adapter` with LoRA weights and trained
embeddings. Keep these together for identity checks. Do not substitute V3
weights or another model size's tokenizer. Base SmolLM2 weights must also be
obtainable by the model loader. Local feedback and DPO rounds remain excluded.

To prepare and train a new experiment, follow [the V4EA experiment guide](../V4EA-synth/README.md).

For comparison inference, the prepared output and best adapter are required. This model is not wired into the native feedback plugin.
