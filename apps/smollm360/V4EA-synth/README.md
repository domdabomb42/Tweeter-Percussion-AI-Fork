# V4EA-synth: full percussion track generation

This is a separate experiment from V3. It trains a new BPE vocabulary on the
**training split only**, including the original 67 target pairs and their
AugmentedSessions children (2,688 training pairs total). It retains the pinned
SmolLM2 pre-tokenizer and special tokens, but learns new vocabulary IDs and
merges. A pretrained SmolLM2 transformer receives newly aligned, trainable
embeddings plus a fresh rank-16 LoRA adapter. V3 adapters and checkpoints are
incompatible with this tokenizer and remain untouched.

Each training input contains the complete lossless V2 conditioning session
with the target track removed and a requested note count. Each target is a
lossless V2 structure containing the **entire percussion track**: its original
track ID, notes and pitches, synth carriers and modulators, full envelopes,
effects, timing, mixing, playback, and unknown fields. `[END_TRACK]` and the
existing EOS token terminate the target. No fixed percussion preset is used
to define the target. Train and validation losses apply only to target tokens.

The 20 fixed validation pairs and five test pairs keep their original source
families. Only training parents and their augmented children enter tokenizer
learning or gradient updates. The trainer rotates 201 distinct augmented
examples alongside all 67 originals per epoch, uses 8-example gradient
accumulation, validates each epoch, and saves complete checkpoints every 10
optimizer steps. Early stopping patience is 5. Preparation rejects any
complete input-plus-target example exceeding the model's 8192-token context;
it never truncates a session or track.

Generated files go to the sibling `MostRecentOutput` folder. Commands below
run from this `V4EA-synth` directory. **Run one line at a time**, checking
each result before starting the next. These commands start a completely new
experiment; they do not resume the V3 epoch-7 run.

```powershell
python -m percussion_generation_v4ea train-tokenizer --vocab-size 16384
python -m percussion_generation_v4ea prepare
python -m percussion_generation_v4ea validate
python -m percussion_generation_v4ea train --epochs 12 --examples-per-epoch 268 --learning-rate 2e-5 --save-steps 10 --patience 5
```

To resume after an interruption:

```powershell
python -m percussion_generation_v4ea train --resume-checkpoint latest --epochs 12 --examples-per-epoch 268 --learning-rate 2e-5 --save-steps 10 --patience 5
```

After training:

```powershell
python -m percussion_generation_v4ea evaluate --partition validation20
python -m percussion_generation_v4ea generate --input .\conditioning.txt --output .\generated.txt --target-note-count 8
```

The default output is `..\MostRecentOutput`. Tokenizer and dataset preparation
refuse to overwrite existing outputs. A new tokenizer changes the model's
token-to-embedding mapping; this pipeline explicitly remaps shared token
strings, initializes new token embeddings from old token pieces, and trains
the embedding matrix. Results may differ substantially from V3. The 1.7B
variant needs more memory than the local 6 GB GPU for practical training.
