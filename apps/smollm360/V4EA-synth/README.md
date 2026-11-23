# V4EA musical percussion experiment

This is a new, isolated experiment. Its default output is
`../MostRecentOutput/V4EA-music-v3`; historical V3 and V4EA outputs are left
untouched. Run the commands below from this model's `V4EA-synth` directory.

The deterministic version-3 input is a positional JSON array containing the
requested note count and, in original order, each conditioning track's role,
instrument name, loop length, and note step/pitch pairs. The
version-2 target contains only a percussion instrument name, loop length,
note duration, transposition, and step/pitch pairs. It ends with `[END_MUSIC]`
and the pinned model's EOS token. Synth, effects, mixing, playback, track IDs,
and unrelated metadata are never output tokens. Instrument usernames remain
in the native source tracks and preset catalog but are excluded from model input. All three model sizes use the
same schema and source sessions; each trains its own BPE tokenizer from the
67 original training pairs and their permitted augmented children.

The preparation step creates a validated instrument preset catalog from
original native tracks, strips notes, and stores the sound settings separately
from model targets. It includes sound templates for held-out instrument names
so native reconstruction can be evaluated; held-out notes are never in the
tokenizer corpus or training pool. Generation selects the named preset, adds
the predicted notes and timing, assigns a free track ID, and verifies export
and reimport. Unknown instruments or malformed music fail clearly. There is
no fallback rhythm. An instrument absent from the training targets may still
be difficult for the model to predict accurately.
The inherited split preserves all 67 original training pairs and the original
10 validation pairs; their broader beat-sequence family overlap remains.

Training uses a fresh rank-16 LoRA adapter and newly remapped, trainable tied
embeddings from the pinned pretrained model. Each forward pass includes the
entire conditioning prefix and target. Prefix labels are `-100`, so target
cross-entropy backpropagates through the prefix. No detached KV cache is used
for training. Gradient checkpointing, disabled training cache, pretrained
BF16/FP16 precision on supported GPUs, 8-example accumulation, and optional
CPU execution control memory. The 8192-token limit is strict. A 6 GB GPU is
unlikely to fit full-context 1.7B training; use a larger GPU or `--cpu` (slow).

Default learning rates are:

| Model | LoRA | Embeddings |
| --- | ---: | ---: |
| 135M | 1e-4 | 1e-4 |
| 360M | 1e-4 | 5e-5 |
| 1.7B | 5e-5 | 5e-5 |

AdamW uses 0.01 weight decay for LoRA, gradient clipping at 1.0, 10% linear
warmup, then cosine decay. With 12 epochs, 268 examples per epoch, and 8
examples per update, the plan is 34 updates per epoch, 408 total, and 41
warmup updates. All 67 originals and 201 rotating augmented examples appear
per epoch. The fixed 20-pair validation set determines the best adapter;
patience is 5. Step checkpoints save every 10 optimizer updates and include
LoRA, embeddings, optimizer, scheduler, scaler, RNG, and example order.

```powershell
python -m percussion_generation_v4ea train-tokenizer --vocab-size 16384
python -m percussion_generation_v4ea prepare
python -m percussion_generation_v4ea validate
python -m percussion_generation_v4ea train --epochs 12 --examples-per-epoch 268 --lora-learning-rate 1e-4 --embedding-learning-rate 5e-5 --save-steps 10 --patience 5
python -m percussion_generation_v4ea train --resume-checkpoint latest --epochs 12 --examples-per-epoch 268 --lora-learning-rate 1e-4 --embedding-learning-rate 5e-5 --save-steps 10 --patience 5
python -m percussion_generation_v4ea evaluate --partition validation20
python -m percussion_generation_v4ea evaluate --partition test
python -m percussion_generation_v4ea generate --input ..\..\L2Ork-Tweeter\sessions\canada-start-session-coll.txt --output .\generated-percussion.txt --target-note-count 8
```

For 360M and 1.7B, change the learning-rate flags to the values in the table.
For CPU use, add `--cpu` to train, resume, evaluate, and generate; resume must
use the same device type as the checkpoint. Run the 1.7B `train --dry-run`
first to inspect the plan. Preparation refuses to overwrite a tokenizer or
dataset; use a new output directory to rebuild. Evaluation writes
`evaluation_validation20.json` or `evaluation_test.json` into the experiment
output. Generated-track metrics are comparable across model sizes; token
cross-entropy is tokenizer-specific and is not directly comparable.

From any one V4EA directory, after all three test evaluations exist:

```powershell
python -m percussion_generation_v4ea compare --reports ..\..\smollm135\MostRecentOutput\V4EA-music-v3\evaluation_test.json ..\..\smollm360\MostRecentOutput\V4EA-music-v3\evaluation_test.json ..\..\smollm1b\MostRecentOutput\V4EA-music-v3\evaluation_test.json --output .\three-model-comparison.json
```
