# V3 implementation report

## Completed before training

- Confirmed V1/V2 locations, source session split, tokenizer, LoRA
  configuration, fixed percussion preset, and V2's frozen starting feature
  adapter. The V3 adapter copy has the same SHA-256 as V2's starting copy.
- Snapshotted 127 V1/V2 files in `preservation_manifest.json`; verification
  finds no changed, added, or missing files.
- Built the V3 position-only dataset from V2's exact rhythm rows: 67 train
  pairs from 26 sessions, 10 validation pairs from 4 sessions, and 5 test
  pairs from 3 sessions. Train note-count median is 10, range 1–64; 24 of 67
  train targets have at most five notes. Maximum V3 train sequence is 8,143
  tokens, within the unchanged 8,192-token context.
- Implemented target-only masked LoRA training, early stopping, best checkpoint
  saving, strict position parsing, fixed-preset reconstruction, native export
  and reimport verification, baseline and constrained decoding, evaluation,
  and a V1/V2/V3 comparison script.
- The `overnight` command dry run confirms both stage paths without creating a
  checkpoint. Automated tests and a read-only eight-token model inference
  smoke test passed. The smoke test ran on CUDA; it did not train V3.

## Pending until the user runs the overnight command

No V3 model has been trained. V3 validation loss, count accuracy, rhythm
quality, output validity, inference latency, and any improvement over V1/V2
are not yet measurable. The constrained stage will reuse the baseline model
weights; it tests whether restricting output tokens changes count accuracy and
latency, not whether a second trained model learned better rhythms.

The data is small, with only three distinct held-out test sessions. Native
Tweeter allows one pitch per step in a single track, so V3's one-track output
cannot represent simultaneous events at the same position. Loop length is
selected from verified percussion or melody tracks, or supplied explicitly;
wrong loop selection remains a possible musical error. A valid export and exact note count do not
establish musical quality. No listening test has been run.

## 6 GB GPU memory fix

The first overnight attempt loaded the model, then failed in backward with a
CUDA/CUBLAS error. A non-persistent reproduction showed 4.65 GB peak allocation
in forward on a 6,904-token example, followed by an allocation failure in
backward. CPU activation offload with gradient checkpointing completed isolated
examples, but the next overnight attempt exhausted system RAM on a 1.5 GB
allocation. The trainer now computes the full conditioning prefix without
gradients and passes its attention cache to a short trainable segment containing
the last prefix token and every rhythm target token. This preserves the full
input and target-only labels, but gradients do not flow through the earlier
prefix tokens. The longest prompt and the largest target both completed
backward in non-persistent CUDA diagnostics. Peak GPU allocation was about
2.0–4.8 GB. Adam moments stay on CPU between updates. The full overnight run
has not yet been completed with this fix.
