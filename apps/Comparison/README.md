# SmolLM2 V3 percussion comparison

The requested `smollm350` is the existing `smollm360` experiment.
Both checkpoints were evaluated with greedy and constrained decoding on
the same five held-out percussion targets from three sessions. They also
received 30 requested-note-count trials each per decoding method.
The constrained adapter is a copy of the rhythm adapter's weights;
constrained decoding is a rule at inference, not a second trained model.
The new augmented sessions were not used to train these checkpoints.

| Decoding | Model | Test loss | Perplexity | Onset F1 | Exact count (30) | Valid (35) | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | smollm135 | 0.739 | 2.094 | 0.082 | 0.0% | 65.7% | 36.345 s (cuda) |
| greedy | smollm360 | 0.690 | 1.995 | 0.172 | 0.0% | 100.0% | 6.596 s (unrecorded) |
| constrained | smollm135 | 0.739 | 2.094 | 0.151 | 100.0% | 100.0% | 34.720 s (cuda) |
| constrained | smollm360 | 0.690 | 1.995 | 0.179 | 100.0% | 100.0% | 10.781 s (unrecorded) |

Constrained decoding achieved the requested count in all 30 sweep trials
for each model; greedy decoding achieved it in none. Both constrained
runs exported 35/35 valid tracks. The 360M checkpoint has lower held-out
loss and higher onset F1 on this small test set. Onset F1
remains below 0.18 in every condition, so count compliance alone does
not establish that the rhythm fits the source music.

Detailed tables cover groove proxies, diversity, training time and peak GPU use:
[greedy results](greedy.md) and [constrained results](constrained.md).
[comparison.json](comparison.json) holds the calculated measurements and source hashes.
The complete 360M evaluations used a 512-token cap, but all outputs
ended before 128 tokens. The 135M runs used a 128-token cap.
The 360M reports did not record the evaluation device, so the latency
figures are not a controlled speed comparison.

No human listening score has been measured. Groove scores are grid-based
proxies and do not establish that a generated percussion line sounds good.

## Reproduce

From this directory run `python compare.py`. The raw
V3 evaluation commands are given in each detailed report. They evaluate
existing checkpoints and do not train a model.
