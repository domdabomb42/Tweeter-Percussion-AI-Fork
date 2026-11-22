# SmolLM2 V3 percussion comparison: greedy

This compares the **135M** model with the existing **360M** model (the requested
“smollm350”). Both were scored with greedy decoding on the same five held-out
percussion targets from three source sessions. Each target was also requested
at six note counts (5, 8, 10, 12, 16, 20), giving 35 generation attempts per
model. The 135M evaluation recorded a GPU device; the earlier 360M report did
not record its device. The saved training runs were not controlled hardware
benchmarks.
Both checkpoints were trained on the original V3 pairs; the new
`L2Ork-Tweeter/AugmentedSessions` files were not used.

The 135M generation limit was 128 new tokens per request. The 360M
evaluation used a 512-token limit, but all 35 outputs ended by token
95, within the 135M limit. Its complete report is preserved as
`smollm360_greedy_512_backup.json`.

| Metric | 135M | 360M |
| --- | ---: | ---: |
| Held-out target loss / perplexity | 0.739 / 2.094 | 0.690 / 1.995 |
| Onset F1, 5 original-count requests | 0.082 | 0.172 |
| Exact note count, 30 count-sweep requests | 0.0% | 0.0% |
| Valid generation, all 35 requests | 65.7% | 100.0% |
| 16-step repetition proxy; real 0.381 | 0.405 | 0.700 |
| 32-step repetition proxy; real 0.368 | 0.457 | 0.700 |
| 4-step pulse alignment proxy; real 36.2% | 100.0% | 95.0% |
| Off-pulse positions proxy; real 63.8% | 0.0% | 5.0% |
| Circular onset-gap variation; real 0.514 | 0.399 | 0.186 |
| Unique rhythms, 5 original-count requests | 100.0% | 60.0% |
| Unique rhythms, 30 count-sweep requests | 35.0% | 16.7% |
| Pairwise rhythm distance; real 0.930 | 0.603 | 0.320 |
| Human listening score | Not measured | Not measured |
| Mean generation latency, 35 requests | 36.345 s (GPU) | 6.596 s (device unrecorded) |
| Peak GPU allocation during evaluation | 6.96 GiB | Not recorded |
| Recorded mean training epoch | 61.1 min | 2.8 min |
| Recorded epoch time, all completed epochs | 9.17 h (9 epochs) | 0.52 h (11 epochs) |
| Recorded training GPU peak | 8.06 GiB | Not recorded |

Lower held-out loss favors **360M** on these five examples. Onset F1 uses
the original requested count and compares generated tracker positions with
the actual withheld percussion track; invalid outputs score zero. Exact count
and validity also include failed generations in their denominators. A
generation is valid only when parsing,
0–63 step bounds, uniqueness and native Tweeter export all succeed.

The groove rows are **descriptive proxies**, computed on valid original-count
outputs (135M: 3/5; 360M:
5/5) and the five real target tracks.
A 4-step pulse is a grid assumption,
not a verified time signature or a claim about musical beat strength.
“Off-pulse” is only a syncopation proxy. A high repetition score need not sound
better; compare its closeness to the real tracks. These statistics and onset F1
do not establish musical coherence. The 135M and 360M models predict tracker
positions only and use a fixed percussion preset, so they do not learn the
source track's sound design.

**Human listening was not performed.** A meaningful score requires rendered
audio from both models over the same accompaniment, randomized presentation,
blinded human ratings and a written scale. No numerical score is assigned.

Generation latency includes parsing/export. The 135M run used the GPU; the
360M report does not identify its device, so these latency numbers are not
a controlled speed comparison. The training times come from separate
historical runs. The 135M history records GPU peak allocation;
the 360M history does not. The peak value therefore cannot be ranked fairly.
The 135M recorded peak exceeds this GPU's 6 GiB of dedicated memory;
Windows can use shared GPU memory, and the historical run's memory setup
was not independently recorded.
See `comparison.json` for individual measurements, exact denominators,
reference-track proxies and source hashes.

## Reproduce

From `apps/Comparison`, run `python compare.py`. The script reads
`../smollm135/V3/output/evaluation/v3_greedy_test.json` and the saved
360M report in this folder, checkpoint histories and held-out row manifests. It checks
that the two reports use identical test targets and generation requests, then
regenerates `comparison.json` and this README. To regenerate the 135M raw
evaluation, run from `apps/smollm135/V3`:

```powershell
$env:USE_TF='0'
python -m percussion_generation_v3 evaluate --partition test --cap 5 --counts 5,8,10,12,16,20 --max-new-tokens 128
```

The analogous command in `apps/smollm360/V3` can create a new GPU evaluation,
but it overwrites that model's original evaluation file. Neither command
trains or changes a checkpoint. Keep the hardware workload similar if
repeating latency measurements.
