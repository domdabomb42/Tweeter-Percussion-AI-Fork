# Five-model percussion comparison

The table uses the same five held-out target tracks for the historical
135M and 360M V3 models and the 135M, 360M, and 1.7B V4EA position models.
Each row covers five original-count requests. Missing V4EA evaluations
are shown as Pending; no scores are estimated.

| Model | Decoder | Structural validity | Exact count | Native export | Rhythmic Jaccard (valid outputs) |
| --- | --- | ---: | ---: | ---: | ---: |
| 135m | greedy | 60% | 20% | 60% | 0.078 (3/5) |
| 135m | constrained | 100% | 100% | 100% | 0.087 (5/5) |
| 360m | greedy | 100% | 0% | 100% | 0.129 (5/5) |
| 360m | constrained | 100% | 100% | 100% | 0.131 (5/5) |
| 135mea | greedy | 80% | 0% | 80% | 0.281 (4/5) |
| 135mea | constrained | 100% | 100% | 100% | 0.202 (5/5) |
| 360mea | greedy | 100% | 60% | 100% | 0.061 (5/5) |
| 360mea | constrained | 100% | 100% | 100% | 0.075 (5/5) |
| 17bea | greedy | 80% | 80% | 80% | 0.196 (4/5) |
| 17bea | constrained | 100% | 100% | 100% | 0.157 (5/5) |

Rhythmic Jaccard is averaged over structurally valid outputs only; its
denominator appears in each cell. Invalid outputs lower the validity rate
but do not enter that Jaccard mean. V3 uses the five reference requests
from its saved 35-request evaluation; sweep requests are excluded.

Token cross-entropy is omitted from the ranking because tokenizers and
conditioning representations differ. Inference time is omitted because the saved V3
latencies include parsing and export while V4EA records generation time.
V3-135M used a 128-token generation cap; the historical V3-360M report
used 512. V4EA evaluation defaults to 512. These are separate training
runs and do not establish a controlled speed comparison.

## Reproduce

From `apps/Comparison`:

```powershell
python compare.py
# After each V4EA model finishes training:
python compare.py --evaluate-ea 135mea
python compare.py --evaluate-ea 360mea
python compare.py --evaluate-ea 17bea
# Require all ten rows to be present:
python compare.py --strict
```

`--evaluate-ea` performs test evaluation only. It does not train models.
The two V3 reports and historical comparison files remain unchanged.
