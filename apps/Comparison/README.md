# Five-model percussion comparison

This folder compares the historical **135M** and **360M** V3 percussion
generators with the new **135MEA**, **360MEA**, and **17BEA** V4EA position
generators. `EA` means the new V4EA experiment. The comparison uses the same
five held-out target tracks and the original requested note count for each.

Run this from `apps/Comparison` to build the report from saved evaluations:

```powershell
python compare.py
```

The output is [five_model_comparison.md](five_model_comparison.md) and
[five_model_comparison.json](five_model_comparison.json). Models without a
finished evaluation appear as **Pending**. The command does not load or train
a model.

After a V4EA model finishes training, evaluate both greedy and constrained
decoding and save distinct report snapshots:

```powershell
python compare.py --evaluate-ea 135mea
python compare.py --evaluate-ea 360mea
python compare.py --evaluate-ea 17bea
python compare.py --strict
```

`--evaluate-ea` runs test evaluation only; it uses the model's current best
adapter and writes snapshots under `reports/`. Use `--cpu` if needed. The
comparison rejects a snapshot when its dataset or best adapter has changed.
Run `--evaluate-ea` again after further training. Wait until training has
finished before evaluating on the same GPU.

The table compares structural validity, exact note count, native export rate,
and rhythmic Jaccard on structurally valid outputs. Invalid outputs affect
the validity rate, while Jaccard lists its own denominator. Loss and timing
are excluded from the ranking because the tokenizers, input formats, and
latency measurement scopes differ. The V3 135M evaluation used a 128-token
generation cap; V3 360M and V4EA used 512.

The original two-model V3 comparison is preserved in
[README_v3_historical.md](README_v3_historical.md),
[comparison.json](comparison.json), [greedy.md](greedy.md), and
[constrained.md](constrained.md). Rebuild it with:

```powershell
python compare.py --historical-v3
```
