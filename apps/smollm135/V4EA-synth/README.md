# V4EA percussion generation

The default new experiment writes to `../MostRecentOutput/V4EA-positions-v1`.
The historical full musical target, its tokenizer, dataset, and checkpoints remain
in `../MostRecentOutput/V4EA-music-v3`. The two output directories are separate.

Both formats use the same `V4EA-MUSIC-INPUT-3` conditioning array: requested
note count, then every remaining track's role, instrument name, loop length,
and `[step,pitch]` pairs in original order. `instrument.username` stays in
native Tweeter files but is omitted from model input. The new target is only
ascending, space separated tracker step numbers from 0 to 63, followed by
`[END_RHYTHM]` and EOS. It is the V3 rhythm target grammar. A fixed, validated
V3 percussion preset supplies instrument, pitch, synth, effects, mixing, and
playback settings during native export. The original `V4EA-MUSIC-TARGET-2`
full musical target, its `[END_MUSIC]` marker, and named preset catalog remain
available through the historical output directory.

Preparation uses the same 2,688 training pairs, held-out source splits, and
augmented sessions. It trains a fresh BPE tokenizer from the chosen target
format and saves a separate dataset. Training keeps the rank-16 LoRA
architecture, tied trainable embeddings, target-only loss masking, learning
rates, and full-pool coverage sampler. The default sampler presents the entire
2,688-pair pool twice, validating every 536 examples. A complete
resumable checkpoint is saved after each epoch; only the latest epoch
checkpoint is retained, alongside the best adapter and final checkpoint.
Periodic step saves are disabled by default. Early stopping uses patience 3
and starts only after one complete dataset pass. The inherited
67 original training pairs and original 10 validation pairs have a broader
beat-sequence family overlap.

Run from this `V4EA-synth` folder. The training command runs the full job.

```powershell
python -m percussion_generation_v4ea train-tokenizer --vocab-size 16384
python -m percussion_generation_v4ea prepare
python -m percussion_generation_v4ea validate
python -m percussion_generation_v4ea train --dataset-passes 2 --examples-per-epoch 536 --lora-learning-rate 1e-4 --embedding-learning-rate 1e-4 --save-steps 0 --patience 3
```

Resume or evaluate:

```powershell
python -m percussion_generation_v4ea train --resume-checkpoint latest --dataset-passes 2 --examples-per-epoch 536 --lora-learning-rate 1e-4 --embedding-learning-rate 1e-4 --save-steps 0 --patience 3
python -m percussion_generation_v4ea evaluate --partition validation20
python -m percussion_generation_v4ea evaluate --partition test --constrained
```

After training, this PowerShell loop creates five native Tweeter text files in
`QuickRunOutputs`. The same source is used with five requested note counts.
`--constrained` enforces ascending positions and the count during inference;
it does not change training.

```powershell
$counts = @(4, 6, 8, 10, 12)
for ($i = 0; $i -lt 5; $i++) {
  python -m percussion_generation_v4ea generate --input ..\..\L2Ork-Tweeter\sessions\1st-beat-fall-2021-session-coll.txt --output ".\QuickRunOutputs\percussion_$($i + 1).txt" --target-note-count $counts[$i] --constrained
  if ($LASTEXITCODE -ne 0) { throw "Generation $($i + 1) failed" }
}
```

The historical musical experiment is selected explicitly:

```powershell
python -m percussion_generation_v4ea --output-dir ..\MostRecentOutput\V4EA-music-v3 evaluate --partition validation20 --checkpoint-run legacy
python -m percussion_generation_v4ea --output-dir ..\MostRecentOutput\V4EA-music-v3 generate --input ..\..\L2Ork-Tweeter\sessions\canada-start-session-coll.txt --output .\QuickRunOutputs\historical.txt --target-note-count 8 --checkpoint-run legacy
```

For a new full musical target dataset, use a fresh output directory and pass
`--target-format musical-v2` to both `train-tokenizer` and `prepare`.
Preparation refuses to overwrite an existing dataset or tokenizer.

For unattended sequential training of all three model sizes from the
repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\run_v4ea_overnight.ps1
```

The runner prepares missing tokenizers and datasets, resumes existing
checkpoints, and advances after successful completion. New 1.7B runs use
the GPU: a local 4,096-token LoRA optimizer-step smoke test peaked at
5,338 MiB reserved on the 6 GiB RTX 3060. Resuming a CPU checkpoint
keeps the original device. Keep other GPU workloads closed while training.
