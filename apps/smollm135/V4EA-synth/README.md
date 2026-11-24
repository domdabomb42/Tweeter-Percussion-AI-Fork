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

## Local AI percussion feedback plugin

The experimental plugin uses the **existing trained 135M V4EA positions-v1**
checkpoint, tokenizer, embedding file, conditioning prompt, note-count target,
constraint, and fixed percussion preset. It does not change any historical
supervised output. The Python bridge is localhost-only. Model generation and
preference training run outside Pd's GUI/audio thread. The bridge refuses
feedback operations unless Tweeter reports an offline session.

From the repository root, launch the bridge and, if installed, PD-L2Ork:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\L2Ork-Tweeter\Start-AIPercussion.ps1
```

If PD-L2Ork is not on `PATH`, pass `-PdExecutable 'C:\path\to\pd-l2ork.exe'`
or open `apps\L2Ork-Tweeter\L2Ork-Tweeter.pd` manually. The script leaves
the bridge running in a hidden process and writes bridge logs in the feedback
directory. In Tweeter, disconnect from multiplayer, then click **RANDOM
SESSION** at the upper right. The bridge picks an eligible augmented training
pair uniformly by original composition, removes its designated percussion
target, verifies the exact supervised prompt and token IDs, and generates five
distinct seeded, constrained alternatives. The first is loaded into the
target's original numeric track slot with native username `AI_A`.

On that track, choose a rating from 1–10 and press **SUBMIT**. A success
indicator follows the durable storage acknowledgment. **NEXT TRACK** queues
the next candidate for the loop boundary and wraps from five to one. The
first NEXT click warns if a candidate has no submitted rating or has an
unsaved changed rating; click NEXT again to leave it. Revisiting a candidate
shows its saved rating; submitting a different rating records a revision.
Other tracks retain their ordinary controls and do not receive candidate
switch messages. Click **AI TRAINING** in the header for collection stats,
manual training controls, and a recovery UUID field. Paste an incomplete
group UUID there and press Enter to reopen it. No training starts from a
rating submission or from completion of five ratings.

Raw feedback is stored at
`$env:LOCALAPPDATA\TweeterAIPercussion\feedback` (normally
`C:\Users\<user>\AppData\Local\TweeterAIPercussion\feedback`). Pass
`-FeedbackDir` to the launcher or `--feedback-dir` to a CLI command to choose
another location **outside the repository**. Each group has an immutable
`group.json` and five native candidate files. Every rating and correction is
an independent fsynced event file under `ratings\<group-id>`. Training only
reads a frozen snapshot; checkpoint cleanup never touches raw feedback.
The 80 completed groups / 400 ratings target is advisory.

From `apps\smollm135\V4EA-synth`, inspect, back up, export, and recover:

```powershell
python -m percussion_generation_v4ea feedback-stats
python -m percussion_generation_v4ea feedback-backup --destination "$env:USERPROFILE\Documents\TweeterAI-feedback-backup-001"
python -m percussion_generation_v4ea feedback-export --destination "$env:USERPROFILE\Documents\TweeterAI-feedback-export-001.json"
python -m percussion_generation_v4ea feedback-recover
python -m percussion_generation_v4ea feedback-recover --group-id '<group-uuid>'
```

Start a separate, immutable DPO round after collecting ratings. Use a new
run directory for every round. The source-family partition keeps rotations
of one composition together, and excludes families belonging to supervised
validation or test. The reference-score cache is shared across rounds under
`..\MostRecentOutput\DPOPreferenceRuns\.reference_scores`; corrupt or
incompatible derived entries are rebuilt. The supervised reference adapter
is never disabled to form a fake reference.

```powershell
$run = '..\MostRecentOutput\DPOPreferenceRuns\round_001'
python -m percussion_generation_v4ea dpo-prepare --run-dir $run --minimum-rating-difference 2
python -m percussion_generation_v4ea dpo-reference --run-dir $run
python -m percussion_generation_v4ea dpo-train --run-dir $run --dry-run
python -m percussion_generation_v4ea dpo-train --run-dir $run --epochs 1 --learning-rate 5e-6 --beta 0.1 --accumulation 4 --save-steps 10 --save-total-limit 3 --save-on-interrupt
```

The DPO loss uses **summed, unmasked completion-token log probabilities**,
including EOS, with correct causal shifting. Each pair weight is its rating
difference divided by all eligible differences within that group. The epoch
objective averages these group losses so a group with more comparisons has
the same total weight as another group. Each optimizer update averages its
four accumulated pair gradients. Only rank-16 LoRA parameters enter AdamW;
trained embeddings, the tied output head, and base weights are frozen and
hashed. A later round may start from an explicitly approved DPO adapter by
passing `--policy-adapter <approved-adapter-path>` to `dpo-train`; the frozen
reference remains the original supervised checkpoint.

Pause at the next successful optimizer boundary or resume exactly:

```powershell
python -m percussion_generation_v4ea dpo-pause --run-dir $run
python -m percussion_generation_v4ea dpo-train --run-dir $run --resume-checkpoint latest --epochs 1 --learning-rate 5e-6 --beta 0.1 --accumulation 4 --save-steps 10 --save-total-limit 3 --save-on-interrupt
python -m percussion_generation_v4ea dpo-checkpoints --run-dir $run
python -m percussion_generation_v4ea dpo-recover --run-dir $run
```

Ctrl+C during training requests a safe boundary save. An unexpected crash
recovers from the latest fully committed checkpoint. If it is corrupt,
`--resume-checkpoint latest` reports the invalid newer checkpoint and falls
back to an earlier compatible one. Periodic checkpoints save every 10
**successful optimizer updates**, retaining the latest three; all epoch,
best, final, and pause checkpoints remain. Training snapshots and
configuration are restored on resume, so ratings submitted after a pause
only enter a **new** round. The progress file is `$run\progress.json`.
The bridge records the active round in `DPOPreferenceRuns\active_round.json`,
so Tweeter's **RESUME TRAINING** control still finds it after restarting Pd.

Evaluate baseline and DPO on identical held-out inputs, sampling settings,
and seeds. The comparison reports rhythmic diversity, native validity,
latency, and teacher-forced token loss. It leaves human preference win rate
unset until actual paired human votes are collected:

```powershell
$best = (Get-ChildItem "$run\checkpoints\best" -Directory | Sort-Object Name | Select-Object -Last 1).FullName
python -m percussion_generation_v4ea dpo-evaluate --adapter "$best\adapter" --partition validation20 --seed 415 --temperature 0.8 --top-p 0.95
python -m percussion_generation_v4ea dpo-approve --checkpoint $best
```

Approval is explicit and only accepts verified non-prunable best, epoch, or
final checkpoints. In Tweeter's AI training panel, **SELECT APPROVED DPO**
uses the approved pointer for future generation; **SELECT SUPERVISED**
returns to the original trained 135M model. Neither action changes the
original supervised checkpoint.
