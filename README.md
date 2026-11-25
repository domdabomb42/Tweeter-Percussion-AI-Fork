# Tweeter Percussion AI Fork

This fork of [Pd-L2Ork](https://github.com/pd-l2ork/pd-l2ork) adds native Tweeter session extraction, augmentation, percussion-model experiments, and an **experimental AI percussion feedback plugin**. The plugin uses the trained SmolLM2-135M V4EA position model. It requests five rhythm candidates for one eligible session and stores 1–10 ratings for later preference training. The 360M and 1.7B models are comparison models; the plugin does not select them.

The Python bridge, feedback storage, and headless Pd switching tests exist. The complete GUI rating and NEXT TRACK workflow has **not** been verified end to end; see [known limitations](apps/smollm135/V4EA-synth/README.md#integration-status-and-limitations). No DPO model has been trained or approved through the plugin workflow.

## Install and launch on Windows

1. Install [Pd-L2Ork](https://l2ork.bukvic.net/main/make-your-own-l2ork/software/) and Python 3.10 or newer. Clone this branch:

   ```powershell
   git clone --branch firstpluginattempt https://github.com/domdabomb42/Tweeter-Percussion-AI-Fork.git
   cd Tweeter-Percussion-AI-Fork
   ```

2. Install the Python dependencies from the repository root:

   ```powershell
   python -m pip install -r .\apps\smollm135\CustomTraining\requirements.txt
   python -m pip install -e .\apps\CustomExtractor
   ```

3. Supply the trained 135M V4EA positions-v1 output locally under `apps\smollm135\MostRecentOutput\V4EA-positions-v1`. It needs its tokenizer, prepared dataset manifest, fixed preset, and `checkpoints\rhythm_coverage_v1\best_adapter` with trained embeddings. The trained V4EA artifacts are **not included in this branch**: GitHub rejected the attempted Git LFS publication to this public fork. There is currently no repository download for them. See [artifact requirements](apps/smollm135/MostRecentOutput/README.md).

4. Launch the local bridge and Tweeter:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\L2Ork-Tweeter\Start-AIPercussion.ps1
   ```

   If Pd-L2Ork is not on `PATH`, pass `-PdExecutable 'C:\path\to\pd-l2ork.exe'`. To stop the bridge, repeat the command with `-Stop`. Disconnect from multiplayer before using AI controls. The bridge listens on localhost and stores feedback outside the repository by default.

On Linux or macOS, install/build Pd-L2Ork using the [upstream instructions](https://github.com/pd-l2ork/pd-l2ork) and open `apps/L2Ork-Tweeter/L2Ork-Tweeter.pd`. The PowerShell bridge launcher is Windows-specific. Native plugin GUI behavior on those platforms has not been tested.

## Plugin and model workflow

**RANDOM SESSION** requests five distinct generated candidates. The designated AI track is named `AI_A`; its rating window offers **SUBMIT** and **NEXT TRACK**. **AI TRAINING** opens collection statistics, recovery, and explicit DPO controls. Ratings are stored as durable event files. DPO preparation freezes submitted preferences; training begins only when requested, supports pause/resume, and saves every 10 successful optimizer steps. The supervised adapter remains the default until an approved DPO adapter is selected. Follow the [135M V4EA guide](apps/smollm135/V4EA-synth/README.md#local-ai-percussion-feedback-plugin) for commands and manual verification.

V4EA's current target is **tracker positions from 0 to 63**, not a complete instrument track. Native export combines those positions with a fixed ElectricKick synthesizer preset. New generations use Tweeter factory #64's output volume, pan, and zero echo amplitude. Historical full-musical-target experiments and V3 outputs remain available.

| Component | Purpose |
| --- | --- |
| [CustomExtractor](apps/CustomExtractor/README.md) | Parse, validate, and export native Tweeter sessions. |
| [AugmentedSessions](apps/L2Ork-Tweeter/AugmentedSessions/README.md) | Lossless shifted copies of source sessions. |
| [135M V4EA](apps/smollm135/V4EA-synth/README.md) | Selected plugin model and feedback/DPO bridge. |
| [360M V4EA](apps/smollm360/V4EA-synth/README.md) and [1.7B V4EA](apps/smollm1b/V4EA-synth/README.md) | Trained comparison experiments. |
| [Comparison](apps/Comparison/README.md) | Five-model held-out evaluation and native samples. |

The [historical V3 experiment](apps/smollm135/V3/README.md), upstream build documentation, and [license](LICENSE.md) remain available.
