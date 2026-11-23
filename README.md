# Tweeter Percussion AI Fork

This repository is a fork of [Pd-L2Ork](https://github.com/pd-l2ork/pd-l2ork), the visual programming environment for sound, music, and multimedia. It keeps the Pd-L2Ork source and the L2Ork Tweeter app while adding tools and experiments for learning percussion patterns from native Tweeter sessions.

## Installation

Clone the fork to get the Tweeter sessions and Python tools:

```sh
git clone https://github.com/domdabomb42/Tweeter-Percussion-AI-Fork.git
cd Tweeter-Percussion-AI-Fork
```

### Pd-L2Ork and Tweeter

For a ready-to-use Pd-L2Ork installation, see the [official downloads](https://l2ork.bukvic.net/main/make-your-own-l2ork/software/) for Linux, macOS, and Windows. Open `apps/L2Ork-Tweeter/L2Ork-Tweeter.pd` from this checkout in Pd-L2Ork to use Tweeter.

To build Pd-L2Ork from this fork, install the platform dependencies in the upstream [Linux](https://github.com/pd-l2ork/pd-l2ork#linux), [macOS](https://github.com/pd-l2ork/pd-l2ork#osx-64-bit-using-homebrew), or [Windows](https://github.com/pd-l2ork/pd-l2ork#windows-64-bit-using-msys2) guide, then run:

```sh
git submodule update --init --recursive
make
```

For a Windows source build, use the MSYS2 MINGW64 shell and clone into an MSYS2 path without spaces, as described in the upstream guide.

### Python tools and model training

Use Python 3.10 or newer. From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. Then install the model dependencies:

```sh
python -m pip install -r apps/smollm135/CustomTraining/requirements.txt
```

The extractor can be installed separately with `python -m pip install -e apps/CustomExtractor`.

To prepare and train a new 135M percussion adapter using the original and augmented sessions:

```sh
cd apps/smollm135/V3
python -m percussion_generation_v3 --output-dir output_augmented prepare
python -m percussion_generation_v3 --output-dir output_augmented train
```

Preparation and training report progress about every three minutes. Training starts from the saved 135M feature adapter. The [135M](apps/smollm135/README.md) and [360M](apps/smollm360/README.md) READMEs have the other model commands and experiment details.

## What this fork adds

- [CustomExtractor](apps/CustomExtractor/README.md) parses native Tweeter session files into editable structured data and exports them back to Tweeter's format.
- [AugmentedSessions](apps/L2Ork-Tweeter/AugmentedSessions/README.md) creates validated time shifts of the original sessions without changing their notes or instruments.
- [SmolLM2 experiments](apps/smollm135/README.md) train feature and percussion adapters for 135M and 360M models; a separate [1.7B experiment](apps/smollm1b/README.md) is also prepared. The percussion trainers can add shifted examples from training sessions while keeping validation and test sessions held out.
- [Comparison](apps/Comparison/README.md) measures the saved 135M and 360M percussion generators with greedy and constrained decoding.

The original Pd-L2Ork project, its build requirements, and its license remain documented upstream. See [LICENSE.md](LICENSE.md) for this repository's license text.
