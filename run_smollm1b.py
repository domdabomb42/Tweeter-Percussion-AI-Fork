"""Prepare, feature-train, then percussion-train the independent 1.7B experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FEATURE = ROOT / "apps/smollm1b/CustomTraining"
PERCUSSION = ROOT / "apps/smollm1b/V3"


def run(command, cwd, dry_run=False):
    print(f"\n[{cwd.relative_to(ROOT)}] {' '.join(command)}", flush=True)
    if not dry_run:
        environment = os.environ.copy()
        environment.setdefault("USE_TF", "0")
        subprocess.run(command, cwd=cwd, env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    python = sys.executable
    feature_dataset = FEATURE / "training_output/dataset/manifest.json"
    feature_adapter = FEATURE / "training_output/checkpoints/best_adapter/adapter_model.safetensors"
    rhythm_dataset = PERCUSSION / "output/manifest.json"
    rhythm_adapter = PERCUSSION / "output/checkpoints/rhythm/best_adapter/adapter_model.safetensors"
    if not feature_dataset.is_file():
        run([python, "-m", "custom_training", "prepare"], FEATURE, args.dry_run)
    if not feature_adapter.is_file():
        run([python, "-m", "custom_training", "train"], FEATURE, args.dry_run)
    if not rhythm_dataset.is_file():
        run([python, "-m", "percussion_generation_v3", "prepare"], PERCUSSION, args.dry_run)
    if not rhythm_adapter.is_file():
        run([python, "run_overnight.py"], PERCUSSION, args.dry_run)
    print("1.7B feature and V3 stages complete." if not args.dry_run else
          "Dry run complete; no training started.", flush=True)


if __name__ == "__main__":
    main()
