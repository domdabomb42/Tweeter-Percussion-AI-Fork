"""Resume 135M V3, then train 1.7B features and 1.7B V3 in sequence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
V3_135 = ROOT / "apps/smollm135/V3"
FEATURE_135 = ROOT / "apps/smollm135/CustomTraining/training_output/checkpoints/best_adapter/adapter_model.safetensors"


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
    if not FEATURE_135.is_file():
        raise SystemExit("135M feature adapter is missing; finish feature training first")

    output = V3_135 / "output"
    history_path = output / "checkpoints/rhythm/history.json"
    adapter = output / "checkpoints/rhythm/best_adapter/adapter_model.safetensors"
    result = output / "overnight_result.json"
    if result.is_file():
        print("135M V3 overnight run already completed; keeping its checkpoints.", flush=True)
    elif history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        completed = len(history.get("epochs", []))
        if not adapter.is_file() or not 1 <= history.get("best_epoch", 0) <= completed:
            raise SystemExit("135M V3 history has no usable saved best adapter")
        if completed >= 20:
            raise SystemExit("135M V3 reached 20 epochs but its final evaluation is incomplete")
        print(f"135M V3 has {completed} completed epochs; saved best is epoch "
              f"{history['best_epoch']}.", flush=True)
        run([sys.executable, "run_overnight.py", "--resume", "--epochs", "20"],
            V3_135, args.dry_run)
    else:
        run([sys.executable, "run_overnight.py"], V3_135, args.dry_run)

    run([sys.executable, str(ROOT / "run_smollm1b.py")]
        + (["--dry-run"] if args.dry_run else []), ROOT, args.dry_run)
    print("Both stages completed." if not args.dry_run else
          "Dry run complete; no training started.", flush=True)


if __name__ == "__main__":
    main()
