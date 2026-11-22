"""One-command V3 baseline training followed by constrained decoding evaluation."""
from pathlib import Path
import sys

from percussion_generation_v3.cli import main


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parent / "output" / "overnight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_out, original_err = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8") as log:
        sys.stdout = Tee(original_out, log)
        sys.stderr = Tee(original_err, log)
        try:
            print(f"V3 overnight log: {log_path}", flush=True)
            code = main(["overnight", *sys.argv[1:]])
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    raise SystemExit(code)
