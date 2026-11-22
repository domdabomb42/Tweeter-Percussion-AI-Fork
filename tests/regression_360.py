"""Compare deterministic 360M inference against committed pre-move Tweeter files."""
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "apps/smollm360/V3"
sys.path.insert(0, str(V3))
from percussion_generation_v3.inference import generate_file


def main():
    source = V3 / "output/POC_inputs/simple-melody-bass.txt"
    for name, cd2 in (("simple-greedy-8.txt", False),
                      ("simple2_constrained_8.txt", True)):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / f"regression-{name}"
            result = generate_file(source, output, 8, V3 / "output",
                                   max_new_tokens=512, cd2=cd2, loop_ms=10800)
            try:
                assert result["export_valid"], result.get("error")
                expected = (V3 / "POC" / name).read_bytes()
                assert output.read_bytes() == expected, name
                print(f"PASS {name}: exact Tweeter bytes", flush=True)
            finally:
                Path(result["diagnostics"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
