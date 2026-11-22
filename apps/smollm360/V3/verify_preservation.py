"""Check the V1/V2 files against the snapshot taken before V3 work."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS = ROOT.parents[1]
MANIFEST = ROOT / "preservation_manifest.json"
VERSIONS = ("PercussionGeneration", "PercussionGenerationV2")


def snapshot():
    rows = {}
    for name in VERSIONS:
        for path in (APPS / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.suffix != ".log":
                key = path.relative_to(APPS).as_posix()
                rows[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def verify():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    current = snapshot()
    return {"missing": sorted(expected.keys() - current.keys()),
            "added": sorted(current.keys() - expected.keys()),
            "changed": sorted(k for k in expected.keys() & current.keys()
                              if expected[k] != current[k])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record:
        if MANIFEST.exists():
            raise SystemExit("snapshot already exists; refusing to replace it")
        files = snapshot()
        MANIFEST.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {len(files)} V1/V2 files")
    else:
        result = verify()
        print(json.dumps(result, indent=2))
        if any(result.values()):
            raise SystemExit(1)
