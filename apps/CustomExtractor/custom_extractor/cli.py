import argparse
import json
from pathlib import Path
import sys
import time
from . import extract_tweeter, export_tweeter, load_json, save_json, dumps, loads, validate
from .exporter import render_tweeter
from .extractor import extract_text
from .reporting import coverage, canonical, musical_features


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CustomExtractor: native Tweeter TXT <-> editable JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("inspect", "Show all extracted features plus diagnostics"),
                            ("extract", "Extract native Tweeter into versioned JSON"),
                            ("export-tweeter", "Reconstruct native Tweeter TXT from JSON"),
                            ("validate", "Validate composition JSON"),
                            ("roundtrip", "Measure TXT -> JSON -> TXT fidelity"),
                            ("coverage", "Show native feature coverage")):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("input", type=Path)
        if name in ("extract", "export-tweeter"):
            cmd.add_argument("--output", "-o", required=True, type=Path)
        if name == "coverage":
            cmd.add_argument("--records", action="store_true", help="Include a diagnostic for every source record")
    args = parser.parse_args(argv)
    try:
        if args.command in ("validate", "export-tweeter"):
            composition = load_json(args.input)
        else:
            start = time.perf_counter()
            composition = extract_tweeter(args.input)
            extract_ms = (time.perf_counter() - start) * 1000
        if args.command == "inspect":
            data = composition.to_dict()
            data.pop("native")
            _print({"composition": data, "derived_features": musical_features(composition), "coverage": coverage(composition)})
        elif args.command == "extract":
            save_json(composition, args.output)
            _print({"output": str(args.output), "extraction_ms": extract_ms})
        elif args.command == "export-tweeter":
            export_tweeter(composition, args.output)
            _print({"output": str(args.output)})
        elif args.command == "validate":
            _print({"valid": True, "warnings": validate(composition)})
        elif args.command == "coverage":
            _print(coverage(composition, args.records))
        else:
            serialized = dumps(composition)
            restored = loads(serialized)
            start = time.perf_counter()
            output = render_tweeter(restored)
            export_ms = (time.perf_counter() - start) * 1000
            equivalent = canonical(composition) == canonical(extract_text(output))
            _print({"semantic_equal": equivalent,
                    "byte_equal": output.encode("utf-8", "surrogateescape") == args.input.read_bytes(),
                    "extraction_ms": extract_ms, "export_ms": export_ms,
                    "original_bytes": args.input.stat().st_size, "json_bytes": len(serialized.encode()),
                    **coverage(composition)})
            if not equivalent:
                return 1
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"CustomExtractor: {error}", file=sys.stderr)
        return 2
