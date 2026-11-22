"""Optional external oracle: installed Pd-L2Ork's real cyclone coll loader.

This runs a temporary headless patch, without audio/network or the Tweeter GUI.
It verifies native message order, atom types/values and routing. It does not
assert audible playback. Pd's text writer uses ~6 significant figures, so its
rewritten numeric output is compared at that precision, independently of the
lossless CustomExtractor round trip.
"""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from .parser import parse, NUMBER


def find_pd():
    candidates = [shutil.which("pd-l2ork"), r"C:\Program Files (x86)\Pd-L2Ork\bin\pd-l2ork.exe"]
    return next((Path(p) for p in candidates if p and Path(p).is_file()), None)


def check_native(input_path, executable=None):
    executable = Path(executable) if executable else find_pd()
    if executable is None:
        return {"status": "unavailable", "playback_verified": False}
    source = Path(input_path).read_bytes()
    expected = parse(source.decode("utf-8", "surrogateescape"))
    with tempfile.TemporaryDirectory(prefix="custom-extractor-native-") as temporary:
        folder = Path(temporary)
        (folder / "input.txt").write_bytes(source)
        patch = """#N canvas 0 0 500 300 10;
#X obj 20 20 loadbang;
#X msg 20 50 read input.txt;
#X obj 20 80 coll;
#X obj 180 80 t b b b;
#X msg 180 110 write output.txt;
#X msg 290 110 dump;
#X obj 20 140 route session-title;
#X obj 20 170 print CE_TITLE;
#X obj 160 170 print CE_TRACK;
#X obj 350 50 delay 150;
#X msg 350 80 \\; pd quit;
#X obj 180 50 spigot 1;
#X msg 260 50 0;
#X obj 20 220 list-drip;
#X obj 20 245 route float symbol;
#X msg 20 270 F;
#X msg 90 270 S;
#X obj 180 270 print CE_TYPE;
#X connect 0 0 1 0;
#X connect 0 0 9 0;
#X connect 1 0 2 0;
#X connect 2 2 11 0;
#X connect 11 0 3 0;
#X connect 3 2 12 0;
#X connect 12 0 11 1;
#X connect 3 0 4 0;
#X connect 3 1 5 0;
#X connect 4 0 2 0;
#X connect 5 0 2 0;
#X connect 2 0 6 0;
#X connect 6 0 7 0;
#X connect 6 1 8 0;
#X connect 9 0 10 0;
#X connect 2 0 13 0;
#X connect 13 0 14 0;
#X connect 14 0 15 0;
#X connect 14 1 16 0;
#X connect 15 0 17 0;
#X connect 16 0 17 0;
"""
        patch_path = folder / "verify.pd"
        patch_path.write_text(patch, encoding="utf-8")
        cyclone = executable.parent.parent / "extra" / "cyclone"
        lists = executable.parent.parent / "extra" / "list-abs"
        command = [str(executable), "-unique", "-nogui", "-batch", "-noaudio", "-nomidi", "-stderr", "-noprefs", "-path", str(cyclone), "-path", str(lists), "-open", str(patch_path)]
        process = subprocess.run(command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log = process.stdout.decode("utf-8", "backslashreplace")
        output = folder / "output.txt"
        if process.returncode or not output.exists() or "couldn't create" in log or "bad file" in log or "error:" in log.lower():
            raise ValueError(f"native Pd loader failed (exit {process.returncode}): {log[-3000:]}")
        actual = parse(output.read_bytes().decode("utf-8", "surrogateescape"))
        if len(actual) != len(expected):
            raise ValueError(f"native coll record loss: {len(expected)} -> {len(actual)}")
        expected_types = ["S" if isinstance(a.value, str) else "F" for r in expected for a in r.atoms]
        actual_types = [line.split(": ", 1)[1] for line in log.splitlines() if line.startswith("CE_TYPE: ")]
        if actual_types != expected_types:
            mismatch = next((i for i, (a, b) in enumerate(zip(expected_types, actual_types)) if a != b), min(len(expected_types), len(actual_types)))
            raise ValueError(f"native input atom type mismatch at {mismatch}: expected {expected_types[mismatch:mismatch+5]}, got {actual_types[mismatch:mismatch+5]}")
        compared, numeric_symbol_write_losses = 0, 0
        for index, (before, after) in enumerate(zip(expected, actual)):
            a = [v.value for v in before.keys + before.atoms]
            b = [v.value for v in after.keys + after.atoms]
            if len(a) != len(b):
                raise ValueError(f"native record {index}: atom count changed: {a!r} -> {b!r}")
            for left, right in zip(a, b):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    okay = math.isclose(left, right, rel_tol=5.1e-6, abs_tol=1e-38)
                elif isinstance(left, str) and isinstance(right, (int, float)) and NUMBER.fullmatch(left):
                    # Pd's own coll writer fails to re-escape numeric-looking
                    # symbols. The live CE_TYPE stream above verifies that the
                    # input symbol reached Pd correctly before its lossy rewrite.
                    okay = math.isclose(float(left), right, rel_tol=5.1e-6, abs_tol=1e-38)
                    numeric_symbol_write_losses += 1
                else:
                    okay = type(left) is type(right) and left == right
                if not okay:
                    raise ValueError(f"native record {index}: atom changed: {left!r} -> {right!r}")
                compared += 1
        routed = sum(line.startswith(("CE_TITLE:", "CE_TRACK:")) for line in log.splitlines())
        if routed != len(expected):
            raise ValueError(f"native dump/routing mismatch: expected {len(expected)}, got {routed}; {log[-1000:]}")
        return {"status": "passed", "executable": str(executable),
                "native_records": len(actual), "native_atoms_compared": compared,
                "native_routed_messages": routed, "native_numeric_relative_tolerance": 5.1e-6,
                "live_atom_types_verified": len(actual_types),
                "pd_writer_numeric_symbol_losses": numeric_symbol_write_losses,
                "playback_verified": False, "method": "actual Pd-L2Ork cyclone coll read/write/dump and session-title routing"}
