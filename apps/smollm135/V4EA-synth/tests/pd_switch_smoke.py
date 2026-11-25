"""Optional headless Pd-L2Ork check of the AI track switch abstraction."""
from pathlib import Path
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[4]
CORE = REPO / "apps/L2Ork-Tweeter/core"
PD = Path(r"C:\Program Files (x86)\Pd-L2Ork\bin\pd-l2ork.exe")


def run_case(playing, stop_pending=False):
    if not PD.is_file():
        raise FileNotFoundError(PD)
    pitches = " ".join(["55"] + ["0"] * 63)
    lines = [
        "#N canvas 0 0 600 400 10;",
        "#X obj 20 20 ai-preview-switch 5 123;",
        "#X obj 20 50 loadbang;",
        "#X obj 20 80 t b b b;",
        f"#X msg 20 110 \\; 5-tracker {int(playing)};",
        f"#X msg 20 140 \\; ai-preview 5 2 {pitches};",
        "#X obj 20 170 delay 600;",
        "#X msg 20 200 \\; pd quit;",
        "#X obj 300 20 r ai-applied;",
        "#X obj 300 50 print AI_APPLIED;",
        "#X obj 300 80 r network-in;",
        "#X obj 300 110 print AI_NETWORK;",
    ]
    if playing:
        lines += ["#X obj 20 230 delay 100;",
                  ("#X msg 20 260 \\; 5-tracker 0;" if stop_pending else
                   "#X msg 20 260 \\; 123-tracker set 63.5;"),
                  "#X msg 20 290 \\; 123-tracker label ignored;"]
    lines += ["#X connect 1 0 2 0;", "#X connect 2 2 3 0;",
              "#X connect 2 1 4 0;", "#X connect 2 0 5 0;",
              "#X connect 5 0 6 0;", "#X connect 7 0 8 0;",
              "#X connect 9 0 10 0;"]
    if playing:
        lines += ["#X connect 2 0 12 0;", "#X connect 12 0 13 0;",
                  "#X connect 2 0 14 0;"]
    with tempfile.TemporaryDirectory(prefix="tweeter-pd-smoke-") as directory:
        patch = Path(directory) / "switch-smoke.pd"
        patch.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = subprocess.run([str(PD), "-nogui", "-stderr", "-path", str(CORE),
                                 str(patch)], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    details = "\n".join(line for line in output.splitlines() if
                        "AI_" in line or "connection failed" in line or "error:" in line)
    if result.returncode or "AI_APPLIED: 2" not in output:
        raise AssertionError(f"playing={playing}, stop_pending={stop_pending}, exit={result.returncode}\n{details}")
    if output.count("AI_NETWORK: 5 tracker-data") != 64:
        raise AssertionError(f"expected 64 targeted tracker data updates\n{details}")
    updates = [line for line in output.splitlines() if line.startswith("AI_NETWORK:")]
    if len(updates) != 128 or any(not line.startswith(("AI_NETWORK: 5 tracker-data ",
                                                      "AI_NETWORK: 5 tracker-char ")) for line in updates):
        raise AssertionError(f"switch touched another track or setting\n{details}")
    if output.count("AI_APPLIED: 2") != 1:
        raise AssertionError(f"expected one switch acknowledgment\n{details}")
    if "select: no method for" in output:
        raise AssertionError(f"tracker label reached the position selector\n{details}")
    return output


def run_panel_case():
    """Load the track UI and feed its identity/rating messages in native Pd."""
    lines = [
        "#N canvas 0 0 600 400 10;",
        "#X obj 20 20 ai-feedback-track 5 123;",
        "#X obj 20 150 loadbang;",
        "#X msg 20 180 \\; network-in 5 username AI_A \\; ai-active-track 5 \\; ai-candidate 1 5 0 0 \\; ai-rating-ack 1 8 1;",
        "#X obj 20 210 delay 600;",
        "#X msg 20 240 \\; pd quit;",
        "#X connect 1 0 2 0;",
        "#X connect 1 0 3 0;",
        "#X connect 3 0 4 0;",
    ]
    with tempfile.TemporaryDirectory(prefix="tweeter-panel-smoke-") as directory:
        patch = Path(directory) / "panel-smoke.pd"
        patch.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = subprocess.run([str(PD), "-nogui", "-stderr", "-path", str(CORE),
                                 str(patch)], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    problems = [line for line in output.splitlines() if
                "no method for" in line or "connection failed" in line or
                "couldn't create" in line]
    if result.returncode or problems:
        raise AssertionError(f"panel load failed: {problems}, exit={result.returncode}")


if __name__ == "__main__":
    for playing, stop_pending in ((False, False), (True, False), (True, True)):
        output = run_case(playing, stop_pending)
        print(f"playing={playing}, stop_pending={stop_pending}: one acknowledgment, 64 targeted note updates")
    run_panel_case()
    print("AI track panel: native patch and rating messages loaded without Pd method errors")
