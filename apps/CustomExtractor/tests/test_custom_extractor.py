import copy
import json
import math
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from custom_extractor import (Composition, extract_text, extract_tweeter, export_tweeter,
                              render_tweeter, dumps, loads, save_json, load_json, validate,
                              set_parameter, set_parameters, add_note, remove_note,
                              set_loop_length, add_track)
from custom_extractor.parser import parse, atom_text, record_text, FormatError
from custom_extractor.parameter_registry import PARAMETERS, BY_NATIVE, WAVEFORMS, get_path
from custom_extractor.reporting import canonical, coverage, musical_features

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent / "L2Ork-Tweeter" / "sessions"


def fixture():
    values = {
        "carrier": 1, "modulator": 4, "ae": [0, .5, 1, -.001, 1.01],
        "me": [1, .3, 0], "harmonicity": 2.3456789012345, "mod-amp": 300,
        "note-dur": 250, "octave": -12, "overdrive": 1.5,
        "overdrive-override": 2, "inst-name": "My Pd; synth, \\ bass $1",
        "reverb-amp": .3, "reverb-type": 2, "echo-amp": .2, "echo-fb": .4,
        "echo-ms": 100.123456789, "flanger-amp": .2, "flanger-hz": .5,
        "flanger-mod-amp": 20, "flanger-ms": 5, "dry-amp": 1,
        "out-amp": 90, "out-pan": .7854, "loop-length": 8000,
        "loop": 1, "solo": 0, "tracker": 1, "keyboard": 0,
        "insert": 0, "delete": 2, "hover": 0,
        "tracker-time": [8, 42.123456789, 1], "preset-id": ["bank-A", 7], "username": "Name Here",
    }
    rows = [record_text([0], ["session-title", "Example Title"])]
    for i, (key, value) in enumerate(values.items(), 1):
        rows.append(record_text([i], [1, key, *(value if isinstance(value, list) else [value])]))
    rows += ["90, 1 tracker-data 0 60;", "91, 1 tracker-data 16 127;",
             "92, 1 tracker-data 32 0;", "93, 1 tracker-char 0 \\,;",
             "94, 1 tracker-char 32;", "95, 1 future-control \\123 hello\\ world 3.1415926535;",
             "96, 2 loop-length 4000;", "97, 2 tracker-data 0 1;"]
    return "\r\n".join(rows) + "\r\n"


class ParserTests(unittest.TestCase):
    def test_native_escapes_and_atom_types(self):
        values = ["hello world", "semi;colon", "comma,", "back\\slash", "$1", "123", "-3.0", "+3", "1_000", "\u00a0", "line\nfeed", "café", 3, -.5, 1e-6]
        self.assertEqual(parse(record_text([0], [1, "future", *values]))[0].values[2:], values)

    def test_unknown_symbolic_coll_keys(self):
        text = "key, 1 loop-length 1000;\n42 symbol, 1 unknown 2;"
        self.assertEqual(render_tweeter(extract_text(text)), text)

    def test_numeric_grammar(self):
        values = parse("0, 1 x +1 1_000 nan inf .2 -3e-4 \\7 １２;")[0].values[2:]
        self.assertEqual(values, ["+1", "1_000", "nan", "inf", .2, -.0003, "7", "１２"])

    def test_malformed_inputs(self):
        for text in ("", "   ", "# comment\n", "0 1 carrier 0;", "0,;", "0, 1 x 2", "0, 1 x 2, 3;", "0.1, 1 x 0;", "0, 1 x \\", "0, 1 x 1e999;", "0, 1 x \x00;"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_not_a_tweeter_snapshot(self):
        with self.assertRaisesRegex(ValueError, "no recognized"):
            extract_text("0, something entirely different;")

    def test_record_numbers_are_not_timing(self):
        c = extract_text("99, 2 loop-length 1000;\n3, 1 tracker-data 63 84;\n99, 2 tracker-data 1 60;")
        self.assertEqual([t["id"] for t in c.tracks], [2, 1])
        self.assertEqual(c.track(1)["notes"], [{"step": 63, "pitch": 84}])

    def test_multiline_envelopes_exact(self):
        text = "0, 1 ae 0\n0.12345678912345678 -0.001\r\n1.00001;"
        self.assertEqual(render_tweeter(loads(dumps(extract_text(text)))), text)

    def test_legacy_multisymbol_names(self):
        c = extract_text("0, 1 inst-name Long Lead;\n1, 1 username My Name;")
        self.assertEqual(c.track(1)["instrument"]["name"], "Long Lead")
        self.assertEqual(c.track(1)["instrument"]["username"], "My Name")
        self.assertEqual(render_tweeter(c), c.native_text)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.c = extract_text(fixture())

    def test_all_parameters_extracted(self):
        self.assertTrue(all(get_path(self.c.track(1), p.path) is not None for p in PARAMETERS))

    def test_waveforms(self):
        for code, name in enumerate(WAVEFORMS):
            c = extract_text(f"0, 1 carrier {code};\n1, 1 modulator {4-code};")
            self.assertEqual(c.track(1)["synth"]["carrier"], name)
            self.assertEqual(c.track(1)["synth"]["modulator"], WAVEFORMS[4-code])

    def test_sparse_notes_and_full_native_pitch_range(self):
        self.assertEqual(self.c.track(1)["notes"], [{"step": 0, "pitch": 60}, {"step": 16, "pitch": 127}])
        self.assertEqual(self.c.track(2)["notes"][0]["pitch"], 1)

    def test_native_pitch_and_independent_timings(self):
        f = musical_features(self.c)
        self.assertEqual(f["tracks"][0]["notes"][0]["sounding_pitch"], 36)
        self.assertEqual(f["tracks"][0]["notes"][1]["onset_ms"], 2000)
        self.assertIsNone(f["tracks"][1]["notes"][0]["sounding_pitch"])
        for key in ("global_loop_ms", "common_track_loop_ms", "tempo", "meter", "velocity", "per_note_gate_duration", "automation"):
            self.assertIsNone(f[key])

    def test_missing_values_not_defaults_or_zero(self):
        self.assertIsNone(self.c.track(2)["synth"]["carrier"])
        c = extract_text("0, 1 carrier 0;\n1, 1 tracker-data 0 0;")
        self.assertEqual(c.track(1)["notes"], [])
        self.assertEqual(c.track(1)["synth"]["carrier"], "sine")
        self.assertIsNone(c.track(1)["timing"]["loop_ms"])

    def test_stale_step_64_preserved_not_played(self):
        c = extract_text(fixture() + "100, 1 tracker-data 64 70;\n")
        self.assertNotIn(64, [n["step"] for n in c.track(1)["notes"]])
        self.assertIn("100, 1 tracker-data 64 70;", render_tweeter(c))
        self.assertTrue(any(r["reason"] and "stale" in r["reason"] for r in coverage(c, True)["records"]))

    def test_unknown_metadata_coverage(self):
        report = coverage(self.c, True)
        self.assertEqual(report["unknown_parameter_names"], ["future-control", "preset-id"])
        self.assertGreater(report["status_counts"]["partial"], 0)

    def test_duplicate_parameters_last_wins_raw_preserved(self):
        c = extract_text("0, 1 carrier 0;\n0, 1 carrier 2;")
        self.assertEqual(c.track(1)["synth"]["carrier"], "triangle")
        self.assertEqual(render_tweeter(c), c.native_text)
        set_parameter(c, 1, "carrier", "square")
        self.assertEqual(render_tweeter(c), "0, 1 carrier 0;\n0, 1 carrier 3;")


class RoundTripTests(unittest.TestCase):
    def test_json_native_exact_bytes(self):
        text = fixture()
        c = loads(dumps(extract_text(text)))
        self.assertEqual(render_tweeter(c), text)
        self.assertEqual(canonical(c), canonical(extract_text(render_tweeter(c))))

    def test_legacy_bytes_and_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.txt"
            data = b"0, 1 tracker-char 0 \xa0;\r\n1, 1 inst-name Caf\xe9;\r\n2, 1 loop-length 1000;\r\n"
            path.write_bytes(data)
            c = loads(dumps(extract_tweeter(path)))
            export_tweeter(c, Path(directory) / "out.txt")
            self.assertEqual((Path(directory) / "out.txt").read_bytes(), data)

    def test_bom_preserved(self):
        text = "\ufeff0, 1 carrier 0;\r\n"
        self.assertEqual(render_tweeter(loads(dumps(extract_text(text)))), text)

    def test_json_disk_round_trip_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            c = extract_text(fixture())
            save_json(c, path)
            self.assertEqual(load_json(path).to_dict(), c.to_dict())
            self.assertEqual(dumps(c), dumps(loads(dumps(c))))

    def test_generated_composition_without_native_text(self):
        c = Composition()
        add_track(c, 1, {"inst-name": "New Synth", "carrier": "square", "loop-length": 2000, "note-dur": 150, "octave": 0}, [{"step": 7, "pitch": 84}])
        text = render_tweeter(c)
        self.assertEqual(canonical(c), canonical(extract_text(text)))
        self.assertEqual(sum("tracker-data" in r.values for r in parse(text)), 64)

    def test_randomized_compositions(self):
        rng = random.Random(415)
        for case in range(50):
            c = Composition()
            for tid in range(1, rng.randrange(2, 5)):
                notes = [{"step": s, "pitch": rng.randrange(1, 128)} for s in sorted(rng.sample(range(64), rng.randrange(0, 20)))]
                add_track(c, tid, {"carrier": rng.choice(WAVEFORMS), "modulator": rng.choice(WAVEFORMS),
                                  "loop-length": rng.uniform(500, 20000), "note-dur": rng.uniform(1, 2000), "octave": rng.uniform(-192, 192),
                                  "ae": [rng.uniform(-.1, 1.1) for _ in range(rng.randrange(1, 115))],
                                  "echo-fb": rng.uniform(0, .999), "inst-name": f"Synth {case}, {tid}; \\ name"}, notes)
            self.assertEqual(canonical(c), canonical(extract_text(render_tweeter(loads(dumps(c))))))

    def test_real_corpus(self):
        files = sorted(CORPUS.glob("*session*.txt"))
        self.assertGreater(len(files), 0, "real Tweeter session corpus not found")
        accepted, empty = 0, 0
        for path in files:
            with self.subTest(path=str(path.relative_to(CORPUS))):
                if path.stat().st_size == 0:
                    with self.assertRaisesRegex(ValueError, "empty"):
                        extract_tweeter(path)
                    empty += 1
                    continue
                c = extract_tweeter(path)
                restored = loads(dumps(c))
                text = render_tweeter(restored)
                self.assertEqual(text.encode("utf-8", "surrogateescape"), path.read_bytes())
                self.assertEqual(canonical(c), canonical(extract_text(text)))
                accepted += 1
        self.assertEqual(accepted + empty, len(files))


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.c = extract_text(fixture())

    def test_edit_every_parameter(self):
        replacements = {"carrier": "square", "modulator": "triangle", "ae": [.1, .7, 0], "me": [.5, 0],
                        "harmonicity": 3.2, "mod-amp": 600, "note-dur": 321, "octave": 12.5,
                        "overdrive": 2.1, "overdrive-override": 3.1, "inst-name": "Pd custom name",
                        "reverb-amp": .55, "reverb-type": 3, "echo-amp": .35, "echo-fb": .62,
                        "echo-ms": 333.3, "flanger-amp": .44, "flanger-hz": 2.5, "flanger-mod-amp": 33,
                        "flanger-ms": 4.2, "dry-amp": .9, "out-amp": 88, "out-pan": .9,
                        "loop-length": 9000, "loop": 0, "solo": 1, "tracker": 0, "keyboard": 1,
                        "insert": 1, "delete": 3, "hover": 2, "tracker-time": {"next_step": 63, "remaining_ms": 12, "running": 0},
                        "username": "A New Name"}
        self.assertEqual(set(replacements), set(BY_NATIVE))
        for name, value in replacements.items():
            with self.subTest(parameter=name):
                c = copy.deepcopy(self.c)
                before = canonical(c)
                set_parameter(c, 1, name, value)
                out = extract_text(render_tweeter(c))
                self.assertEqual(canonical(c), canonical(out))
                self.assertEqual(get_path(out.track(1), BY_NATIVE[name].path), value)
                self.assertEqual(before["tracks"][1], out.track(2))
                for p in PARAMETERS:
                    if p.native != name:
                        self.assertEqual(get_path(self.c.track(1), p.path), get_path(out.track(1), p.path))

    def test_add_remove_and_loop(self):
        remove_note(self.c, 1, 0, 60)
        add_note(self.c, 1, 63, 1)
        set_loop_length(self.c, 2, 1234.5678)
        out = extract_text(render_tweeter(self.c))
        self.assertEqual(canonical(self.c), canonical(out))
        self.assertEqual(out.track(1)["notes"], [{"step": 16, "pitch": 127}, {"step": 63, "pitch": 1}])
        self.assertIn("1 tracker-data 0 0;", render_tweeter(self.c))

    def test_rest_clears_duplicates(self):
        c = extract_text("0, 1 tracker-data 1 60;\n1, 1 tracker-data 1 62;")
        remove_note(c, 1, 1)
        self.assertEqual(extract_text(render_tweeter(c)).track(1)["notes"], [])

    def test_numeric_display_character_remains_symbol(self):
        add_note(self.c, 1, 2, 66)
        rows = [r.values for r in parse(render_tweeter(self.c)) if r.values[:3] == [1, "tracker-char", 2]]
        self.assertEqual(rows, [[1, "tracker-char", 2, "2"]])

    def test_unknown_records_unchanged_after_edit(self):
        set_parameter(self.c, 1, "carrier", "pink_noise")
        self.assertIn("95, 1 future-control \\123 hello\\ world 3.1415926535;", render_tweeter(self.c))

    def test_add_track_keeps_existing_tracks(self):
        before = copy.deepcopy(self.c.tracks)
        add_track(self.c, 14, {"inst-name": "Bass", "loop-length": 1000, "note-dur": 200, "octave": 0}, [{"step": 3, "pitch": 63}])
        out = extract_text(render_tweeter(self.c))
        self.assertEqual(out.tracks[:2], before)
        self.assertEqual(out.track(14), self.c.track(14))

    def test_remove_parameter_clears_all_duplicates(self):
        c = extract_text("0, 1 carrier 0;\n1, 1 carrier 1;\n2, 1 loop-length 1000;")
        set_parameter(c, 1, "carrier", None)
        self.assertNotIn("carrier", render_tweeter(c))

    def test_title_edit(self):
        self.c.global_settings["metadata"]["session_title"] = ["New; title", "123"]
        self.assertEqual(extract_text(render_tweeter(self.c)).global_settings, self.c.global_settings)

    def test_multi_edit_atomic_failure(self):
        before = self.c.to_dict()
        with self.assertRaises(ValueError):
            set_parameters(self.c, 1, {"carrier": "square", "echo-fb": 1.0})
        self.assertEqual(before, self.c.to_dict())

    def test_legacy_out_of_range_is_preserved_not_clamped(self):
        c = extract_text("0, 1 overdrive 11;\n1, 1 loop-length 100;")
        self.assertEqual(len(validate(c)), 2)
        self.assertEqual(render_tweeter(c), c.native_text)
        with self.assertRaises(ValueError):
            set_parameter(c, 1, "overdrive", 11)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.c = extract_text(fixture())

    def test_bad_parameter_values(self):
        bad = [("carrier", "noise"), ("carrier", 0), ("harmonicity", -1), ("mod-amp", 0),
               ("echo-fb", 1), ("out-pan", -1), ("out-amp", 101), ("flanger-ms", 51),
               ("reverb-type", 1.5), ("ae", []), ("ae", [1, "x"]), ("me", [math.nan]),
               ("overdrive", math.inf), ("note-dur", 0), ("inst-name", ""),
               ("inst-name", "x" * 1000), ("octave", True), ("echo-amp", .000000000000000000000000000000000000000000000000000001),
               ("tracker-time", {"next_step": 65, "remaining_ms": 1, "running": 1})]
        for name, value in bad:
            with self.subTest(name=name, value=value):
                before = self.c.to_dict()
                with self.assertRaises(ValueError):
                    set_parameter(self.c, 1, name, value)
                self.assertEqual(self.c.to_dict(), before)

    def test_note_boundaries_and_occupied_step(self):
        for step, pitch in [(-1, 60), (64, 60), (1, 0), (1, math.inf), (1, True), (1.5, 60), (True, 60), (0, 62), ("1", 60)]:
            before = self.c.to_dict()
            with self.subTest(step=step, pitch=pitch), self.assertRaises(ValueError):
                add_note(self.c, 1, step, pitch)
            self.assertEqual(self.c.to_dict(), before)

    def test_invalid_tracks_and_parameters(self):
        for tid in (0, 15, 17, 3, "1", True):
            with self.subTest(tid=tid), self.assertRaises(ValueError):
                set_parameter(self.c, tid, "carrier", "sine")
        with self.assertRaisesRegex(ValueError, "unsupported parameter"):
            set_parameter(self.c, 1, "velocity", 127)

    def test_bad_json_and_versions(self):
        for text in ('{"format": "CustomExtractor", "format": "other"}', '{"x": NaN}'):
            with self.assertRaises(ValueError):
                loads(text)
        for version in (0, 2, True, "1"):
            data = self.c.to_dict()
            data["version"] = version
            with self.assertRaises(ValueError):
                loads(json.dumps(data))

    def test_schema_rejects_unexported_extra_fields(self):
        data = self.c.to_dict()
        data["tracks"][0]["notes"][0]["velocity"] = 80
        with self.assertRaises(ValueError):
            loads(json.dumps(data))
        data = self.c.to_dict()
        data["global"]["tempo"] = 120
        with self.assertRaises(ValueError):
            loads(json.dumps(data))

    def test_track_reorder_or_delete_rejected(self):
        self.c.tracks.reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            render_tweeter(self.c)

    def test_new_track_requires_timing_with_notes(self):
        before = self.c.to_dict()
        with self.assertRaises(ValueError):
            add_track(self.c, 3, {"carrier": "sine"}, [{"step": 0, "pitch": 60}])
        self.assertEqual(self.c.to_dict(), before)

    def test_failed_export_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.txt"
            path.write_bytes(b"existing")
            self.c.track(1)["synth"]["carrier"] = "bad"
            with self.assertRaises(ValueError):
                export_tweeter(self.c, path)
            self.assertEqual(path.read_bytes(), b"existing")

    def test_malformed_native_fields(self):
        for text in ("0, 1 carrier 6;", "0, 1 tracker-data 65 60;", "0, 1 tracker-data 0 nope;", "0, 1 tracker-time 1 2;", "0, 1 tracker-data 0;", "0, 1 note-dur hello;"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate(extract_text(text))

    def test_zero_echo_delay_native_default(self):
        set_parameter(self.c, 1, "echo-ms", 0)
        self.assertEqual(extract_text(render_tweeter(self.c)).track(1)["effects"]["echo"]["delay_ms"], 0)

    def test_native_fractional_and_non_midi_pitches(self):
        for step, pitch in [(1, 60.5), (2, -12), (3, 130.25)]:
            add_note(self.c, 1, step, pitch)
        restored = extract_text(render_tweeter(self.c))
        self.assertEqual(restored.tracks, self.c.tracks)

    def test_only_fourteen_supported_native_parts(self):
        for tid in (15, 16, 17):
            with self.assertRaises(ValueError):
                add_track(self.c, tid, {"carrier": "sine"})


class CLITests(unittest.TestCase):
    def test_complete_cli_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original, structured, output = [folder / name for name in ("in.txt", "c.json", "out.txt")]
            original.write_bytes(fixture().encode())
            commands = [["extract", str(original), "-o", str(structured)],
                        ["inspect", str(original)], ["coverage", str(original), "--records"],
                        ["validate", str(structured)], ["export-tweeter", str(structured), "-o", str(output)],
                        ["roundtrip", str(original)]]
            for command in commands:
                result = subprocess.run([sys.executable, "-m", "custom_extractor", *command], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)
            self.assertEqual(original.read_bytes(), output.read_bytes())

    def test_cli_error(self):
        result = subprocess.run([sys.executable, "-m", "custom_extractor", "inspect", "does-not-exist.txt"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
