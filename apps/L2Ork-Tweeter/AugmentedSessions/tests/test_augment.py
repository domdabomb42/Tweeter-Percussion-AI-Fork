"""Native Tweeter fixtures for exact synchronized rotation."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import augment


def session(tracks, title="unit test"):
    """Write a small but native coll document, with 64 notes and display cells."""
    rows = [f"0, session-title {title};"]
    key = 1
    for track_id, loop_ms, notes in tracks:
        rows.extend((f"{key}, {track_id} inst-name Test{track_id};",
                     f"{key + 1}, {track_id} carrier 0;",
                     f"{key + 2}, {track_id} tracker-time 7 5 1;"))
        key += 3
        if loop_ms is not None:
            rows.append(f"{key}, {track_id} loop-length {loop_ms};")
            key += 1
        for step in range(64):
            rows.append(f"{key}, {track_id} tracker-data {step} {notes.get(step, 0)};")
            key += 1
            rows.append(f"{key}, {track_id} tracker-char {step} {'x' if step in notes else 'rest'};")
            key += 1
    return ("\n".join(rows) + "\n").encode("utf-8")


class AugmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sessions"
        self.sources.mkdir()
        self.output = self.root / "generated"
        self.manifest = self.root / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, tracks):
        source = self.sources / "example.txt"
        source_bytes = session(tracks)
        source.write_bytes(source_bytes)
        dry = augment.generate(self.sources, self.output, self.manifest, dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())
        actual = augment.generate(self.sources, self.output, self.manifest)
        self.assertEqual(dry["summary"]["valid_candidates"], actual["summary"]["valid_candidates"])
        self.assertEqual(dry["summary"]["skipped_offsets"], actual["summary"]["skipped_offsets"])
        self.assertEqual(dry["summary"]["successfully_generated"], 0)
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual((self.output / "example/example_shift_00.txt").read_bytes(), source_bytes)
        self.assertEqual(actual["summary"]["validation_failures"], 0)
        self.assertEqual(actual["summary"]["successfully_generated"],
                         len(list(self.output.rglob("*.txt"))))
        self.assertEqual(json.loads(self.manifest.read_text())["summary"], actual["summary"])
        return actual

    def test_equal_loop_lengths_and_wraparound(self):
        result = self.run_case([(1, 6400, {63: 60, 0: 62}), (2, 6400, {63: 70})])
        self.assertEqual(result["summary"]["successfully_generated"], 64)
        changed = augment.extract_tweeter(self.output / "example/example_shift_01.txt")
        self.assertEqual({n["step"] for n in changed.track(1)["notes"]}, {62, 63})
        self.assertEqual(changed.track(1)["playback"]["cursor"]["next_step"], 7)

    def test_twice_as_fast(self):
        result = self.run_case([(1, 6400, {0: 60}), (2, 3200, {0: 70})])
        self.assertEqual(result["summary"]["successfully_generated"], 64)
        changed = augment.extract_tweeter(self.output / "example/example_shift_01.txt")
        self.assertEqual(changed.track(1)["notes"][0]["step"], 63)
        self.assertEqual(changed.track(2)["notes"][0]["step"], 62)
        self.assertEqual(result["generated"][1]["time_shift_ms_exact"], "100")
        cells = [augment.identify(record) for record in augment.parse(changed.native_text)]
        self.assertIn((1, "tracker-char", [63, "x"]), cells)

    def test_four_times_as_fast(self):
        result = self.run_case([(1, 6400, {2: 60}), (2, 1600, {2: 70})])
        self.assertEqual(result["summary"]["successfully_generated"], 64)
        changed = augment.extract_tweeter(self.output / "example/example_shift_01.txt")
        self.assertEqual(changed.track(2)["notes"][0]["step"], 62)

    def test_several_loop_lengths_and_fractional_skips(self):
        result = self.run_case([(1, 6400, {0: 60}), (2, 3200, {0: 70}),
                                (3, 4800, {0: 80})])
        self.assertEqual(result["summary"]["successfully_generated"], 22)
        self.assertEqual(result["summary"]["skipped_offsets"], 42)
        self.assertIn("fractional tracker shift", str(result["skipped"][0]))
        changed = augment.extract_tweeter(self.output / "example/example_shift_03.txt")
        self.assertEqual(changed.track(3)["notes"][0]["step"], 60)

    def test_empty_tracks_and_repeated_rhythms(self):
        result = self.run_case([(1, 6400, {0: 60, 16: 60, 32: 60, 48: 60}),
                                (2, 3200, {})])
        self.assertEqual(result["summary"]["successfully_generated"], 64)
        zero = self.output / "example/example_shift_00.txt"
        sixteen = self.output / "example/example_shift_16.txt"
        self.assertNotEqual(zero.read_bytes(), sixteen.read_bytes())
        self.assertEqual(augment.extract_tweeter(sixteen).track(2)["notes"], [])

    def test_missing_loop_allows_only_offset_zero(self):
        result = self.run_case([(1, 6400, {0: 60}), (2, None, {1: 70})])
        self.assertEqual(result["summary"]["successfully_generated"], 1)
        self.assertEqual(result["summary"]["skipped_offsets"], 63)
        self.assertIn("missing loop-length", str(result["skipped"][0]))

    def test_exact_native_decimal_ratio(self):
        result = self.run_case([(1, "1000.0000000000001", {0: 60}),
                                (2, "500.00000000000005", {0: 70})])
        self.assertEqual(result["summary"]["successfully_generated"], 64)
        self.assertEqual(result["generated"][1]["tracks"][1]["tracker_shift"], 2)

    def test_rerun_is_deterministic(self):
        self.run_case([(1, 6400, {0: 60}), (2, 4800, {63: 70})])
        files = {p.relative_to(self.output): hashlib.sha256(p.read_bytes()).digest()
                 for p in self.output.rglob("*.txt")}
        manifest_bytes = self.manifest.read_bytes()
        augment.generate(self.sources, self.output, self.manifest)
        self.assertEqual(files, {p.relative_to(self.output): hashlib.sha256(p.read_bytes()).digest()
                                 for p in self.output.rglob("*.txt")})
        self.assertEqual(manifest_bytes, self.manifest.read_bytes())


if __name__ == "__main__":
    unittest.main()
