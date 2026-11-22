import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare


class ComparisonTests(unittest.TestCase):
    def test_validity_requires_unique_in_range_exported_steps(self):
        good = {"parse_valid": True, "export_valid": True, "loop_valid": True,
                "positions": [0, 4, 63]}
        self.assertTrue(compare.valid_entry(good))
        for change in ({"positions": [0, 0]}, {"positions": [64]},
                       {"parse_valid": False}, {"export_valid": False},
                       {"loop_valid": False}):
            self.assertFalse(compare.valid_entry({**good, **change}))

    def test_onset_f1_and_structure(self):
        self.assertEqual(compare.onset_f1({0, 4}, {0, 8}), .5)
        self.assertEqual(compare.onset_f1(set(), set()), 1)
        grid = compare.rhythm_structure([0, 16, 32, 48])
        self.assertEqual(grid["pulse_alignment_4_step"], 1)
        self.assertEqual(grid["off_pulse_4_step"], 0)
        self.assertEqual(grid["repeat_16_step"], 1)
        self.assertAlmostEqual(compare.pairwise_jaccard_distance([[0, 4], [0, 8]]), 2 / 3)

    def test_elapsed_time_reset_on_continuation(self):
        history = {"epochs": [{"elapsed_seconds": 100},
                              {"elapsed_seconds": 210},
                              {"elapsed_seconds": 90},
                              {"elapsed_seconds": 185}]}
        self.assertEqual(compare.duration_summary(history)["epoch_seconds"],
                         [100, 110, 90, 95])

    def test_original_360m_budget_was_never_reached(self):
        for decoding in compare.DECODINGS:
            backup = Path(__file__).resolve().parents[1] / f"smollm360_{decoding}_512_backup.json"
            report = compare.read_json(backup)
            self.assertEqual(len(report["examples"]), 35)
            self.assertTrue(all(item["ended_with_eos"] for item in report["examples"]))
            self.assertLess(max(item["generated_tokens"] for item in report["examples"]), 128)


if __name__ == "__main__":
    unittest.main()
