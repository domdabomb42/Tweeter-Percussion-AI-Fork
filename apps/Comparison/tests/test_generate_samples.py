"""The sample set is exactly the source-verified held-out split."""
import json
from pathlib import Path
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_samples import APPS, HERE, MODELS, heldout_plan, sha256


class SampleSelectionTests(unittest.TestCase):
    def test_fixed_seed_uses_all_five_heldout_pairs(self):
        first, second = heldout_plan(), heldout_plan()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({(r["source"], r["target_track_id"]) for r in first}), 5)
        source = APPS / "smollm135/V3/output/rhythm/test.jsonl"
        expected = {(r["source"], r["target_track_id"])
                    for r in (json.loads(line) for line in source.read_text().splitlines())}
        self.assertEqual({(r["source"], r["target_track_id"]) for r in first}, expected)

    def test_all_native_outputs_reimport_with_unchanged_accompaniment(self):
        sys.path.insert(0, str(APPS / "smollm135/V3"))
        from percussion_generation_v3.evaluation import remove_target
        from custom_extractor import extract_tweeter
        from custom_extractor.validator import validate

        root = HERE / "SampleOutputs"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        samples = [sample for model in manifest["models"] for sample in model["samples"]]
        self.assertEqual(len(samples), 25)
        self.assertEqual({model["model"] for model in manifest["models"]}, set(MODELS))
        self.assertEqual({path.name for path in root.glob("*.txt")},
                         {sample["file"] for sample in samples})
        for sample in samples:
            with self.subTest(file=sample["file"]):
                path = root / sample["file"]
                self.assertEqual(sha256(path), sample["sha256"])
                original = extract_tweeter(APPS / "L2Ork-Tweeter/sessions" / sample["source"])
                condition = remove_target(original, sample["target_track_id"])
                restored = extract_tweeter(path)
                self.assertEqual(restored.global_settings, condition.global_settings)
                self.assertEqual(restored.tracks[:-1], condition.tracks)
                self.assertEqual([note["step"] for note in restored.tracks[-1]["notes"]],
                                 sample["positions"])
                candidate = deepcopy(condition)
                candidate.tracks.append(restored.tracks[-1])
                validate(candidate)  # The appended track has no historical range exemption.


if __name__ == "__main__":
    unittest.main()
