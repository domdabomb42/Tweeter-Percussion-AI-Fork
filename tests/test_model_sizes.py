"""Independent model identity, retained data, and model-local decoding checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
MODELS = {"smollm135": "HuggingFaceTB/SmolLM2-135M",
          "smollm360": "HuggingFaceTB/SmolLM2-360M",
          "smollm1b": "HuggingFaceTB/SmolLM2-1.7B"}


def rows(root, split):
    path = root / "V3/output/rhythm" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ModelSizeTests(unittest.TestCase):
    def test_folders_and_base_model_ids(self):
        self.assertFalse((APPS / "Smollm").exists())
        for name, model_id in MODELS.items():
            root = APPS / name
            self.assertTrue((root / "CustomTraining/custom_training/train.py").is_file())
            self.assertTrue((root / "V3/percussion_generation_v3/cd2.py").is_file())
            code = (root / "CustomTraining/custom_training/__init__.py").read_text(encoding="utf-8")
            self.assertIn(f'MODEL_ID = "{model_id}"', code)
            self.assertNotIn("-Instruct", code)
        for name in ("smollm135", "smollm1b"):
            root = APPS / name
            self.assertFalse(list(root.rglob("adapter_model.safetensors")))

    def test_retained_v3_targets_splits_and_tokenization(self):
        reference = APPS / "smollm360"
        tokenizer_hash = hashlib.sha256(
            (reference / "V3/output/tokenizer/tokenizer.json").read_bytes()).hexdigest()
        for name in MODELS:
            root = APPS / name
            manifest = json.loads((root / "V3/output/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["max_length"], 8192)
            self.assertEqual(hashlib.sha256(
                (root / "V3/output/tokenizer/tokenizer.json").read_bytes()).hexdigest(), tokenizer_hash)
            seen = []
            for split, count in (("train", 67), ("validation", 10), ("test", 5)):
                original, actual = rows(reference, split), rows(root, split)
                self.assertEqual(len(actual), count)
                self.assertEqual([(r["source"], r["source_sha256"], r["target_track_id"],
                                   r["prefix"], r["target"]) for r in actual],
                                 [(r["source"], r["source_sha256"], r["target_track_id"],
                                   r["prefix"], r["target"]) for r in original])
                self.assertTrue(all(r["tokens"] <= 8192 for r in actual))
                self.assertTrue(all(r["tokens"] == len(r["prefix_ids"]) + len(r["target_ids"])
                                    for r in actual))
                seen.append({r["source"] for r in actual})
            self.assertFalse(seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])

    def test_feature_split_and_model_local_paths(self):
        reference = json.loads((APPS / "smollm360/CustomTraining/training_output/dataset/manifest.json")
                               .read_text(encoding="utf-8"))
        for name in ("smollm135", "smollm1b"):
            root = APPS / name
            manifest = json.loads((root / "CustomTraining/training_output/dataset/manifest.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(manifest["split"], reference["split"])
            self.assertEqual(manifest["tokenizer"]["model_id"], MODELS[name])
            self.assertEqual(manifest["max_training_length"], 4096)
            code = "from percussion_generation_v3.data import FEATURE_ADAPTER, SMOLLM; from custom_training import MODEL_ID; print(SMOLLM.name, MODEL_ID, FEATURE_ADAPTER)"
            proc = subprocess.run([sys.executable, "-c", code], cwd=root / "V3",
                                  text=True, capture_output=True, check=True)
            self.assertIn(name, proc.stdout)
            self.assertIn(MODELS[name], proc.stdout)
            self.assertIn(str(root / "CustomTraining/training_output/checkpoints/best_adapter"), proc.stdout)

    def test_each_tokenizer_drives_both_constraints(self):
        code = """from percussion_generation_v3.data import ROOT, load_tokenizer, parse_positions
from percussion_generation_v3.inference import PositionConstraint
from percussion_generation_v3.cd2 import PositionConstraintCD2
t = load_tokenizer(ROOT / 'output')
for cls in (PositionConstraint, PositionConstraintCD2):
    grammar = cls(t, 3)
    consumed = []
    for piece in ('0', ' 4', ' 8', '\\n[END_RHYTHM]'):
        for token in t.encode(piece, add_special_tokens=False):
            assert token in grammar.allowed(consumed), (cls.__name__, piece)
            consumed.append(token)
    assert grammar.allowed(consumed) == [t.eos_token_id]
    assert parse_positions(t.decode(consumed, clean_up_tokenization_spaces=False)) == [0, 4, 8]
print('OK')"""
        # Each subprocess starts with an empty module cache, so cross-model imports
        # cannot accidentally reuse another folder's custom_training package.
        for name in MODELS:
            proc = subprocess.run([sys.executable, "-c", code], cwd=APPS / name / "V3",
                                  text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
