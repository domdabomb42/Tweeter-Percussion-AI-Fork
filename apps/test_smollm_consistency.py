"""Fail when shared V3/V4EA or feature-training source drifts by model."""
import ast
import hashlib
from pathlib import Path
import unittest


APPS = Path(__file__).resolve().parent
VARIANTS = ("smollm135", "smollm360", "smollm1b")
PINNED = {
    "smollm135": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "smollm360": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
    "smollm1b": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConsistencyTests(unittest.TestCase):
    def test_v4ea_implementation_is_byte_identical(self):
        names = {p.name for variant in VARIANTS
                 for p in (APPS / variant / "V4EA-synth/percussion_generation_v4ea").glob("*.py")}
        for name in sorted(names):
            paths = [APPS / variant / "V4EA-synth/percussion_generation_v4ea" / name
                     for variant in VARIANTS]
            with self.subTest(file=name):
                self.assertTrue(all(p.is_file() for p in paths))
                self.assertEqual(len({digest(p) for p in paths}), 1)

    def test_v3_implementation_is_byte_identical(self):
        names = {p.name for variant in VARIANTS
                 for p in (APPS / variant / "V3/percussion_generation_v3").glob("*.py")}
        for name in sorted(names):
            paths = [APPS / variant / "V3/percussion_generation_v3" / name
                     for variant in VARIANTS]
            with self.subTest(file=name):
                self.assertTrue(all(p.is_file() for p in paths))
                self.assertEqual(len({digest(p) for p in paths}), 1)

    def test_feature_training_and_runner_are_identical(self):
        names = {p.name for variant in VARIANTS for p in
                 (APPS / variant / "CustomTraining/custom_training").glob("*.py")}
        for name in sorted(names - {"__init__.py"}):
            paths = [APPS / variant / "CustomTraining/custom_training" / name
                     for variant in VARIANTS]
            with self.subTest(file=name):
                self.assertTrue(all(p.is_file() for p in paths))
                self.assertEqual(len({digest(p) for p in paths}), 1)
        paths = [APPS / variant / "V3/run_overnight.py" for variant in VARIANTS]
        self.assertEqual(len({digest(p) for p in paths}), 1)

    def test_only_pinned_model_identity_differs(self):
        for variant, expected in PINNED.items():
            path = APPS / variant / "CustomTraining/custom_training/__init__.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            values = {assignment.targets[0].id: ast.literal_eval(assignment.value)
                      for assignment in tree.body if isinstance(assignment, ast.Assign)}
            self.assertEqual((values["MODEL_ID"], values["MODEL_REVISION"]), expected)
            self.assertEqual(set(values), {"MODEL_ID", "MODEL_REVISION"})


if __name__ == "__main__":
    unittest.main()
