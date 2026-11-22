"""CD2 permits unordered, unique positions and exports ordered native notes."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from percussion_generation_v3.data import ROOT, load_tokenizer, parse_positions
from custom_extractor import extract_tweeter
from percussion_generation_v3.cd2 import PositionConstraintCD2
from percussion_generation_v3.inference import generate_file


class CD2Tests(unittest.TestCase):
    def test_unordered_distinct_positions_and_exact_count(self):
        tokenizer = load_tokenizer(ROOT / "output")
        for positions in ([32, 0, 16, 8], [63, 62, 61, 0], [8, 0, 63]):
            grammar = PositionConstraintCD2(tokenizer, len(positions))
            consumed = []
            for index, step in enumerate(positions):
                piece = str(step) if index == 0 else " " + str(step)
                for token in tokenizer.encode(piece, add_special_tokens=False):
                    self.assertIn(token, grammar.allowed(consumed))
                    consumed.append(token)
            for token in tokenizer.encode("\n[END_RHYTHM]", add_special_tokens=False):
                self.assertIn(token, grammar.allowed(consumed))
                consumed.append(token)
            self.assertEqual(grammar.allowed(consumed), [tokenizer.eos_token_id])
            raw = tokenizer.decode(consumed, clean_up_tokenization_spaces=False)
            self.assertEqual(parse_positions(raw, requested_count=len(positions),
                                             allow_unordered=True), list(positions))
            with self.assertRaisesRegex(ValueError, "out of order"):
                parse_positions(raw)

        grammar = PositionConstraintCD2(tokenizer, 3)
        consumed = []
        for piece in ("32", " 0"):
            for token in tokenizer.encode(piece, add_special_tokens=False):
                self.assertIn(token, grammar.allowed(consumed))
                consumed.append(token)
        duplicate = tokenizer.encode(" 32", add_special_tokens=False)
        trial = consumed.copy()
        rejected = False
        for token in duplicate:
            if token not in grammar.allowed(trial):
                rejected = True
                break
            trial.append(token)
        self.assertTrue(rejected)

    def test_duplicate_step_is_rejected_by_parser(self):
        with self.assertRaisesRegex(ValueError, "one pitch per step"):
            parse_positions("32 0 32\n[END_RHYTHM]", allow_unordered=True)

    def test_cd2_export_sorts_only_native_notes(self):
        input_path = ROOT / "output/POC_inputs/simple-melody-bass.txt"
        raw = "32 0 16 8\n[END_RHYTHM]"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("manifest.json", "fixed_preset.json"):
                shutil.copy2(ROOT / "output" / name, root / name)
            adapter = root / "checkpoints/constrained/best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"test only")
            output_path = root / "simple2_constrained_4.txt"
            with patch("percussion_generation_v3.inference.load_feature_model",
                       return_value=(object(), "cpu")), patch(
                       "percussion_generation_v3.inference.load_tokenizer",
                       return_value=object()), patch(
                       "percussion_generation_v3.inference.generate_rhythm",
                       return_value=(raw, {"input_processing_seconds": 0.01,
                                          "generated_tokens": 12})) as generate:
                result = generate_file(input_path, output_path, 4, root,
                                       loop_ms=10800, cd2=True)
            self.assertTrue(generate.call_args.kwargs["cd2"])
            self.assertTrue(result["export_valid"])
            self.assertEqual(result["decoder"], "cd2")
            self.assertEqual(result["generated_positions"], [32, 0, 16, 8])
            self.assertEqual(result["export_positions"], [0, 8, 16, 32])
            restored = extract_tweeter(output_path)
            original = extract_tweeter(input_path)
            self.assertEqual(restored.tracks[:-1], original.tracks)
            self.assertEqual([n["step"] for n in restored.tracks[-1]["notes"]],
                             [0, 8, 16, 32])
            diagnostics = json.loads(Path(result["diagnostics"]).read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["raw_output"], raw)
            with self.assertRaises(ValueError):
                generate_file(input_path, root / "invalid.txt", 4, root,
                              constrained=True, cd2=True)


if __name__ == "__main__":
    unittest.main()
