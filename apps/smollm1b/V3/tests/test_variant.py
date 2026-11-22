"""Model-local tokenizer, masking, fixed-preset export, and decoder behavior."""
import json
from pathlib import Path
import tempfile
import unittest

from percussion_generation_v3.data import ROOT, load_rows, load_tokenizer, parse_positions
from custom_extractor import extract_tweeter
from percussion_generation_v3.cd2 import PositionConstraintCD2
from percussion_generation_v3.inference import PositionConstraint, export_rhythm
from percussion_generation_v3.model import masked_example


class VariantTests(unittest.TestCase):
    def test_model_local_tokenizer_and_both_constraints(self):
        tokenizer = load_tokenizer(ROOT / "output")
        for grammar_type in (PositionConstraint, PositionConstraintCD2):
            grammar = grammar_type(tokenizer, 3)
            consumed = []
            for piece in ("0", " 4", " 8", "\n[END_RHYTHM]"):
                for token in tokenizer.encode(piece, add_special_tokens=False):
                    self.assertIn(token, grammar.allowed(consumed))
                    consumed.append(token)
            self.assertEqual(grammar.allowed(consumed), [tokenizer.eos_token_id])
            self.assertEqual(parse_positions(tokenizer.decode(
                consumed, clean_up_tokenization_spaces=False)), [0, 4, 8])

    def test_target_only_labels_and_native_export(self):
        row = load_rows(ROOT / "output", "rhythm", "train")[0]
        ids, labels = masked_example(row, "cpu")
        self.assertEqual(ids.shape[-1], row["tokens"])
        self.assertTrue(all(v == -100 for v in labels[0, :len(row["prefix_ids"])].tolist()))
        self.assertEqual(labels[0, len(row["prefix_ids"]):].tolist(), row["target_ids"])
        source = (ROOT.parents[1] / "smollm360/V3/output/POC_inputs/simple-melody-bass.txt")
        original = extract_tweeter(source)
        preset = json.loads((ROOT / "output/fixed_preset.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.txt"
            export_rhythm(original, [0, 16, 32, 48], preset, 10800, output)
            result = extract_tweeter(output)
            self.assertEqual(result.tracks[:-1], original.tracks)
            self.assertEqual([note["step"] for note in result.tracks[-1]["notes"]],
                             [0, 16, 32, 48])


if __name__ == "__main__":
    unittest.main()
