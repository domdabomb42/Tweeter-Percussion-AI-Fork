"""Model-local tokenizer, masking, fixed-preset export, and decoder behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil
import io
from contextlib import redirect_stdout

from percussion_generation_v3.data import (AUGMENTED_SESSIONS, ROOT,
    augmented_training_rows, load_rows, load_tokenizer, parse_positions)
from custom_extractor import extract_tweeter
from percussion_generation_v3.cd2 import PositionConstraintCD2
from percussion_generation_v3.inference import PositionConstraint, export_rhythm
from percussion_generation_v3.model import masked_example, train
from transformers import AutoTokenizer


class VariantTests(unittest.TestCase):
    def test_augmented_pair_keeps_its_training_parent(self):
        original = load_rows(ROOT / "output", "rhythm", "train")[0]
        manifest = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
        child = next(entry for entry in manifest["generated"]
                     if entry["original_session"] == original["source"]
                     and entry["reference_offset"] != 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "generated" / child["augmented_filename"]
            destination.parent.mkdir(parents=True)
            shutil.copyfile(AUGMENTED_SESSIONS / "generated" / child["augmented_filename"], destination)
            (root / "manifest.json").write_text(json.dumps({
                "generated": [child], "tracker_steps": 64, "validation_failures": []}), encoding="utf-8")
            with patch("percussion_generation_v3.data.AUGMENTED_SESSIONS", root):
                output = io.StringIO()
                with redirect_stdout(output):
                    rows, summary = augmented_training_rows(
                        [original], AutoTokenizer.from_pretrained(ROOT / "output/tokenizer",
                                                                  local_files_only=True), 8192,
                        progress_seconds=1e-9)
        self.assertIn("1/1 files checked", output.getvalue())
        self.assertEqual(summary["added_pairs"], 1)
        self.assertEqual(rows[0]["source"], original["source"])
        self.assertEqual(rows[0]["augmentation"]["reference_offset"], child["reference_offset"])
        self.assertEqual(rows[0]["target_note_count"], original["target_note_count"])

    def test_resume_can_start_from_saved_best_before_latest_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rhythm").mkdir()
            (root / "manifest.json").write_bytes((ROOT / "output/manifest.json").read_bytes())
            source = ROOT / "output/rhythm/train.jsonl"
            first = source.read_text(encoding="utf-8").splitlines()[0]
            for partition in ("train", "validation"):
                (root / "rhythm" / f"{partition}.jsonl").write_text(first + "\n", encoding="utf-8")
            checkpoint = root / "checkpoints/rhythm"
            adapter = checkpoint / "best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"test checkpoint")
            history = {"plan": {"mode": "rhythm"}, "epochs": [{"epoch": 1}, {"epoch": 2}],
                       "best_epoch": 1, "best_validation_loss": 0.5}
            (checkpoint / "history.json").write_text(json.dumps(history), encoding="utf-8")
            plan = train(root, "rhythm", epochs=20, resume=True, dry_run=True)
            self.assertEqual(plan["completed_epochs"], 2)
            self.assertEqual(plan["starting_checkpoint_epoch"], 1)
            self.assertTrue(plan["resume_from_prior_best"])
            self.assertEqual(plan["prior_stale_epochs"], 1)
            self.assertEqual((checkpoint / "history.json").read_text(encoding="utf-8"),
                             json.dumps(history))

    def test_model_local_tokenizer_and_both_constraints(self):
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "output/tokenizer", local_files_only=True)
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
