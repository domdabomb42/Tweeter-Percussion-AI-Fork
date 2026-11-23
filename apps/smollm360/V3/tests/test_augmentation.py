"""Augmented rhythm pairs retain their original training parent."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import io
from contextlib import redirect_stdout

from percussion_generation_v3.data import (AUGMENTED_SESSIONS, ROOT,
    augmented_training_rows, load_rows, load_tokenizer)


class AugmentationTests(unittest.TestCase):
    def test_shifted_training_pair(self):
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
                        [original], load_tokenizer(ROOT / "output"), 8192,
                        progress_seconds=1e-9)
        self.assertIn("1/1 files checked", output.getvalue())
        self.assertEqual(summary["added_pairs"], 1)
        self.assertEqual(rows[0]["source"], original["source"])
        self.assertEqual(rows[0]["augmentation"]["reference_offset"], child["reference_offset"])
        self.assertEqual(rows[0]["target_note_count"], original["target_note_count"])
