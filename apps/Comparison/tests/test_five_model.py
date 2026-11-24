import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import five_model


class FiveModelTests(unittest.TestCase):
    def test_historical_reports_use_five_reference_requests(self):
        result = five_model.build_comparison()
        self.assertEqual(len(result["test_targets"]), 5)
        self.assertEqual(tuple(result["models"]), five_model.MODEL_ORDER)
        for slug in ("135m", "360m"):
            for decoding in five_model.DECODINGS:
                self.assertEqual(result["models"][slug][decoding]["pairs"], 5)
                self.assertIn("rhythmic_jaccard_valid", result["models"][slug][decoding])
        for slug in five_model.EA_MODELS:
            for decoding in five_model.DECODINGS:
                key = f"{slug}_{decoding}"
                item = result["models"][slug][decoding]
                if item is None:
                    self.assertIn(key, result["pending"])
                else:
                    self.assertEqual(item["pairs"], 5)
                    self.assertNotIn(key, result["pending"])

    def test_ea_summary_rejects_wrong_identity_and_rates(self):
        report = {"model_id": five_model.EA_MODELS["135mea"][1],
                  "partition": "test", "examples": 5, "test_loss": 0.6,
                  "structurally_valid_rate": 0.6, "exact_note_count_rate": 0.4,
                  "native_export_success_rate": 0.4, "rhythmic_jaccard": 0.25}
        summary = five_model.summarize_ea(report, report["model_id"])
        self.assertEqual(summary["valid_jaccard_denominator"], 3)
        self.assertEqual(summary["rhythmic_jaccard_valid"], 0.25)
        with self.assertRaises(ValueError):
            five_model.summarize_ea({**report, "examples": 20}, report["model_id"])
        with self.assertRaises(ValueError):
            five_model.summarize_ea({**report, "structurally_valid_rate": 0.5}, report["model_id"])

    def test_snapshot_checks_target_and_checkpoint_identity(self):
        targets = five_model.targets_from_rows(
            five_model.APPS / "smollm135/V3/output/rhythm/test.jsonl")
        report = {"model_id": five_model.EA_MODELS["135mea"][1],
                  "partition": "test", "examples": 5, "test_loss": 0.6,
                  "structurally_valid_rate": 1.0, "exact_note_count_rate": 0.8,
                  "native_export_success_rate": 1.0, "rhythmic_jaccard": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            adapter = output / "checkpoints/rhythm_coverage_v1/best_adapter"
            adapter.mkdir(parents=True)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"adapter")
            metadata = adapter / "experiment_metadata.json"
            metadata.write_text("{}")
            manifest_path = output / "manifest.json"
            manifest_path.write_text(json.dumps({"encoding": "positions-v1",
                "model_id": report["model_id"], "prepared_dataset_sha256": "dataset",
                "tokenizer_sha256": "tokenizer"}))
            (root / "reports").mkdir()
            snapshot = {"schema_version": 1, "model": "135mea", "decoding": "greedy",
                        "target_format": "positions-v1", "report": report,
                        "test_targets": targets, "dataset_sha256": "dataset",
                        "tokenizer_sha256": "tokenizer",
                        "manifest_sha256": five_model.sha256(manifest_path),
                        "adapter_sha256": five_model.sha256(weights),
                        "adapter_metadata_sha256": five_model.sha256(metadata),
                        "report_content_sha256": hashlib.sha256(json.dumps(
                            report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            path = root / "reports/135mea_greedy.json"
            path.write_text(json.dumps(snapshot))
            with patch.object(five_model, "HERE", root), patch.object(five_model, "ea_output", return_value=output):
                item, reason = five_model.load_ea_snapshot("135mea", "greedy", targets)
                self.assertIsNone(reason)
                self.assertEqual(item["pairs"], 5)
                weights.write_bytes(b"new adapter")
                item, reason = five_model.load_ea_snapshot("135mea", "greedy", targets)
                self.assertIsNone(item)
                self.assertIn("adapter changed", reason)
                with self.assertRaises(ValueError):
                    five_model.load_ea_snapshot("135mea", "greedy", targets[::-1])

    def test_evaluation_saves_both_decoders_separately(self):
        targets = five_model.targets_from_rows(
            five_model.APPS / "smollm135/V3/output/rhythm/test.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (output / "rhythm").mkdir(parents=True)
            (output / "rhythm/test.jsonl").write_text("".join(json.dumps({
                "source": row["source"], "target_track_id": row["target_track_id"],
                "target_positions": row["positions"]}) + "\n" for row in targets))
            (output / "manifest.json").write_text(json.dumps({
                "encoding": "positions-v1", "model_id": five_model.EA_MODELS["135mea"][1],
                "prepared_dataset_sha256": "dataset", "tokenizer_sha256": "tokenizer"}))
            adapter = output / "checkpoints/rhythm_coverage_v1/best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "experiment_metadata.json").write_text("{}")
            commands = []
            def fake_run(command, **kwargs):
                commands.append(command)
                constrained = "--constrained" in command
                report = {"model_id": five_model.EA_MODELS["135mea"][1],
                          "partition": "test", "examples": 5, "test_loss": 0.6,
                          "structurally_valid_rate": 1.0,
                          "exact_note_count_rate": 1.0 if constrained else 0.8,
                          "native_export_success_rate": 1.0,
                          "rhythmic_jaccard": 0.3 if constrained else 0.2}
                (output / "evaluation_test.json").write_text(json.dumps(report))
            with patch.object(five_model, "HERE", root), \
                 patch.object(five_model, "ea_output", return_value=output), \
                 patch.object(five_model.subprocess, "run", side_effect=fake_run):
                saved = five_model.evaluate_ea("135mea")
                self.assertEqual(len(saved), 2)
                self.assertNotIn("--constrained", commands[0])
                self.assertIn("--constrained", commands[1])
                commands.clear()
                self.assertEqual(len(five_model.evaluate_ea("135mea", decodings=("constrained",))), 1)
                self.assertEqual(len(commands), 1)
                self.assertIn("--constrained", commands[0])
                greedy, _ = five_model.load_ea_snapshot("135mea", "greedy", targets)
                constrained, _ = five_model.load_ea_snapshot("135mea", "constrained", targets)
                self.assertEqual(greedy["rhythmic_jaccard_valid"], 0.2)
                self.assertEqual(constrained["rhythmic_jaccard_valid"], 0.3)


if __name__ == "__main__":
    unittest.main()
