"""V3 dataset, grammar, reconstruction, and independent-output checks."""
from copy import deepcopy
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from percussion_generation_v3.data import (APPS, DEFAULT_SESSIONS, FEATURE_ADAPTER,
    ROOT, V2_OUTPUT, load_rows, load_tokenizer, parse_positions, recount_prefix,
    serialize_positions, v3_prefix)
from percussion_generation_v3.evaluation import compare, remove_target, score_positions, summarize, source_for_row
from percussion_generation_v3.inference import (PositionConstraint, build_track,
    choose_loop_ms, export_rhythm, generate_file, prompt)
from percussion_generation_v3.model import cached_target_loss, masked_example, move_adam_statistics, train
from percussion_generation_v3.cli import overnight
from custom_extractor import extract_tweeter
from verify_preservation import verify

OUTPUT = ROOT / "output"


class V3Tests(unittest.TestCase):
    @unittest.skipUnless(V2_OUTPUT.is_dir(), "removed V1/V2 source is not bundled")
    def test_v1_v2_unchanged_and_initial_weights_match(self):
        self.assertEqual(verify(), {"missing": [], "added": [], "changed": []})
        original = APPS / "PercussionGenerationV2/initial_feature_adapter/adapter_model.safetensors"
        copied = FEATURE_ADAPTER / "adapter_model.safetensors"
        self.assertEqual(hashlib.sha256(original.read_bytes()).digest(),
                         hashlib.sha256(copied.read_bytes()).digest())
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            self.assertEqual(hashlib.sha256((V2_OUTPUT / "tokenizer" / name).read_bytes()).digest(),
                             hashlib.sha256((OUTPUT / "tokenizer" / name).read_bytes()).digest())

    @unittest.skipUnless(V2_OUTPUT.is_dir(), "removed V2 source is not bundled")
    def test_same_v2_rows_and_exact_position_targets(self):
        seen = {}
        for partition in ("train", "validation", "test"):
            v2 = load_rows(V2_OUTPUT, "rhythm", partition)
            v3 = load_rows(OUTPUT, "rhythm", partition)
            self.assertEqual(len(v2), len(v3))
            seen[partition] = {row["source"] for row in v3}
            for old, new in zip(v2, v3):
                self.assertEqual((old["source"], old["source_sha256"], old["target_track_id"]),
                                 (new["source"], new["source_sha256"], new["target_track_id"]))
                positions = [note["step"] for note in json.loads(old["target"])["notes"]]
                self.assertEqual(new["target_positions"], positions)
                self.assertEqual(new["target_note_count"], len(positions))
                self.assertEqual(new["target"], serialize_positions(positions))
                self.assertEqual(new["prefix"], v3_prefix(old["prefix"], len(positions)))
                self.assertLessEqual(new["tokens"], 8192)
                self.assertNotIn("pitch", new["target"])
        self.assertFalse(seen["train"] & seen["validation"])
        self.assertFalse(seen["train"] & seen["test"])
        self.assertFalse(seen["validation"] & seen["test"])
        self.assertEqual([len(load_rows(OUTPUT, "rhythm", part)) for part in
                          ("train", "validation", "test")], [67, 10, 5])

    def test_duplicate_preservation_and_native_rejection(self):
        text = serialize_positions([0, 0, 8])
        self.assertEqual(text, "0 0 8\n[END_RHYTHM]")
        self.assertEqual(parse_positions(text, allow_duplicates=True), [0, 0, 8])
        with self.assertRaisesRegex(ValueError, "native Tweeter"):
            parse_positions(text)
        for bad in ("0 4", "64\n[END_RHYTHM]", "8 4\n[END_RHYTHM]",
                    "00 4\n[END_RHYTHM]", "0  4\n[END_RHYTHM]"):
            with self.assertRaises(ValueError):
                parse_positions(bad)
        with self.assertRaisesRegex(ValueError, "requested 5"):
            parse_positions(serialize_positions([0, 8]), requested_count=5)

    def test_prefix_and_masking(self):
        row = load_rows(OUTPUT, "rhythm", "train")[0]
        self.assertIn(f"RHYTHM_TARGET={row['target_note_count']}\n", row["prefix"])
        self.assertEqual(recount_prefix(row["prefix"], row["target_note_count"], 16)
                         .split("\nRHYTHM_TARGET=")[0], row["prefix"].split("\nRHYTHM_TARGET=")[0])
        ids, labels = masked_example(row, "cpu")
        self.assertEqual(labels[0, :len(row["prefix_ids"])].tolist(),
                         [-100] * len(row["prefix_ids"]))
        self.assertEqual(labels[0, len(row["prefix_ids"]):].tolist(), row["target_ids"])
        source = extract_tweeter(DEFAULT_SESSIONS / row["source"])
        condition = remove_target(source, row["target_track_id"])
        self.assertEqual(prompt(condition, row["target_note_count"]), row["prefix"])

    def test_constrained_token_grammar(self):
        tokenizer = load_tokenizer(OUTPUT)
        grammar = PositionConstraint(tokenizer, 3)
        generated = []
        for piece in ("0", " 4", " 8", "\n[END_RHYTHM]"):
            for token in tokenizer.encode(piece, add_special_tokens=False):
                self.assertIn(token, grammar.allowed(generated))
                generated.append(token)
        self.assertEqual(grammar.allowed(generated), [tokenizer.eos_token_id])
        self.assertEqual(parse_positions(tokenizer.decode(generated,
                         clean_up_tokenization_spaces=False)), [0, 4, 8])
        dense = PositionConstraint(tokenizer, 64)
        self.assertEqual(dense.allowed([]), tokenizer.encode("0", add_special_tokens=False)[:1])
        for count in (1, 5, 8, 10, 12, 16, 20, 64):
            grammar = PositionConstraint(tokenizer, count)
            consumed = []
            for index in range(count):
                piece = str(index) if index == 0 else " " + str(index)
                for token in tokenizer.encode(piece, add_special_tokens=False):
                    self.assertIn(token, grammar.allowed(consumed))
                    consumed.append(token)
            for token in tokenizer.encode("\n[END_RHYTHM]", add_special_tokens=False):
                self.assertIn(token, grammar.allowed(consumed))
                consumed.append(token)
            self.assertEqual(grammar.allowed(consumed), [tokenizer.eos_token_id])
            self.assertEqual(len(parse_positions(tokenizer.decode(
                consumed, clean_up_tokenization_spaces=False))), count)

    def test_fixed_preset_export_and_metrics(self):
        row = load_rows(OUTPUT, "rhythm", "test")[0]
        source = extract_tweeter(DEFAULT_SESSIONS / row["source"])
        condition = remove_target(source, row["target_track_id"])
        preset = json.loads((OUTPUT / "fixed_preset.json").read_text(encoding="utf-8"))
        positions = [0, 16, 32, 48]
        track = build_track(condition, positions, preset, 1500, row["target_track_id"])
        self.assertEqual(track["synth"], preset["track"]["synth"])
        self.assertEqual(track["effects"], preset["track"]["effects"])
        self.assertEqual([note["step"] for note in track["notes"]], positions)
        self.assertEqual(len({note["pitch"] for note in track["notes"]}), 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.txt"
            exported = export_rhythm(condition, positions, preset, 1500,
                                     output, row["target_track_id"])
            restored = extract_tweeter(output)
            self.assertEqual(restored.tracks[:-1], condition.tracks)
            self.assertEqual(restored.tracks[-1], exported)
        scored = score_positions(condition, source.track(row["target_track_id"]),
                                 positions, 5, 1500, preset)
        self.assertTrue(scored["export_valid"])
        self.assertEqual(scored["absolute_count_error"], 1)
        self.assertEqual(summarize([scored])["within_one_fraction"], 1)

    def test_heldout_loop_selection_matches_source_targets(self):
        for row in load_rows(OUTPUT, "rhythm", "test"):
            condition, reference = source_for_row(row)
            self.assertEqual(choose_loop_ms(condition), reference["timing"]["loop_ms"])

    def test_dry_run_separate_paths_and_preserves_checkpoint(self):
        checkpoint = OUTPUT / "checkpoints/rhythm/best_adapter"
        existing = checkpoint.exists()
        weights = checkpoint / "adapter_model.safetensors"
        before = hashlib.sha256(weights.read_bytes()).digest() if weights.is_file() else None
        plan = train(OUTPUT, "rhythm", epochs=20, patience=3,
                     min_delta=.005, learning_rate=.00005,
                     restart_from_feature=True, dry_run=True)
        self.assertEqual((plan["train_pairs"], plan["validation_pairs"]), (67, 10))
        self.assertEqual(plan["starting_adapter"], str(FEATURE_ADAPTER))
        self.assertIn("prefix cached without gradients", plan["training_memory"])
        self.assertIn("CPU", plan["optimizer_state_memory"])
        self.assertIn("V3", plan["checkpoint"])
        self.assertEqual(checkpoint.exists(), existing)
        self.assertEqual(hashlib.sha256(weights.read_bytes()).digest()
                         if weights.is_file() else None, before)

    def test_cached_loss_uses_full_prefix_and_target_labels(self):
        import torch
        from types import SimpleNamespace

        class Recorder:
            def __init__(self):
                self.scale = torch.nn.Parameter(torch.tensor(1.0))
                self.calls = []

            def __call__(self, input_ids, use_cache, past_key_values=None,
                         attention_mask=None, labels=None):
                self.calls.append((input_ids.tolist(), use_cache,
                                   None if labels is None else labels.tolist(),
                                   None if attention_mask is None else attention_mask.tolist(),
                                   torch.is_grad_enabled()))
                if use_cache:
                    return SimpleNamespace(past_key_values=(torch.tensor([1.0]),))
                return SimpleNamespace(loss=self.scale * (labels != -100).sum())

        model = Recorder()
        row = {"prefix_ids": [10, 11, 12], "target_ids": [20, 21, 0]}
        loss = cached_target_loss(model, row, "cpu")
        self.assertEqual(float(loss), 3.0)
        loss.backward()
        self.assertEqual(float(model.scale.grad), 3.0)
        self.assertEqual(model.calls[0], ([[10, 11]], True, None, None, False))
        self.assertEqual(model.calls[1], ([[12, 20, 21, 0]], False,
                                          [[-100, 20, 21, 0]], [[1] * 6], True))

    def test_adam_moments_can_leave_gpu_between_updates(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda"))
        reference = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda"))
        optimizer = torch.optim.AdamW([parameter], lr=.01, foreach=False)
        reference_optimizer = torch.optim.AdamW([reference], lr=.01, foreach=False)
        parameter.grad = torch.ones_like(parameter)
        reference.grad = torch.ones_like(reference)
        optimizer.step()
        reference_optimizer.step()
        expected = optimizer.state[parameter]["exp_avg"].clone()
        move_adam_statistics(optimizer, torch.device("cpu"))
        self.assertEqual(optimizer.state[parameter]["exp_avg"].device.type, "cpu")
        move_adam_statistics(optimizer, torch.device("cuda"))
        self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], expected))
        parameter.grad = torch.full_like(parameter, 2.0)
        reference.grad = torch.full_like(reference, 2.0)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(parameter, reference)

    def test_inference_keeps_raw_output_and_reports_count(self):
        row = load_rows(OUTPUT, "rhythm", "test")[0]
        source = extract_tweeter(DEFAULT_SESSIONS / row["source"])
        condition = remove_target(source, row["target_track_id"])
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            input_path = temp / "input.txt"
            input_path.write_bytes(condition.native_text.encode("utf-8", errors="surrogateescape"))
            root = temp / "v3_output"
            root.mkdir()
            for name in ("manifest.json", "fixed_preset.json"):
                (root / name).write_bytes((OUTPUT / name).read_bytes())
            adapter = root / "checkpoints/rhythm/best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
            output_path = temp / "result.txt"
            with patch("percussion_generation_v3.inference.load_feature_model",
                       return_value=(object(), "cpu")), patch(
                       "percussion_generation_v3.inference.load_tokenizer",
                       return_value=object()), patch(
                       "percussion_generation_v3.inference.generate_rhythm",
                       return_value=(serialize_positions([0, 16, 32, 48]),
                                     {"input_tokens": 100, "generated_tokens": 12,
                                      "input_processing_seconds": .01, "decode_seconds": .1,
                                      "prefill_seconds": .01})):
                result = generate_file(input_path, output_path, 8, root,
                                       loop_ms=1500)
            self.assertTrue(result["export_valid"])
            self.assertEqual(result["generated_note_count"], 4)
            self.assertEqual(result["absolute_count_error"], 4)
            self.assertIn("0 16 32 48", json.loads(Path(result["diagnostics"])
                          .read_text(encoding="utf-8"))["raw_output"])
            self.assertEqual(extract_tweeter(output_path).tracks[:-1], condition.tracks)

    def test_overnight_automatically_saves_separate_decoder_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            args = Namespace(output_dir=root, epochs=20, learning_rate=.00005,
                             accumulation=8, patience=3, min_delta=.005,
                             progress_seconds=180, cpu=True, resume=False,
                             dry_run=False, cap=1, counts=[5, 8], max_new_tokens=256)

            def fake_train(*_args, **_kwargs):
                baseline = root / "checkpoints/rhythm/best_adapter"
                baseline.mkdir(parents=True)
                (baseline / "adapter_model.safetensors").write_bytes(b"test weights")
                return {"best_epoch": 3, "best_validation_loss": .25}

            with patch("percussion_generation_v3.model.train", side_effect=fake_train), patch(
                    "percussion_generation_v3.evaluation.evaluate_v3",
                    side_effect=[{"report": "baseline"}, {"report": "constrained"},
                                 {"report": "cd2"}]) as evaluate, patch(
                    "percussion_generation_v3.evaluation.compare",
                    return_value={"report": "comparison"}):
                result = overnight(args)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(evaluate.call_args_list[0].args[1], False)
            self.assertEqual(evaluate.call_args_list[1].args[1], True)
            copied = root / "checkpoints/constrained/best_adapter/adapter_model.safetensors"
            self.assertEqual(copied.read_bytes(), b"test weights")
            config = json.loads((root / "checkpoints/constrained/decoding_config.json")
                                .read_text(encoding="utf-8"))
            self.assertTrue(config["same_weights_as_baseline"])
            self.assertEqual(result["constrained_evaluation"]["report"], "constrained")
            self.assertEqual(result["cd2_evaluation"]["report"], "cd2")
            self.assertEqual(result["v1_v2_v3_comparison"]["status"], "unavailable")

    @unittest.skipUnless(V2_OUTPUT.is_dir(), "removed V1/V2 source is not bundled")
    def test_comparison_uses_aligned_rows_and_separate_v3_reports(self):
        import sys
        sys.path.insert(0, str(APPS / "PercussionGenerationV2"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rhythm").mkdir()
            row = load_rows(OUTPUT, "rhythm", "test")[0]
            (root / "rhythm/test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "fixed_preset.json").write_bytes((OUTPUT / "fixed_preset.json").read_bytes())
            (root / "evaluation").mkdir()
            sample = {"parse_valid": True, "export_valid": True,
                      "generated_note_count": 1, "absolute_count_error": 4,
                      "exact_count": False, "within_one": False,
                      "positions": [0], "melody_onset_alignment": 1.0,
                      "reference_onset_f1": .2, "loop_valid": True,
                      "loop_ms_match": True, "wrap_gap_ratio": None,
                      "generated_tokens": 3, "total_seconds": .1}
            for name in ("greedy", "constrained"):
                (root / "evaluation" / f"v3_{name}_test.json").write_text(
                    json.dumps({"examples": [sample]}), encoding="utf-8")
            prediction = {"timing": {"loop_ms": 2700},
                          "notes": [{"step": 0, "pitch": 48}]}
            with patch("percussion_generation_v3.evaluation.load_feature_model",
                       return_value=(object(), "cpu")), patch(
                       "percussion_generation_v2.inference.generate_object",
                       return_value=(prediction, {"generated_tokens": 20,
                                                  "latency_seconds": .1})):
                result = compare(root, cap=1, counts=(5,), max_new_tokens=64, cpu=True)
            self.assertEqual(set(result["summaries"]), {"v1", "v2", "v3", "v3_constrained"})
            self.assertEqual(result["summaries"]["v1"]["attempts"], 1)
            self.assertEqual(result["summaries"]["v2"]["attempts"], 2)
            self.assertTrue(Path(result["report"]).is_file())


if __name__ == "__main__":
    unittest.main()
