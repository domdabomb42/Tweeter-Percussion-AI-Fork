"""Musical schema, gradients, splits, presets, scheduling, and resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from percussion_generation_v4ea import END_MARKER, MODEL_ID, MODEL_REVISION, MODEL_ROOT, OUTPUT, VERSION
from percussion_generation_v4ea.music import (TARGET_TAG, SCHEMA_SHA256,
    decode_target, encode_input, encode_target, reconstruct)
from percussion_generation_v4ea.pipeline import (prompt_for_composition,
    source_groups, texts_for_pair, train_tokenizer, prepare, verify_dataset, sha256)
from percussion_generation_v4ea.trainer import (train, full_sequence_loss,
    optimizer_groups, warmup_cosine_scheduler)
from percussion_generation_v4ea.inference import _decode_generated
from percussion_generation_v3.checkpoints import load_checkpoint_state, resolve_checkpoint
from percussion_generation_v3.rotation import select_epoch_indices
from percussion_generation_v3.lossless_dataset import _source_path
from custom_extractor import extract_tweeter, export_tweeter
from custom_extractor.models import Composition


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.lora_A = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
        self.config = SimpleNamespace(use_cache=False)
        self.seen_labels = None

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, labels=None, use_cache=False):
        self.seen_labels = labels.detach().clone()
        hidden = self.embed(input_ids).cumsum(dim=1)
        logits = hidden @ self.lora_A @ self.lora_B
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 16),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter_model.safetensors")


class V4EATests(unittest.TestCase):
    def test_real_musical_pairs_and_splits(self):
        _, groups, sources, generated, _ = source_groups()
        self.assertEqual(tuple(len(groups[key]) for key in
            ("train", "validation", "validation20", "test")), (2688, 10, 20, 5))
        self.assertFalse(sources["train"] & sources["validation"])
        self.assertFalse(sources["train"] & sources["test"])
        self.assertEqual(groups["validation20"][:10], groups["validation"])
        rows = [next(row for row in groups["train"] if not row.get("augmentation")),
                next(row for row in groups["train"] if row.get("augmentation"))]
        for row in rows:
            composition = extract_tweeter(_source_path(row, generated))
            prefix, target, track = texts_for_pair(row, generated, composition)
            packed = target.split("\n" + END_MARKER)[0]
            musical = decode_target(packed, len(track["notes"]))
            self.assertEqual(musical["notes"], track["notes"])
            self.assertEqual(musical["instrument"], track["instrument"]["name"])
            self.assertEqual(encode_target(track), packed)
            self.assertNotIn("synth", prefix + target)
            self.assertNotIn("effects", prefix + target)
            remaining = Composition(composition.global_settings,
                [item for item in composition.tracks if item["id"] != row["target_track_id"]])
            self.assertEqual(prefix, prompt_for_composition(remaining, len(track["notes"])))
            self.assertEqual(prefix, prompt_for_composition(remaining, len(track["notes"])))

    def test_target_validation(self):
        example = [TARGET_TAG, "electrikick", 1000, 100, 0, [[2, 40], [4, 42]]]
        self.assertEqual(len(decode_target(json.dumps(example), 2)["notes"]), 2)
        for bad in (["wrong", *example[1:]],
                    [*example[:5], [[2, 40], [2, 42]]],
                    [*example[:5], [[64, 40]]],
                    [*example[:5], [[2, 0]]]):
            with self.assertRaises(ValueError):
                decode_target(json.dumps(bad))
        with self.assertRaises(ValueError):
            decode_target(json.dumps(example), 3)
        with self.assertRaises(ValueError):
            decode_target(json.dumps(example), 2, {"other": {}})

    def test_rotating_268_and_tokenizer_plan(self):
        _, groups, _, _, _ = source_groups()
        rows = groups["train"]
        first = select_epoch_indices(rows, 0, 415, 268)
        second = select_epoch_indices(rows, 1, 415, 268)
        self.assertEqual((len(first), len(set(first))), (268, 268))
        self.assertEqual(sum(not rows[i].get("augmentation") for i in first), 67)
        self.assertNotEqual(set(first), set(second))
        with tempfile.TemporaryDirectory() as directory:
            plan = train_tokenizer(directory, dry_run=True)
            self.assertEqual(plan["train_pairs"], 2688)
            self.assertTrue(plan["heldout_sources_excluded"])

    def test_fresh_bpe_preserves_eos(self):
        from transformers import AutoTokenizer
        old = AutoTokenizer.from_pretrained(MODEL_ROOT / "V3/output/tokenizer",
                                            local_files_only=True)
        new = old.train_new_from_iterator(
            ['["V4EA-MUSIC-INPUT-2",8,[["non_percussion","Bass",null,1000,[[0,42]]]]]',
             '["V4EA-MUSIC-TARGET-2","electrikick",1000,100,0,[[2,40]]]'],
            vocab_size=256, new_special_tokens=[END_MARKER])
        self.assertEqual(new.eos_token, old.eos_token)
        self.assertIn(END_MARKER, new.get_vocab())

    def test_small_real_preparation_and_identity(self):
        from transformers import AutoTokenizer
        root, groups, sources, generated, augmentation_sha = source_groups()
        original = next(row for row in groups["train"] if not row.get("augmentation"))
        shifted = next(row for row in groups["train"] if row.get("augmentation"))
        mini = {"train": [original, shifted], "validation": groups["validation"][:1],
                "validation20": groups["validation"][:1], "test": groups["test"][:1]}
        old_path = MODEL_ROOT / "V3/output/tokenizer"
        old = AutoTokenizer.from_pretrained(old_path, local_files_only=True)
        prefix, target, _ = texts_for_pair(original, generated)
        fresh = old.train_new_from_iterator([prefix, target], vocab_size=512,
                                            new_special_tokens=[END_MARKER])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "tokenizer").mkdir()
            fresh.save_pretrained(output / "tokenizer")
            identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                        "tokenizer_sha256": sha256(output / "tokenizer/tokenizer.json"),
                        "eos_token_id": fresh.eos_token_id,
                        "tokenizer_kind": "fresh-bpe-smoke"}
            (output / "tokenizer_identity.json").write_text(json.dumps(identity), encoding="utf-8")
            with patch("percussion_generation_v4ea.pipeline.source_groups",
                       return_value=(root, mini, sources, generated, augmentation_sha)):
                result = prepare(output)
            manifest, converted = verify_dataset(output)
            self.assertEqual(result["train_pairs"], 2)
            self.assertEqual(len(converted["validation20"]), 1)
            self.assertEqual(manifest["codec_schema_sha256"], SCHEMA_SHA256)
            self.assertEqual(converted["train"][0]["target"], target)
            self.assertTrue((output / "preset_catalog.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare(output)

    def test_constrained_opening_matches_tokenized_target(self):
        from transformers import AutoTokenizer
        _, groups, _, generated, _ = source_groups()
        row = next(item for item in groups["train"] if not item.get("augmentation"))
        original = extract_tweeter(_source_path(row, generated))
        prefix, target, track = texts_for_pair(row, generated, original)
        old = AutoTokenizer.from_pretrained(MODEL_ROOT / "V3/output/tokenizer",
                                            local_files_only=True)
        tokenizer = old.train_new_from_iterator([prefix, target], vocab_size=512,
                                                new_special_tokens=[END_MARKER])
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        class FakeGenerator:
            def generate(self, input_ids, prefix_allowed_tokens_fn, **kwargs):
                sequence = input_ids[0]
                for token_id in target_ids:
                    self_allowed = prefix_allowed_tokens_fn(0, sequence)
                    if token_id not in self_allowed:
                        raise AssertionError("constraint rejects actual tokenizer target")
                    sequence = torch.cat([sequence, torch.tensor([token_id])])
                return sequence.unsqueeze(0)
        conditioning = Composition(original.global_settings,
            [item for item in original.tracks if item["id"] != row["target_track_id"]])
        catalog = {track["instrument"]["name"]: track}
        predicted, _ = _decode_generated(SimpleNamespace(peft_model=FakeGenerator()),
            tokenizer, torch.device("cpu"), conditioning, len(track["notes"]), catalog, 512)
        self.assertEqual(predicted["notes"], track["notes"])

    def test_full_gradient_and_target_only_shift(self):
        torch.manual_seed(13)
        model = TinyCausal()
        row = {"prefix_ids": [4, 2, 3], "target_ids": [5, 6, 7]}
        loss = full_sequence_loss(model, row, torch.device("cpu"))
        self.assertEqual(model.seen_labels.tolist(), [[-100, -100, -100, 5, 6, 7]])
        ids = torch.tensor([[4, 2, 3, 5, 6, 7]])
        logits = model(ids, labels=model.seen_labels).logits
        expected = F.cross_entropy(logits[:, 2:5].reshape(-1, 16), torch.tensor([5, 6, 7]))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(model.embed.weight.grad[4].abs().sum().item(), 0)
        self.assertGreater(model.embed.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.lora_A.grad.abs().sum().item(), 0)
        self.assertGreater(model.lora_B.grad.abs().sum().item(), 0)

    def test_optimizer_groups_and_warmup_cosine(self):
        model = TinyCausal()
        optimizer = optimizer_groups(model, 1e-4, 5e-5)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["lora", "embedding"])
        self.assertEqual([group["initial_lr"] for group in optimizer.param_groups], [1e-4, 5e-5])
        scheduler, warmup = warmup_cosine_scheduler(optimizer, 408, 0.1)
        self.assertEqual(warmup, 41)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"], 2)
        for _ in range(41):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4, places=6)
        for _ in range(367):
            optimizer.step(); scheduler.step()
        self.assertLess(optimizer.param_groups[0]["lr"], 1e-6)

    def test_native_preset_reconstruction(self):
        _, groups, _, generated, _ = source_groups()
        row = next(item for item in groups["train"] if not item.get("augmentation"))
        original = extract_tweeter(_source_path(row, generated))
        target = original.track(row["target_track_id"])
        catalog = {target["instrument"]["name"]: {**target, "id": 0, "notes": []}}
        musical = decode_target(encode_target(target), len(target["notes"]), catalog)
        remaining = Composition(original.global_settings,
            [item for item in original.tracks if item["id"] != row["target_track_id"]])
        candidate, predicted = reconstruct(remaining, musical, catalog)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.txt"
            export_tweeter(candidate, path)
            restored = extract_tweeter(path)
            self.assertEqual(restored.tracks[:-1], remaining.tracks)
            self.assertEqual(restored.tracks[-1], predicted)

    def test_training_plan_and_exact_resume(self):
        rows = []
        for index in range(67):
            source = f"train-{index}"
            rows.append({"source": source, "target_track_id": index,
                         "prefix_ids": [1, 2], "target_ids": [3, 4]})
            for shift in range(1, 4):
                rows.append({"source": source, "target_track_id": index,
                             "augmentation": {"file": f"{index}/shift_{shift}",
                                              "reference_offset": shift},
                             "prefix_ids": [1, 2], "target_ids": [3, 4]})
        groups = {"train": rows, "validation": rows[:10],
                  "validation20": rows[:20], "test": rows[:5]}
        manifest = {"tokenizer_sha256": "a", "prepared_dataset_sha256": "b",
                    "codec_schema_sha256": SCHEMA_SHA256, "preset_catalog_sha256": "d"}
        def loader(output, adapter, cpu, trainable):
            model = TinyCausal()
            if adapter is not None:
                model.load_state_dict(torch.load(Path(adapter) / "adapter_model.safetensors",
                                                 map_location="cpu", weights_only=True))
            return model, torch.device("cpu")
        def loss(model, row, device):
            return (model.embed.weight[0].sum() + model.lora_A.sum() - 0.2) ** 2
        def validation(model, rows, device):
            return {"loss": float(model.embed.weight.detach()[0].square().sum()),
                    "pairs": len(rows), "target_tokens": 2 * len(rows)}
        with tempfile.TemporaryDirectory() as directory, \
             patch("percussion_generation_v4ea.trainer.verify_dataset", return_value=(manifest, groups)), \
             patch("percussion_generation_v4ea.trainer.fingerprints", return_value={}), \
             patch("percussion_generation_v4ea.trainer._validate_pool"):
            options = {"output_dir": directory, "cpu": True, "save_steps": 1,
                       "model_loader": loader, "loss_fn": loss, "validation_fn": validation}
            plan = train(dry_run=True, **options)
            self.assertEqual((plan["examples_per_epoch"], plan["optimizer_updates_per_epoch"],
                              plan["warmup_updates"]), (268, 34, 41))
            self.assertEqual(plan["lora_learning_rate"], 5e-5 if "1.7B" in MODEL_ID else 1e-4)
            self.assertEqual(plan["embedding_learning_rate"], 1e-4 if "135M" in MODEL_ID else 5e-5)
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train(stop_after_steps=1, **options)
            root = Path(directory) / "checkpoints/rhythm"
            first = resolve_checkpoint(root, "latest")
            metadata = json.loads((first / "metadata.json").read_text())
            state = load_checkpoint_state(first, metadata["config"], metadata["fingerprints"])
            self.assertEqual((state["global_step"], state["next_position"]), (1, 8))
            self.assertEqual(state["scheduler"]["last_epoch"], 1)
            self.assertEqual(len(state["optimizer"]["param_groups"]), 2)
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train(resume_checkpoint="latest", stop_after_steps=2, **options)
            second = resolve_checkpoint(root, "latest")
            metadata = json.loads((second / "metadata.json").read_text())
            resumed = load_checkpoint_state(second, metadata["config"], metadata["fingerprints"])
            self.assertEqual((resumed["global_step"], resumed["next_position"]), (2, 16))
            self.assertEqual(resumed["scheduler"]["last_epoch"], 2)
            self.assertEqual(resumed["selected_order"], state["selected_order"])
            with tempfile.TemporaryDirectory() as second_directory:
                independent = {**options, "output_dir": second_directory}
                with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                    train(stop_after_steps=2, **independent)
                direct_path = resolve_checkpoint(
                    Path(second_directory) / "checkpoints/rhythm", "latest")
                direct_meta = json.loads((direct_path / "metadata.json").read_text())
                direct = load_checkpoint_state(direct_path, direct_meta["config"],
                                               direct_meta["fingerprints"])
                self.assertEqual(direct["selected_order"], resumed["selected_order"])
                self.assertEqual(direct["scheduler"], resumed["scheduler"])
                direct_weights = torch.load(direct_path / "adapter/adapter_model.safetensors",
                                            weights_only=True)
                resumed_weights = torch.load(second / "adapter/adapter_model.safetensors",
                                             weights_only=True)
                for name in direct_weights:
                    self.assertTrue(torch.equal(direct_weights[name], resumed_weights[name]))
                for parameter_id, values in direct["optimizer"]["state"].items():
                    for key, value in values.items():
                        other = resumed["optimizer"]["state"][parameter_id][key]
                        self.assertTrue(torch.equal(value, other) if torch.is_tensor(value)
                                        else value == other)


if __name__ == "__main__":
    unittest.main()
