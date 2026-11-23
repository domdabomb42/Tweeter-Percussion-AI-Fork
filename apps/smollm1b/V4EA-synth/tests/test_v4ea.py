"""Real-data split/target checks plus lightweight tokenizer and trainer smoke tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from percussion_generation_v4ea import END_MARKER, MODEL_ID, MODEL_ROOT, OUTPUT, VERSION
from percussion_generation_v4ea.pipeline import (prompt_for_composition,
    source_groups, texts_for_pair, train_tokenizer)
from percussion_generation_v4ea.trainer import train
from percussion_generation_v3.checkpoints import load_checkpoint_state, resolve_checkpoint
from percussion_generation_v3.codec_v2 import decode_v2
from percussion_generation_v3.rotation import select_epoch_indices
from custom_extractor import extract_tweeter
from custom_extractor.models import Composition


class V4EATests(unittest.TestCase):
    def test_real_original_and_augmented_full_track_round_trip(self):
        _, groups, sources, generated, _ = source_groups()
        self.assertEqual((len(groups["train"]), len(groups["validation"]),
                          len(groups["validation20"]), len(groups["test"])),
                         (2688, 10, 20, 5))
        self.assertFalse(sources["train"] & sources["validation"])
        self.assertFalse(sources["train"] & sources["test"])
        self.assertEqual(groups["validation20"][:10], groups["validation"])
        rows = [next(r for r in groups["train"] if not r.get("augmentation")),
                next(r for r in groups["train"] if r.get("augmentation"))]
        from percussion_generation_v3.lossless_dataset import _source_path
        for row in rows:
            composition = extract_tweeter(_source_path(row, generated))
            prefix, target, track = texts_for_pair(row, generated, composition)
            self.assertEqual(target.split("\n" + END_MARKER)[-1], "")
            decoded = decode_v2(target[:-(len(END_MARKER) + 1)])
            self.assertEqual(decoded["tracks"], [{"role": "percussion", "track": track}])
            remaining = Composition(composition.global_settings,
                                    [item for item in composition.tracks
                                     if item["id"] != row["target_track_id"]])
            self.assertEqual(prefix, prompt_for_composition(remaining, len(track["notes"])))
            self.assertIn("synth", track)
            self.assertIn("effects", track)
            self.assertIn("mixing", track)
            self.assertIn("playback", track)

    def test_tokenizer_plan_uses_augmented_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = train_tokenizer(directory, dry_run=True)
            self.assertEqual(plan["model_id"], MODEL_ID)
            self.assertEqual(plan["train_pairs"], 2688)
            self.assertTrue(plan["heldout_sources_excluded"])
            self.assertFalse((Path(directory) / "tokenizer").exists())

    def test_rotating_268_schedule(self):
        _, groups, _, _, _ = source_groups()
        rows = groups["train"]
        first = select_epoch_indices(rows, 0, 415, 268)
        second = select_epoch_indices(rows, 1, 415, 268)
        self.assertEqual((len(first), len(set(first))), (268, 268))
        self.assertEqual(sum(not rows[i].get("augmentation") for i in first), 67)
        self.assertNotEqual(set(first), set(second))

    def test_fresh_bpe_template_retains_eos(self):
        from transformers import AutoTokenizer
        old = AutoTokenizer.from_pretrained(MODEL_ROOT / "V3/output/tokenizer",
                                            local_files_only=True)
        new = old.train_new_from_iterator(
            ['["L2V2",{"notes":[29,32,33]}]\n[END_TRACK]',
             '["L2V2",{"notes":[1,2,3]}]\n[END_TRACK]'],
            vocab_size=256, new_special_tokens=[END_MARKER])
        self.assertEqual(new.eos_token, old.eos_token)
        self.assertIn(END_MARKER, new.get_vocab())
        text = '["L2V2",{"notes":[29,32,33]}]'
        self.assertEqual(new.decode(new.encode(text, add_special_tokens=False)), text)

    def test_trainer_dry_run_does_not_load_model(self):
        rows = [{"source": "train", "target_track_id": i} for i in range(2688)]
        groups = {"train": rows, "validation": [{}] * 10,
                  "validation20": [{}] * 20, "test": [{}] * 5}
        manifest = {"tokenizer_sha256": "a", "prepared_dataset_sha256": "b",
                    "codec_schema_sha256": "c"}
        with patch("percussion_generation_v4ea.trainer.verify_dataset", return_value=(manifest, groups)), \
             patch("percussion_generation_v4ea.trainer.fingerprints", return_value={}), \
             patch("percussion_generation_v4ea.trainer._validate_pool"):
            plan = train(OUTPUT, dry_run=True)
        self.assertEqual(plan["version"], VERSION)
        self.assertEqual(plan["original_training_pairs"], 67)
        self.assertEqual(plan["examples_per_epoch"], 268)
        self.assertEqual(plan["optimizer_updates_per_epoch"], 34)
        self.assertEqual(plan["patience"], 5)

    def test_step_checkpoint_restores_full_track_trainer(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))
                self.config = SimpleNamespace(use_cache=False)

            def save_pretrained(self, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), path / "adapter_model.safetensors")

        rows = []
        for i in range(67):
            source = f"train-{i}"
            rows.append({"source": source, "target_track_id": i,
                         "target_ids": [1, 2]})
            for shift in range(1, 4):
                rows.append({"source": source, "target_track_id": i,
                             "augmentation": {"file": f"{i}/shift_{shift}",
                                              "reference_offset": shift},
                             "target_ids": [1, 2]})
        groups = {"train": rows, "validation": rows[:10],
                  "validation20": rows[:20], "test": rows[:5]}
        manifest = {"tokenizer_sha256": "a", "prepared_dataset_sha256": "b",
                    "codec_schema_sha256": "c"}

        def loader(output, adapter, cpu, trainable):
            model = TinyModel()
            if adapter is not None:
                model.load_state_dict(torch.load(Path(adapter) / "adapter_model.safetensors",
                                                 map_location="cpu", weights_only=True))
            return model, torch.device("cpu")

        def loss(model, row, device):
            return (model.weight - 0.2) ** 2

        def validation(model, rows, device):
            return {"loss": float(model.weight.detach() ** 2), "pairs": len(rows),
                    "target_tokens": sum(len(row["target_ids"]) for row in rows)}

        with tempfile.TemporaryDirectory() as directory, \
             patch("percussion_generation_v4ea.trainer.verify_dataset", return_value=(manifest, groups)), \
             patch("percussion_generation_v4ea.trainer.fingerprints", return_value={}), \
             patch("percussion_generation_v4ea.trainer._validate_pool"):
            options = {"output_dir": directory, "cpu": True, "save_steps": 1,
                       "model_loader": loader, "loss_fn": loss,
                       "validation_fn": validation}
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train(stop_after_steps=1, **options)
            root = Path(directory) / "checkpoints/rhythm"
            first = resolve_checkpoint(root, "latest")
            metadata = json.loads((first / "metadata.json").read_text())
            state = load_checkpoint_state(first, metadata["config"], metadata["fingerprints"])
            self.assertEqual((state["global_step"], state["next_position"]), (1, 8))
            self.assertEqual(state["scheduler"]["last_epoch"], 1)
            self.assertTrue(state["optimizer"]["state"])
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train(resume_checkpoint="latest", stop_after_steps=2, **options)
            second = resolve_checkpoint(root, "latest")
            metadata2 = json.loads((second / "metadata.json").read_text())
            resumed = load_checkpoint_state(second, metadata2["config"], metadata2["fingerprints"])
            self.assertEqual((resumed["global_step"], resumed["next_position"]), (2, 16))
            self.assertEqual(resumed["scheduler"]["last_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
