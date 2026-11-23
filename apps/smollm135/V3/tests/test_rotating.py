"""Rotation, split, warm-start, and complete step-resume regression tests."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from percussion_generation_v3.checkpoints import (load_checkpoint_state,
    resolve_checkpoint, save_checkpoint, sha256_file)
from percussion_generation_v3.data import ROOT, load_rows, validate_epoch7_splits
from percussion_generation_v3.rotation import example_id, select_epoch_indices
from percussion_generation_v3.rotating_train import (HISTORICAL_ADAPTER,
    BASELINE_ADAPTER_SHA256, BASELINE_DATASET_SHA256,
    cosine_scheduler, normalize_history, overall_epoch, train_rotating,
    verify_epoch7_adapter)


PREPARED = ROOT / "output_augmented_from_epoch7"


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(use_cache=False)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter_model.safetensors")


class RotationTests(unittest.TestCase):
    def test_epoch7_baseline_and_global_epoch_labels(self):
        config = {"historical_best_epoch": 7,
                  "prepared_dataset_sha256": BASELINE_DATASET_SHA256,
                  "init_adapter_sha256": BASELINE_ADAPTER_SHA256}
        initial = normalize_history({"epochs": []}, config)
        self.assertEqual(initial["best_epoch"], 7)
        self.assertAlmostEqual(initial["best_validation_loss"], 0.7691895557198547)
        self.assertEqual(overall_epoch(config, 2), 9)
        previous = {"epochs": [{"epoch": 1, "validation20": {"loss": 0.7566005504793591}}],
                    "best_epoch": 1, "best_validation_loss": 0.7566005504793591,
                    "stale_epochs": 0}
        normalized = normalize_history(previous, config)
        self.assertEqual(normalized["epochs"][0]["epoch"], 8)
        self.assertEqual(normalized["epochs"][0]["experiment_epoch"], 1)
        self.assertEqual(normalized["best_epoch"], 8)
        self.assertEqual(normalize_history(normalized, config), normalized)

    @classmethod
    def setUpClass(cls):
        if (PREPARED / "rhythm/train.jsonl").is_file():
            cls.rows = load_rows(PREPARED, "rhythm", "train")
        else:
            cls.rows = []
            for index in range(67):
                source = f"synthetic-{index}.txt"
                cls.rows.append({"source": source, "target_track_id": index})
                for offset in range(1, 11):
                    cls.rows.append({"source": source, "target_track_id": index,
                                     "augmentation": {"file": f"{index}/shift_{offset}.txt",
                                                      "reference_offset": offset}})

    def test_268_distinct_with_all_67_originals_and_rotation(self):
        original_ids = {i for i, row in enumerate(self.rows) if "augmentation" not in row}
        self.assertEqual(len(original_ids), 67)
        schedules = [select_epoch_indices(self.rows, epoch, 415, 268) for epoch in range(4)]
        for order in schedules:
            self.assertEqual(len(order), 268)
            self.assertEqual(len(set(order)), 268)
            self.assertTrue(original_ids.issubset(order))
            self.assertEqual(sum("augmentation" in self.rows[i] for i in order), 201)
        self.assertEqual(schedules, [select_epoch_indices(self.rows, epoch, 415, 268)
                                     for epoch in range(4)])
        first_shifts = set(schedules[0]) - original_ids
        second_shifts = set(schedules[1]) - original_ids
        self.assertGreater(len(second_shifts - first_shifts), 150)

    def test_fixed_validation_and_source_assignments(self):
        if not (PREPARED / "rhythm/validation20.jsonl").is_file():
            self.skipTest("run epoch-7 preparation for real-data split verification")
        result = validate_epoch7_splits(PREPARED)
        self.assertEqual((result["train_original_pairs"], result["validation_fixed_pairs"],
                          result["validation_original_pairs"], result["test_pairs"]),
                         (67, 20, 10, 5))
        # The inherited beat-sequence overlap was explicitly accepted by the user.
        self.assertEqual(result["inherited_family_overlap"], ["beat-sequence"])
        self.assertEqual(verify_epoch7_adapter(HISTORICAL_ADAPTER),
                         sha256_file(HISTORICAL_ADAPTER / "adapter_model.safetensors"))

    def test_checkpoint_retention_and_latest_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyAdapter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = cosine_scheduler(optimizer, 408)
            state = {"global_step": 0, "epoch_index": 0, "next_position": 0,
                     "selected_order": [], "config": {"seed": 415},
                     "fingerprints": {"train": "test"}}
            for step in (10, 20, 30, 40):
                state["global_step"] = step
                save_checkpoint(root, "periodic", f"step_{step:08d}",
                                model, optimizer, scheduler, state)
            self.assertEqual(sorted(p.name for p in (root / "periodic").iterdir()),
                             ["step_00000020", "step_00000030", "step_00000040"])
            self.assertEqual(resolve_checkpoint(root, "latest").name, "step_00000040")
            for epoch in (1, 2):
                state["epoch_index"] = epoch
                save_checkpoint(root, "epoch_boundaries", f"epoch_{epoch:03d}_step_00000040",
                                model, optimizer, scheduler, state)
            self.assertEqual([p.name for p in (root / "epoch_boundaries").iterdir()],
                             ["epoch_002_step_00000040"])
            save_checkpoint(root, "final", "step_00000040", model, optimizer, scheduler, state)
            self.assertEqual(resolve_checkpoint(root, "latest").parent.name, "final")

    def test_checkpoint_retries_transient_windows_file_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyAdapter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = cosine_scheduler(optimizer, 408)
            state = {"global_step": 10, "epoch_index": 0, "next_position": 80,
                     "selected_order": list(range(268)), "config": {"seed": 415},
                     "fingerprints": {"train": "test"}}
            original_save = torch.save
            failed = False

            def locked_once(payload, path):
                nonlocal failed
                if Path(path).name == "training_state.pt" and not failed:
                    failed = True
                    error = OSError("file in use")
                    error.winerror = 32
                    raise error
                return original_save(payload, path)

            with patch("torch.save", side_effect=locked_once), patch(
                    "percussion_generation_v3.checkpoints.time.sleep") as sleep:
                saved = save_checkpoint(root, "periodic", "step_00000010",
                                        model, optimizer, scheduler, state)
            self.assertTrue(failed)
            sleep.assert_called_once()
            self.assertEqual(resolve_checkpoint(root, "latest"), saved)
            restored = load_checkpoint_state(saved, state["config"], state["fingerprints"])
            self.assertEqual((restored["global_step"], restored["next_position"]), (10, 80))

    def test_short_interrupted_training_restores_exact_position(self):
        if not (PREPARED / "rhythm/validation20.jsonl").is_file():
            self.skipTest("run epoch-7 preparation for real-data resume smoke test")
        before = sha256_file(HISTORICAL_ADAPTER / "adapter_model.safetensors")
        originals = [row for row in self.rows if "augmentation" not in row]
        shifts = [row for row in self.rows if "augmentation" in row][:201]
        self.assertEqual((len(originals), len(shifts)), (67, 201))
        processed = []

        def fake_loader(adapter, cpu, trainable):
            model = TinyAdapter()
            adapter = Path(adapter)
            if adapter != HISTORICAL_ADAPTER and (adapter / "adapter_model.safetensors").is_file():
                model.load_state_dict(torch.load(adapter / "adapter_model.safetensors",
                                                 map_location="cpu", weights_only=True))
            return model, torch.device("cpu")

        def fake_loss(model, row, device):
            processed.append(example_id(row))
            return (model.weight - 0.2) ** 2

        def fake_validation(model, rows, device):
            return {"loss": float(model.weight.detach() ** 2),
                    "pairs": len(rows), "target_tokens": sum(len(r["target_ids"]) for r in rows)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rhythm").mkdir()
            for name in ("validation.jsonl", "validation20.jsonl", "test.jsonl"):
                shutil.copy2(PREPARED / "rhythm" / name, root / "rhythm" / name)
            with (root / "rhythm/train.jsonl").open("w", encoding="utf-8") as stream:
                for row in originals + shifts:
                    stream.write(json.dumps(row) + "\n")
            shutil.copytree(PREPARED / "tokenizer", root / "tokenizer")
            shutil.copy2(PREPARED / "tokenizer_identity.json", root / "tokenizer_identity.json")
            shutil.copy2(PREPARED / "manifest.json", root / "manifest.json")
            options = {"output_dir": root, "examples_per_epoch": 268,
                       "learning_rate": 2e-5, "save_steps": 10, "epochs": 12,
                       "model_loader": fake_loader, "loss_fn": fake_loss,
                       "validation_fn": fake_validation, "cpu": True}
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train_rotating(init_adapter=HISTORICAL_ADAPTER, stop_after_steps=10, **options)
            checkpoint_root = root / "checkpoints/rhythm"
            first = resolve_checkpoint(checkpoint_root, "latest")
            self.assertEqual(first.name, "step_00000010")
            metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
            state = load_checkpoint_state(first, metadata["config"], metadata["fingerprints"])
            self.assertEqual((state["global_step"], state["epoch_index"], state["next_position"]),
                             (10, 0, 80))
            self.assertEqual(state["scheduler"]["last_epoch"], 10)
            self.assertTrue(state["optimizer"]["state"])
            self.assertEqual(state["config"]["learning_rate"], 2e-5)
            self.assertEqual(state["config"]["accumulation"], 8)
            self.assertEqual(state["config"]["save_steps"], 10)
            order = state["selected_order"]
            pool = originals + shifts
            self.assertEqual(processed, [example_id(pool[i]) for i in order[:80]])
            with self.assertRaisesRegex(InterruptedError, "smoke-test"):
                train_rotating(resume_checkpoint="latest", stop_after_steps=11,
                               patience=5, **options)
            second = resolve_checkpoint(checkpoint_root, "latest")
            metadata2 = json.loads((second / "metadata.json").read_text(encoding="utf-8"))
            resumed = load_checkpoint_state(second, metadata2["config"], metadata2["fingerprints"])
            self.assertEqual((resumed["global_step"], resumed["next_position"]), (11, 88))
            self.assertEqual(resumed["scheduler"]["last_epoch"], 11)
            self.assertEqual(resumed["config"]["patience"], 5)
            self.assertEqual(resumed["history"]["patience_changes"], [
                {"at_global_step": 10, "from": 3, "to": 5}])
            self.assertEqual(processed[80:], [example_id(pool[i]) for i in order[80:88]])
            manual = TinyAdapter()
            manual.load_state_dict(torch.load(first / "adapter/adapter_model.safetensors",
                                              map_location="cpu", weights_only=True))
            manual_optimizer = torch.optim.AdamW(manual.parameters(), lr=2e-5, foreach=False)
            manual_scheduler = cosine_scheduler(manual_optimizer, 408)
            manual_optimizer.load_state_dict(state["optimizer"])
            manual_scheduler.load_state_dict(state["scheduler"])
            for _ in order[80:88]:
                ((manual.weight - 0.2) ** 2 / 8).backward()
            torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.0, foreach=False)
            manual_optimizer.step()
            manual_scheduler.step()
            actual = torch.load(second / "adapter/adapter_model.safetensors",
                                map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(manual.state_dict()["weight"], actual["weight"]))
            self.assertEqual(manual_scheduler.state_dict()["last_epoch"], resumed["scheduler"]["last_epoch"])
            self.assertEqual(sha256_file(HISTORICAL_ADAPTER / "adapter_model.safetensors"), before)
