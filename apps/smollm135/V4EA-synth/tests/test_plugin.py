"""Focused regression tests for immutable feedback and resumable DPO."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

from percussion_generation_v4ea import OUTPUT
from percussion_generation_v4ea.plugin_bridge import Bridge, _load_active_run, _save_active_run
from percussion_generation_v4ea.plugin_checkpoints import inspect, recover, verify
from percussion_generation_v4ea.plugin_dpo import train_dpo, lora_optimizer
from percussion_generation_v4ea.plugin_generation import (prepare_feedback_pair,
                                                            render_candidate,
                                                            sample_training_row,
                                                            GenerationService, decode_sample)
from percussion_generation_v4ea.plugin_preference import (completion_logprob,
    eligible_pairs, load_snapshot, make_snapshot, precompute_reference)
from percussion_generation_v4ea.plugin_store import FeedbackStore, canonical, digest


def fake_group(manifest, source="source-A", family="family-A"):
    group_id = str(uuid.uuid4())
    candidates = [{"candidate_id": str(uuid.uuid4()), "positions": [i, i+16],
                   "completion_ids": [i+1, 0]} for i in range(5)]
    group = {"group_id": group_id, "source": source, "source_family": family,
             "tokenizer_sha256": manifest["tokenizer_sha256"],
             "dataset_sha256": manifest["prepared_dataset_sha256"],
             "codec_schema_sha256": manifest["codec_schema_sha256"],
             "fixed_preset_sha256": manifest["fixed_preset_sha256"],
             "prompt_ids": [1, 2], "candidates": candidates}
    sessions = {c["candidate_id"]: f"native {i};\n" for i, c in enumerate(candidates)}
    return group, sessions


class TinyPeft:
    def __init__(self, value=0.0):
        import torch
        self.module = torch.nn.Module()
        self.module.embedding = torch.nn.Embedding(10, 4)
        self.module.head = torch.nn.Linear(4, 10, bias=False)
        with torch.no_grad():
            self.module.embedding.weight.copy_(torch.arange(40, dtype=torch.float32).reshape(10, 4) / 50)
        self.module.head.weight = self.module.embedding.weight
        self.module.register_parameter("lora_delta", torch.nn.Parameter(torch.tensor(value)))
        self.config = SimpleNamespace(use_cache=False)

    def named_parameters(self):
        return self.module.named_parameters()

    def parameters(self):
        return self.module.parameters()

    def get_input_embeddings(self):
        return self.module.embedding

    def get_output_embeddings(self):
        return self.module.head

    def gradient_checkpointing_enable(self, **_kwargs):
        pass

    def train(self, mode=True):
        self.module.train(mode)

    def __call__(self, input_ids, **_kwargs):
        import torch
        embedding = self.module.embedding(input_ids)
        logits = embedding @ self.module.head.weight.T
        logits = logits + self.module.lora_delta * torch.arange(10).reshape(1, 1, 10)
        return SimpleNamespace(logits=logits)


class TinyModel:
    def __init__(self, value=0.0):
        self.peft_model = TinyPeft(value)
        self.config = self.peft_model.config

    def __call__(self, **kwargs):
        return self.peft_model(**kwargs)

    def train(self, mode=True):
        self.peft_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save_pretrained(self, path):
        import torch
        path = Path(path)
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        torch.save({"lora": self.peft_model.module.lora_delta.detach().clone()},
                   path / "adapter_model.safetensors")
        (path / "embedding.safetensors").write_bytes(b"frozen-test-embedding")


def tiny_loader(_output, adapter=None, cpu=False, trainable=True):
    import torch
    value = 0.0
    adapter = Path(adapter)
    if (adapter / "adapter_config.json").read_text(encoding="utf-8") == "{}":
        value = float(torch.load(adapter / "adapter_model.safetensors", weights_only=True)["lora"])
    return TinyModel(value), torch.device("cpu")


class PluginTests(unittest.TestCase):
    def test_bridge_candidate_switch_rating_ack_and_recovery(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback")
            group, sessions = fake_group(self.manifest)
            group.update(target_track_id=5, pitch=55)
            store.save_group(group, sessions)
            bridge = object.__new__(Bridge)
            bridge.store, bridge.group = store, group
            bridge.active_index, bridge.pending_index = 0, None
            bridge.draft_rating, bridge.warned_unsaved = None, False
            bridge.offline, bridge.busy = True, False
            bridge.lock, bridge.cancel_event = threading.RLock(), threading.Event()
            events = []
            bridge.broadcast = lambda *pieces: events.append(pieces)
            bridge.handle(["choice", "8"])
            bridge.handle(["next"])
            self.assertIsNone(bridge.pending_index)
            self.assertEqual(events[-1][0], "ai_warn")
            bridge.handle(["rate", "8"])
            self.assertEqual(events[-2][0], "ai_rating_ack")
            self.assertEqual(store.current_ratings(group["group_id"])[group["candidates"][0]["candidate_id"]]["rating"], 8)
            for index in (2, 3, 4, 5, 1):
                bridge.handle(["next"])
                if bridge.pending_index is None:  # Unsaved rating requires the explicit second click.
                    self.assertEqual(events[-1][0], "ai_warn")
                    bridge.handle(["next"])
                self.assertEqual(bridge.pending_index, index-1)
                self.assertEqual(events[-1][0], "ai_preview")
                bridge.handle(["applied", str(index)])
                self.assertEqual(bridge.active_index, index-1)
            bridge.group = None
            bridge.handle(["recover", group["group_id"]])
            self.assertEqual(bridge.active_index, 0)
            self.assertEqual(bridge.group["group_id"], group["group_id"])
            self.assertIn(("ai_group", group["group_id"]), events)
            bridge.handle(["next"])
            if bridge.pending_index is None:
                bridge.handle(["next"])
            bridge._expire_switch(group["group_id"], 1)
            self.assertIsNone(bridge.pending_index)
            self.assertEqual(events[-1][0], "ai_error")
            bridge.offline = False
            with self.assertRaises(RuntimeError):
                bridge.handle(["next"])
            with self.assertRaises(RuntimeError):
                bridge.handle(["rate", "9"])

    def test_incomplete_group_uuid_is_advertised_after_bridge_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback")
            group, sessions = fake_group(self.manifest)
            store.save_group(group, sessions)
            bridge = object.__new__(Bridge)
            bridge.store, bridge.group = store, None
            bridge.dpo_root = Path(directory) / "dpo"
            bridge.dpo_root.mkdir()
            bridge.training_run, bridge.training = None, None
            bridge.service = SimpleNamespace(output_dir=OUTPUT, adapter=self.reference)
            events = []
            bridge.broadcast = lambda *pieces: events.append(pieces)
            bridge.handle(["stats"])
            self.assertIn(("ai_group", group["group_id"]), events)

    def test_bridge_ready_status_clears_loading_progress(self):
        bridge = object.__new__(Bridge)
        bridge.training, bridge.training_run, bridge.offline = None, None, False
        bridge.service = SimpleNamespace(model=None)
        bridge.service.load = lambda: setattr(bridge.service, "model", object())
        events = []
        bridge.broadcast = lambda *pieces: events.append(pieces)
        bridge.handle(["init"])
        bridge.handle(["offline", "1"])
        self.assertIn(("ai_progress", "idle", 0, 0, 5), events)
        self.assertEqual(events[-1], ("ai_status", "model_ready_offline"))

    def test_training_status_distinguishes_pause_completion_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "round"
            run.mkdir()
            bridge = object.__new__(Bridge)
            bridge.training_run, bridge.training = run, None
            (run / "trainer.pid").write_text(str(os.getpid()), encoding="ascii")
            self.assertTrue(bridge._training_active())
            (run / "trainer.pid").write_text("2147483647", encoding="ascii")
            self.assertFalse(bridge._training_active())
            self.assertFalse((run / "trainer.pid").exists())
            events = []
            bridge.broadcast = lambda *pieces: events.append(pieces)
            for stored, expected in (("paused", "paused"), ("complete", "complete"),
                                     ("failed", "failed"), ("training", "failed")):
                (run / "progress.json").write_text(json.dumps({"status": stored,
                    "epoch": 2, "global_optimizer_step": 10, "examples": 40}), encoding="utf-8")
                bridge._training_progress()
                self.assertEqual(events[-2], ("ai_training", expected))
                self.assertEqual(events[-1][0], "ai_training_progress")
            with patch.object(bridge, "_training_active", return_value=True):
                bridge.handle(["init"])
            self.assertIn(("ai_status", "training_running"), events)
            self.assertNotIn(("ai_progress", "loading_model", 0, 0, 5), events)

    def test_active_dpo_round_survives_bridge_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "round_001"
            run.mkdir()
            (run / "snapshot.json").write_text("{}", encoding="utf-8")
            _save_active_run(root, run)
            self.assertEqual(_load_active_run(root), run.resolve())
            outside = root.parent / "outside_round"
            (root / "active_round.json").write_bytes(canonical({"run_dir": str(outside)}))
            with self.assertRaises(ValueError):
                _load_active_run(root)

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        cls.reference = OUTPUT / "checkpoints/rhythm_coverage_v1/best_adapter"

    def test_feedback_is_durable_append_only_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback")
            group, sessions = fake_group(self.manifest)
            store.save_group(group, sessions)
            first = group["candidates"][0]["candidate_id"]
            event = store.submit(group["group_id"], first, 8, str(uuid.uuid4()))
            same = store.submit(group["group_id"], first, 8, event["submission_id"])
            self.assertEqual(event, same)
            correction = store.submit(group["group_id"], first, 10)
            self.assertEqual(correction["revision"], 2)
            self.assertEqual(store.current_ratings(group["group_id"])[first]["rating"], 10)
            self.assertEqual(len(list(store.events())), 2)
            self.assertEqual(store.incomplete_groups()[0]["rated"], 1)
            for candidate in group["candidates"][1:]:
                store.submit(group["group_id"], candidate["candidate_id"], 2)
            self.assertEqual(store.statistics()["completed_groups"], 1)
            backup = store.backup(Path(directory) / "backup")
            self.assertGreater(backup["files"], 5)
            exported = store.export(Path(directory) / "export.json")
            self.assertEqual(exported["rating_events"], 6)
            self.assertEqual(store.get_group(group["group_id"])["group_id"], group["group_id"])

    def test_pairs_exclude_ties_and_normalize_each_group(self):
        group, _ = fake_group(self.manifest)
        candidates = group["candidates"]
        ratings = {item["candidate_id"]: {"rating": rating}
                   for item, rating in zip(candidates, (10, 2, 8, 6, 6))}
        pairs = eligible_pairs(group, ratings, 2)
        self.assertTrue(pairs)
        self.assertAlmostEqual(sum(pair["group_weight"] for pair in pairs), 1)
        self.assertTrue(all(pair["rating_difference"] >= 2 for pair in pairs))
        self.assertFalse(any(pair["chosen_rating"] == pair["rejected_rating"] for pair in pairs))
        strong = next(p for p in pairs if p["rating_difference"] == 8)
        weak = next(p for p in pairs if p["rating_difference"] == 2)
        self.assertGreater(strong["group_weight"], weak["group_weight"])

    def test_snapshot_remains_frozen_after_rating_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback")
            groups = [fake_group(self.manifest, f"source-{i}", f"family-{i}") for i in range(3)]
            for group, sessions in groups:
                store.save_group(group, sessions)
                for candidate, rating in zip(group["candidates"], (10, 2, 8, 4, 6)):
                    store.submit(group["group_id"], candidate["candidate_id"], rating)
            run = Path(directory) / "round"
            summary = make_snapshot(store, run, self.manifest, self.reference)
            frozen, before = load_snapshot(run)
            self.assertGreater(summary["train_pairs"], 0)
            self.assertGreater(summary["validation_pairs"], 0)
            self.assertFalse(set(frozen["train_families"]) & set(frozen["validation_families"]))
            store.submit(groups[0][0]["group_id"], groups[0][0]["candidates"][0]["candidate_id"], 1)
            again, after = load_snapshot(run)
            self.assertEqual(before, after)
            self.assertEqual(frozen, again)
            self.assertGreater(store.statistics(previous_snapshot_events=frozen["raw_rating_event_count"])["new_rating_events_since_snapshot"], 0)

    def test_seeded_sampling_restores_rng_state(self):
        import torch
        class FakeTokenizer:
            eos_token_id = 0
            def decode(self, ids, skip_special_tokens=False):
                return "7\n[END_RHYTHM]<eos>"
        class FakePeft:
            def generate(self, input_ids, **_options):
                sampled = torch.randint(1, 100, (1, 1), device=input_ids.device)
                return torch.cat((input_ids, sampled, torch.zeros((1, 1), dtype=torch.long, device=input_ids.device)), 1)
        model = SimpleNamespace(peft_model=FakePeft())
        before = torch.get_rng_state().clone()
        with patch("percussion_generation_v4ea.plugin_generation.PositionConstraint") as constraint:
            constraint.return_value.allowed.return_value = list(range(100))
            first = decode_sample(model, FakeTokenizer(), torch.device("cpu"), [1, 2], 1, 42)
            second = decode_sample(model, FakeTokenizer(), torch.device("cpu"), [1, 2], 1, 42)
        self.assertEqual(first[0], [7])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_five_candidates_retry_duplicates_and_preserve_track_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            service = object.__new__(GenerationService)
            service.store = FeedbackStore(Path(directory) / "feedback")
            service.rows = [{"source": "source-A", "augmentation": {"file": "shift"}}]
            service.rng = __import__("random").Random(415)
            service.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "target")
            service.generated = {}
            service.manifest = self.manifest
            service.adapter = self.reference
            service.preset = {"example_pitch": 55}
            service.load = lambda: (object(), "cpu")
            source = Path(directory) / "source.txt"
            source.write_text("source", encoding="utf-8")
            original = SimpleNamespace(to_dict=lambda: {"native": "original"})
            condition = SimpleNamespace(to_dict=lambda: {"native": "condition"})
            removed = {"id": 5, "notes": [{"step": 0}], "timing": {"loop_ms": 1500}}
            patterns = iter(([0], [0], [1], [2], [3], [4]))
            def fake_decode(*_args, **_kwargs):
                return next(patterns), [7, 0], .01
            def fake_render(_condition, _original, _removed, positions, _preset):
                return {"id": 5, "notes": positions, "instrument": {"username": "AI_A"}}, "native"+str(positions)
            with patch("percussion_generation_v4ea.plugin_generation.prepare_feedback_pair",
                       return_value=(source, original, condition, removed, "prompt", [1, 2])), \
                 patch("percussion_generation_v4ea.plugin_generation.decode_sample", side_effect=fake_decode), \
                 patch("percussion_generation_v4ea.plugin_generation.render_candidate", side_effect=fake_render):
                group = service.generate_group(max_attempts=6, seeds=iter(range(6)))
            self.assertEqual(len(group["candidates"]), 5)
            self.assertEqual(group["sampling"]["duplicate_attempts"], 1)
            self.assertEqual({c["native_track"]["id"] for c in group["candidates"]}, {5})
            self.assertEqual(len(list((service.store.groups / group["group_id"]).glob("*.txt"))), 5)

    def test_training_prompt_and_native_target_slot(self):
        from percussion_generation_v4ea.pipeline import load_new_tokenizer
        from percussion_generation_v3.data import AUGMENTED_SESSIONS
        tokenizer, _ = load_new_tokenizer(OUTPUT)
        with (OUTPUT / "rhythm/train.jsonl").open(encoding="utf-8") as stream:
            row = next(candidate for line in stream
                       if (candidate := json.loads(line)).get("augmentation"))
        augmentation = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
        generated = {item["augmented_filename"]: item for item in augmentation["generated"]}
        _path, original, condition, removed, prompt, ids = prepare_feedback_pair(row, generated, tokenizer)
        self.assertEqual(prompt, row["prefix"])
        self.assertEqual(ids, row["prefix_ids"])
        self.assertEqual(len(condition.tracks)+1, len(original.tracks))
        preset = json.loads((OUTPUT / "fixed_preset.json").read_text(encoding="utf-8"))
        track, native = render_candidate(condition, original, removed, row["target_positions"], preset)
        self.assertEqual(track["id"], removed["id"])
        self.assertEqual(track["instrument"]["username"], "AI_A")
        self.assertIn("AI_A", native)

    def test_uniform_source_sampling_not_variant_weighting(self):
        import random
        rows = ([{"source": "A", "augmentation": {"file": str(i)}} for i in range(100)] +
                [{"source": "B", "augmentation": {"file": "only"}}])
        draws = [sample_training_row(rows, random.Random(i))["source"] for i in range(100)]
        self.assertGreater(draws.count("B"), 30)
        self.assertLess(draws.count("B"), 70)

    def test_only_lora_tensors_enter_optimizer(self):
        model = TinyModel()
        optimizer = lora_optimizer(model)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertTrue(model.peft_model.module.lora_delta.requires_grad)
        self.assertFalse(model.peft_model.get_input_embeddings().weight.requires_grad)
        self.assertFalse(model.peft_model.get_output_embeddings().weight.requires_grad)

    def test_pd_connection_indices_and_offline_gate(self):
        repo = Path(__file__).resolve().parents[4]
        paths = [repo / "apps/L2Ork-Tweeter/L2Ork-Tweeter.pd",
                 repo / "apps/L2Ork-Tweeter/core/L2OrkTweet.pd",
                 *sorted((repo / "apps/L2Ork-Tweeter/core").glob("ai-*.pd"))]
        self.assertEqual(len(paths), 5)
        for path in paths:
            stack = []
            for line in path.read_text(encoding="utf-8", errors="surrogateescape").splitlines():
                if line.startswith("#N canvas"):
                    stack.append([0, []])
                elif line.startswith("#X restore"):
                    count, edges = stack.pop()
                    self.assertTrue(all(src < count and dst < count for src, dst in edges), path)
                    if stack:
                        stack[-1][0] += 1
                elif line.startswith("#X connect"):
                    atoms = line.removesuffix(";").split()
                    stack[-1][1].append((int(atoms[2]), int(atoms[4])))
                elif line.startswith("#X ") and not line.startswith(("#X coords", "#X array", "#X declare", "#X f ")):
                    stack[-1][0] += 1
            for count, edges in stack:
                self.assertTrue(all(src < count and dst < count for src, dst in edges), path)
        bridge = paths[2].read_text(encoding="utf-8")
        self.assertIn("connect 127.0.0.1 19785", bridge)
        self.assertIn("r network-status", bridge)
        self.assertIn("v ai-network-status", bridge)
        panel = (repo / "apps/L2Ork-Tweeter/core/ai-feedback-track.pd").read_text(encoding="utf-8")
        self.assertIn("#X coords 0 -1 1 1 268 108 2 850 0", panel)
        self.assertIn("dirty 0", panel)
        main = paths[0].read_text(encoding="utf-8", errors="surrogateescape")
        self.assertIn("#X obj 850 59 spigot 1;", main)

    def test_reference_logprob_causal_shift_and_target_only(self):
        import torch
        import torch.nn.functional as F
        model = TinyModel()
        prompt, completion = [1, 2], [3, 0]
        actual = completion_logprob(model, prompt, completion, torch.device("cpu"))
        ids = torch.tensor([prompt+completion])
        logits = model(input_ids=ids).logits
        expected = (F.log_softmax(logits[0, 1].float(), -1)[3] +
                    F.log_softmax(logits[0, 2].float(), -1)[0])
        self.assertTrue(torch.allclose(actual, expected))

    def _snapshot(self, directory):
        run = Path(directory) / "round"
        pairs = []
        for group in range(8):
            for _ in range(6):
                pairs.append({"group_id": f"g{group}", "source": f"s{group}",
                              "source_family": f"f{group}", "chosen_id": "c",
                              "rejected_id": "r", "rating_difference": 4,
                              "group_weight": 1/6, "prompt_ids": [1, 2],
                              "chosen_ids": [3, 0], "rejected_ids": [4, 0]})
        snapshot = {"format": "TweeterDPO-snapshot-v1",
                    "reference_adapter": str(self.reference.resolve()),
                    "reference_adapter_sha256": digest((self.reference / "adapter_model.safetensors").read_bytes()),
                    "embedding_sha256": digest((self.reference / "embedding.safetensors").read_bytes()),
                    "tokenizer_sha256": self.manifest["tokenizer_sha256"],
                    "dataset_sha256": self.manifest["prepared_dataset_sha256"],
                    "codec_schema_sha256": self.manifest["codec_schema_sha256"],
                    "fixed_preset_sha256": self.manifest["fixed_preset_sha256"],
                    "train_families": [f"f{i}" for i in range(8)],
                    "validation_families": ["validation"],
                    "pairs": {"train": pairs, "validation": [{**pairs[0],
                            "group_id": "validation", "source_family": "validation", "group_weight": 1.0}]}}
        run.mkdir()
        raw = canonical(snapshot) + b"\n"
        (run / "snapshot.json").write_bytes(raw)
        (run / "snapshot.sha256").write_text(digest(raw), encoding="ascii")
        return run

    def test_interrupt_resume_matches_uninterrupted_and_falls_back(self):
        import torch
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            full = self._snapshot(first)
            interrupted = self._snapshot(second)
            for run in (full, interrupted):
                precompute_reference(run, OUTPUT, cpu=True, model_loader=tiny_loader)
            with patch("percussion_generation_v4ea.plugin_dpo.verify_dataset", return_value=(self.manifest, {})):
                complete = train_dpo(full, OUTPUT, epochs=1, accumulation=4,
                                     save_steps=10, model_loader=tiny_loader, cpu=True)
                paused = train_dpo(interrupted, OUTPUT, epochs=1, accumulation=4,
                                   save_steps=10, max_optimizer_steps=11,
                                   model_loader=tiny_loader, cpu=True)
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["global_step"], 11)
                resumed = train_dpo(interrupted, OUTPUT, epochs=1, accumulation=4,
                                    save_steps=10, resume_checkpoint="latest",
                                    model_loader=tiny_loader, cpu=True)
            self.assertEqual(complete["global_step"], resumed["global_step"])
            self.assertEqual(complete["global_step"], 12)
            a = verify(Path(complete["final_checkpoint"]))
            b = verify(Path(resumed["final_checkpoint"]))
            self.assertEqual(a["global_step"], b["global_step"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            lora_a = torch.load(Path(complete["final_checkpoint"]) / "adapter/adapter_model.safetensors", weights_only=True)["lora"]
            lora_b = torch.load(Path(resumed["final_checkpoint"]) / "adapter/adapter_model.safetensors", weights_only=True)["lora"]
            self.assertTrue(torch.allclose(lora_a, lora_b, atol=1e-7))
            periodic = list((interrupted / "checkpoints/periodic").glob("step_*"))
            self.assertEqual([p.name for p in periodic], ["step_00000010"])
            self.assertTrue(list((interrupted / "checkpoints/epochs").glob("epoch_*")))
            pointer = interrupted / "checkpoints/latest.json"
            final = Path(resumed["final_checkpoint"])
            (final / "state.pt").write_bytes(b"broken")
            recovered, state, failures = recover(interrupted / "checkpoints")
            self.assertNotEqual(recovered, final)
            self.assertTrue(failures)
            self.assertEqual(state["global_step"], 12)
            self.assertTrue(any(not row["valid"] for row in inspect(interrupted / "checkpoints")))


if __name__ == "__main__":
    unittest.main()
