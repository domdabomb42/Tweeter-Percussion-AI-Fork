import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from custom_training.data_loader import Session, family_name, split_sessions
from custom_training.prepare import tokenizer_for_model, segment_session, load_prepared
from custom_training.serializer import serialize, serialize_object, leaf_spans, whole_track_object
from custom_training.baselines import feature_instances, feature_name, train_statistics
from custom_training.evaluate import _value, _native_valid, select_feature_queries
from custom_training.train import train

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "CustomExtractor"))
from custom_extractor import Composition, add_track
from custom_extractor.reporting import canonical


def composition(name, shift=0):
    c = Composition()
    c.global_settings["metadata"]["session_title"] = [name]
    add_track(c, 1, {"carrier": "sine", "modulator": "square", "ae": [0, .25, .75, 1] * 26,
                      "me": [1, .7, .3, 0] * 26, "harmonicity": 2.123456789, "mod-amp": 500,
                      "reverb-amp": .25, "echo-fb": .456789, "flanger-ms": 3.12345678,
                      "inst-name": "Original Pd Synth", "loop-length": 6000, "note-dur": 350,
                      "octave": -12},
              [{"step": 0, "pitch": 60+shift}, {"step": 16, "pitch": 64+shift}])
    return c


def session(name, family, shift=0):
    c = composition(name, shift)
    return Session(Path(name + "-session-coll.txt"), c, family, name, str(shift))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tokenizer_for_model(local_only=True)

    def test_serializer_is_canonical_and_lossless(self):
        c = composition("A")
        original = copy.deepcopy(canonical(c))
        text = serialize(c)
        self.assertEqual(json.loads(text), original)
        self.assertEqual(text, serialize(c))
        self.assertIn('2.123456789', text)
        self.assertEqual(canonical(c), original)

    def test_leaf_spans_hold_complete_values(self):
        c = composition("A")
        text, spans = leaf_spans(canonical(c))
        self.assertEqual(text, serialize(c))
        harmonicity = next(s for s in spans if s["path"][-1] == "harmonicity")
        self.assertEqual(text[harmonicity["start"]:harmonicity["end"]], "2.123456789")
        self.assertEqual(harmonicity["value"], 2.123456789)

    def test_segmentation_preserves_tracks(self):
        c = composition("A")
        for tid in (2, 3):
            track = copy.deepcopy(c.tracks[0])
            track["id"] = tid
            c.tracks.append(track)
        s = Session(Path("synthetic"), c, "synthetic", "sha", "notes")
        single = len(self.tokenizer.encode(serialize_object(whole_track_object(c, [c.tracks[0]])), add_special_tokens=False)) + 1
        segments = segment_session(s, self.tokenizer, single + 20)
        self.assertGreater(len(segments), 1)
        self.assertEqual(sorted(tid for r in segments for tid in r["track_ids"]), [1,2,3])
        self.assertTrue(all(len(r["token_ids"]) <= single + 20 for r in segments))
        self.assertEqual([t for r in segments for t in json.loads(r["text"])["tracks"]], c.tracks)

    def test_no_silent_truncation(self):
        c = composition("A")
        s = Session(Path("synthetic"), c, "synthetic", "sha", "notes")
        with self.assertRaisesRegex(ValueError, "no truncation"):
            segment_session(s, self.tokenizer, 100)

    def test_family_and_arrangement_groups(self):
        self.assertEqual(family_name("insterstellar-end-session-coll.txt"), "interstellar")
        self.assertEqual(family_name("interstellar-intro-session-coll.txt"), "interstellar")
        sessions = [session("a1", "a", 0), session("a2", "a", 0), session("b1", "b", 1),
                    session("c1", "c", 2), session("d1", "d", 3), session("e1", "e", 4)]
        split, manifest = split_sessions(sessions, 415)
        self.assertEqual({a.name for part in split.values() for a in part}, {a.name for a in sessions})
        locations = {a.name: name for name, part in split.items() for a in part}
        self.assertEqual(locations[sessions[0].name], locations[sessions[1].name])
        self.assertEqual(split_sessions(sessions, 415)[1]["partitions"], manifest["partitions"])

    def test_feature_values_are_actual_complete_scalars(self):
        c = composition("A")
        row = {"text": serialize(c), "source": "A", "segment": 0}
        features = list(feature_instances([row]))
        value = next(x for x in features if x["name"] == "synth.harmonicity")
        self.assertEqual(value["value"], 2.123456789)
        self.assertTrue(value["prefix"].endswith('"harmonicity":'))
        self.assertIn("note.pitch", {x["name"] for x in features})
        self.assertNotIn("synth.carrier_envelope", {x["name"] for x in features})
        self.assertEqual(_value('2.123456789,"next"'), 2.123456789)
        self.assertEqual(_value('"square",'), "square")
        self.assertIsNone(_value("garbage"))
        self.assertTrue(_native_valid("note.pitch", -12.5))
        self.assertFalse(_native_valid("note.step", 64))
        self.assertFalse(_native_valid("effects.echo.feedback", 1.0))
        self.assertEqual(select_feature_queries([row], 1)[0]["source"], "A")

    def test_dry_run_never_loads_model(self):
        from custom_training import MODEL_ID, MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"max_training_length":4096,
                "tokenizer": {"model_id": MODEL_ID, "revision": MODEL_REVISION}}))
            (root / "tokenizer").mkdir()
            (root / "tokenizer/tokenizer.json").write_bytes(b"fixture")
            row = {"text": serialize(composition("A")), "source": "A", "segment": 0,
                   "token_ids": self.tokenizer.encode(serialize(composition("A")), add_special_tokens=False) + [self.tokenizer.eos_token_id]}
            for part in ("train", "validation"):
                (root / (part + ".jsonl")).write_text(json.dumps(row) + "\n")
            plan = train(root, root / "ckpt", cpu=True, dry_run=True)
            self.assertEqual(plan["lora_rank"],16)
            self.assertEqual(plan["train_tokens"],len(row["token_ids"]))
            self.assertFalse((root / "ckpt").exists())

    def test_resume_dry_run_counts_completed_epochs_without_writing(self):
        from custom_training import MODEL_ID, MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"max_training_length": 4096,
                "tokenizer": {"model_id": MODEL_ID, "revision": MODEL_REVISION}}))
            (root / "tokenizer").mkdir()
            (root / "tokenizer/tokenizer.json").write_bytes(b"fixture")
            row = {"text": serialize(composition("A")), "source": "A", "segment": 0,
                   "token_ids": [1, 2, 3]}
            for part in ("train", "validation"):
                (root / (part + ".jsonl")).write_text(json.dumps(row) + "\n")
            checkpoint = root / "ckpt"
            adapter = checkpoint / "best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"dry-run marker")
            history = {"plan": {"model": MODEL_ID, "revision": MODEL_REVISION,
                                "train_segments": 1, "validation_segments": 1,
                                "max_sequence_length": 4096},
                       "epochs": [{"epoch": number, "updates": number} for number in range(1, 6)],
                       "best_epoch": 5, "best_validation_loss": 0.4}
            history_path = checkpoint / "history.json"
            history_path.write_text(json.dumps(history))
            plan = train(root, checkpoint, epochs=20, resume=True, dry_run=True, cpu=True)
            self.assertEqual((plan["completed_epochs"], plan["additional_epochs"]), (5, 15))
            self.assertFalse((checkpoint / "best_adapter_epoch5_backup").exists())
            self.assertEqual(json.loads(history_path.read_text()), history)


if __name__ == "__main__":
    unittest.main()
