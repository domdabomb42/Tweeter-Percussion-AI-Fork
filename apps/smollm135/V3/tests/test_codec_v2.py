"""Exact V2 codec and preserved V1 checks on real and unusual structures."""
from pathlib import Path
import json
import unittest
import tempfile
import hashlib
import math

from percussion_generation_v3 import data as _data  # install model-local import paths
from custom_extractor import extract_tweeter
from percussion_generation_v3.codec import pack_condition, unpack_condition
from percussion_generation_v3.codec_v2 import SCHEMA_SHA256, decode_v2, dumps, encode_v2
from percussion_generation_v3.data import DEFAULT_SESSIONS, ROOT, load_rows, load_tokenizer, verify_adapter_codec
from percussion_generation_v3.evaluation import remove_target
from percussion_generation_v3.inference import prompt, sparse
from percussion_generation_v3.lossless_dataset import (_row_groups, _statistics, _source_path,
    sequence_dictionary_variant, restore_sequence_dictionary)
from percussion_generation_v3.roles import role


def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def condition(composition):
    return {"global": composition.global_settings,
            "tracks": [{"role": role(track), "track": track} for track in composition.tracks]}


class CodecV2Tests(unittest.TestCase):
    def test_unusual_values_and_all_envelope_modes(self):
        value = {"global": {"tracker_steps": 64, "metadata": {
            "session_title": None, "new": {"x": [None, -2, 1.0, 1, []]}},
            "unknown": {"nested": {"missing_or_null": None, "negative_zero": -0.0}}},
            "tracks": [{"role": "percussion", "track": {
                "id": -3, "synth": {"carrier_envelope": [1, 1.0, -0.25, -0.25, 0],
                                    "modulator_envelope": [], "extra": [None]},
                "timing": {"loop_ms": 1.23456789012345},
                "effects": {"overdrive": {"level": 0, "override": None},
                             "reverb": {}, "echo": None, "flanger": {"delay_ms": -1.0}},
                "instrument": {"name": "奇妙"}, "mixing": {},
                "playback": {"cursor": {"running": None}},
                "notes": [{"step": 0, "pitch": 37, "velocity": 0.5},
                          {"step": 3, "pitch": 42}]}},
                       {"role": "other", "track": {"notes": []}, "extra": None}],
            "future": {"value": 5}}
        for mode in ("raw", "rle", "dictionary", "zlib"):
            encoded = encode_v2(value, mode)
            self.assertTrue(exact(decode_v2(encoded), value), mode)
            self.assertEqual(math.copysign(1, decode_v2(encoded)["global"]["unknown"]["nested"]["negative_zero"]), -1)
            self.assertEqual(encoded, encode_v2(value, mode))
        with self.assertRaisesRegex(ValueError, "version"):
            decode_v2('["L2V3",0,[],[]]')
        self.assertEqual(len(SCHEMA_SHA256), 64)
        sequence = [1, 2.0, -3, 0] * 6 + [None, 4.25]
        self.assertTrue(exact(restore_sequence_dictionary(sequence_dictionary_variant(sequence)), sequence))

    def test_every_original_session_and_legacy(self):
        files = sorted(DEFAULT_SESSIONS.rglob("*.txt"))
        self.assertGreater(len(files), 25)
        for path in files:
            source = extract_tweeter(path)
            value = condition(source)
            self.assertTrue(exact(decode_v2(encode_v2(value)), value), path.name)
            legacy_value = {"global": sparse(source.global_settings),
                            "tracks": [{"role": role(t), "track": sparse(t)} for t in source.tracks]}
            self.assertEqual(unpack_condition(pack_condition(legacy_value)), legacy_value)

    def test_prepared_prompt_and_splits(self):
        source_root = ROOT / "output_augmented_from_epoch7"
        groups, sources = _row_groups(source_root)
        self.assertEqual(len(groups["train"]), 2688)
        self.assertEqual(len(groups["validation20"]), 20)
        self.assertFalse(sources["train"] & sources["validation"])
        self.assertFalse(sources["train"] & sources["test"])
        row = groups["validation"][0]
        composition = remove_target(extract_tweeter(DEFAULT_SESSIONS / row["source"]), row["target_track_id"])
        prefix = prompt(composition, row["target_note_count"], "lossless-v2")
        self.assertEqual(decode_v2(prefix.split("\nRHYTHM_TARGET=")[0]), condition(composition))
        tokenizer = load_tokenizer(source_root)
        legacy_prefix = prompt(composition, row["target_note_count"], "legacy-v1")
        self.assertEqual(prefix, encode_v2(condition(composition)) +
                         "\nRHYTHM_TARGET=" + str(row["target_note_count"]) + "\n")
        self.assertNotEqual(tokenizer.encode(prefix, add_special_tokens=False),
                            tokenizer.encode(legacy_prefix, add_special_tokens=False))
        self.assertEqual(tokenizer.encode(row["target"], add_special_tokens=False) +
                         [tokenizer.eos_token_id], row["target_ids"])
        self.assertEqual(_statistics([2, 2, 5, 9])["total"], 18)

    def test_checkpoint_codec_identity(self):
        historical = ROOT / "output/checkpoints/rhythm/best_adapter"
        self.assertEqual(verify_adapter_codec(ROOT / "output", historical)["encoding"], "compact")
        self.assertEqual(load_tokenizer(ROOT / "output").encode("RHYTHM_TARGET=5"),
                         load_tokenizer(ROOT / "output_augmented").encode("RHYTHM_TARGET=5"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rhythm").mkdir()
            (root / "tokenizer").mkdir()
            (root / "tokenizer/tokenizer.json").write_bytes(b"test tokenizer")
            tokenizer_sha = hashlib.sha256(b"test tokenizer").hexdigest()
            digest = hashlib.sha256()
            for split in ("train", "validation", "test", "validation20"):
                path = root / "rhythm" / f"{split}.jsonl"
                path.write_bytes(b"")
                digest.update(split.encode() + b"\0")
            manifest = {"encoding": "lossless-v2", "codec_version": "lossless-v2",
                        "codec_schema_sha256": SCHEMA_SHA256,
                        "envelope_mode": "zlib",
                        "tokenizer_sha256": tokenizer_sha, "prepared_dataset_sha256": digest.hexdigest()}
            from custom_training import MODEL_ID, MODEL_REVISION
            manifest.update(model_id=MODEL_ID, revision=MODEL_REVISION)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter is not validated"):
                verify_adapter_codec(root, historical)
            adapter = root / "adapter"
            adapter.mkdir()
            metadata = {"codec_version": "lossless-v2", "codec_schema_sha256": SCHEMA_SHA256,
                        "tokenizer_sha256": tokenizer_sha, "prepared_dataset_sha256": digest.hexdigest()}
            (adapter / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(verify_adapter_codec(root, adapter), manifest)
            metadata["tokenizer_sha256"] = "changed"
            (adapter / "experiment_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter is not validated"):
                verify_adapter_codec(root, adapter)

    def test_prepared_v2_full_pool_smoke(self):
        root = ROOT / "output_lossless_v2"
        if not (root / "manifest.json").is_file():
            self.skipTest("V2 full preparation has not completed")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["codec_schema_sha256"], SCHEMA_SHA256)
        groups, _ = _row_groups(root)
        self.assertEqual((len(groups["train"]), len(groups["validation"]),
                          len(groups["test"]), len(groups["validation20"])),
                         (2688, 10, 5, 20))
        tokenizer = load_tokenizer(root)
        from percussion_generation_v3.data import AUGMENTED_SESSIONS, verify_v2_dataset
        verify_v2_dataset(root, manifest)
        generated = {entry["augmented_filename"]: entry for entry in
                     json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))["generated"]}
        for row in (groups["train"][0], next(r for r in groups["train"] if "augmentation" in r),
                    groups["validation20"][-1]):
            source = extract_tweeter(_source_path(row, generated))
            conditioned = remove_target(source, row["target_track_id"])
            self.assertEqual(row["prefix"], prompt(conditioned, row["target_note_count"], "lossless-v2"))
            self.assertEqual(row["prefix_ids"], tokenizer.encode(row["prefix"], add_special_tokens=False))
            self.assertEqual(row["target_ids"], tokenizer.encode(row["target"], add_special_tokens=False) +
                             [tokenizer.eos_token_id])


if __name__ == "__main__":
    unittest.main()
