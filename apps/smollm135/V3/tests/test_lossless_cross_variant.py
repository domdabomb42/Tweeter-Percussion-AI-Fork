"""Every model can prepare V2 prompts from the same source split metadata."""
import json
import unittest

from percussion_generation_v3.codec_v2 import decode_v2, encode_v2
from percussion_generation_v3.data import APPS, ROOT, load_rows, load_tokenizer
from percussion_generation_v3.inference import prompt
from percussion_generation_v3.lossless_dataset import _source_output, _source_path
from percussion_generation_v3.roles import role
from custom_extractor import extract_tweeter
from custom_extractor.models import Composition


class CrossVariantV2Tests(unittest.TestCase):
    def test_source_split_and_model_local_tokenizer(self):
        source_root = _source_output()
        self.assertEqual(source_root.resolve(),
                         (APPS / "smollm135/V3/output_augmented_from_epoch7").resolve())
        row = load_rows(source_root, "rhythm", "train")[0]
        augmentation = json.loads((APPS / "L2Ork-Tweeter/AugmentedSessions/manifest.json")
                                  .read_text(encoding="utf-8"))
        lookup = {item["augmented_filename"]: item for item in augmentation["generated"]}
        source = extract_tweeter(_source_path(row, lookup))
        conditioned = Composition(source.global_settings,
                                  [track for track in source.tracks
                                   if track["id"] != row["target_track_id"]])
        original = {"global": conditioned.global_settings,
                    "tracks": [{"role": role(track), "track": track}
                               for track in conditioned.tracks]}
        text = encode_v2(original)
        self.assertEqual(decode_v2(text), original)
        self.assertEqual(prompt(conditioned, row["target_note_count"], "lossless-v2"),
                         text + "\nRHYTHM_TARGET=" + str(row["target_note_count"]) + "\n")
        tokenizer = load_tokenizer(ROOT / "output")
        target_ids = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        self.assertEqual(target_ids, row["target_ids"])


if __name__ == "__main__":
    unittest.main()
