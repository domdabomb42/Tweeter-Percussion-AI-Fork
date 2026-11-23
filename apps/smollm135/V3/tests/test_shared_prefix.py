"""Shared-prefix evaluation must reproduce uncached generation exactly."""
import unittest

from percussion_generation_v3.data import ROOT, load_tokenizer
from percussion_generation_v3.inference import SharedPrefixSession, generate_rhythm


class SharedPrefixTests(unittest.TestCase):
    def test_token_ids_and_generation_match_uncached(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        tokenizer = load_tokenizer(ROOT / "output")
        config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32,
                             intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=4,
                             max_position_embeddings=256)
        config._attn_implementation = "eager"
        torch.manual_seed(11)
        model = LlamaForCausalLM(config).eval()
        prefixes = ['{"g":1,"x":[1,2,3]}\nRHYTHM_TARGET=' + str(count) + '\n'
                    for count in (5, 8)]
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            model.to(device)
            for mode in ("greedy", "ordered", "cd2"):
                with self.subTest(device=str(device), mode=mode):
                    expected = [generate_rhythm(model, tokenizer, device, prefix,
                                                max_new_tokens=6,
                                                constrained_count=count if mode != "greedy" else None,
                                                cd2=mode == "cd2")
                                for prefix, count in zip(prefixes, (5, 8))]
                    session = SharedPrefixSession(model, tokenizer, device, prefixes)
                    self.assertGreater(session.length, 2)
                    observed = [generate_rhythm(model, tokenizer, device, prefix,
                                                max_new_tokens=6,
                                                constrained_count=count if mode != "greedy" else None,
                                                cd2=mode == "cd2", shared_session=session)
                                for prefix, count in zip(prefixes, (5, 8))]
                    self.assertEqual([x[0] for x in observed], [x[0] for x in expected])
                    self.assertEqual([x[1]["generated_tokens"] for x in observed],
                                     [x[1]["generated_tokens"] for x in expected])
                    self.assertEqual(session.prefill_count, 1)
                    self.assertEqual(session.cache.get_seq_length(), session.length)


if __name__ == "__main__":
    unittest.main()
