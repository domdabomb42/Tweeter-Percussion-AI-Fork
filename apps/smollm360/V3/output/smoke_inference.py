"""Read-only model API smoke test; never trains or saves an adapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from percussion_generation_v3.data import FEATURE_ADAPTER, ROOT, load_rows, load_tokenizer, parse_positions, recount_prefix
from percussion_generation_v3.inference import generate_rhythm
from percussion_generation_v3.model import load_feature_model

row = load_rows(ROOT / "output", "rhythm", "test")[-1]
model, device = load_feature_model(FEATURE_ADAPTER)
tokenizer = load_tokenizer(ROOT / "output")
raw, metrics = generate_rhythm(model, tokenizer, device, row["prefix"], max_new_tokens=8)
print({"raw_prefix": raw[:80], "input_tokens": metrics["input_tokens"],
       "generated_tokens": metrics["generated_tokens"], "device": str(device),
       "prefill_seconds": metrics["prefill_seconds"],
       "decode_seconds": metrics["decode_seconds"]})
one_prefix = recount_prefix(row["prefix"], row["target_note_count"], 1)
raw, metrics = generate_rhythm(model, tokenizer, device, one_prefix,
                               max_new_tokens=32, constrained_count=1)
assert len(parse_positions(raw, requested_count=1)) == 1
print({"constrained_raw": raw, "generated_tokens": metrics["generated_tokens"],
       "decode_seconds": metrics["decode_seconds"]})
