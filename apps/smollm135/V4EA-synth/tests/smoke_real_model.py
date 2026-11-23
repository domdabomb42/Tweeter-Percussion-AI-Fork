"""One small real-model backward pass; no training run or checkpoint writes."""
import json
from pathlib import Path
import tempfile
import gc
import sys

import torch

from transformers import AutoTokenizer

from percussion_generation_v4ea import END_MARKER, MODEL_ID, MODEL_REVISION, MODEL_ROOT, VERSION
from percussion_generation_v4ea.model import load_model
from percussion_generation_v4ea.pipeline import sha256
from percussion_generation_v4ea.trainer import full_sequence_loss, optimizer_groups


def main():
    old_path = MODEL_ROOT / "V3/output/tokenizer"
    old = AutoTokenizer.from_pretrained(old_path, local_files_only=True)
    corpus = ['["V4EA-MUSIC-INPUT-3",3,[["non_percussion","Bass",1000,[[0,42]]]]]'
              '\nPERCUSSION_MUSIC_TARGET\n',
              '["V4EA-MUSIC-TARGET-2","electrikick",1000,100,0,[[0,40],[16,42],[32,40]]]'
              '\n[END_MUSIC]']
    fresh = old.train_new_from_iterator(corpus, vocab_size=512,
                                        new_special_tokens=[END_MARKER])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tokenizer").mkdir()
        fresh.save_pretrained(root / "tokenizer")
        identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                    "pretrained_tokenizer_sha256": sha256(old_path / "tokenizer.json"),
                    "tokenizer_sha256": sha256(root / "tokenizer/tokenizer.json"),
                    "eos_token_id": fresh.eos_token_id}
        (root / "tokenizer_identity.json").write_text(json.dumps(identity), encoding="utf-8")
        cpu = "--gpu" not in sys.argv
        model, device = load_model(root, cpu=cpu, trainable=True)
        model.peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        prefix = fresh.encode(corpus[0], add_special_tokens=False)
        target = fresh.encode(corpus[1], add_special_tokens=False) + [fresh.eos_token_id]
        exclusive = next(token for token in prefix if token not in target)
        loss = full_sequence_loss(model, {"prefix_ids": prefix, "target_ids": target}, device)
        loss.backward()
        embedding_grad = model.peft_model.get_input_embeddings().weight.grad
        lora_grads = [parameter.grad for name, parameter in model.peft_model.named_parameters()
                      if "lora_" in name and parameter.requires_grad]
        optimizer = optimizer_groups(model, 1e-4, 1e-4)
        assert embedding_grad is not None and embedding_grad.abs().sum() > 0
        assert embedding_grad[exclusive].abs().sum() > 0
        assert lora_grads and any(grad is not None and grad.abs().sum() > 0 for grad in lora_grads)
        assert len(optimizer.param_groups) == 2
        result = {"model_id": MODEL_ID, "loss": float(loss),
                          "prefix_only_token_gradient": float(embedding_grad[exclusive].abs().sum()),
                          "embedding_gradient": float(embedding_grad.abs().sum()),
                          "nonzero_lora_gradients": sum(grad is not None and bool(grad.abs().sum() > 0)
                                                        for grad in lora_grads)}
        before = model.peft_model.get_input_embeddings().weight.detach().cpu().clone()
        adapter = root / "adapter"
        model.save_pretrained(adapter)
        (adapter / "experiment_metadata.json").write_text(json.dumps({
            "version": VERSION, "model_id": MODEL_ID,
            "revision": MODEL_REVISION, "tokenizer_sha256": identity["tokenizer_sha256"]}),
            encoding="utf-8")
        del model, optimizer, loss, embedding_grad, lora_grads
        gc.collect()
        reloaded, _ = load_model(root, adapter=adapter, cpu=cpu, trainable=True)
        assert torch.equal(before, reloaded.peft_model.get_input_embeddings().weight.detach().cpu())
        result["embedding_checkpoint_reload"] = True
        result["device"] = str(device)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
