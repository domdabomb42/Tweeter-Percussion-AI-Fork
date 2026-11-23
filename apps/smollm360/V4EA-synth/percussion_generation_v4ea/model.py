"""Fresh LoRA plus trainable remapped embeddings for a newly trained tokenizer."""
from pathlib import Path

from . import MODEL_ID, MODEL_REVISION, MODEL_ROOT, VERSION
from .pipeline import load_new_tokenizer, sha256


class FullTrackModel:
    """Delegate the causal model API and save both LoRA and the new vocabulary."""
    def __init__(self, peft_model):
        self.peft_model = peft_model

    @property
    def config(self):
        return self.peft_model.config

    def __call__(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)

    def parameters(self):
        return self.peft_model.parameters()

    def train(self, mode=True):
        self.peft_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save_pretrained(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        self.peft_model.save_pretrained(path, save_embedding_layers=False)
        weight = self.peft_model.get_input_embeddings().weight.detach().cpu().contiguous()
        save_file({"embed_tokens": weight}, str(path / "embedding.safetensors"))


def _remap_embeddings(base, old_tokenizer, new_tokenizer):
    """Reuse embeddings for shared token strings; compose new ones from old IDs."""
    import torch
    old_weight = base.get_input_embeddings().weight.detach().cpu().clone()
    old_vocab = old_tokenizer.get_vocab()
    base.resize_token_embeddings(len(new_tokenizer), mean_resizing=False)
    new_weight = base.get_input_embeddings().weight
    shared = 0
    with torch.no_grad():
        for index in range(len(new_tokenizer)):
            token = new_tokenizer.convert_ids_to_tokens(index)
            if token in old_vocab:
                value = old_weight[old_vocab[token]]
                shared += 1
            else:
                text = new_tokenizer.convert_tokens_to_string([token])
                pieces = old_tokenizer.encode(text, add_special_tokens=False)
                value = old_weight[pieces].float().mean(0).to(old_weight.dtype) if pieces else old_weight.float().mean(0).to(old_weight.dtype)
            new_weight[index].copy_(value.to(new_weight.device, dtype=new_weight.dtype))
    base.tie_weights()
    return shared


def load_model(output_dir, adapter=None, cpu=False, trainable=True):
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file
    from custom_training.train import load_model as load_base

    tokenizer, identity = load_new_tokenizer(output_dir)
    old_tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "V3/output/tokenizer",
                                                  local_files_only=True)
    if sha256(MODEL_ROOT / "V3/output/tokenizer/tokenizer.json") != identity.get("pretrained_tokenizer_sha256"):
        raise ValueError("pretrained tokenizer template changed since V4EA training")
    base, device, _ = load_base(cpu=cpu)
    shared = _remap_embeddings(base, old_tokenizer, tokenizer)
    if adapter is None:
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                          lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        peft_model = get_peft_model(base, lora)
    else:
        path = Path(adapter)
        import json
        metadata_path = path / "experiment_metadata.json"
        if metadata_path.exists():
            info = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif (path.parent / "metadata.json").exists():
            info = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))["config"]
        else:
            raise ValueError("adapter lacks musical experiment identity metadata")
        manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
        if (info.get("version"), info.get("model_id"), info.get("revision"),
                info.get("tokenizer_sha256")) != (
                manifest["version"], MODEL_ID, MODEL_REVISION, identity["tokenizer_sha256"]):
            raise ValueError("adapter belongs to another musical experiment or tokenizer")
        if not (path / "embedding.safetensors").is_file():
            raise FileNotFoundError("V4EA checkpoint lacks its trained embeddings")
        peft_model = PeftModel.from_pretrained(base, path, is_trainable=trainable)
        saved = load_file(str(path / "embedding.safetensors"), device="cpu")["embed_tokens"]
        if tuple(saved.shape) != tuple(peft_model.get_input_embeddings().weight.shape):
            raise ValueError("checkpoint embedding shape differs from trained tokenizer")
        with torch.no_grad():
            peft_model.get_input_embeddings().weight.copy_(saved)
    peft_model.get_input_embeddings().weight.requires_grad_(trainable)
    if peft_model.get_input_embeddings().weight.data_ptr() != peft_model.get_output_embeddings().weight.data_ptr():
        raise ValueError("SmolLM2 input/output embeddings are no longer tied")
    peft_model.to(device)
    peft_model.config.use_cache = not trainable
    model = FullTrackModel(peft_model)
    model.shared_pretrained_tokens = shared
    model.tokenizer_sha256 = identity["tokenizer_sha256"]
    return model, device
