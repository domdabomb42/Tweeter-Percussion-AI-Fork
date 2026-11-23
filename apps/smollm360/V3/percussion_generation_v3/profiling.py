"""Opt-in, identical CPU/CUDA stage timing for all SmolLM V3 variants."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time


class StageProfiler:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.values = defaultdict(list)
        self.pending_cuda = []
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def stage(self, name):
        if self.cuda:
            import torch
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.pending_cuda.append((name, start, end))
        else:
            started = time.perf_counter()
            try:
                yield
            finally:
                self.values[name].append(time.perf_counter() - started)

    def report(self, model=None, examples=None, generated_tokens=None):
        import torch
        if self.pending_cuda:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.pending_cuda:
                self.values[name].append(start.elapsed_time(end) / 1000)
            self.pending_cuda.clear()
        stages = {name: {"calls": len(values), "total_seconds": sum(values),
                         "mean_seconds": statistics.mean(values)}
                  for name, values in sorted(self.values.items()) if values}
        elapsed = time.perf_counter() - self.started
        return {"model_variant": (getattr(getattr(model, "config", None), "_name_or_path", None)
                                  if model is not None else None),
                "hardware": torch.cuda.get_device_name(self.device) if self.cuda else "CPU",
                "device": str(self.device), "torch_version": torch.__version__,
                "attention_backend": getattr(getattr(model, "config", None), "_attn_implementation", None),
                "dtype": str(next(model.parameters()).dtype) if model is not None else None,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
                "elapsed_seconds": elapsed, "stages": stages,
                "examples_per_second": examples / elapsed if examples is not None and elapsed else None,
                "generated_tokens_per_second": generated_tokens / elapsed
                if generated_tokens is not None and elapsed else None}

    def save(self, path, model=None, examples=None, generated_tokens=None, extra=None):
        report = self.report(model, examples, generated_tokens)
        if extra:
            report.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def benchmark_inference(output_dir, report_path, partition="validation", cap=1,
                        counts=(5, 8), max_new_tokens=16, cpu=False):
    """Read-only, same-request baseline versus shared-prefix benchmark."""
    import torch
    from .data import load_rows, load_tokenizer, recount_prefix, verify_adapter_codec
    from .inference import SharedPrefixSession, generate_rhythm
    from .model import load_feature_model
    from custom_training import MODEL_ID, MODEL_REVISION

    root = Path(output_dir)
    rows = load_rows(root, "rhythm", partition)[:cap]
    if not rows or cap < 1 or max_new_tokens < 1:
        raise ValueError("benchmark needs at least one held-out pair and one decoding token")
    adapter = root / "checkpoints/rhythm/best_adapter"
    verify_adapter_codec(root, adapter)
    model, device = load_feature_model(adapter, cpu=cpu)
    tokenizer = load_tokenizer(root)
    results = []
    for row in rows:
        prefixes = [recount_prefix(row["prefix"], row["target_note_count"], count)
                    for count in counts]
        case = {"source": row["source"], "target_track_id": row["target_track_id"],
                "requested_counts": list(counts),
                "input_lengths": [len(tokenizer.encode(p, add_special_tokens=False)) for p in prefixes]}
        outputs = {}
        for name, use_shared in (("baseline", False), ("shared_prefix", True)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            session = SharedPrefixSession(model, tokenizer, device, prefixes) if use_shared else None
            generated = [generate_rhythm(model, tokenizer, device, prefix, max_new_tokens,
                                         shared_session=session)
                         for prefix in prefixes]
            elapsed = time.perf_counter() - started
            outputs[name] = [text for text, _ in generated]
            case[name] = {"seconds": elapsed,
                          "prefill_seconds": sum(meta["prefill_seconds"] for _, meta in generated),
                          "decode_seconds": sum(meta["decode_seconds"] for _, meta in generated),
                          "generated_tokens": sum(meta["generated_tokens"] for _, meta in generated),
                          "shared_prefix_tokens": session.length if session else 0,
                          "shared_prefills": session.prefill_count if session else 0,
                          "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                          if device.type == "cuda" else None}
        if outputs["baseline"] != outputs["shared_prefix"]:
            raise ValueError("shared-prefix generation changed model output")
        case["identical_generated_text"] = True
        results.append(case)
    report = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "device": str(device), "dtype": str(next(model.parameters()).dtype),
              "torch_version": torch.__version__,
              "attention_backend": getattr(model.config, "_attn_implementation", None),
              "partition": partition, "max_new_tokens": max_new_tokens,
              "cases": results,
              "baseline_total_seconds": sum(r["baseline"]["seconds"] for r in results),
              "shared_total_seconds": sum(r["shared_prefix"]["seconds"] for r in results)}
    report["speedup_ratio"] = (report["baseline_total_seconds"] / report["shared_total_seconds"]
                               if report["shared_total_seconds"] else None)
    path = Path(report_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report": str(path), "speedup_ratio": report["speedup_ratio"],
            "cases": len(results)}
