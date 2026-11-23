# V3 SmolLM consistency and efficiency

The three V3 Python package trees are byte-for-byte identical. Their
`CustomTraining/custom_training` packages are also identical except for
`__init__.py`, which pins each model's existing Hugging Face ID and revision.
`V3/run_overnight.py` is identical. Run `python apps/test_smollm_consistency.py`
from the repository root to check this after any edit.

## Audit and preserved historical differences

Before standardization, V3's codec, CD2 decoder, roles, package entry points,
and overnight runner matched. The 135M tree alone had lossless V2 preparation,
epoch-7 rotating training, and full step checkpoints. The 360M legacy
preparation added shifted training sessions; the 1.7B legacy preparation did
not. Their historical datasets and adapters are still separate artifacts and
have not been rewritten. The source trees now expose the same capabilities;
the saved manifest chooses the input encoding and validates the adapter.

`CustomTraining` had the same training, preparation, loader, serializer, and
baseline code. Evaluation and report files differed only by hard-coded model
labels, and the 360M CLI had a different description. Labels now derive from
each model's pinned ID. Model-specific V3 test fixtures remain different
because the saved checkpoints and augmented datasets differ; the new shared
regression tests are mirrored across all three test folders.

| Variant | Pinned model ID | Pinned revision |
|---|---|---|
| 135M | HuggingFaceTB/SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| 360M | HuggingFaceTB/SmolLM2-360M | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` |
| 1.7B | HuggingFaceTB/SmolLM2-1.7B | `effd688a12921b4cc83e3312b6feb579f70f9c71` |

The existing 135M epoch-7 adapter remains a special, verified warm start.
360M may warm start from its model-local V3 best adapter; 1.7B has a feature
adapter but no saved V3 best adapter here. These starting weights are a
historical comparability limit. No training was run during this change.

## Implemented optimizations

**Evaluation prefill reuse.** `SharedPrefixSession` tokenizes the requested
prompts once and finds their longest common prefix using actual tokenizer IDs.
It prefills that prefix once. Each request processes its own suffix and starts
with a cache cropped back to the common prefix. If the model's cache cannot
be cropped, generation falls back to the original complete prefill. The
uncached path remains available with `evaluate --no-shared-prefix-cache`.
The per-request constraint state is new, so greedy, ordered, and CD2 requests
cannot contaminate each other.

**Incremental constraints.** Ordered and CD2 decoders advance their current
grammar state one selected token at a time. Their original `allowed(generated)`
API remains for compatibility and tests.

**Training synchronization.** Detached loss scalars are copied from GPU to
CPU once per gradient-accumulation group instead of once per example. Each
scalar is still accumulated in original order for reporting. Nonfinite losses
are detected before gradient clipping and the optimizer update; detection can
occur later within that group. The target-only objective, detached-prefix KV
strategy, gradient accumulation, AdamW, scheduler, and checkpoint boundaries
are unchanged. Opt-in `train --profile` records prefix forward, target
forward, backward, optimizer step, optimizer-state transfer, and validation
timings in JSON at each epoch boundary. `evaluate --profile` records inference
timings, throughput, device, dtype, attention backend, and peak GPU memory.

The installed 135M model reported **SDPA** attention with the existing loader,
so no attention backend change was made. Flash Attention was not added as a
dependency. Static caches, `torch.compile`, CUDA graphs, and GPU-resident Adam
moments remain unselected because the current dynamic decoder and small GPU
need separate compatibility and memory measurements.

## Measured read-only benchmark

The saved 135M historical adapter was tested on one validation composition
with two requests (5 and 8 notes), 6,602 input tokens per request, and at most
four new tokens. Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, BF16, PyTorch
2.5.1+cu121, SDPA. Generated text matched exactly.

| Measurement | Complete prefill | Shared prefix |
|---|---:|---:|
| Total for two requests | 52.56 s | 24.16 s |
| Prefill time | 50.57 s | 22.55 s |
| Peak allocated GPU memory | 5,829,083,136 B | 5,825,886,208 B |

This one-case result is **2.18× faster** overall. It is not a full evaluation
or a training-speed measurement. The JSON result is
`smollm135/V3/benchmarks/v3_135m_legacy_gpu.json`. No 360M or 1.7B runtime
benchmark is claimed on the available 6 GB GPU. To benchmark any variant
without training, run from its `V3` folder:

```powershell
python -m percussion_generation_v3 --output-dir .\output benchmark-inference --report .\benchmarks\v3_inference.json --cap 1 --counts 5,8 --max-new-tokens 4
```

The report path must be fresh; the benchmark refuses to overwrite it.

## Lossless V2 preparation for all three variants

The two commands below use the original sessions and the 135M verified
source/split metadata, but each variant re-tokenizes with its own pinned local
tokenizer. They write fresh model-local directories and preserve all historical
outputs. The original 67 training pairs, 2,621 shifted training pairs,
20 fixed validation examples, and 5 test examples are inherited by source.
The original beat-sequence train/validation family overlap remains documented.

```powershell
cd .\apps\smollm360\V3
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 prepare --encoding lossless-v2
cd ..\..\smollm1b\V3
python -m percussion_generation_v3 --output-dir .\output_lossless_v2 prepare --encoding lossless-v2
```

The 135M and 1.7B V2 datasets already exist locally; the 1.7B dataset has
2,713 examples and its saved fingerprint verifies. The 360M full preparation
job has not been run here; its command and source/tokenizer smoke test pass.
No new adapters were trained. Historical inference keeps the legacy codec,
while V2 inference requires an adapter explicitly fine-tuned for V2.
