"""Seeded, grammar-constrained V4EA generation for local Tweeter feedback."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

from . import OUTPUT
from .model import load_model
from .pipeline import load_new_tokenizer, prompt_for_composition, verify_dataset
from .target_format import decode, end_marker_for
from .plugin_store import FeedbackStore, digest, timestamp
from percussion_generation_v3.data import AUGMENTED_SESSIONS, _experiment_family
from percussion_generation_v3.evaluation import remove_target
from percussion_generation_v3.inference import PositionConstraint, build_track, factory64_electrikick_preset
from percussion_generation_v3.lossless_dataset import _source_path
from custom_extractor import extract_tweeter
from custom_extractor.exporter import render_tweeter
from custom_extractor.extractor import extract_text
from custom_extractor.validator import validate


def sample_training_row(rows, rng):
    """Choose a source uniformly, then one of its augmented pairs uniformly."""
    by_source = defaultdict(list)
    for row in rows:
        if row.get("augmentation"):
            by_source[row["source"]].append(row)
    if not by_source:
        raise ValueError("no eligible augmented training pairs")
    source = rng.choice(sorted(by_source))
    return rng.choice(by_source[source])


def prepare_feedback_pair(row, generated, tokenizer, requested_note_count=None):
    source_path = _source_path(row, generated)
    original = extract_tweeter(source_path)
    removed = original.track(row["target_track_id"])
    condition = remove_target(original, row["target_track_id"])
    if len(condition.tracks) + 1 != len(original.tracks):
        raise ValueError("designated target was not removed")
    original_count = len(removed["notes"])
    original_prompt = prompt_for_composition(condition, original_count, "positions-v1")
    if original_prompt != row["prefix"]:
        raise ValueError("plugin prompt differs from supervised training prompt")
    original_ids = tokenizer.encode(original_prompt, add_special_tokens=False)
    if original_ids != row["prefix_ids"]:
        raise ValueError("plugin token IDs differ from supervised training tokens")
    count = original_count if requested_note_count is None else requested_note_count
    if type(count) is not int or not 1 <= count <= 63:
        raise ValueError("requested note count must be an integer from 1 to 63")
    prompt = prompt_for_composition(condition, count, "positions-v1")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return source_path, original, condition, removed, prompt, ids


def render_candidate(condition, original, removed, positions, preset):
    loop_ms = removed["timing"]["loop_ms"]
    track = build_track(condition, positions, preset, loop_ms,
                        target_track_id=removed["id"])
    track["instrument"]["username"] = "AI_A"
    candidate = deepcopy(condition)
    # The native exporter preserves baseline record order and requires a new
    # track to be appended. Its ID is the original native track slot.
    candidate.tracks.append(track)
    validate(candidate)
    native = render_tweeter(candidate)
    restored = extract_text(native)
    if (restored.global_settings != original.global_settings or
            restored.tracks != candidate.tracks or
            [item for item in restored.tracks if item["id"] != removed["id"]] != condition.tracks):
        raise ValueError("native candidate changed accompaniment or generated track")
    return track, native


def decode_sample(model, tokenizer, device, ids, count, seed, temperature=.8,
                  top_p=.95, max_new_tokens=256, cancel_event=None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    if not 0 < temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("temperature must be 0..2 and top-p must be 0..1")
    if len(ids) + max_new_tokens > 8192:
        max_new_tokens = 8192 - len(ids)
    if max_new_tokens < count + 2:
        raise ValueError("conditioning prompt leaves too little context")
    constraint = PositionConstraint(tokenizer, count)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return bool(cancel_event and cancel_event.is_set())
    started = time.perf_counter()
    # This pinned Transformers version does not accept generate(generator=...).
    # fork_rng gives each candidate an independent seed and restores the
    # process RNG afterward, so repeated requests remain reproducible.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        output = model.peft_model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=top_p, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([Cancelled()]),
            prefix_allowed_tokens_fn=lambda _batch, sequence: constraint.allowed(
                sequence[len(ids):].tolist()))
    completion = output[0, len(ids):].tolist()
    text = tokenizer.decode(completion, skip_special_tokens=False)
    ending = "\n" + end_marker_for("positions-v1")
    if ending not in text or tokenizer.eos_token_id not in completion:
        raise ValueError("sampled target did not terminate correctly")
    positions = decode(text.split(ending, 1)[0] + ending, count, "positions-v1")
    return positions, completion, time.perf_counter() - started


class GenerationService:
    def __init__(self, output_dir=OUTPUT, feedback_dir=None, cpu=False, seed=None):
        self.output_dir = Path(output_dir)
        self.store = FeedbackStore(feedback_dir)
        self.cpu = cpu
        self.rng = random.Random(seed) if seed is not None else random.SystemRandom()
        self.manifest, groups = verify_dataset(self.output_dir)
        if self.manifest["encoding"] != "positions-v1":
            raise ValueError("AI preference plugin requires V4EA positions-v1")
        self.rows = groups["train"]
        held_out = set(self.manifest["source_partitions"]["validation"] +
                       self.manifest["source_partitions"]["test"])
        held_out_families = {_experiment_family(source) for source in held_out}
        self.rows = [row for row in self.rows if _experiment_family(row["source"]) not in held_out_families]
        if not self.rows or any(row["source"] in held_out for row in self.rows):
            raise ValueError("no isolated augmented training pool")
        augmentation = json.loads((AUGMENTED_SESSIONS / "manifest.json").read_text(encoding="utf-8"))
        self.generated = {item["augmented_filename"]: item for item in augmentation["generated"]}
        self.tokenizer, _ = load_new_tokenizer(self.output_dir)
        self.source_preset = json.loads((self.output_dir / "fixed_preset.json").read_text(encoding="utf-8"))
        self.preset = factory64_electrikick_preset(self.source_preset)
        self.adapter = self.output_dir / "checkpoints/rhythm_coverage_v1/best_adapter"
        self.model = self.device = None

    def select_adapter(self, adapter):
        adapter = Path(adapter).resolve()
        if not (adapter / "embedding.safetensors").is_file():
            raise FileNotFoundError("selected adapter lacks trained embeddings")
        self.release_model()
        self.adapter = adapter

    def load(self):
        if self.model is None:
            self.model, self.device = load_model(self.output_dir, adapter=self.adapter,
                                                 cpu=self.cpu, trainable=False)
            self.model.eval()
        return self.model, self.device

    def release_model(self):
        self.model = self.device = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_group(self, temperature=.8, top_p=.95, max_attempts=30,
                       seeds=None, on_progress=None, cancel_event=None,
                       requested_note_count=None, source_row=None):
        if max_attempts < 5:
            raise ValueError("at least five attempts are required")
        if source_row is not None and source_row not in self.rows:
            raise ValueError("current composition is not an eligible training source")
        row = source_row if source_row is not None else sample_training_row(self.rows, self.rng)
        source_path, original, condition, removed, prompt, ids = prepare_feedback_pair(
            row, self.generated, self.tokenizer, requested_note_count)
        count = len(removed["notes"]) if requested_note_count is None else requested_note_count
        model, device = self.load()
        candidate_rows, sessions, seen, attempted = [], {}, set(), set()
        duplicates = invalid = 0
        last_failure = None
        seed_iter = iter(seeds) if seeds is not None else None
        for attempt in range(max_attempts):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("generation cancelled")
            if len(candidate_rows) == 5:
                break
            if on_progress:
                on_progress({"phase": "sampling", "attempt": attempt+1,
                             "max_attempts": max_attempts,
                             "accepted": len(candidate_rows), "total": 5})
            try:
                try:
                    seed = next(seed_iter) if seed_iter is not None else self.rng.randrange(0, 2**63)
                except StopIteration as exc:
                    raise ValueError("provided generation seeds were exhausted") from exc
                if seed in attempted:
                    duplicates += 1
                    continue
                attempted.add(seed)
                positions, completion, seconds = decode_sample(
                    model, self.tokenizer, device, ids, count, seed,
                    temperature, top_p, cancel_event=cancel_event)
                key = tuple(positions)
                if key in seen:
                    duplicates += 1
                    continue
                track, native = render_candidate(condition, original, removed, positions, self.preset)
            except (ValueError, RuntimeError, KeyError) as exc:
                invalid += 1
                last_failure = str(exc)
                continue
            else:
                seen.add(key)
                candidate_id = str(uuid.uuid4())
                candidate_rows.append({"candidate_id": candidate_id, "seed": seed,
                                       "positions": positions, "completion_ids": completion,
                                       "completion_text": self.tokenizer.decode(completion, skip_special_tokens=False),
                                       "native_track": track, "generation_seconds": seconds})
                sessions[candidate_id] = native
            finally:
                if on_progress:
                    on_progress({"phase": "sampling", "attempt": attempt+1,
                                 "max_attempts": max_attempts,
                                 "accepted": len(candidate_rows), "total": 5})
        if len(candidate_rows) != 5:
            detail = f"; last error: {last_failure}" if last_failure else ""
            raise RuntimeError(f"only {len(candidate_rows)}/5 valid candidates after {max_attempts} attempts "
                               f"({invalid} invalid, {duplicates} duplicate){detail}")
        group = {"group_id": str(uuid.uuid4()), "created_at": timestamp(),
                 "source": row["source"], "source_family": _experiment_family(row["source"]),
                 "augmentation": row["augmentation"], "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                 "source_file": str(source_path), "original_composition": original.to_dict(),
                 "conditioning_composition": condition.to_dict(),
                 "removed_target_track": deepcopy(removed), "target_track_id": removed["id"],
                 "original_target_note_count": len(removed["notes"]),
                 "requested_note_count": count,
                 "prompt": prompt, "prompt_ids": ids, "model_id": self.manifest["model_id"],
                 "model_revision": self.manifest["revision"],
                 "dataset_sha256": self.manifest["prepared_dataset_sha256"],
                 "codec_schema_sha256": self.manifest["codec_schema_sha256"],
                 "tokenizer_sha256": self.manifest["tokenizer_sha256"],
                 "adapter_path": str(self.adapter),
                 "adapter_sha256": digest((self.adapter / "adapter_model.safetensors").read_bytes()),
                 "embedding_sha256": digest((self.adapter / "embedding.safetensors").read_bytes()),
                 "fixed_preset": self.source_preset,
                 "fixed_preset_sha256": self.manifest["fixed_preset_sha256"],
                 "generation_preset": self.preset,
                 "generation_preset_sha256": digest(self.preset),
                 "loop_ms": removed["timing"]["loop_ms"], "pitch": self.preset["example_pitch"],
                 "sampling": {"temperature": temperature, "top_p": top_p,
                              "max_attempts": max_attempts, "attempts_used": len(attempted),
                              "duplicate_attempts": duplicates, "invalid_attempts": invalid},
                 "candidates": candidate_rows}
        return self.store.save_group(group, sessions)
