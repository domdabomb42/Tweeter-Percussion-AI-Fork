"""Position-only generation and verified fixed-preset Tweeter reconstruction."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from .data import END_MARKER, V3_MARKER, load_tokenizer, parse_positions, verify_adapter_codec
from custom_extractor import extract_tweeter, export_tweeter
from custom_extractor.validator import validate
from custom_training.serializer import serialize_object

from .codec import pack_condition
from .codec_v2 import encode_v2
from .model import load_feature_model
from .roles import role

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT.parents[1]


def sparse(value):
    if isinstance(value, dict):
        return {key: sparse(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [sparse(item) for item in value]
    return value


def prompt(composition, target_note_count, encoding="compact"):
    if type(target_note_count) is not int or not 1 <= target_note_count <= 64:
        raise ValueError("target note count must be 1..64")
    if encoding == "lossless-v2":
        condition = {"global": composition.global_settings,
                     "tracks": [{"role": role(track), "track": track}
                                for track in composition.tracks]}
        return encode_v2(condition) + V3_MARKER + str(target_note_count) + "\n"
    condition = {"global": sparse(composition.global_settings),
                 "tracks": [{"role": role(track), "track": sparse(track)}
                            for track in composition.tracks]}
    if encoding in ("compact", "legacy-v1"):
        condition = pack_condition(condition)
    elif encoding != "canonical":
        raise ValueError("unknown input encoding")
    return serialize_object(condition) + V3_MARKER + str(target_note_count) + "\n"


class PositionConstraint:
    """Token-level grammar: ascending 0..63 positions, exact count, end marker, EOS."""

    def __init__(self, tokenizer, count):
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("count must be 1..64")
        self.tokenizer = tokenizer
        self.count = count
        self.eos = tokenizer.eos_token_id
        self.pieces = {(first, step): tuple(tokenizer.encode(
            ("" if first else " ") + str(step), add_special_tokens=False))
            for first in (False, True) for step in range(64)}
        self.end = tuple(tokenizer.encode("\n" + END_MARKER, add_special_tokens=False))
        if not self.end or any(not ids for ids in self.pieces.values()):
            raise ValueError("tokenizer cannot encode rhythm grammar")
        self.cache = {(): frozenset({(0, -1, (), -1, False)})}
        self.current_states = self.cache[()]

    def _options(self, completed, last):
        if completed == self.count:
            return [(self.end, -1, True)]
        remaining = self.count - completed
        return [(self.pieces[(completed == 0, step)], step, False)
                for step in range(last + 1, 65 - remaining)]

    def _advance(self, states, token):
        result = set()
        for completed, last, pending, next_step, ending in states:
            if ending and not pending:
                if token == self.eos:
                    result.add((completed, last, (), -1, True))
                continue
            options = [(pending, next_step, ending)] if pending else self._options(completed, last)
            for ids, step, end_flag in options:
                if ids and ids[0] == token:
                    tail = ids[1:]
                    if tail:
                        result.add((completed, last, tail, step, end_flag))
                    elif end_flag:
                        result.add((completed, last, (), -1, True))
                    else:
                        result.add((completed + 1, step, (), -1, False))
        return frozenset(result)

    def states(self, generated):
        key = tuple(generated)
        if key not in self.cache:
            self.cache[key] = self._advance(self.states(key[:-1]), key[-1])
        return self.cache[key]

    def allowed(self, generated):
        return self._allowed_states(self.states(generated))

    def allowed_incremental(self):
        return self._allowed_states(self.current_states)

    def advance(self, token):
        self.current_states = self._advance(self.current_states, token)

    def _allowed_states(self, states):
        allowed = set()
        for completed, last, pending, _, ending in states:
            if ending and not pending:
                allowed.add(self.eos)
            elif pending:
                allowed.add(pending[0])
            else:
                allowed.update(ids[0] for ids, _, _ in self._options(completed, last))
        if not allowed:
            raise ValueError("constrained decoder has no legal next token")
        return sorted(allowed)


class SharedPrefixSession:
    """Reuse a verified token-ID prefix between sequential generation requests.

    DynamicCache is cropped back to the common prefix after each request, so
    no note-count suffix or generated token can enter the next request.
    """

    def __init__(self, model, tokenizer, device, prefixes):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        started = time.perf_counter()
        self.ids = {prefix: tokenizer.encode(prefix, add_special_tokens=False)
                    for prefix in prefixes}
        self.tokenization_seconds = time.perf_counter() - started
        sequences = list(self.ids.values())
        common = 0
        if len(sequences) > 1:
            for values in zip(*sequences):
                if len(set(values)) != 1:
                    break
                common += 1
        # Retain one real suffix token even when all prompts are identical.
        self.length = min(common, min(map(len, sequences), default=0) - 1)
        self.enabled = self.length >= 2
        self.cache = None
        self.prefill_count = 0

    def begin(self):
        if not self.enabled:
            return None
        if self.cache is None:
            import torch
            shared = next(iter(self.ids.values()))[:self.length]
            inputs = torch.tensor([shared], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                result = self.model(input_ids=inputs,
                                    attention_mask=torch.ones_like(inputs), use_cache=True)
            cache = result.past_key_values
            del result
            if not hasattr(cache, "crop"):
                self.enabled = False
                return None
            self.cache = cache
            self.prefill_count += 1
        try:
            self.cache.crop(self.length)
        except (AttributeError, NotImplementedError, ValueError):
            self.enabled = False
            self.cache = None
            return None
        return self.cache

    def end(self, cache):
        if cache is not None:
            try:
                cache.crop(self.length)
                self.cache = cache
            except (AttributeError, NotImplementedError, ValueError):
                self.enabled = False
                self.cache = None


def generate_rhythm(model, tokenizer, device, prefix, max_new_tokens=3072,
                    constrained_count=None, cd2=False, shared_session=None):
    """Greedy cached decoding with separately synchronized prefill/decode times."""
    import torch
    started = time.perf_counter()
    ids = (shared_session.ids[prefix] if shared_session is not None
           else tokenizer.encode(prefix, add_special_tokens=False))
    input_seconds = time.perf_counter() - started
    if not ids or len(ids) >= 8192:
        raise ValueError(f"prompt length {len(ids)} exceeds model context")
    budget = min(max_new_tokens, 8192 - len(ids))
    if cd2 and constrained_count is None:
        raise ValueError("CD2 requires a requested note count")
    if cd2:
        from .cd2 import PositionConstraintCD2
        constraint = PositionConstraintCD2(tokenizer, constrained_count)
    else:
        constraint = PositionConstraint(tokenizer, constrained_count) if constrained_count else None
    inputs = torch.tensor([ids], dtype=torch.long, device=device)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    model.eval()
    sync()
    prefill_start = time.perf_counter()
    shared_cache = None
    with torch.inference_mode():
        if shared_session is not None:
            shared_cache = shared_session.begin()
        if shared_cache is None:
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True)
        else:
            suffix = inputs[:, shared_session.length:]
            output = model(input_ids=suffix, past_key_values=shared_cache,
                           attention_mask=torch.ones_like(inputs), use_cache=True)
        sync()
        prefill_seconds = time.perf_counter() - prefill_start
        decode_start = time.perf_counter()
        generated = []
        cache = output.past_key_values
        try:
            for index in range(budget):
                logits = output.logits[0, -1]
                if constraint:
                    allowed = constraint.allowed_incremental()
                    choice = allowed[int(torch.argmax(logits[allowed]).item())]
                else:
                    choice = int(torch.argmax(logits).item())
                generated.append(choice)
                if constraint:
                    constraint.advance(choice)
                if choice == tokenizer.eos_token_id:
                    break
                if index + 1 < budget:
                    next_id = torch.tensor([[choice]], dtype=torch.long, device=device)
                    output = model(input_ids=next_id, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
        finally:
            if shared_cache is not None:
                shared_session.end(cache)
        sync()
        decode_seconds = time.perf_counter() - decode_start
    ended = bool(generated and generated[-1] == tokenizer.eos_token_id)
    text = tokenizer.decode(generated[:-1] if ended else generated,
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, {"input_tokens": len(ids), "generated_tokens": len(generated),
                  "generation_budget": budget, "ended_with_eos": ended,
                  "input_processing_seconds": input_seconds,
                  "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                  "reused_prefix_tokens": shared_session.length if shared_cache is not None else 0,
                  "decode_tokens_per_second": len(generated) / decode_seconds if decode_seconds else None}


def choose_loop_ms(composition, reference_track_id=None, loop_ms=None):
    if loop_ms is not None:
        if type(loop_ms) is not int or loop_ms <= 0:
            raise ValueError("loop-ms must be a positive integer")
        return loop_ms
    if reference_track_id is not None:
        value = composition.track(reference_track_id)["timing"]["loop_ms"]
        if value is None:
            raise ValueError("reference track has no loop length")
        return value
    drum_loops = [track["timing"]["loop_ms"] for track in composition.tracks
                  if role(track) == "percussion" and track["timing"]["loop_ms"] is not None]
    melody_loops = [track["timing"]["loop_ms"] for track in composition.tracks
                    if role(track) == "non_percussion" and track["timing"]["loop_ms"] is not None]
    loops = drum_loops or melody_loops
    if not loops:
        raise ValueError("no verified loop length; supply --loop-ms or --reference-track-id")
    counts = Counter(loops)
    peak = max(counts.values())
    winners = [length for length, frequency in counts.items() if frequency == peak]
    if len(winners) != 1:
        raise ValueError("multiple equally common loop lengths; supply --loop-ms or --reference-track-id")
    return winners[0]


def factory64_electrikick_preset(preset):
    """Use output level 95 with Tweeter factory #64 pan and echo settings."""
    selected = deepcopy(preset)
    if selected.get("instrument_name") == "electrikick":
        track = selected["track"]
        track["mixing"]["output_amplitude"] = 95
        track["mixing"]["pan"] = 0.765262
        track["effects"]["echo"]["amplitude"] = 0
    return selected


def build_track(composition, positions, preset, loop_ms, target_track_id=None):
    used = {track["id"] for track in composition.tracks}
    track_id = target_track_id or next((n for n in range(1, 15) if n not in used), None)
    if track_id is None or track_id in used or not 1 <= track_id <= 14:
        raise ValueError("no available native track slot")
    if len(set(positions)) != len(positions):
        raise ValueError("one native percussion track cannot hold two notes at one step")
    track = factory64_electrikick_preset(preset)["track"]
    track["id"] = track_id
    track["timing"]["loop_ms"] = loop_ms
    track["notes"] = [{"step": p, "pitch": preset["example_pitch"]} for p in positions]
    return track


def export_rhythm(composition, positions, preset, loop_ms, output_path,
                  target_track_id=None):
    candidate = deepcopy(composition)
    track = build_track(composition, positions, preset, loop_ms, target_track_id)
    candidate.tracks.append(track)
    validate(candidate)
    export_tweeter(candidate, output_path)
    restored = extract_tweeter(output_path)
    if restored.tracks[:-1] != composition.tracks or restored.tracks[-1] != track:
        raise ValueError("Tweeter export changed source tracks or generated notes")
    return track


def generate_file(input_path, output_path, target_note_count, output_dir=ROOT / "output",
                  constrained=False, max_new_tokens=3072, cpu=False, loop_ms=None,
                  reference_track_id=None, target_track_id=None, preset_path=None,
                  cd2=False):
    if constrained and cd2:
        raise ValueError("choose ordered constrained decoding or CD2, not both")
    output_dir = Path(output_dir)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    for variant in ("smollm135", "smollm360", "smollm1b"):
        other = APPS / variant
        if other != ROOT.parent and (output_path.resolve().is_relative_to(other.resolve()) or
                                     output_dir.resolve().is_relative_to(other.resolve())):
            raise ValueError("generation output belongs to another model experiment")
    for name in ("PercussionGeneration", "PercussionGenerationV2"):
        if output_path.resolve().is_relative_to((APPS / name).resolve()):
            raise ValueError("V3 cannot write into V1/V2")
    composition = extract_tweeter(input_path)
    if not any(role(track) == "non_percussion" for track in composition.tracks):
        raise ValueError("no verified non-percussion track in input")
    chosen_loop = choose_loop_ms(composition, reference_track_id, loop_ms)
    preset = json.loads((Path(preset_path) if preset_path else output_dir / "fixed_preset.json")
                        .read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    adapter = output_dir / "checkpoints" / ("constrained" if constrained or cd2 else "rhythm") / "best_adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"V3 checkpoint missing: {adapter}")
    verify_adapter_codec(output_dir, adapter)
    model, device = load_feature_model(adapter, cpu=cpu)
    tokenizer = load_tokenizer(output_dir)
    started = time.perf_counter()
    raw, metrics = generate_rhythm(model, tokenizer, device,
                                   prompt(composition, target_note_count, manifest["encoding"]),
                                   max_new_tokens, target_note_count if constrained or cd2 else None,
                                   cd2=cd2)
    parse_start = time.perf_counter()
    result = {"input": str(input_path), "output": str(output_path),
              "requested_note_count": target_note_count,
              "decoder": "cd2" if cd2 else "constrained" if constrained else "greedy",
              "raw_output": raw, "loop_ms": chosen_loop, **metrics}
    try:
        positions = parse_positions(raw,
                                    requested_count=target_note_count if cd2 else None,
                                    allow_unordered=cd2)
        export_positions = sorted(positions) if cd2 else positions
        result["generated_positions"] = positions
        result["export_positions"] = export_positions
        result["generated_note_count"] = len(positions)
        result["absolute_count_error"] = abs(len(positions) - target_note_count)
        result["exact_count"] = len(positions) == target_note_count
        result["parse_valid"] = True
        result["parse_seconds"] = time.perf_counter() - parse_start
        export_start = time.perf_counter()
        track = export_rhythm(composition, export_positions, preset, chosen_loop,
                              output_path, target_track_id)
        result["export_seconds"] = time.perf_counter() - export_start
        result["export_valid"] = True
        result["track_id"] = track["id"]
        result["instrument_name"] = track["instrument"]["name"]
    except (ValueError, TypeError, OSError) as error:
        result.setdefault("parse_valid", False)
        result["error"] = str(error)
    result["total_seconds"] = time.perf_counter() - started + metrics["input_processing_seconds"]
    diagnostics = output_dir / "generated" / (output_path.stem + ".diagnostics.json")
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["diagnostics"] = str(diagnostics)
    return result
