"""Checkpointable shuffle without replacement over the complete training pool."""
import random

SAMPLER_VERSION = "full-pool-shuffle-v1"


def new_sampler(pool_size, seed):
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("training pool must be nonempty")
    rng = random.Random(seed)
    order = list(range(pool_size))
    rng.shuffle(order)
    return {"version": SAMPLER_VERSION, "pool_size": pool_size, "seed": seed,
            "order": order, "cursor": 0, "pass_index": 0,
            "rng_state": rng.getstate()}


def validate_sampler(state, pool_size, seed):
    if (not isinstance(state, dict) or state.get("version") != SAMPLER_VERSION or
            state.get("pool_size") != pool_size or state.get("seed") != seed or
            type(state.get("cursor")) is not int or
            not 0 <= state["cursor"] <= pool_size or
            type(state.get("pass_index")) is not int or state["pass_index"] < 0 or
            sorted(state.get("order", [])) != list(range(pool_size))):
        raise ValueError("checkpoint sampler is incompatible with training pool")
    rng = random.Random()
    rng.setstate(state["rng_state"])


def draw(state, count):
    """Advance the saved permutation, reshuffling only after exhaustion."""
    validate_sampler(state, state["pool_size"], state["seed"])
    if type(count) is not int or count < 0:
        raise ValueError("sample count must be nonnegative")
    rng = random.Random()
    rng.setstate(state["rng_state"])
    selected = []
    while len(selected) < count:
        if state["cursor"] == state["pool_size"]:
            state["order"] = list(range(state["pool_size"]))
            rng.shuffle(state["order"])
            state["cursor"] = 0
            state["pass_index"] += 1
        take = min(count - len(selected), state["pool_size"] - state["cursor"])
        selected.extend(state["order"][state["cursor"]:state["cursor"] + take])
        state["cursor"] += take
    state["rng_state"] = rng.getstate()
    return selected
