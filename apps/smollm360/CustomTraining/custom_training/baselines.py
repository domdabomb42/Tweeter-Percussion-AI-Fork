"""Train-only frequency/median predictors for the same held-out targets."""
from collections import Counter, defaultdict
from statistics import median
import json
from .prepare import load_prepared
from .serializer import leaf_spans


def feature_name(path):
    if not path or path[0] != "tracks":
        return None
    if "notes" in path:
        if path[-1] in ("step", "pitch"):
            return "note." + path[-1]
        return None
    if any(section in path for section in ("synth", "effects", "timing")):
        if path[-1] in ("carrier_envelope", "modulator_envelope") or any(isinstance(item, int) for item in path[3:]):
            return None
        return ".".join(str(x) for x in path[2:])
    return None


def feature_instances(rows):
    for row in rows:
        text, spans = leaf_spans(json.loads(row["text"]))
        for span in spans:
            name = feature_name(span["path"])
            value = span["value"]
            if name and value is not None and type(value) in (int, float, str):
                yield {"name": name, "value": value, "prefix": text[:span["start"]],
                       "source": row["source"], "segment": row["segment"],
                       "path": span["path"]}


def train_statistics(dataset_dir):
    values = defaultdict(list)
    for instance in feature_instances(load_prepared(dataset_dir, "train")):
        values[instance["name"]].append(instance["value"])
    result = {}
    for name, observed in values.items():
        if type(observed[0]) is str:
            counts = Counter(observed)
            prediction = min(((-n, str(value), value) for value, n in counts.items()))[2]
            result[name] = {"type": "categorical", "prediction": prediction,
                            "count": len(observed), "distinct": len(counts)}
        else:
            result[name] = {"type": "numeric", "prediction": median(observed),
                            "count": len(observed), "distinct": len(set(observed))}
    return result
