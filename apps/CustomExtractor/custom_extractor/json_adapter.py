"""Deterministic JSON: ASCII escapes retain legacy non-UTF-8 Pd symbol bytes."""
import json
from pathlib import Path
from .models import Composition
from .validator import validate
from .exporter import atomic_write


def dumps(composition):
    validate(composition)
    return json.dumps(composition.to_dict(), ensure_ascii=True, allow_nan=False, indent=2) + "\n"


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    composition = Composition.from_dict(json.loads(text, object_pairs_hook=_unique, parse_constant=bad_constant))
    validate(composition)
    return composition


def save_json(composition, path):
    atomic_write(path, dumps(composition).encode("utf-8"))
    return Path(path)


def load_json(path):
    return loads(Path(path).read_text(encoding="utf-8-sig"))
