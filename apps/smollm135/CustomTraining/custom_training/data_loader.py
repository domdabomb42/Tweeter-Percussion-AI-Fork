"""Read original sessions and group related work before tokenization."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import sys

_EXTRACTOR = Path(__file__).resolve().parents[3] / "CustomExtractor"
if str(_EXTRACTOR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTOR))
from custom_extractor import extract_tweeter


@dataclass
class Session:
    path: Path
    composition: object
    family: str
    source_sha256: str
    note_signature: str

    @property
    def name(self):
        return self.path.name


def family_name(filename):
    name = filename.lower().removesuffix("-session-coll.txt")
    if name.startswith("0s-and-1s-apart-"):
        return "0s-and-1s-apart"
    if name.startswith(("1st-beat-", "2nd-beat-", "3rd-beat-", "4th-beat-")):
        return "beat-sequence"
    if name.startswith("canada-"):
        return "canada"
    if name.startswith("grapevine-"):
        return "grapevine"
    if name.startswith(("interstellar-", "insterstellar-")):
        return "interstellar"
    if name.startswith("jamminlicious-"):
        return "jamminlicious"
    if name.startswith("l2ork-001."):
        return "l2ork-001"
    if name.startswith("mus3314+l2ork-ie-"):
        return "mus3314-ie"
    if name.startswith("mus3314-spring-2025-") or name.startswith("spring-2025-"):
        return "spring-2025"
    for prefix in ("new-era-", "raye-"):
        if name.startswith(prefix):
            return prefix.rstrip("-")
    return name


def _note_signature(composition):
    material = [(t["id"], t["timing"]["loop_ms"], t["notes"]) for t in composition.tracks]
    return hashlib.sha256(repr(material).encode()).hexdigest()


def load_sessions(directory):
    paths = sorted(Path(directory).glob("*session*.txt"))
    if not paths:
        raise ValueError(f"no native Tweeter session files in {directory}")
    result = []
    for path in paths:
        composition = extract_tweeter(path)
        result.append(Session(path, composition, family_name(path.name),
                              hashlib.sha256(path.read_bytes()).hexdigest(),
                              _note_signature(composition)))
    return result


def _note_material(session):
    return {(t["id"], n["step"], n["pitch"]) for t in session.composition.tracks for n in t["notes"]}


def connected_groups(sessions):
    """Family variants, exact arrangements and substantial near-overlap stay together."""
    parent = list(range(len(sessions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    material = [_note_material(s) for s in sessions]
    overlap_edges = []
    for i, a in enumerate(sessions):
        for j in range(i + 1, len(sessions)):
            b = sessions[j]
            intersection = len(material[i] & material[j])
            union_size = len(material[i] | material[j])
            near = intersection >= 20 and union_size and intersection / union_size >= 0.65
            if a.family == b.family or a.note_signature == b.note_signature or near:
                union(i, j)
                overlap_edges.append({"a": a.name, "b": b.name,
                                      "reason": "family" if a.family == b.family else "exact_arrangement" if a.note_signature == b.note_signature else "note_jaccard>=0.65",
                                      "shared_note_events": intersection})
    groups = defaultdict(list)
    for i, session in enumerate(sessions):
        groups[find(i)].append(session)
    return sorted(groups.values(), key=lambda g: min(s.name for s in g)), overlap_edges


def split_sessions(sessions, seed=415):
    groups, edges = connected_groups(sessions)
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    total = len(sessions)
    targets = [total - 2 * round(total / 10), round(total / 10), round(total / 10)]
    # Dynamic program balances the counts exactly where grouping permits it.
    choices = {(0, 0): ()}
    processed = 0
    for group_index in order:
        size = len(groups[group_index])
        updated = {}
        for (validation, test), assignment in choices.items():
            for bucket in range(3):
                v = validation + (size if bucket == 1 else 0)
                t = test + (size if bucket == 2 else 0)
                if v > targets[1] + 3 or t > targets[2] + 3:
                    continue
                updated.setdefault((v, t), assignment + (bucket,))
        if not updated:
            raise ValueError("cannot split session families without leakage")
        choices = updated
        processed += size
    key = min(choices, key=lambda vt: (sum((actual - target) ** 2 for actual, target in zip((total-vt[0]-vt[1], *vt), targets)),
                                       abs(vt[0]-vt[1]), vt))
    split = {name: [] for name in ("train", "validation", "test")}
    for group_index, bucket in zip(order, choices[key]):
        split[("train", "validation", "test")[bucket]].extend(groups[group_index])
    for part in split.values():
        part.sort(key=lambda s: s.name)
    signatures = [{s.note_signature for s in part} for part in split.values()]
    if any(signatures[i] & signatures[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("duplicate arrangements crossed a split")
    manifest = {"seed": seed, "target_counts": dict(zip(split, targets)),
                "actual_counts": {key: len(value) for key, value in split.items()},
                "groups": [[s.name for s in group] for group in groups],
                "overlap_edges": edges,
                "partitions": {key: [{"name": s.name, "family": s.family,
                                       "sha256": s.source_sha256, "note_signature": s.note_signature}
                                      for s in value] for key, value in split.items()}}
    return split, manifest


def dataset_summary(sessions):
    from custom_extractor.parameter_registry import get_path
    from statistics import median
    waveform_pairs = Counter()
    distributions = defaultdict(list)
    file_rows = []
    numeric = ("synth.harmonicity", "synth.mod_amplitude",
               "effects.reverb.amplitude", "effects.reverb.type",
               "effects.echo.amplitude", "effects.echo.feedback", "effects.echo.delay_ms",
               "effects.flanger.amplitude", "effects.flanger.frequency_hz",
               "effects.flanger.mod_amplitude", "effects.flanger.delay_ms",
               "effects.overdrive.level")
    for session in sessions:
        tracks = session.composition.tracks
        loops = [t["timing"]["loop_ms"] for t in tracks if t["timing"]["loop_ms"] is not None]
        file_rows.append({"file": session.name, "tracks": len(tracks),
                          "notes": sum(len(t["notes"]) for t in tracks),
                          "max_track_loop_ms": max(loops, default=None),
                          "distinct_track_loop_ms": sorted(set(loops)),
                          "family": session.family})
        for track in tracks:
            pair = (track["synth"]["carrier"], track["synth"]["modulator"])
            if None not in pair:
                waveform_pairs["/".join(pair)] += 1
            for path in numeric:
                value = get_path(track, path)
                if value is not None:
                    distributions[path].append(value)
    return {"sessions": len(sessions), "tracks": sum(r["tracks"] for r in file_rows),
            "notes": sum(r["notes"] for r in file_rows),
            "duration_note": "No recorded end time; max_track_loop_ms is a cycle length proxy, not total playback duration.",
            "files": file_rows, "waveform_combinations": dict(sorted(waveform_pairs.items())),
            "distinct_waveform_combinations": len(waveform_pairs),
            "numeric_distributions": {path: {"count": len(values), "distinct": len(set(values)),
                                             "min": min(values), "median": median(values), "max": max(values)}
                                      for path, values in distributions.items()}}
