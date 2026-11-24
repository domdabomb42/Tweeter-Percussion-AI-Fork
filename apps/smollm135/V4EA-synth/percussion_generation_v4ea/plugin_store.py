"""Immutable generation groups and append-only, durable rating events.

Raw feedback is deliberately independent of datasets and training checkpoints.
Every group and rating is an individual fsynced file published with os.replace.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid


def default_feedback_dir():
    root = os.environ.get("LOCALAPPDATA")
    return Path(root) / "TweeterAIPercussion" / "feedback" if root else Path.home() / ".local" / "share" / "TweeterAIPercussion" / "feedback"


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, root=None):
        self.root = Path(root or default_feedback_dir()).resolve()
        # A misplaced feedback directory would be lost with model output.
        repo = Path(__file__).resolve().parents[4]
        if self.root.is_relative_to(repo):
            raise ValueError("feedback directory must be outside the Git working tree")
        self.groups = self.root / "groups"
        self.ratings = self.root / "ratings"
        self.groups.mkdir(parents=True, exist_ok=True)
        self.ratings.mkdir(parents=True, exist_ok=True)

    def save_group(self, group, sessions):
        group_id = group["group_id"]
        uuid.UUID(group_id)
        if len(group["candidates"]) != 5 or len({c["candidate_id"] for c in group["candidates"]}) != 5:
            raise ValueError("generation group requires five distinct candidates")
        directory = self.groups / group_id
        directory.mkdir(exist_ok=False)
        try:
            session_hashes = {}
            for candidate in group["candidates"]:
                key = candidate["candidate_id"]
                content = sessions[key]
                raw = content.encode("utf-8", errors="surrogateescape")
                atomic_bytes(directory / f"{key}.txt", raw)
                session_hashes[key] = digest(raw)
            record = {**group, "native_session_sha256": session_hashes}
            atomic_bytes(directory / "group.json", canonical(record) + b"\n")
            return record
        except Exception:
            # A directory without group.json is never a published group.
            shutil.rmtree(directory)
            raise

    def get_group(self, group_id):
        uuid.UUID(group_id)
        record = json.loads((self.groups / group_id / "group.json").read_text(encoding="utf-8"))
        for candidate_id, expected in record["native_session_sha256"].items():
            if digest((self.groups / group_id / f"{candidate_id}.txt").read_bytes()) != expected:
                raise ValueError(f"native candidate file changed: {group_id}/{candidate_id}")
        return record

    def all_groups(self):
        for file in sorted(self.groups.glob("*/group.json")):
            yield self.get_group(file.parent.name)

    def events(self, group_id=None):
        if group_id is not None:
            uuid.UUID(group_id)
        directories = [self.ratings / group_id] if group_id else sorted(self.ratings.iterdir())
        for directory in directories:
            if directory.is_dir():
                for file in sorted(directory.glob("*.json")):
                    yield json.loads(file.read_text(encoding="utf-8"))

    def current_ratings(self, group_id):
        current = {}
        for event in self.events(group_id):
            previous = current.get(event["candidate_id"])
            if previous is None or (event["revision"], event["submitted_at"]) > (previous["revision"], previous["submitted_at"]):
                current[event["candidate_id"]] = event
        return current

    def submit(self, group_id, candidate_id, rating, submission_id=None):
        if type(rating) is not int or not 1 <= rating <= 10:
            raise ValueError("rating must be an integer 1..10")
        group = self.get_group(group_id)
        if candidate_id not in {c["candidate_id"] for c in group["candidates"]}:
            raise ValueError("candidate does not belong to this generation group")
        submission_id = submission_id or str(uuid.uuid4())
        uuid.UUID(submission_id)
        directory = self.ratings / group_id
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{submission_id}.json"
        if file.exists():
            saved = json.loads(file.read_text(encoding="utf-8"))
            if saved["candidate_id"] != candidate_id or saved["rating"] != rating:
                raise ValueError("submission ID reused for a different rating")
            return saved
        current = self.current_ratings(group_id)
        event = {"group_id": group_id, "candidate_id": candidate_id, "rating": rating,
                 "submission_id": submission_id, "submitted_at": timestamp(),
                 "revision": current.get(candidate_id, {}).get("revision", 0) + 1}
        atomic_bytes(file, canonical(event) + b"\n")
        return event

    def statistics(self, target=80, previous_snapshot_events=0):
        groups = list(self.all_groups())
        ratings = {group["group_id"]: self.current_ratings(group["group_id"]) for group in groups}
        completed = sum(len(value) == 5 for value in ratings.values())
        raw_events = list(self.events())
        sources = {group["source"] for group in groups}
        pairs = sum(sum(1 for i, a in enumerate(items) for b in items[i+1:]
                        if abs(a["rating"] - b["rating"]) >= 2)
                    for items in ([event for event in current.values()] for current in ratings.values()))
        counts = Counter(event["rating"] for current in ratings.values() for event in current.values())
        attempts = sum(group.get("sampling", {}).get("attempts_used", 0) for group in groups)
        duplicates = sum(group.get("sampling", {}).get("duplicate_attempts", 0) for group in groups)
        return {"completed_groups": completed, "incomplete_groups": len(groups) - completed,
                "groups": len(groups), "submitted_ratings": sum(map(len, ratings.values())),
                "rating_events": len(raw_events), "new_rating_events_since_snapshot": max(0, len(raw_events)-previous_snapshot_events),
                "unique_original_sources": len(sources), "eligible_pairs_min_difference_2": pairs,
                "rating_distribution": dict(sorted(counts.items())),
                "duplicate_generation_rate": duplicates / attempts if attempts else None,
                "collection_target": target,
                "collection_progress": min(1.0, completed / target) if target else None}

    def incomplete_groups(self):
        return [{"group_id": group["group_id"], "source": group["source"],
                 "rated": len(self.current_ratings(group["group_id"])),
                 "candidate_ids": [item["candidate_id"] for item in group["candidates"]]}
                for group in self.all_groups() if len(self.current_ratings(group["group_id"])) < 5]

    def backup(self, destination):
        destination = Path(destination).resolve()
        if destination.exists() or destination.is_relative_to(self.root):
            raise ValueError("backup destination must be new and outside feedback store")
        temporary = destination.with_name(".pending_" + uuid.uuid4().hex)
        try:
            shutil.copytree(self.root, temporary)
            report = {str(path.relative_to(self.root)): digest(path.read_bytes())
                      for path in self.root.rglob("*") if path.is_file()}
            for relative, expected in report.items():
                if digest((temporary / relative).read_bytes()) != expected:
                    raise ValueError(f"backup verification failed: {relative}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return {"destination": str(destination), "files": len(report), "sha256": digest(report)}

    def export(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        archive = {"format": "TweeterAIPreferences-v1", "groups": list(self.all_groups()),
                   "rating_events": list(self.events())}
        atomic_bytes(destination, canonical(archive) + b"\n")
        return {"destination": str(destination), "groups": len(archive["groups"]),
                "rating_events": len(archive["rating_events"]), "sha256": digest(destination.read_bytes())}
