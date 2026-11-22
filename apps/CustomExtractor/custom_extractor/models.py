"""The public model uses ordinary nested dictionaries for convenient editing."""
from dataclasses import dataclass, field
from copy import deepcopy

FORMAT = "CustomExtractor"
VERSION = 1


@dataclass
class Composition:
    global_settings: dict = field(default_factory=lambda: {
        "tracker_steps": 64,
        "metadata": {"session_title": None},
    })
    tracks: list[dict] = field(default_factory=list)
    native_text: str = ""

    def to_dict(self):
        return deepcopy({"format": FORMAT, "version": VERSION,
                         "global": self.global_settings, "tracks": self.tracks,
                         "native": {"format": "pd-coll-session", "text": self.native_text}})

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"format", "version", "global", "tracks", "native"}:
            raise ValueError("composition requires exactly format, version, global, tracks, native")
        if data["format"] != FORMAT or type(data["version"]) is not int or data["version"] != VERSION:
            raise ValueError("unsupported CustomExtractor format/version")
        native = data["native"]
        if not isinstance(native, dict) or set(native) != {"format", "text"} or native["format"] != "pd-coll-session" or not isinstance(native["text"], str):
            raise ValueError("native must contain pd-coll-session format and preservation text")
        return cls(deepcopy(data["global"]), deepcopy(data["tracks"]), native["text"])

    def track(self, track_id):
        if type(track_id) is not int:
            raise ValueError(f"track id must be integer, got {track_id!r}")
        for track in self.tracks:
            if track["id"] == track_id:
                return track
        raise ValueError(f"track {track_id}: not present")
