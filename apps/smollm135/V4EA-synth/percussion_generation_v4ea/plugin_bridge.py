"""Local-only FUDI/TCP control bridge; model work never runs in Pd's GUI thread."""
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time
import uuid

from . import OUTPUT
from .plugin_generation import GenerationService
from .plugin_preference import make_snapshot, precompute_reference
from .plugin_store import FeedbackStore, canonical, timestamp


def fudi_symbol(value):
    return str(value).replace("\\", "/").replace(" ", "\\ ").replace(";", "\\;").replace(",", "\\,")


def _save_active_run(root, run):
    path = root / "active_round.json"
    temporary = root / f".active_round_{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical({"run_dir": str(run.resolve())}) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_active_run(root):
    path = root / "active_round.json"
    if not path.is_file():
        return None
    run = Path(json.loads(path.read_text(encoding="utf-8"))["run_dir"]).resolve()
    if not run.is_relative_to(root.resolve()) or not (run / "snapshot.json").is_file():
        raise ValueError("active DPO round pointer is invalid")
    return run


class Bridge:
    def __init__(self, output_dir=OUTPUT, feedback_dir=None, dpo_root=None, cpu=False):
        self.service = GenerationService(output_dir, feedback_dir, cpu)
        self.store = self.service.store
        self.dpo_root = Path(dpo_root or (Path(output_dir).parent / "DPOPreferenceRuns"))
        self.dpo_root.mkdir(parents=True, exist_ok=True)
        self.group = None
        self.active_index = 0
        self.pending_index = None
        self.draft_rating = None
        self.warned_unsaved = False
        self.training = None
        self.training_run = _load_active_run(self.dpo_root)
        self.busy = False
        self.offline = False
        self.cancel_event = threading.Event()
        self.lock = threading.RLock()
        self.clients = set()
        self.stop_callback = None

    def broadcast(self, *pieces):
        line = " ".join(fudi_symbol(p) for p in pieces) + ";\n"
        data = line.encode("utf-8")
        with self.lock:
            for client in list(self.clients):
                try:
                    client.sendall(data)
                except OSError:
                    self.clients.discard(client)

    def _training_active(self):
        return self.training is not None and self.training.poll() is None

    def _current(self):
        if self.group is None:
            raise ValueError("no feedback group is active")
        return self.group["candidates"][self.active_index]

    def _candidate_status(self):
        candidate = self._current()
        ratings = self.store.current_ratings(self.group["group_id"])
        current = ratings.get(candidate["candidate_id"])
        self.broadcast("ai_candidate", self.active_index+1, 5,
                       current["rating"] if current else 0, len(ratings))
        if len(ratings) == 5:
            self.broadcast("ai_complete", self.group["group_id"])

    def _generation_worker(self):
        try:
            group = self.service.generate_group(on_progress=lambda p: self.broadcast(
                "ai_progress", p["phase"], p["attempt"], p["accepted"], p["total"]),
                cancel_event=self.cancel_event)
            with self.lock:
                self.group, self.active_index, self.pending_index = group, 0, None
                self.draft_rating, self.warned_unsaved = None, False
            candidate = group["candidates"][0]
            path = self.store.groups / group["group_id"] / f"{candidate['candidate_id']}.txt"
            self.broadcast("ai_session", path)
            self.broadcast("ai_track", group["target_track_id"])
            self._candidate_status()
            self.handle(["stats"])
        except Exception as exc:
            self.broadcast("ai_error", str(exc))
        finally:
            self.busy = False

    def _training_worker(self, resume=False):
        try:
            if not resume:
                name = time.strftime("round_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
                run = self.dpo_root / name
                manifest = self.service.manifest
                reference = Path(self.service.output_dir) / "checkpoints/rhythm_coverage_v1/best_adapter"
                approved = self.dpo_root / "approved.json"
                policy_adapter = (Path(json.loads(approved.read_text(encoding="utf-8"))["adapter"])
                                  if approved.is_file() and self.service.adapter != reference else reference)
                self.broadcast("ai_training", "snapshot")
                make_snapshot(self.store, run, manifest, reference)
                self.training_run = run
                _save_active_run(self.dpo_root, run)
                self.broadcast("ai_training", "reference")
                precompute_reference(run, self.service.output_dir, cpu=self.service.cpu)
            elif self.training_run is None:
                raise ValueError("no paused DPO round to resume")
            run = self.training_run
            self.broadcast("ai_training", "running")
            command = [sys.executable, "-m", "percussion_generation_v4ea", "dpo-train",
                       "--run-dir", str(run), "--epochs", "1", "--save-steps", "10",
                       "--save-total-limit", "3", "--save-on-interrupt"]
            if not resume and policy_adapter != reference:
                command += ["--policy-adapter", str(policy_adapter)]
            elif resume:
                # The initial policy is recorded in the first checkpoint config.
                from .plugin_checkpoints import recover
                _, state, _ = recover(run / "checkpoints")
                command += ["--policy-adapter", state["config"]["policy_initial_adapter"]]
            if resume:
                command += ["--resume-checkpoint", "latest"]
            if self.service.cpu:
                command += ["--cpu"]
            self.training = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            status = self.training.wait()
            self.broadcast("ai_training", "complete" if status == 0 else "failed", status)
        except Exception as exc:
            self.broadcast("ai_error", str(exc))
        finally:
            self.busy = False

    def handle(self, parts):
        if not parts:
            return
        command = parts[0]
        if command == "offline":
            self.offline = len(parts) == 2 and parts[1] == "1"
            self.broadcast("ai_status", "offline" if self.offline else "disabled_online")
        elif command == "init":
            if self._training_active():
                raise RuntimeError("training owns the model device")
            self.broadcast("ai_progress", "loading_model", 0, 0, 5)
            self.service.load()
            self.broadcast("ai_status", "model_ready")
        elif command == "random":
            if not self.offline:
                raise RuntimeError("AI feedback requires an offline local Tweeter session")
            if self.busy or self._training_active():
                raise RuntimeError("model or training process is busy")
            self.busy = True
            self.cancel_event.clear()
            threading.Thread(target=self._generation_worker, daemon=True).start()
            self.broadcast("ai_progress", "starting", 0, 0, 5)
        elif command == "next":
            if not self.offline:
                raise RuntimeError("AI feedback is disabled during multiplayer use")
            if self.pending_index is not None:
                raise ValueError("candidate switch is waiting for loop boundary")
            if self.group is None:
                raise ValueError("no active feedback group")
            current = self._current()
            saved = self.store.current_ratings(self.group["group_id"]).get(current["candidate_id"])
            unsaved = saved is None or (self.draft_rating is not None and self.draft_rating != saved["rating"])
            if unsaved and not self.warned_unsaved:
                self.warned_unsaved = True
                self.broadcast("ai_warn", "unsubmitted_rating_click_NEXT_again")
                return
            self.pending_index = (self.active_index + 1) % 5
            self.warned_unsaved = False
            candidate = self.group["candidates"][self.pending_index]
            positions = set(candidate["positions"])
            pitch = self.group["pitch"]
            self.broadcast("ai_preview", self.group["target_track_id"], self.pending_index+1,
                           *[pitch if step in positions else 0 for step in range(64)])
        elif command == "applied":
            if self.pending_index is None or len(parts) != 2 or int(parts[1]) != self.pending_index+1:
                raise ValueError("unexpected candidate switch acknowledgment")
            self.active_index, self.pending_index = self.pending_index, None
            self.draft_rating = None
            self._candidate_status()
        elif command == "choice":
            if len(parts) != 2 or not 1 <= int(parts[1]) <= 10:
                raise ValueError("rating choice must be 1..10")
            self.draft_rating = int(parts[1])
            self.warned_unsaved = False
        elif command == "rate":
            if not self.offline:
                raise RuntimeError("AI feedback is disabled during multiplayer use")
            if self.pending_index is not None:
                raise ValueError("wait for candidate switch to complete before rating")
            if len(parts) != 2:
                raise ValueError("rate requires one integer")
            candidate = self._current()
            rating = int(parts[1])
            previous = self.store.current_ratings(self.group["group_id"]).get(candidate["candidate_id"])
            event = previous if previous and previous["rating"] == rating else self.store.submit(
                self.group["group_id"], candidate["candidate_id"], rating)
            self.draft_rating = rating
            self.warned_unsaved = False
            # The success message is sent only after the fsynced record is readable.
            self.broadcast("ai_rating_ack", self.active_index+1, event["rating"], event["revision"])
            self._candidate_status()
        elif command == "recover":
            if not self.offline:
                raise RuntimeError("AI feedback is disabled during multiplayer use")
            if len(parts) != 2:
                raise ValueError("recover requires a group UUID")
            self.group = self.store.get_group(parts[1])
            self.active_index, self.pending_index = 0, None
            self.draft_rating, self.warned_unsaved = None, False
            candidate = self._current()
            path = self.store.groups / self.group["group_id"] / f"{candidate['candidate_id']}.txt"
            self.broadcast("ai_session", path)
            self.broadcast("ai_track", self.group["target_track_id"])
            self._candidate_status()
        elif command == "stats":
            snapshots = sorted(self.dpo_root.glob("round_*/snapshot.json"), key=lambda p: p.stat().st_mtime)
            previous = (json.loads(snapshots[-1].read_text(encoding="utf-8")).get("raw_rating_event_count", 0)
                        if snapshots else 0)
            values = self.store.statistics(previous_snapshot_events=previous)
            self.broadcast("ai_stats", values["completed_groups"], values["incomplete_groups"],
                           values["submitted_ratings"], values["unique_original_sources"],
                           values["eligible_pairs_min_difference_2"], values["collection_target"],
                           values["new_rating_events_since_snapshot"],
                           round(100 * values["collection_progress"]))
            supervised = self.service.output_dir / "checkpoints/rhythm_coverage_v1/best_adapter"
            self.broadcast("ai_model", "supervised" if self.service.adapter == supervised else "approved_dpo")
            if self.training_run:
                progress = self.training_run / "progress.json"
                if progress.is_file():
                    state = json.loads(progress.read_text(encoding="utf-8"))
                    self.broadcast("ai_training_progress", state.get("status", "unknown"),
                                   state.get("epoch", state.get("epochs", 0)),
                                   state.get("global_optimizer_step", 0),
                                   state.get("examples", 0), state.get("epoch_examples", 0))
        elif command == "train":
            if self.busy or self._training_active():
                raise RuntimeError("generation or training is busy")
            self.service.release_model()
            self.busy = True
            threading.Thread(target=self._training_worker, daemon=True).start()
        elif command == "pause":
            if not self._training_active() or self.training_run is None:
                raise ValueError("no DPO training process is active")
            (self.training_run / "pause.request").write_text(timestamp(), encoding="utf-8")
            self.broadcast("ai_training", "pause_requested")
        elif command == "resume":
            if self.busy or self._training_active():
                raise RuntimeError("generation or training is busy")
            self.service.release_model()
            self.busy = True
            threading.Thread(target=lambda: self._training_worker(resume=True), daemon=True).start()
        elif command == "select":
            if self.busy or self._training_active():
                raise RuntimeError("cannot switch model while busy")
            if len(parts) != 2:
                raise ValueError("select requires supervised or approved")
            if parts[1] == "supervised":
                adapter = self.service.output_dir / "checkpoints/rhythm_coverage_v1/best_adapter"
            elif parts[1] == "approved":
                pointer = json.loads((self.dpo_root / "approved.json").read_text(encoding="utf-8"))
                adapter = Path(pointer["adapter"])
            else:
                raise ValueError("select requires supervised or approved")
            self.service.select_adapter(adapter)
            self.broadcast("ai_model", parts[1])
        elif command == "cancel":
            self.cancel_event.set()
            self.broadcast("ai_progress", "cancel_requested", 0, 0, 5)
        elif command == "shutdown":
            self.cancel_event.set()
            if self._training_active() and self.training_run:
                (self.training_run / "pause.request").write_text(timestamp(), encoding="utf-8")
            self.service.release_model()
            self.broadcast("ai_status", "shutting_down")
            if self.stop_callback:
                threading.Thread(target=self.stop_callback, daemon=True).start()
        else:
            raise ValueError(f"unknown AI bridge command: {command}")


def serve(bridge, port=19785):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            if self.client_address[0] != "127.0.0.1":
                return
            bridge.clients.add(self.request)
            bridge.broadcast("ai_status", "ready")
            try:
                for raw in self.rfile:
                    for segment in raw.decode("utf-8", errors="replace").split(";"):
                        parts = segment.strip().split()
                        if not parts:
                            continue
                        try:
                            bridge.handle(parts)
                        except Exception as exc:
                            bridge.broadcast("ai_error", str(exc))
            finally:
                bridge.clients.discard(self.request)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with Server(("127.0.0.1", port), Handler) as server:
        bridge.stop_callback = server.shutdown
        print(f"Tweeter AI bridge ready on 127.0.0.1:{port}; feedback {bridge.store.root}", flush=True)
        try:
            server.serve_forever()
        finally:
            bridge.cancel_event.set()
            if bridge._training_active() and bridge.training_run:
                (bridge.training_run / "pause.request").write_text(timestamp(), encoding="utf-8")
            bridge.service.release_model()
