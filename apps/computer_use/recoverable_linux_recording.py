"""Opt-in task recorder; frozen LinuxRecording and trial runners stay unchanged.

The wrapper owns one subprocess and publishes its verified movie. Encoding
completion, uninterrupted fresh sampling and task completion remain distinct.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import kvmem_osworld_queue as owned
import graceful_recoverable_vnc_recording as recorder
from replay_recording_evidence import recording_evidence


ROOT = Path(__file__).resolve().parents[2]
RECORDER_SHA256 = "cfddde9b86858dc6ce98b067cf103b0ec6713db76a5506210f8899de7e498f58"


@dataclass(frozen=True)
class RecordingProfile:
    duration_seconds: float = 1830
    fps: int = 2
    max_wait_seconds: float = 30
    reconnect_budget: int = 3
    startup_seconds: float = 35

    def __post_init__(self):
        for value, low, high in ((self.duration_seconds, 1, 3600),
                                 (self.max_wait_seconds, 3, 120),
                                 (self.startup_seconds, .1, 120)):
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Declare finite recording duration and wait budgets")
        if type(self.fps) is not int or not 1 <= self.fps <= 30:
            raise ValueError("Declare an integer sampling rate")
        if type(self.reconnect_budget) is not int or not 0 <= self.reconnect_budget <= 10:
            raise ValueError("Declare a lifetime reconnect budget")


class RecoverableLinuxRecording:
    def __init__(self, directory, container_id, name="qw3-computer-linux", *, profile=None, port=5900):
        self.directory = Path(directory).resolve()
        self.container_id, self.name, self.port = container_id, name, port
        self.profile = profile or RecordingProfile()
        if not isinstance(self.profile, RecordingProfile) or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Require a declared recording profile and console port")
        self.worker = self.identity = self.owner = self.receipt = None
        self.collected = False
        self.published = False

    def inspect(self):
        info = json.loads(subprocess.check_output(["docker", "inspect", self.name], text=True, timeout=3))
        if len(info) != 1:
            raise ValueError("Ambiguous owned VM")
        info = info[0]
        address = info["NetworkSettings"]["Networks"]["bridge"]["IPAddress"]
        if self.owner is None:
            self.owner = recorder.OwnedEndpoint(self.name, self.container_id, address)
        self.owner(time.monotonic() + 3)
        return info

    def worker_live(self):
        if self.worker is None or self.worker.poll() is not None:
            return False
        actual = owned.process_identity(self.worker.pid)
        if not owned.same_process(self.identity, actual) or actual.get("uid") != os.getuid() or actual["state"] == "Z":
            raise RuntimeError("Recorder process identity changed; replacement preserved")
        return True

    def stop_owned_worker(self, timeout=110):
        if self.worker_live():
            self.worker.terminate()
        self.worker.wait(timeout=timeout)

    def first_frame_ready(self):
        video = self.directory / "video"
        for name in ("recording.json", "capture.jsonl", "framebuffer-geometry.jsonl"):
            if not (video / name).is_file():
                return False
        lines = []
        for name in ("capture.jsonl", "framebuffer-geometry.jsonl"):
            with (video / name).open() as stream:
                line = stream.readline()
            if not line.endswith("\n"):
                return False
            lines.append(json.loads(line))
        capture, geometry = lines
        manifest = json.loads((video / "recording.json").read_text())
        return (manifest.get("status") == "recording" and capture.get("capture") == 0
                and geometry.get("capture") == 0 and geometry.get("full_framebuffer_received") is True
                and geometry.get("pixel_scaling") is False
                and type(capture.get("available_from_frame")) is int
                and capture["captured_monotonic_ns"] >= manifest["started_monotonic_ns"])

    def start(self):
        if self.worker is not None or self.collected or (self.directory / "video").exists():
            raise ValueError("Refuse to replace an existing recorder or media")
        recorder.verify_sources()
        if hashlib.sha256(Path(recorder.__file__).read_bytes()).hexdigest() != RECORDER_SHA256:
            raise ValueError("Qualified recoverable recorder changed")
        import imageio_ffmpeg
        self.inspect()
        command = [sys.executable, "-m", "graceful_recoverable_vnc_recording", "--output", str(self.directory / "video"),
                   "--ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(), "--container-name", self.name,
                   "--container-id", self.container_id, "--host", self.owner.host, "--port", str(self.port),
                   "--duration", str(self.profile.duration_seconds), "--fps", str(self.profile.fps),
                   "--max-wait-seconds", str(self.profile.max_wait_seconds),
                   "--reconnect-budget", str(self.profile.reconnect_budget)]
        with (self.directory / "recorder.log").open("xb") as log:
            self.worker = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        try:
            self.identity = owned.register_child(self.worker.pid, command)
            launch = {"kind": "declared_owned_recoverable_linux_task_recorder", "identity": self.identity,
                      "container_id": self.container_id, "endpoint_identity": self.owner.identity,
                      "profile": asdict(self.profile), "recorder_source_sha256": RECORDER_SHA256,
                      "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "started_at": time.time(), "input_events": 0, "model_calls": 0,
                      "readiness_requires_complete_fresh_frame": True}
            with (self.directory / "recorder-launch.json").open("x") as stream:
                json.dump(launch, stream, indent=2); stream.write("\n")
            deadline = time.monotonic() + self.profile.startup_seconds
            while time.monotonic() < deadline:
                if not self.worker_live():
                    raise RuntimeError("Recorder exited before a complete fresh frame; original artifacts retained")
                if self.first_frame_ready():
                    self.inspect()
                    return launch
                time.sleep(.05)
            raise TimeoutError("No complete fresh frame within recorder readiness budget")
        except BaseException as error:
            # If exec was not registered, never infer authority from a bare PID.
            failure = {"error_type": type(error).__name__, "error": str(error),
                       "registered_identity": self.identity, "input_events": 0, "model_calls": 0}
            if self.identity is not None:
                try:
                    self.stop_owned_worker(timeout=15)
                    failure["owned_worker_retired"] = True
                except Exception as cleanup:
                    failure.update(owned_worker_retired=False, cleanup_error=str(cleanup))
            with (self.directory / "recorder-start-failure.json").open("x") as stream:
                json.dump(failure, stream, indent=2); stream.write("\n")
            raise

    def finish(self, name="continuous.mp4"):
        if name not in {"continuous.mp4", "interrupted.mp4"}:
            raise ValueError("Unsupported recording publication")
        if self.collected:
            if not self.published:
                raise RuntimeError("Prior recording publication failed; original artifacts retained")
            return self.receipt
        if self.worker is None or self.identity is None:
            raise ValueError("Recorder was not started and registered")
        self.stop_owned_worker()
        self.collected = True
        self.inspect()
        receipt_path = self.directory / "video/recoverable-recording.json"
        self.receipt = json.loads(receipt_path.read_text())
        source = self.directory / "video/screen.mp4"
        evidence = recording_evidence(self.directory, source, self.receipt.get("video_duration_seconds"))
        if (self.worker.returncode != 0 or self.receipt.get("encoding_complete") is not True
                or self.receipt.get("validated_complete_captures", 0) < 1
                or evidence.get("recording_evidence_status") != "verified_recovery_timeline"):
            raise RuntimeError("Recorded movie or fresh-frame evidence failed; original artifacts retained")
        # Same filesystem, terminal media only. Do not allocate another full movie.
        os.link(source, self.directory / name)
        self.receipt.update(publication=name, publication_evidence=evidence,
                            qualified_for_task_latency_comparison=False,
                            full_task_continuous_observation_proven=False)
        with (self.directory / "recording-publication.json").open("x") as stream:
            json.dump(self.receipt, stream, indent=2); stream.write("\n")
        self.published = True
        return self.receipt
