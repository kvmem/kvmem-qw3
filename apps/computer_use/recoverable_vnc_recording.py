"""Standalone read-only recording candidate with bounded transport recovery.

Reuse the qualified RFB geometry/parser and encoder unchanged. A reconnect
discards incomplete pixels and requires the same owned endpoint plus a complete
fresh framebuffer. Held video frames remain stale, explicitly logged evidence.
The process-local factories below are for a dedicated recorder process only.
"""
import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import resizable_vnc_recording as geometry
from vm import record_console


SOURCE_SHA256 = {
    "resizable_vnc_recording.py": "ff96022ccdd73756bb99781540e7f69d9b3fa4ba26a91c6de8e6ee1f89ea084d",
    "vm/record_console.py": "580f32b7bb949d5b6de74a269c8ecb7e94e3c111d325aa6d153c439b95fe5b15",
    "vm/vnc_admin.py": "6b1db14692d7f078833b3cd7ee89c99b486cf30fe2213f1d9c133aa33d230cfe",
}
TRANSPORT_ERRORS = (TimeoutError, EOFError, ConnectionError)


def verify_sources():
    root = Path(__file__).resolve().parent
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Qualified recording source changed: " + name)


class OwnedEndpoint:
    """Pin the registered Docker instance, image, start time and bridge address."""
    def __init__(self, name, container_id, host):
        if not re.fullmatch(r"qw3-[a-zA-Z0-9_.-]{1,100}", name):
            raise ValueError("Use a registered task-owned container name")
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ValueError("Require the full registered container identity")
        if not ipaddress.ip_address(host).is_private:
            raise ValueError("Require an owned private console address")
        self.name, self.container_id, self.host = name, container_id, host
        self.identity = None

    def __call__(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Endpoint identity deadline expired")
        data = json.loads(subprocess.check_output(
            ["docker", "inspect", self.name], text=True, timeout=min(3, remaining)))
        if len(data) != 1:
            raise RuntimeError("Ambiguous endpoint identity")
        info = data[0]
        if (info["Id"] != self.container_id or info["Name"] != "/" + self.name
                or info["Config"].get("Labels", {}).get("qw3.goal") != "cross-device-20261003"
                or not info["State"]["Running"]
                or info["NetworkSettings"]["Networks"]["bridge"]["IPAddress"] != self.host):
            raise RuntimeError("Owned recording endpoint changed")
        current = (info["Id"], info["Image"], info["State"]["StartedAt"], self.host)
        if self.identity is not None and current != self.identity:
            raise RuntimeError("Owned container restarted or changed")
        self.identity = current
        return True


class BoundedConsole(geometry.ResizableConsole):
    """Bound each connection/handshake/capture including slow partial reads."""
    def __init__(self, port, host, *, canvas, on_capture, deadline, io_timeout, stop):
        self.deadline, self.io_timeout, self.stop = deadline, io_timeout, stop
        # Inject only this module's connection factory, not socket globally.
        api = SimpleNamespace(create_connection=lambda address, timeout: socket.create_connection(
            address, min(timeout, self.remaining())))
        with patch.object(geometry, "socket", api):
            super().__init__(port, host, canvas=canvas, on_capture=on_capture)

    def remaining(self):
        if self.stop.is_set():
            raise InterruptedError("Owned recorder was stopped during an unobserved interval")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Capture/recovery deadline expired")
        return remaining

    def read(self, count):
        data = bytearray()
        while len(data) < count:
            self.socket.settimeout(min(self.io_timeout, self.remaining()))
            chunk = self.socket.recv(count - len(data))
            if not chunk:
                raise EOFError("Owned console disconnected during a frame")
            data.extend(chunk)
        return bytes(data)


class RecoverableConsole:
    def __init__(self, port, host, *, owner_check, canvas=(1920, 1080),
                 max_wait_seconds=30, reconnect_budget=3, io_timeout=3, stop=None,
                 on_capture=None, on_event=None):
        verify_sources()
        if (not callable(owner_check) or not ipaddress.ip_address(host).is_private
                or type(port) is not int or not 1 <= port <= 65535
                or len(canvas) != 2 or any(type(v) is not int or v <= 0 for v in canvas)
                or canvas[0] > 3840 or canvas[1] > 2160):
            raise ValueError("Declare a bounded canvas and verified private endpoint")
        if (type(max_wait_seconds) not in (int, float) or not math.isfinite(max_wait_seconds)
                or not 0.1 <= max_wait_seconds <= 120
                or type(io_timeout) not in (int, float) or not math.isfinite(io_timeout)
                or not 0 < io_timeout <= max_wait_seconds
                or type(reconnect_budget) is not int or not 0 <= reconnect_budget <= 10):
            raise ValueError("Declare finite capture and lifetime reconnect budgets")
        self.port, self.host, self.canvas = port, host, canvas
        self.width, self.height = canvas
        self.owner_check, self.on_capture, self.on_event = owner_check, on_capture, on_event
        self.max_wait_seconds, self.reconnect_budget, self.io_timeout = max_wait_seconds, reconnect_budget, io_timeout
        self.stop = stop or threading.Event()
        self.delegate, self.pending_geometry, self.gap = None, None, None
        self.events, self.gaps = [], []
        self.reconnects, self.connections, self.captures = 0, 0, 0
        self.last_capture_ns = None
        deadline = time.monotonic() + max_wait_seconds
        try:
            try:
                self.connect(deadline)
            except TRANSPORT_ERRORS as exc:
                self.recover(deadline, exc)
        except BaseException:
            self.close()
            raise

    @property
    def socket(self):
        return self.delegate.socket

    def emit(self, kind, **fields):
        event = {"kind": kind, "at_monotonic_ns": time.monotonic_ns(), "input_events": 0, **fields}
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def guard(self, deadline):
        if self.stop.is_set():
            raise InterruptedError("Owned recorder stopped; no fresh observation claimed")
        if deadline <= time.monotonic():
            raise TimeoutError("Capture/recovery deadline expired")
        try:
            accepted = self.owner_check(deadline)
        except Exception as exc:
            raise RuntimeError("Recording endpoint identity could not be established") from exc
        if accepted is not True:
            raise RuntimeError("Recording endpoint identity was not positively verified")

    def connect(self, deadline):
        self.guard(deadline)
        self.close()
        self.pending_geometry = None
        self.connections += 1
        self.delegate = BoundedConsole(self.port, self.host, canvas=self.canvas,
            on_capture=lambda event: setattr(self, "pending_geometry", event),
            deadline=deadline, io_timeout=self.io_timeout, stop=self.stop)

    def finish_gap(self, recovered, **fields):
        if self.gap is None:
            return
        gap = {**self.gap, "until_monotonic_ns": time.monotonic_ns(), "recovered": recovered,
               "held_frames_are_new_observations": False, "missing_pixels_reconstructed": False, **fields}
        self.gaps.append(gap)
        self.emit("observation_gap_closed", **gap)
        self.gap = None

    def recover(self, deadline, error):
        if self.gap is None:
            self.gap = {"from_monotonic_ns": self.last_capture_ns or time.monotonic_ns(),
                        "first_error_type": type(error).__name__, "last_fresh_capture": self.captures - 1}
        self.emit("transport_failure", error_type=type(error).__name__, capture=self.captures)
        self.close()
        self.pending_geometry = None
        try:
            while self.reconnects < self.reconnect_budget:
                self.guard(deadline)
                self.reconnects += 1
                self.emit("reconnect_attempt", attempt=self.reconnects)
                try:
                    self.connect(deadline)
                    return
                except TRANSPORT_ERRORS as exc:
                    self.emit("transport_failure", error_type=type(exc).__name__, capture=self.captures)
                    self.close()
            raise TimeoutError("Lifetime reconnect budget exhausted")
        except BaseException as exc:
            self.finish_gap(False, terminal_error_type=type(exc).__name__)
            raise

    def screenshot(self):
        deadline = time.monotonic() + self.max_wait_seconds
        try:
            while True:
                self.guard(deadline)
                self.delegate.deadline = deadline
                self.delegate.socket.settimeout(min(self.io_timeout, self.delegate.remaining()))
                self.pending_geometry = None
                try:
                    image = self.delegate.screenshot()
                except TRANSPORT_ERRORS as exc:
                    self.recover(deadline, exc)
                    continue
                self.guard(deadline)
                if self.pending_geometry is None:
                    raise RuntimeError("Capture lacks complete-frame geometry evidence")
                event = {**self.pending_geometry, "capture": self.captures,
                         "connection": self.connections, "validated_monotonic_ns": time.monotonic_ns()}
                self.last_capture_ns = event["validated_monotonic_ns"]
                if self.on_capture is not None:
                    self.on_capture(event)
                self.captures += 1
                self.finish_gap(True, fresh_capture=event["capture"], connection=self.connections,
                                full_framebuffer_received=True)
                return image
        except BaseException as exc:
            self.finish_gap(False, terminal_error_type=type(exc).__name__)
            self.close()
            raise

    def close(self):
        if self.delegate is not None:
            self.delegate.close()
            self.delegate = None


def record(output, ffmpeg, *, owner_check, canvas=(1920, 1080), host="127.0.0.1",
           port=15900, fps=2, duration=None, stop=None, max_wait_seconds=30,
           reconnect_budget=3, io_timeout=3):
    verify_sources()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite a recording")
    clients, events, captures = [], [], []
    def sink(filename, target, event):
        target.append(event)
        with (output / filename).open("a") as stream:
            stream.write(json.dumps(event) + "\n")
    def factory(port, host):
        client = RecoverableConsole(port, host, owner_check=owner_check, canvas=canvas,
            stop=stop, max_wait_seconds=max_wait_seconds, reconnect_budget=reconnect_budget,
            io_timeout=io_timeout, on_capture=lambda e: sink("framebuffer-geometry.jsonl", captures, e),
            on_event=lambda e: sink("transport-recovery.jsonl", events, e))
        clients.append(client)
        return client
    result = None
    try:
        with patch.object(record_console, "Console", factory):
            result = record_console.record(output, ffmpeg, duration=duration, fps=fps,
                                           port=port, host=host, stop=stop)
        if not captures:
            raise RuntimeError("Recording has no positively validated complete frame")
        return result
    finally:
        if (output / "recording.json").exists():
            raw = json.loads((output / "recording.json").read_text())
            gaps = [e for e in events if e["kind"] == "observation_gap_closed"]
            report = {**raw, "kind": "isolated_bounded_recoverable_vnc_recording",
                      "declared_canvas": list(canvas), "pixel_scaling": False,
                      "encoding_complete": raw.get("complete") is True,
                      "fresh_sampling_uninterrupted": raw.get("complete") is True and bool(captures) and not gaps and not any(e["kind"] == "transport_failure" for e in events),
                      "full_task_continuous_observation_proven": False,
                      "transport_gaps": gaps, "reconnects": sum(e["kind"] == "reconnect_attempt" for e in events),
                      "validated_complete_captures": len(captures), "geometry_validated_per_capture": bool(captures),
                      "recording_scope": "Sampled timeline; held frame intervals are stale, never new observations.",
                      "qualified_for_task_latency_comparison": False,
                      "reused_source_sha256": SOURCE_SHA256,
                      "recorder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "max_wait_seconds_per_capture": max_wait_seconds,
                      "lifetime_reconnect_budget": reconnect_budget,
                      "model_calls": 0, "input_events": 0}
            (output / "recoverable-recording.json").write_text(json.dumps(report, indent=2) + "\n")
            if result is not None:
                result.update(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--ffmpeg", required=True)
    p.add_argument("--container-name", required=True); p.add_argument("--container-id", required=True)
    p.add_argument("--host", required=True); p.add_argument("--port", type=int, default=5900)
    p.add_argument("--duration", type=float, required=True); p.add_argument("--fps", type=int, default=2)
    p.add_argument("--max-wait-seconds", type=float, default=30); p.add_argument("--reconnect-budget", type=int, default=3)
    args = p.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set()); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    owner = OwnedEndpoint(args.container_name, args.container_id, args.host)
    result = record(args.output, args.ffmpeg, owner_check=owner, host=args.host, port=args.port,
                    duration=args.duration, fps=args.fps, stop=stop,
                    max_wait_seconds=args.max_wait_seconds, reconnect_budget=args.reconnect_budget)
    print(json.dumps({k: result[k] for k in ("encoding_complete", "fresh_sampling_uninterrupted", "reconnects", "input_events")}))


if __name__ == "__main__": main()
