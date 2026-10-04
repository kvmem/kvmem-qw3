"""Isolated graceful-stop variant of the qualified recoverable recorder.

Only an InterruptedError with this recorder's stop event set ends capture
normally. The unfinished frame is discarded; the encoded tail retains the
last complete image and is explicitly an unobserved interval.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import signal
import threading
import time
from unittest.mock import patch

import recoverable_vnc_recording as base


BASE_SHA256 = "a3b65a70a414a9e28eaf503ab81e878196a6843c7a95fb6f1e32762d5b49c35e"
OwnedEndpoint = base.OwnedEndpoint
SOURCE_SHA256 = base.SOURCE_SHA256
ORIGINAL_RECORD = base.record


def verify_sources():
    base.verify_sources()
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Qualified recoverable recording source changed")


class GracefulStopConsole(base.RecoverableConsole):
    def screenshot(self):
        try:
            return super().screenshot()
        except InterruptedError:
            if self.stop.is_set():
                # The base already discards and closes the partial connection.
                # Add a tail only if it did not close an existing recovery gap.
                if not self.gaps or self.gaps[-1].get("terminal_error_type") != "InterruptedError":
                    self.gap = {"from_monotonic_ns": self.last_capture_ns or time.monotonic_ns(),
                                "first_error_type": "InterruptedError", "last_fresh_capture": self.captures - 1,
                                "reason": "stop_during_capture"}
                    self.finish_gap(False, terminal_error_type="InterruptedError",
                                    full_framebuffer_received=False)
                self.emit("graceful_stop_discarded_incomplete_capture", capture=self.captures,
                          held_frames_are_new_observations=False, missing_pixels_reconstructed=False)
            raise


def graceful_encoder():
    """Derive one guarded change from the pinned encoder; never edit its file."""
    verify_sources()
    source = inspect.getsource(base.record_console.record)
    anchor = "                screen = console.screenshot()\n"
    if source.count(anchor) != 1:
        raise ValueError("Qualified encoder capture anchor changed")
    source = source.replace(anchor, """                try:
                    screen = console.screenshot()
                except InterruptedError:
                    if not stop.is_set():
                        raise
                    break  # No capture count, timestamp or pixels from this partial frame.
""")
    namespace = dict(base.record_console.__dict__)
    exec(compile(source, str(Path(__file__).resolve()) + "::pinned_encoder", "exec"), namespace)
    function = namespace["record"]
    def encode(*args, **kwargs):
        # ORIGINAL_RECORD installs the positively verified console factory
        # immediately before this call. Use that exact factory in the clone.
        namespace["Console"] = base.record_console.Console
        return function(*args, **kwargs)
    return encode


def record(output, ffmpeg, **options):
    encoder = graceful_encoder()
    result = None
    try:
        with patch.object(base, "RecoverableConsole", GracefulStopConsole), patch.object(base.record_console, "record", encoder):
            result = ORIGINAL_RECORD(output, ffmpeg, **options)
        return result
    finally:
        path = Path(output) / "recoverable-recording.json"
        if path.is_file():
            report = json.loads(path.read_text())
            report.update(graceful_stop_variant_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          qualified_base_recorder_source_sha256=BASE_SHA256,
                          graceful_stop_discarded_capture=any(g.get("terminal_error_type") == "InterruptedError" for g in report["transport_gaps"]),
                          graceful_stop_does_not_publish_incomplete_pixels=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            if result is not None:
                result.update(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5900)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-wait-seconds", type=float, default=30)
    parser.add_argument("--reconnect-budget", type=int, default=3)
    args = parser.parse_args()
    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    result = record(args.output, args.ffmpeg, owner_check=OwnedEndpoint(args.container_name, args.container_id, args.host),
                    host=args.host, port=args.port, duration=args.duration, fps=args.fps, stop=stop,
                    max_wait_seconds=args.max_wait_seconds, reconnect_budget=args.reconnect_budget)
    print(json.dumps({k: result[k] for k in ("encoding_complete", "fresh_sampling_uninterrupted", "reconnects", "input_events", "graceful_stop_discarded_capture")}))


if __name__ == "__main__":
    main()
