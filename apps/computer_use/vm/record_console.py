"""Record the task-owned VM's shared VNC screen without observing or injecting input.

Video timestamps use the host monotonic clock. Capture delays retain the previous
frame, with the delay recorded in capture.jsonl. This is continuous sampling,
not a reconstruction from agent observations. No executor credentials are read.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time

from vm.vnc_admin import Console


def frames_before(elapsed_seconds, fps):
    """Slots strictly before a newly captured frame became available."""
    return max(0, math.ceil(elapsed_seconds * fps))


def record(output, ffmpeg, duration=None, fps=10, port=15900, stop=None, host="127.0.0.1"):
    if not 1 <= fps <= 30 or (duration is not None and duration <= 0):
        raise ValueError("Use 1–30 fps and a positive duration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stop = stop or threading.Event()
    console = None
    encoder = None
    started = None
    frames = captures = 0
    own_cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    error = None
    metadata = {"kind": "continuous_read_only_vnc_recording", "fps": fps,
                "clock": "host_monotonic_ns", "status": "starting",
                "executor_observe_calls": 0, "input_events": 0,
                "capture_gap_policy": "repeat_last_available_frame; initial frame is black",
                "video": "screen.mkv", "complete": False, "console_host":host,"console_port":port}
    with Path(ffmpeg).open("rb") as stream:
        metadata["ffmpeg_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = output / "recording.json"

    def save():
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(metadata, indent=2))
        temp.replace(manifest)

    save()
    try:
        console = Console(port,host)
        # A failed capture must terminate promptly instead of producing a
        # seemingly valid movie with an unreported frozen desktop.
        console.socket.settimeout(3)
        width, height = console.width, console.height
        with (output / "encoder.log").open("wb") as stderr, (output / "capture.jsonl").open("w") as events:
            command = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "warning", "-n",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                       "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "18", "-threads", "2", "-pix_fmt", "yuv420p",
                       str(output / "screen.mkv")]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
            started = time.monotonic_ns()
            metadata.update(status="recording", started_monotonic_ns=started,
                            started_unix_ns=time.time_ns(), width=width, height=height,
                            encoder_command=command)
            save()
            previous = bytes(width * height * 3)
            deadline = started + int(duration * 1e9) if duration else None
            next_capture = started
            while not stop.is_set():
                if deadline is not None and time.monotonic_ns() >= deadline:
                    break
                request_ns = time.monotonic_ns()
                screen = console.screenshot()
                captured_ns = time.monotonic_ns()
                if screen.size != (width, height):
                    raise RuntimeError("VM resolution changed during recording")
                # Never show a future capture earlier on the video timeline.
                target = frames_before((captured_ns - started) / 1e9, fps)
                emitted = target - frames
                while frames < target:
                    encoder.stdin.write(previous)
                    frames += 1
                events.write(json.dumps({"capture": captures, "request_monotonic_ns": request_ns,
                                         "captured_monotonic_ns": captured_ns, "available_from_frame": frames,
                                         "elapsed_seconds": (captured_ns - started) / 1e9,
                                         "preceding_frame_slots": emitted,
                                         "rgb_sha256": hashlib.sha256(screen.tobytes()).hexdigest()}) + "\n")
                events.flush()
                captures += 1
                previous = screen.tobytes()
                next_capture = max(next_capture + round(1e9 / fps), captured_ns)
                if deadline is not None:
                    next_capture = min(next_capture, deadline)
                stop.wait(max(0, (next_capture - time.monotonic_ns()) / 1e9))
            ended = time.monotonic_ns()
            target = frames_before((ended - started) / 1e9, fps)
            while frames < target:
                encoder.stdin.write(previous)
                frames += 1
            encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                raise RuntimeError("Video encoder failed; see encoder.log")
            metadata.update(status="captured", stopped_monotonic_ns=ended,
                            elapsed_seconds=(ended - started) / 1e9)
        # MKV remains the recovery original; MP4 makes browser playback easy.
        subprocess.run([str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(output / "screen.mkv"), "-c", "copy", "-movflags", "+faststart",
                        str(output / "screen.mp4")], check=True, timeout=60)
        metadata.update(status="complete", complete=True, video="screen.mp4")
    except BaseException as exc:
        error = exc
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if console:
            console.close()
        if encoder and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                with contextlib.suppress(BrokenPipeError):
                    encoder.stdin.close()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
        metadata.update(captures=captures, video_frames=frames, video_duration_seconds=frames / fps)
        own_cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        child_cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        metadata["recorder_process_cpu_seconds"] = (own_cpu_end.ru_utime + own_cpu_end.ru_stime -
                                                    own_cpu_start.ru_utime - own_cpu_start.ru_stime)
        metadata["encoder_and_remux_cpu_seconds"] = (child_cpu_end.ru_utime + child_cpu_end.ru_stime -
                                                    child_cpu_start.ru_utime - child_cpu_start.ru_stime)
        metadata["process_peak_rss_kib"] = own_cpu_end.ru_maxrss
        metadata["resource_scope"] = "standalone Linux recorder process and its encoder/remux children; excludes VM overhead"
        metadata["artifacts"] = {}
        for name in ("screen.mkv", "screen.mp4", "capture.jsonl", "encoder.log"):
            path = output / name
            if path.exists():
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                metadata["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": sha}
        save()
    if error:
        raise error
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--port", type=int, default=15900)
    parser.add_argument("--host", default="127.0.0.1", help="Private IP of an already verified task-owned VM console")
    args = parser.parse_args()
    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    result = record(args.output, args.ffmpeg, args.duration, args.fps, args.port, stop, args.host)
    print(json.dumps({k: result[k] for k in ("status", "captures", "video_frames", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
