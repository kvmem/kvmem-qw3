"""Check a live changing VM screen, video decoding, and observation noninterference."""
import argparse
import asyncio
import json
from pathlib import Path
import statistics
import subprocess
import time
import uuid

from qw3_computer.desktop import Desktop
from vm.record_console import record


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    desktop = Desktop(args.desktop, args.token.read_text().strip(), output / "desktop")
    result = {"kind": "live_windows_recorder_qualification", "passed": False,
              "host_started_unix_ns": time.time_ns()}
    try:
        await desktop.start()
        await desktop.control("resume")
        before = await desktop.observe()
        assert before["foreground"]["process"] == "msedge", "Open the synthetic Edge counter first"
        assert "Recorder qualification" in before["foreground"]["title"], "Unexpected probe window"
        result["initial_observation_id"] = before["observation_id"]
        video = await asyncio.to_thread(record, output / "video", args.ffmpeg, 10, 10)
        # Use the exact pre-recording observation. A recorder that called
        # /observe would invalidate it and fail this actual executor action.
        receipt = await desktop.act({"desktop_id": before["desktop_id"],
            "observation_id": before["observation_id"], "action_id": "recording-probe-" + uuid.uuid4().hex,
            "action": {"kind": "key", "key": "CTRL+L"}})
        result["post_recording_action_status"] = receipt["status"]
        result["pre_recording_observation_accepted"] = receipt["status"] == "completed"
        rows = [json.loads(line) for line in (output / "video/capture.jsonl").read_text().splitlines()]
        result["distinct_captured_images"] = len(set(row["rgb_sha256"] for row in rows))
        gaps = [(b["captured_monotonic_ns"]-a["captured_monotonic_ns"])/1e9 for a,b in zip(rows, rows[1:])]
        result["capture"] = {"count": len(rows), "elapsed_seconds": video["elapsed_seconds"],
                             "mean_hz": (len(rows)-1)/sum(gaps), "max_gap_seconds": max(gaps),
                             "median_gap_seconds": statistics.median(gaps)}
        decoded = subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-i",
            str(output / "video/screen.mp4"), "-f", "framemd5", "-"], capture_output=True, text=True, check=True)
        (output / "decoded-frames.md5").write_text(decoded.stdout)
        frames = [line for line in decoded.stdout.splitlines() if line and not line.startswith("#")]
        distinct = len(set(line.rsplit(",",1)[1].strip() for line in frames))
        result["decoded_frames"] = len(frames)
        result["distinct_decoded_frames"] = distinct
        result["duration_rounding_error_seconds"] = video["video_duration_seconds"] - video["elapsed_seconds"]
        assert len(frames) == video["video_frames"], "Encoder frame count mismatch"
        assert result["distinct_captured_images"] > 20, "Changing screen not adequately captured"
        assert max(gaps) < 0.5, "Too many capture gaps on the changing screen"
        assert 0 <= result["duration_rounding_error_seconds"] < 0.101, "Video time scale differs from real time"
        assert result["pre_recording_observation_accepted"]
        result["passed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if desktop.session:
                await desktop.control("pause")
        finally:
            await desktop.close()
        result["host_finished_unix_ns"] = time.time_ns()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
