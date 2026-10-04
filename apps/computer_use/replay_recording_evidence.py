"""Public, bounded recording-gap evidence; never promote component to task success."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid recording counter or timestamp")
    return value


def recording_evidence(root, source, duration=None):
    root, source = Path(root), Path(source)
    directories = (root / "video", root / "recording", root)
    directory = next((p for p in directories if (p / "recoverable-recording.json").is_file()), None)
    recoverable = directory is not None
    if directory is None:
        directory = next((p for p in directories if (p / "recording.json").is_file()), None)
    if directory is None:
        return {"recording_evidence_status": "not_available", "video_observation_gaps": []}
    receipt = directory / ("recoverable-recording.json" if recoverable else "recording.json")
    result = {"recording_evidence_status": "invalid", "video_observation_gaps": [],
              "recording_manifest_sha256": digest(receipt), "video_usable_for_full_task_trace": False}
    try:
        data = json.loads(receipt.read_text())
        expected_kind = "isolated_bounded_recoverable_vnc_recording" if recoverable else "continuous_read_only_vnc_recording"
        if data.get("kind") != expected_kind:
            return result | {"recording_evidence_status": "unsupported"}
        artifact_name = "screen.mkv" if source.suffix == ".mkv" else "screen.mp4"
        artifact = data["artifacts"][artifact_name]
        if artifact["sha256"] != digest(source) or integer(artifact["bytes"], 1) != source.stat().st_size:
            raise ValueError("Media does not match its terminal recording receipt")
        if type(data["complete"]) is not bool:
            raise ValueError("Unknown encoding completion state")
        result.update(recording_evidence_status="verified_source", recording_encoding_complete=data["complete"],
                      original_recording_video_sha256=artifact["sha256"], partial_video=not data["complete"])
        if not recoverable:
            return result
        if (type(data["fresh_sampling_uninterrupted"]) is not bool
                or not isinstance(data["transport_gaps"], list) or len(data["transport_gaps"]) > 11):
            raise ValueError("Invalid bounded recovery evidence")
        if data["fresh_sampling_uninterrupted"] and (data["transport_gaps"] or not data["complete"]):
            raise ValueError("Contradictory sampling completion state")
        if data["clock"] != "host_monotonic_ns":
            raise ValueError("Unknown recording clock domain")
        fps = data["fps"]
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 30:
            raise ValueError("Invalid video clock")
        start = integer(data["started_monotonic_ns"], 1)
        frame_count = integer(data["video_frames"])
        video_end = frame_count / fps
        if duration is not None:
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid decoded movie duration")
            if abs(duration - video_end) > 1 / fps + .05:
                raise ValueError("Decoded video and source clock disagree")
        capture_path = directory / "capture.jsonl"
        expected = data["artifacts"]["capture.jsonl"]
        if digest(capture_path) != expected["sha256"] or capture_path.stat().st_size != expected["bytes"]:
            raise ValueError("Capture timeline changed")
        captures = {}
        for line in capture_path.read_text().splitlines():
            capture = json.loads(line); index = integer(capture["capture"])
            slot = integer(capture["available_from_frame"])
            timestamp = integer(capture["captured_monotonic_ns"], start)
            if index in captures or slot > frame_count:
                raise ValueError("Ambiguous capture timeline")
            captures[index] = (slot, timestamp)
        if sorted(captures) != list(range(integer(data["captures"]))) or not captures:
            raise ValueError("Capture counters disagree")
        gaps = []
        for gap in data["transport_gaps"]:
            left = integer(gap["from_monotonic_ns"]); right = integer(gap["until_monotonic_ns"], left)
            if (type(gap["recovered"]) is not bool or gap["held_frames_are_new_observations"] is not False
                    or gap["missing_pixels_reconstructed"] is not False):
                raise ValueError("Recovery cannot manufacture observations")
            begin = max(0, (left - start) / 1e9)
            end = video_end
            if gap["recovered"]:
                slot, captured = captures[integer(gap["fresh_capture"])]
                if captured < right:
                    raise ValueError("Fresh frame preceded completed recovery")
                # The new image is visible only at its actual encoded slot,
                # possibly later than the parser's recovery completion time.
                end = slot / fps
            if end < begin:
                if gap["recovered"]:
                    raise ValueError("Recovery interval runs backwards")
                continue  # Failed tail can lie beyond the retained fragment.
            if end > begin:
                gaps.append({"start_seconds": begin, "end_seconds": end,
                             "recovered": gap["recovered"], "displayed_frames_are_stale": True})
        result.update(video_observation_gaps=gaps, recording_gap_count=len(data["transport_gaps"]),
                      fresh_sampling_uninterrupted=data["fresh_sampling_uninterrupted"],
                      recording_evidence_status="verified_recovery_timeline")
        if gaps or not data["complete"]:
            result["video_usable_for_full_task_trace"] = False
        return result
    except (KeyError, TypeError, ValueError, OSError):
        # Keep grades untouched and do not advertise an unverified timeline.
        return result | {"recording_evidence_status": "invalid", "video_observation_gaps": [],
                         "video_usable_for_full_task_trace": False}


def retained_mkv(root):
    """Admit a partial source only through its failed original VNC receipt."""
    root = Path(root); source = root / "video/screen.mkv"
    if not source.is_file():
        return None
    evidence = recording_evidence(root, source)
    if (evidence.get("recording_evidence_status") in ("verified_source", "verified_recovery_timeline")
            and evidence.get("recording_encoding_complete") is False):
        return source, evidence
    return None
