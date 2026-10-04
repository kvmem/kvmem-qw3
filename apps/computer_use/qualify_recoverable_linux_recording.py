"""Reproducible zero-input recorder qualification on the registered idle pilot.

Lifecycle exercises start and SIGTERM collection. Stability samples an idle
native VM for 1800 seconds; it is not an active GUI task or model benchmark.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import execution_retest_queue as base
import recoverable_linux_recording as wrapper_module
import retired_051_successors as successors


CID = "bcb52434987f397b5f2d9ccf03a3d51feed3ce33776fb710aa78000b6e552b20"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ledger(item):
    path = Path(item["token_file"]).parent / "receipts.sqlite"
    with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        rows = db.execute("SELECT * FROM actions ORDER BY rowid").fetchall()
        counts = dict(db.execute("SELECT status,COUNT(*) FROM actions GROUP BY status").fetchall())
    return {"rows_sha256": hashlib.sha256(repr(rows).encode()).hexdigest(), "counts": counts}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"kind": "native_recoverable_linux_task_wrapper_qualification", "mode": args.mode,
              "status": "preparing", "started_at": time.time(),
              "producer": base.owned.process_identity(os.getpid()), "recipe_sha256": digest(__file__),
              "model_calls": 0, "native_device_inputs": 0,
              "native_device_task_or_qw3_retest": False,
              "task_success_or_latency_improvement_proven": False,
              "scope": "Registered paused idle Linux pilot; no active GUI task or model input.",
              "native_1800_second_idle_sampling_stability_proven": False}
    state = output / "state.json"
    def save():
        temp = state.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2) + "\n"); temp.replace(state)
    save()
    wrapper = None
    try:
        qualification = base.read(args.qualification)
        if (qualification.get("passed") is not True or qualification.get("tests_run", 0) < 8
                or qualification.get("model_calls") != 0 or qualification.get("native_device_inputs") != 0):
            raise ValueError("Require completed wrapper subprocess/RFB/codec qualification")
        for name, sha in qualification["source_sha256"].items():
            if digest(Path(wrapper_module.__file__).parent / name) != sha:
                raise ValueError("Qualified wrapper source changed")
        record["source_sha256"] = qualification["source_sha256"]
        protocol = base.read(args.protocol)
        gateway = protocol["previous_gateways"]["linux-vm"]
        item = gateway["item"]
        base.require_identity(gateway["identity"])
        base.owned.assert_apps_idle(protocol["previous_app"])
        declaration, _, _, _ = successors.validate_followup(args.followup)
        parent = successors.continued_parent(declaration, digest(args.followup), args.followup)
        child = parent.get("child")
        if child is None:
            raise ValueError("Require a verified active main trial to prove separate VM identity")
        child_identity = base.owned.process_identity(child["pid"])
        if not base.owned.same_process(child, child_identity):
            raise ValueError("Main trial identity changed during preparation")
        active_output = Path(child["command"][child["command"].index("--output") + 1])
        active_vm = base.read(active_output / "vm.json")
        if active_vm["container_id"] == CID:
            raise ValueError("Read-only pilot is the active model VM")
        before = base.gateway_api(item, "/observe")
        if before.get("paused") is not True:
            raise ValueError("Existing pilot must already be paused")
        before_ledger = ledger(item)
        profile = wrapper_module.RecordingProfile(duration_seconds=1800 if args.mode == "stability" else 1830)
        wrapper = wrapper_module.RecoverableLinuxRecording(output, CID, profile=profile)
        launch = wrapper.start()
        record.update(status="recording", recorder=launch, container_id=CID,
                      other_active_osworld_container=active_vm["container_id"],
                      before_receipt_ledger=before_ledger, observed_dimensions=[before["width"], before["height"]])
        save()
        started = time.monotonic()
        if args.mode == "lifecycle":
            time.sleep(8)
        else:
            while wrapper.worker_live():
                record["observed_running_seconds"] = time.monotonic() - started
                save()
                if record["observed_running_seconds"] > 1945:
                    raise TimeoutError("Declared stability recorder did not terminate within its bounded profile")
                time.sleep(5)
        receipt = wrapper.finish()
        after = base.gateway_api(item, "/observe")
        after_ledger = ledger(item)
        base.require_identity(gateway["identity"])
        base.owned.assert_apps_idle(protocol["previous_app"])
        wrapper.inspect()
        geometry = [json.loads(line) for line in (output / "video/framebuffer-geometry.jsonl").read_text().splitlines()]
        import av
        frames, sizes, visible = 0, set(), False
        with av.open(str(output / "continuous.mp4")) as video:
            video.streams.video[0].codec_context.thread_count = 2
            for frame in video.decode(video=0):
                frames += 1; sizes.add((frame.width, frame.height))
                if not visible:
                    visible = bool((frame.to_ndarray(format="rgb24") > 20).mean() > .05)
        checks = {
            "recording_owner_is_separate_from_active_model_vm": active_vm["container_id"] != CID,
            "existing_device_remained_paused": before["paused"] is True and after["paused"] is True,
            "existing_device_and_session_unchanged": all(before[k] == after[k] for k in ("device_id", "desktop_id", "session_id")),
            "all_original_receipt_rows_unchanged": before_ledger == after_ledger,
            "worker_registered_with_complete_command": bool(launch["identity"]["command"]),
            "first_complete_fresh_frame_required": launch["readiness_requires_complete_fresh_frame"] is True,
            "owned_worker_actually_reaped": wrapper.worker.poll() == 0 and base.owned.process_identity(wrapper.worker.pid) is None,
            "encoding_completed_with_source_evidence": receipt["encoding_complete"] is True and receipt["publication_evidence"]["recording_evidence_status"] == "verified_recovery_timeline",
            "all_video_frames_decode": frames == receipt["video_frames"],
            "fixed_canvas_without_scaling": sizes == {(1920, 1080)} and all(e["pixel_scaling"] is False for e in geometry),
            "all_complete_native_framebuffers_match_device": len(geometry) > 1 and all(e["framebuffer"] == [before["width"], before["height"]] and e["full_framebuffer_received"] is True for e in geometry),
            "decoded_video_contains_real_desktop": visible,
            "published_movie_uses_original_terminal_inode": (output / "continuous.mp4").stat().st_ino == (output / "video/screen.mp4").stat().st_ino,
            "sampling_state_and_gaps_not_promoted_to_task_success": receipt["qualified_for_task_latency_comparison"] is False and receipt["full_task_continuous_observation_proven"] is False,
            "source_pins_unchanged": all(digest(Path(wrapper_module.__file__).parent / name) == sha for name, sha in record["source_sha256"].items()),
            "declared_duration_covered": receipt["video_duration_seconds"] >= (1800 if args.mode == "stability" else 8),
        }
        record.update(status="passed" if all(checks.values()) else "failed", passed=all(checks.values()),
                      checks=checks, checks_passed=sum(checks.values()), finished_at=time.time(),
                      decoded_frames=frames, video_seconds=receipt["video_duration_seconds"],
                      fresh_sampling_uninterrupted=receipt["fresh_sampling_uninterrupted"],
                      reconnects=receipt["reconnects"], transport_gaps=receipt["transport_gaps"],
                      actual_complete_captures=len(geometry), after_receipt_ledger=after_ledger,
                      video_sha256=digest(output / "continuous.mp4"),
                      recorder_process_cpu_seconds=receipt["recorder_process_cpu_seconds"],
                      encoder_and_remux_cpu_seconds=receipt["encoder_and_remux_cpu_seconds"],
                      process_peak_rss_kib=receipt["process_peak_rss_kib"],
                      native_1800_second_idle_sampling_stability_proven=args.mode == "stability" and all(checks.values()))
        save()
        if not record["passed"]:
            raise RuntimeError("Native qualification failed; original evidence retained")
        return record
    except BaseException as error:
        record.update(status="failed", passed=False, error_type=type(error).__name__, error=str(error), finished_at=time.time())
        if wrapper is not None and wrapper.identity is not None and not wrapper.collected:
            try:
                wrapper.stop_owned_worker()
                record["owned_worker_retired_after_failure"] = wrapper.worker.poll() is not None
            except Exception as cleanup:
                record.update(owned_worker_retired_after_failure=False, cleanup_error=str(cleanup))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("lifecycle", "stability"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    record = run(parser.parse_args())
    print(json.dumps({key: record[key] for key in ("passed", "checks_passed", "mode", "decoded_frames", "video_seconds", "model_calls", "native_device_inputs")}))


if __name__ == "__main__":
    main()
