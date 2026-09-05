#!/usr/bin/env python3
"""Audit one OSWorld/QW3 run and seal its trace bundle with SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from analyze_compaction import find_task_trace, load_condition, parse_server_log, read_jsonl


MANIFEST_NAME = "trace_integrity_manifest.json"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",")]


def require_number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    check(isinstance(value, (int, float)) and math.isfinite(value), f"{context}: missing numeric {key}")
    check(float(value) >= 0.0, f"{context}: negative {key}")
    return float(value)


def required_file(path: Path) -> Path:
    check(path.is_file(), f"missing file: {path}")
    check(path.stat().st_size > 0, f"empty file: {path}")
    return path


def audit(
    run_dir: Path,
    expected_steps: int,
    expected_active_images: list[int],
    expected_compaction_steps: list[int],
    expected_history_n: int | None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    required_file(run_dir / "runner.log")
    required_file(run_dir / "qw3_server.log")
    manifest_path = required_file(run_dir / "run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(manifest.get("schema_version") == 1, "unexpected run manifest schema")
    check(manifest.get("endpoints", {}).get("hf_endpoint") == "https://hf-mirror.com", "HF endpoint is not hf-mirror")
    proxies = manifest.get("proxy_environment_present", {})
    expected_proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"}
    check(expected_proxy_keys <= set(proxies), "run manifest does not cover all proxy variables")
    check(not any(bool(proxies[key]) for key in expected_proxy_keys), "a proxy variable was present during the run")
    runtime = manifest.get("runtime", {})
    check(runtime.get("route_expected") == "plain", "unexpected runtime route")
    check(isinstance(runtime.get("kvmem_enabled"), bool), "manifest omits KVMem state")

    task_dir, timing_rows = find_task_trace(run_dir)
    agent_rows = read_jsonl(required_file(task_dir / "agent_step_trace.jsonl"))
    trajectory_rows = read_jsonl(required_file(task_dir / "traj.jsonl"))
    guest_rows = read_jsonl(required_file(task_dir / "trace.jsonl"))
    agent_config = json.loads(required_file(task_dir / "agent_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads(required_file(task_dir / "evaluation_timing.json").read_text(encoding="utf-8"))
    results = json.loads(required_file(run_dir / "osworld" / "summary" / "results.json").read_text(encoding="utf-8"))

    check(len(timing_rows) == expected_steps, f"expected {expected_steps} timing rows, got {len(timing_rows)}")
    check(len(agent_rows) == expected_steps, f"expected {expected_steps} agent rows, got {len(agent_rows)}")
    check([row.get("step_num") for row in timing_rows] == list(range(1, expected_steps + 1)), "step timing sequence is not contiguous")
    check([row.get("step_num") for row in agent_rows] == list(range(1, expected_steps + 1)), "agent trace sequence is not contiguous")
    check(all(row.get("status") == "success" for row in timing_rows), "a step timing row is not successful")
    check(all(row.get("status") == "success" for row in agent_rows), "an agent trace row is not successful")

    observed_images = [int(row.get("compaction", {}).get("active_image_count", -1)) for row in agent_rows]
    observed_events = [int(row["step_num"]) for row in agent_rows if row.get("compaction", {}).get("triggered")]
    check(observed_images == expected_active_images, f"active images {observed_images} != {expected_active_images}")
    check(observed_events == expected_compaction_steps, f"compaction events {observed_events} != {expected_compaction_steps}")
    manifest_history_n = manifest.get("agent_compaction", {}).get("history_n")
    history_n = manifest_history_n if expected_history_n is None else expected_history_n
    check(isinstance(history_n, int) and history_n > 0, "invalid expected history_n")
    check(agent_config.get("history_n") == history_n, f"history_n is not {history_n}")

    action_count = 0
    screenshot_names: set[str] = set()
    for timing, agent in zip(timing_rows, agent_rows, strict=True):
        step = int(timing["step_num"])
        context = f"step {step}"
        check(timing.get("agent_trace") == agent, f"{context}: embedded and standalone agent traces differ")
        require_number(timing, "predict_wall_time_ms", context)
        require_number(timing, "action_wall_time_ms", context)
        require_number(timing, "turn_wall_time_ms", context)
        for key in ("image_prepare", "input_dump", "message_build", "message_dump", "api_call", "parse", "predict_total"):
            require_number(agent.get("timing_ms", {}), key, f"{context} agent timing")

        request = agent.get("request", {})
        input_path = required_file(task_dir / str(request.get("input_file")))
        message_path = required_file(task_dir / str(request.get("message_file")))
        check(sha256_file(input_path) == request.get("input_sha256"), f"{context}: exact input image hash mismatch")
        messages = json.loads(message_path.read_text(encoding="utf-8"))
        check(isinstance(messages, list), f"{context}: message trace is not a list")
        check(len(messages) == request.get("message_count"), f"{context}: message count mismatch")
        image_parts = [
            part
            for message in messages
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        check(len(image_parts) == observed_images[step - 1], f"{context}: sanitized message image count mismatch")
        check(image_parts[-1].get("image_url", {}).get("sha256") == request.get("input_sha256"), f"{context}: newest message image is not the exact step input")

        api = agent.get("api", {})
        check(api.get("finish_reason") in {"tool_calls", "stop"}, f"{context}: unexpected finish reason")
        check(api.get("response_id"), f"{context}: missing API response id")
        require_number(api.get("usage", {}), "prompt_tokens", f"{context} API usage")
        require_number(api.get("usage", {}), "completion_tokens", f"{context} API usage")
        require_number(api.get("server_timing", {}), "request_total_sec", f"{context} server timing")
        require_number(api.get("server_timing", {}), "server_ttft_sec", f"{context} server timing")
        require_number(api.get("server_timing", {}), "engine_ttft_sec", f"{context} server timing")
        check(api.get("attempts") and api["attempts"][-1].get("status") == "success", f"{context}: API attempt did not succeed")

        actions = timing.get("actions", [])
        check(len(actions) == timing.get("action_count"), f"{context}: action count mismatch")
        action_count += len(actions)
        for action in actions:
            env_timing = action.get("environment_timing_ms", {})
            for key in ("action_dispatch_ms", "post_action_pause_ms", "observation_ms", "total_ms"):
                require_number(env_timing, key, f"{context} environment timing")

    check(len(trajectory_rows) == action_count, "trajectory row count does not match executed actions")
    for row in trajectory_rows:
        name = str(row.get("screenshot_file"))
        check(name not in screenshot_names, f"duplicate post-action screenshot name: {name}")
        screenshot_names.add(name)
        required_file(task_dir / name)
    required_file(task_dir / "initial_state.png")

    expected_guest_rows = action_count + 2
    check(len(guest_rows) == expected_guest_rows, f"expected {expected_guest_rows} guest samples, got {len(guest_rows)}")
    check(guest_rows[0].get("sample_type") == "initial_after_reset", "missing initial guest resource sample")
    check(guest_rows[-1].get("sample_type") == "final_before_evaluate", "missing final guest resource sample")
    check(sum(row.get("sample_type") == "post_action" for row in guest_rows) == action_count, "post-action guest sample count mismatch")
    check(all(row.get("collector_error") is None for row in guest_rows), "guest resource collector reported an error")

    require_number(evaluation, "wall_time_ms", "evaluation timing")
    check(len(results) == 1 and results[0].get("task_id") == task_dir.name, "task result does not match trace directory")
    check(results[0].get("status") == "success", "task result status is not success")

    server_requests = parse_server_log(run_dir / "qw3_server.log")
    check(len(server_requests) == expected_steps, f"expected {expected_steps} server requests, got {len(server_requests)}")
    merged_rows, _ = load_condition(run_dir.name, run_dir)
    check(len(merged_rows) == expected_steps, "analyzer did not merge every step")
    for row in merged_rows:
        step = int(row["step_num"])
        check(row.get("server_rid") is not None, f"step {step}: no matching server request")
        check(row.get("server_route") == "plain", f"step {step}: server route is not plain")
        if not runtime["kvmem_enabled"]:
            check(row.get("server_kvmem_fallback_reason") == "disabled", f"step {step}: unexpected fallback reason")
        for key in ("server_total_ms", "server_preprocess_ms", "server_engine_ms", "turn_wall_ms"):
            require_number(row, key, f"step {step} merged timing")

    return {
        "task_id": task_dir.name,
        "step_count": expected_steps,
        "action_count": action_count,
        "agent_trace_rows": len(agent_rows),
        "step_timing_rows": len(timing_rows),
        "trajectory_rows": len(trajectory_rows),
        "guest_resource_rows": len(guest_rows),
        "server_request_rows": len(server_requests),
        "active_image_sequence": observed_images,
        "compaction_event_steps": observed_events,
        "score": results[0].get("score"),
    }


def seal(run_dir: Path, audit_summary: dict[str, Any]) -> Path:
    run_dir = run_dir.resolve()
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output = run_dir / MANIFEST_NAME
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                "run_dir": str(run_dir),
                "audit": audit_summary,
                "file_count": len(files),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-active-images", required=True)
    parser.add_argument("--expected-compaction-steps", default="")
    parser.add_argument(
        "--expected-history-n",
        type=int,
        help="Expected screenshot history; defaults to run_manifest.json.",
    )
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()

    summary = audit(
        args.run_dir,
        args.expected_steps,
        parse_int_list(args.expected_active_images),
        parse_int_list(args.expected_compaction_steps),
        args.expected_history_n,
    )
    if not args.no_seal:
        summary["integrity_manifest"] = str(seal(args.run_dir, summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
