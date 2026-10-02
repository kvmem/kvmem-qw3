"""Summarize preserved software-product evidence without changing trial results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def latency(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    # Nearest-rank P95; small trials are descriptive, not stable estimates.
    import math
    return {"count": len(values), "sum_seconds": sum(values), "p50_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(.95 * len(values)) - 1], "max_seconds": max(values)}


def summarize(root):
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    guest = result.get("guest_result", {})
    task = guest.get("terminal_state", {})
    jobs = task.get("software", {}).get("jobs", [])
    requests = [r for r in rows(root / "model-requests.jsonl") if r["event"] == "request_terminal"]
    resources = rows(root / "resources.jsonl")
    tools = [r["params"]["item"] for r in rows(root / "guest/app/codex-events.jsonl")
             if r.get("method") == "item/completed" and r["params"]["item"].get("type") == "mcpToolCall"]
    steps = [dict(job_id=j["id"], **s) for j in jobs for s in j["steps"]]
    checks = guest.get("grade", {}).get("checks", {})
    return {
        "attempt": root.name, "status": result["status"], "passed": result["passed"],
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "execution_mode": result["execution_mode"], "model_used": result["model_used"],
        "advisor_used": result["advisor_used"], "bundle_sha256": result.get("native_bundle", {}).get("archive", {}).get("sha256"),
        "engine_binary_sha256": result.get("engine", {}).get("binary_sha256"),
        "kvmem": result.get("engine", {}).get("kvmem"), "thinking": task.get("thinking"),
        "outcome": guest.get("task", {}).get("outcome"), "checks": checks,
        "task_seconds": task.get("finished_at", 0) - task.get("started_at", 0) if task.get("finished_at") else None,
        "host_total_seconds": result.get("finished_at", 0) - result["started_at"] if result.get("finished_at") else None,
        "inference": latency([(r["host_finished_monotonic_ns"] - r["host_received_monotonic_ns"]) / 1e9 for r in requests]),
        "first_byte": latency([(r["first_response_byte_monotonic_ns"] - r["host_received_monotonic_ns"]) / 1e9
                                for r in requests if r.get("first_response_byte_monotonic_ns")]),
        "tool_calls": len(tools), "tool_errors": sum(t["status"] == "failed" for t in tools),
        "wait_argument_type_errors": sum(t.get("arguments", {}).get("operation") == "job_status"
                                        and type(t["arguments"].get("wait_seconds", 5)) is not int for t in tools),
        "jobs": [{k: j.get(k) for k in ("id", "status", "attempt", "stage", "error", "needs_reconciliation")} for j in jobs],
        "stages": [{"name": s["name"], "attempt": s["attempt"], "status": s["status"],
                    "seconds": s.get("finished_at", 0) - s["started_at"] if s.get("finished_at") else None}
                   for s in steps],
        "device_gpu_peak_sampled_bytes": max((r.get("gpu_memory_bytes", 0) for r in resources), default=None),
        "engine_rss_peak_sampled_bytes": max((r.get("rss_bytes", 0) for r in resources), default=None),
        "evidence_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "video_seconds": result.get("recording", {}).get("video_duration_seconds"),
        "grade_error": guest.get("grade", {}).get("error"),
        "limitations": ["Inference and background installation overlap; their times are not additive.",
                        "RSS samples the host engine process; GPU memory samples the device. Guest/application memory is not included.",
                        "First byte is the HTTP streaming response's first chunk, not guaranteed first generated text token.",
                        "This is controlled-software-tool assistance, not a pure GUI or general reasoning score."]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"kind": "software_product_evidence_summary", "trials": [summarize(p) for p in args.runs]}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
