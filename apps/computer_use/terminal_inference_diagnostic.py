"""Summarize terminal model streams without exporting any reasoning text."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(campaign, queue, output):
    campaign, queue, output = (Path(p).resolve() for p in (campaign, queue, output))
    if queue not in (campaign / "queue.json", campaign / "continued_queue.json"):
        raise ValueError("Use the declared original or continued campaign queue")
    queue_bytes = queue.read_bytes()
    state = json.loads(queue_bytes)
    if state["protocol_sha256"] != digest(campaign / "protocol.json"):
        raise ValueError("Frozen campaign protocol changed")
    protocol = json.loads((campaign / "protocol.json").read_text())
    declared = protocol["trials"]
    identities = [row["id"] for row in declared]
    order = protocol.get("execution_order", identities)
    if len(order) != len(identities) or set(order) != set(identities):
        raise ValueError("Declared execution order differs from the frozen trials")
    if [row["id"] for row in state["trials"]] != order[:len(state["trials"])]:
        raise ValueError("Terminal rows differ from declared trial order")
    results = []
    for row in state["trials"]:
        attempt = campaign / "attempts" / row["id"]
        result_path = attempt / "result.json"; result_sha = digest(result_path)
        business = json.loads(result_path.read_text())
        if business != row["result"]: raise ValueError("Recorded terminal result differs")
        records = []
        for metric_path in (attempt / "app/inference").glob("*/metrics.json"):
            metric = json.loads(metric_path.read_text()); backend = metric_path.parent / "backend.jsonl"
            if not backend.is_file(): raise ValueError("Model metrics lack their original stream")
            counts, finishes = collections.Counter(), collections.Counter()
            with backend.open() as stream:
                for line in stream:
                    chunk = json.loads(line)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        for key in ("reasoning_content", "content"):
                            if isinstance(delta.get(key), str): counts[key + "_characters"] += len(delta[key])
                        if isinstance(delta.get("tool_calls"), list): counts["tool_call_delta_events"] += len(delta["tool_calls"])
                        if choice.get("finish_reason") is not None: finishes[choice["finish_reason"]] += 1
            records.append({"trace": metric_path.parent.name, "metrics_sha256": digest(metric_path),
                "backend_stream_sha256": digest(backend), "started_at": metric["started_at"],
                "status": metric["status"], "elapsed_seconds": metric.get("elapsed_seconds"),
                "first_model_delta_seconds": metric.get("first_model_delta_seconds"), "usage": metric.get("usage"),
                "stream_counts": dict(counts), "finish_reasons": dict(finishes),
                "length_stop_without_action_or_answer": bool(finishes["length"] and counts["reasoning_content_characters"]
                    and not counts["content_characters"] and not counts["tool_call_delta_events"])})
        records.sort(key=lambda item: item["started_at"])
        elapsed = sum(r["elapsed_seconds"] for r in records if isinstance(r["elapsed_seconds"], (int, float)))
        if digest(result_path) != result_sha: raise ValueError("Terminal result changed during read-only analysis")
        results.append({"trial": row["id"], "terminal_result_sha256": result_sha, "official_score": business.get("score"),
            "official_status": business.get("status"), "business_success": business.get("unassisted_business_passed", False),
            "task_seconds": business.get("seconds"), "tool_calls": business.get("tool_calls"),
            "trace_count": len(records), "summed_model_seconds": elapsed,
            "length_stops_without_action_or_answer": sum(r["length_stop_without_action_or_answer"] for r in records),
            "records": records})
    report = {"kind": "read_only_terminal_inference_stage_diagnostic", "at": time.time(), "passed": True,
        "queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(), "protocol_sha256": state["protocol_sha256"],
        "recipe_sha256": digest(Path(__file__)), "terminal_trials": len(results), "results": results,
        "model_calls_dispatched": 0, "device_inputs": 0, "reasoning_text_exported": False,
        "limits": ["Character counts do not estimate reasoning tokens; usage tokens describe the complete backend output.",
                   "Long reasoning may follow visual or planning confusion; this does not isolate model capability.",
                   "Model durations include image encoding/prefill/decoding and shared-resource effects.",
                   "No runtime policy was changed and no latency or success improvement was demonstrated."]}
    with output.open("x") as file: file.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "queue", "output"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); report = diagnose(args.campaign, args.queue, args.output)
    print(json.dumps({"terminal_trials":report["terminal_trials"], "model_calls_dispatched":0,
        "reasoning_only_length_stop_trials":[r["trial"] for r in report["results"] if r["length_stops_without_action_or_answer"]]}))


if __name__ == "__main__": main()
