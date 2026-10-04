"""Summarize terminal local computer-use traces without exporting their content.

Only timing, token counts, image counts, channel lengths and finish reasons are
exported. These are descriptive development diagnostics, not causal ablations.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_count(value):
    if isinstance(value, list):
        return sum(image_count(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "input_image"}:
            return 1
        return sum(image_count(item) for item in value.values())
    return 0


def archive_image_count(node):
    """Count short type markers without decoding private image/text blobs."""
    if set(node) == {"d"}:
        fields = dict(node["d"])
        if fields.get("type", {}).get("s") in {"image_url", "input_image"}:
            return 1
        return sum(archive_image_count(value) for value in fields.values())
    if set(node) == {"l"}:
        return sum(archive_image_count(value) for value in node["l"])
    if set(node) in ({"s"}, {"p"}, {"blob"}):
        return 0
    raise ValueError("Unsupported request archive node")


def backend_summary(path):
    characters, deltas, finishes = Counter(), Counter(), Counter()
    timing = None
    if not path.exists():
        return {"available": False}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            for choice in value.get("choices", []):
                for name, delta in choice.get("delta", {}).items():
                    deltas[name] += 1
                    if isinstance(delta, str) and name != "role":
                        characters[name] += len(delta)
                if choice.get("finish_reason"):
                    finishes[choice["finish_reason"]] += 1
            if "timing" in value:
                timing = {k: v for k, v in value["timing"].items()
                          if k in {"engine_ttft_sec", "request_total_sec", "response_ttft_sec", "server_ttft_sec"}}
    return {"available": True, "delta_occurrences": dict(deltas), "channel_characters": dict(characters),
            "finish_reasons": dict(finishes), "timing": timing,
            "reasoning_only_length_stop": finishes.get("length", 0) > 0
            and characters.get("reasoning_content", 0) > 0
            and characters.get("content", 0) == 0 and deltas.get("tool_calls", 0) == 0}


def describe(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "sum": sum(values), "median": statistics.median(values),
            "p95_nearest_rank": ordered[math.ceil(len(ordered) * .95) - 1],
            "min": ordered[0], "max": ordered[-1]}


def summarize(attempt):
    attempt = attempt.resolve()
    result_path = attempt / "result.json"
    result = json.loads(result_path.read_text())
    if not result.get("finished_at"):
        raise ValueError("Summarize terminal attempts only")
    manifest = json.loads((attempt / "manifest.json").read_text())
    budget = manifest.get("model_output_tokens")
    rows = []
    for path in (attempt / "app/inference").glob("*/metrics.json"):
        metrics = json.loads(path.read_text())
        archive_path = path.parent / "request.archive.json"
        archive = json.loads(archive_path.read_text()) if archive_path.exists() else None
        if archive is not None and archive.get("version") != 1:
            raise ValueError("Unsupported private archive format")
        stream = backend_summary(path.parent / "backend.jsonl")
        usage = metrics.get("usage") or {}
        rows.append({"request_id": path.parent.name, "metrics_sha256": sha(path),
                     "request_archive_sha256": sha(archive_path) if archive is not None else None,
                     "started_at": metrics.get("started_at"), "status": metrics.get("status"),
                     "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                     "input_images": archive_image_count(archive["tree"]) if archive is not None else None,
                     "first_model_delta_seconds": metrics.get("first_model_delta_seconds"),
                     "elapsed_seconds": metrics.get("elapsed_seconds"), "stream": stream,
                     "reached_output_budget": type(budget) is int and usage.get("output_tokens") == budget})
    rows.sort(key=lambda row: row["started_at"] or 0)
    def series(key):
        return [row[key] for row in rows if type(row[key]) in (int, float)]
    elapsed = describe(series("elapsed_seconds"))
    duration = result.get("seconds")
    return {"attempt": attempt.name, "path": str(attempt), "result_sha256": sha(result_path),
            "manifest_sha256": sha(attempt / "manifest.json"), "model_output_tokens": budget,
            "task_seconds": duration, "agent_status": result.get("agent_status"),
            "official_score": result.get("score"), "uncertain_actions": result.get("uncertain_actions"),
            "requests": len(rows), "request_statuses": dict(Counter(row["status"] for row in rows)),
            "input_tokens": describe(series("input_tokens")), "output_tokens": describe(series("output_tokens")),
            "first_model_delta_seconds": describe(series("first_model_delta_seconds")),
            "elapsed_seconds": elapsed,
            "backend_elapsed_over_task_duration": elapsed["sum"] / duration if elapsed and duration else None,
            "first_request_input_tokens": rows[0]["input_tokens"] if rows else None,
            "last_request_input_tokens": rows[-1]["input_tokens"] if rows else None,
            "first_request_images": rows[0]["input_images"] if rows else None,
            "last_request_images": rows[-1]["input_images"] if rows else None,
            "output_budget_hits": sum(row["reached_output_budget"] for row in rows),
            "reasoning_only_length_stops": sum(row["stream"].get("reasoning_only_length_stop", False) for row in rows),
            "trace": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic file; never overwrite an earlier result")
    result = {"kind": "read_only_terminal_model_trace_diagnostic", "created_at": time.time(),
              "analyzer_sha256": sha(Path(__file__)), "model_requests_dispatched": 0, "device_input_sent": False,
              "raw_prompts_outputs_or_images_exported": False,
              "limitations": ["First model delta includes backend work; engine timing is reported separately when available.",
                              "Serial trace totals describe observed overhead and do not isolate prefill, thinking or shared GPU effects.",
                              "Image counts and channel characters are descriptive; they do not measure visual accuracy."],
              "attempts": [summarize(attempt) for attempt in args.attempt]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"attempts": len(result["attempts"]), "output": str(args.output),
                      "reasoning_only_length_stops": sum(row["reasoning_only_length_stops"] for row in result["attempts"])}))


if __name__ == "__main__":
    main()
