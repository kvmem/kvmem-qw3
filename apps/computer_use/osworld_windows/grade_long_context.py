"""Host-only evidence audit for actual-model native compaction diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

from .source import sha256


def grade(directory):
    directory = Path(directory)
    report = json.loads((directory / "result.json").read_text())
    native = json.loads((directory / "native/result.json").read_text())
    observations_path = directory / "native/observations.jsonl"
    observations = [json.loads(line) for line in observations_path.read_text().splitlines()]
    events_path = directory / "native/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    expected_count = report["observations_requested"]
    expected = dict(observations=expected_count,
        first_marker=hashlib.sha256((directory.name + ":1").encode()).hexdigest()[:16],
        latest_marker=hashlib.sha256((directory.name + ":" + str(expected_count)).encode()).hexdigest()[:16])
    receipts = {row["backend_trace_tag"]: row for line in (directory / "relay/requests.jsonl").read_text().splitlines()
                if (row := json.loads(line)).get("event") == "request_terminal" and row.get("backend_trace_tag")}
    requests = []
    source_hashes = {"result.json": sha256(directory / "result.json"),
                     "native/result.json": sha256(directory / "native/result.json"),
                     "native/observations.jsonl": sha256(observations_path),
                     "native/events.jsonl": sha256(events_path)}
    for path in (directory / "native/guest-runtime/inference").glob("*/request.json"):
        body = json.loads(path.read_text())
        metrics_path = path.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text())
        last = body["messages"][-1]
        content = last.get("content", "")
        text = content if isinstance(content, str) else "\n".join(part.get("text", "") for part in content)
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        receipt = receipts.get(body.get("kvmem_trace_tag"), {})
        requests.append(dict(id=path.parent.name, started_at=metrics["started_at"],
            status=metrics["status"], input_tokens=metrics.get("usage", {}).get("input_tokens", 0),
            tool_count=len(body.get("tools", [])),
            is_compaction=not body.get("tools") and last["role"] == "user" and "CONTEXT CHECKPOINT COMPACTION" in text,
            receipt_bound=receipt.get("canonical_request_sha256") == hashlib.sha256(canonical).hexdigest() and
                          receipt.get("status") == "complete"))
        for source in (path, metrics_path):
            source_hashes[str(source.relative_to(directory))] = sha256(source)
    requests.sort(key=lambda row: row["started_at"])
    compactions = [row for row in requests if row["is_compaction"]]
    completed_compactions = [event for event in events if event.get("method") == "item/completed" and
        event.get("params", {}).get("item", {}).get("type") == "contextCompaction"]
    post = [row for row in requests if compactions and row["started_at"] > compactions[-1]["started_at"] and row["tool_count"]]
    engine_directory = directory.parent / "engine"
    engine_manifest = engine_directory / "manifest.json"
    traces = {}
    retrieval = engine_directory / "retrieval.jsonl"
    if retrieval.exists():
        with retrieval.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") == "meta":
                    traces.setdefault(row.get("trace_tag"), set()).add(row.get("prompt_tokens"))
        source_hashes[str(retrieval.relative_to(directory.parent))] = sha256(retrieval)
    engine_bound = (engine_manifest.exists() and json.loads(engine_manifest.read_text()) == report.get("engine") and
                    report.get("engine", {}).get("binary_sha256") is not None)
    if engine_manifest.exists():
        source_hashes["engine/manifest.json"] = sha256(engine_manifest)
    checks = dict(actual_model_declared=native.get("real_long_context_inference") is True,
        engine_manifest_bound=engine_bound,
        native_windows_codex=report.get("launch_receipt", {}).get("started") is True and native.get("initial_health", {}).get("platform") == "windows",
        observation_count_and_markers=(len(observations) == expected_count and all(row["sequence"] == index and
            row["marker"] == hashlib.sha256((directory.name + ":" + str(index)).encode()).hexdigest()[:16]
            for index, row in enumerate(observations, 1))),
        summary_retains_first_and_latest=native.get("answer") == expected,
        near_1m_compaction_request=any(row["status"] == "completed" and row["input_tokens"] >= 900000 for row in compactions),
        compaction_usage_matches_engine=(report.get("engine", {}).get("kvmem") != "on" or
            any(row["input_tokens"] in traces.get(row["id"], set()) for row in compactions)),
        compaction_event_completed=bool(completed_compactions),
        tools_restored_with_reduced_context=bool(post) and all(row["status"] == "completed" for row in post) and
            min((row["input_tokens"] for row in post), default=1048576) < 200000,
        all_backend_requests_bound=bool(requests) and all(row["receipt_bound"] for row in requests),
        native_turn_completed=native.get("turn", {}).get("status") == "completed",
        continuous_recording_complete=report.get("recording", {}).get("complete") is True,
        desktop_remains_paused=report.get("final_health", {}).get("paused") is True)
    result = dict(kind="independent_actual_model_compaction_audit", formal_run=False,
                  passed=all(checks.values()), checks=checks, requests=requests, source_sha256=source_hashes,
                  expected=expected, grader_sha256=sha256(Path(__file__)),
                  limitation="Synthetic padded live observations qualify the component; not task success or representative latency.")
    (directory / "independent-grade.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    result = grade(parser.parse_args().directory)
    print(json.dumps(dict(passed=result["passed"], checks=result["checks"])))
    raise SystemExit(0 if result["passed"] else 1)
