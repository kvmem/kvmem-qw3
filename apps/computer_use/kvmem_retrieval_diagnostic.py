"""Read native KVMem routing evidence without exporting task or query text."""
import argparse
import hashlib
import json
from pathlib import Path
import re


class EvidenceError(ValueError):
    pass


def fields(text):
    result = {}
    for key, value in re.findall(r"([a-z_]+)=([^\s]+)", text):
        if key in result:
            raise EvidenceError(f"Duplicate metadata field: {key}")
        result[key] = value
    return result


def integer(values, key):
    value = values.get(key, "")
    if not re.fullmatch(r"[0-9]+", value):
        raise EvidenceError(f"Missing or invalid numeric metadata: {key}")
    return int(value)


def summarize(events):
    return dict(
        scorer_events=len(events),
        window_fallback_events=sum(e["used"] == "window-profile" and e["fallback"] == 1 for e in events),
        successful_query_events=sum(e["fallback"] == 0 and e["query_ready"] == 1
                                    and e["used"] == e["requested"] for e in events),
        captured_query_events=sum(e["query_captured"] > 0 for e in events),
        identity_window_events=sum(e["selected_blocks"] == e["source_blocks"] for e in events),
    )


def analyze(data, allow_partial=False):
    if not isinstance(data, bytes):
        raise TypeError("Analyze an exact log byte snapshot")
    incomplete_tail = bool(data and not data.endswith(b"\n"))
    if incomplete_tail and not allow_partial:
        raise EvidenceError("Incomplete log line; retry the same live producer")
    lines = data.splitlines()
    if incomplete_tail:
        lines = lines[:-1]
    requests, events = [], []
    pending_gate, current = None, None
    sequences = set()
    for number, raw in enumerate(lines, 1):
        line = raw.decode("utf-8", errors="replace")
        if line.startswith("[qw3-serve] incoming tools="):
            pending_gate = None
        elif line.startswith("[qw3-serve] KVMem harness continuation gate "):
            gate = fields(line)
            pending_gate = dict(reselect=gate.get("reselect"), line=number,
                                headroom_grace=integer(gate, "headroom_grace"))
        elif line.startswith("[qw3-serve] KVMem request plan stage=draft "):
            plan = fields(line)
            current = dict(ordinal=len(requests) + 1, line=number,
                           prompt_tokens=integer(plan, "prompt_tokens"),
                           selection_budget=integer(plan, "selection_budget"),
                           generation_budget=integer(plan, "generation_budget"),
                           draft_action=plan["action"], continuation_gate=pending_gate,
                           query_span=None, final_action=None, pressure_ingest=None,
                           native_query_capture_modes=[], events=[])
            pending_gate = None
            requests.append(current)
        elif line.startswith("[qw3-serve] kvmem query span "):
            match = re.fullmatch(r"\[qw3-serve\] kvmem query span \[([0-9]+),([0-9]+)\) of ([0-9]+) prompt tokens", line)
            if current is None or match is None:
                raise EvidenceError("Query metadata lacks a matching request plan")
            begin, end, size = map(int, match.groups())
            if size != current["prompt_tokens"] or not 0 <= begin < end <= size:
                raise EvidenceError("Query metadata differs from the request plan")
            current["query_span"] = [begin, end]
        elif line.startswith("[qw3] KVMem request plan stage=final "):
            if current is None:
                raise EvidenceError("Final plan lacks a draft request")
            plan = fields(line)
            current.update(final_action=plan["action"], pressure_ingest=integer(plan, "pressure_ingest"))
        elif line.startswith("[qw3] native kvmem query-conditioned: "):
            if current is None:
                raise EvidenceError("Query capture lacks a draft request")
            current["native_query_capture_modes"].append(fields(line).get("mode"))
        elif line.startswith("[kvmem-scorer]"):
            if current is None:
                raise EvidenceError("Scorer event lacks a draft request")
            values = fields(line)
            event = {key: integer(values, key) for key in (
                "seq", "position", "fallback", "query_expected", "query_captured", "query_ready",
                "indexed_blocks", "source_blocks", "selected_blocks")}
            for key in ("requested", "used", "reason"):
                if key not in values:
                    raise EvidenceError(f"Scorer metadata missing: {key}")
                event[key] = values[key]
            if event["seq"] in sequences:
                raise EvidenceError("Duplicate scorer sequence; engine lineage is ambiguous")
            sequences.add(event["seq"])
            # expected/captured are row counts; only readiness/fallback are flags.
            if any(event[k] not in (0, 1) for k in ("fallback", "query_ready")):
                raise EvidenceError("Invalid scorer state flag")
            event["line"] = number
            current["events"].append(event)
            events.append(event)
    pressure_events = []
    for request in requests:
        request["above_selection_budget"] = request["prompt_tokens"] > request["selection_budget"]
        request["summary"] = summarize(request["events"])
        request["gate_suppressed_query_selection"] = bool(
            request["above_selection_budget"] and request["query_span"]
            and request["continuation_gate"]
            and request["continuation_gate"]["reselect"] == "off"
            and request["pressure_ingest"] == 1
            and request["events"]
            and all(e["query_expected"] == 0 and e["query_captured"] == 0 for e in request["events"]))
        if request["above_selection_budget"]:
            pressure_events.extend(request["events"])
        # Export counts and exact structural evidence, never raw task/query text.
        request["scorer_sequences"] = [e["seq"] for e in request.pop("events")]
    return dict(kind="readonly_native_kvmem_retrieval_routing_diagnostic",
                snapshot_sha256=hashlib.sha256(data).hexdigest(), snapshot_bytes=len(data),
                incomplete_tail_ignored=incomplete_tail, scope="log_snapshot_not_business_grade",
                summary=summarize(events), above_budget_summary=summarize(pressure_events),
                above_budget_requests=sum(r["above_selection_budget"] for r in requests),
                gated_pressure_requests=sum(r["gate_suppressed_query_selection"] for r in requests),
                requests=requests, model_requests_submitted=0, device_inputs=0,
                business_or_latency_improvement_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.log.read_bytes(), args.allow_partial)
    result["source_log"] = str(args.log.absolute())
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({key: result[key] for key in ("summary", "above_budget_summary", "above_budget_requests", "gated_pressure_requests")}))


if __name__ == "__main__":
    main()
