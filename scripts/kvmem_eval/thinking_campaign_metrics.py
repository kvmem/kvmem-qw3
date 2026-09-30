"""Per-request and per-trial metrics for the thinking compaction campaign."""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def fields(line):
    result = {}
    for key, value in re.findall(r'([a-z_][a-z_0-9]*)=("(?:\\.|[^"\\])*"|[^\s]+)', line):
        try:
            result[key] = json.loads(value)
        except (ValueError, TypeError):
            result[key] = value
    return result


class RequestParser:
    """Associate serialized engine snapshots with the following HTTP accounting.

    Host observation time is sampled, not an exact engine timestamp. Request
    ordinal and accounting durations remain exact to the precision of the log.
    """
    def __init__(self):
        self.pending = {}
        self.rows = []
        self.offset = 0
        self.partial = b""

    def line(self, line, elapsed=None):
        if "[thinking-summary-output]" in line:
            item = json.loads(line.split("[thinking-summary-output]", 1)[1])
            # Full text remains in the server log; keep compact diagnostics in
            # the metric trace rather than duplicating long generated payloads.
            self.pending.setdefault("summary_outputs", []).append(
                {key: value for key, value in item.items() if key != "text"})
        elif "kvmem bounded history=" in line:
            self.pending["pool"] = fields(line)
        elif "[thinking-compaction]" in line:
            event = fields(line)
            outputs = self.pending.get("summary_outputs", [])
            if outputs and "summary_masked_steps" not in outputs[-1]:
                event["summary_masked_steps"] = outputs[-1].get("masked_steps", 0)
                outputs[-1]["summary_masked_steps"] = event["summary_masked_steps"]
            self.pending.setdefault("compactions", []).append(event)
        elif "kvmem prefix-cache HIT (" in line or "kvmem prefix-cache MISS (why)" in line:
            self.pending["prefix_cache"] = {
                "hit": "prefix-cache HIT" in line, **fields(line)}
        elif "kvmem prefix-cache CAPTURE (" in line:
            self.pending["prefix_capture"] = fields(line)
        elif "native generate:" in line:
            item = fields(line)
            for key in ("prefill", "decode"):
                if key in item:
                    item[key + "_sec"] = float(str(item.pop(key)).removesuffix("s"))
            self.pending["generation"] = item
        elif "[qw3-server-ttft]" in line:
            self.pending["ttft"] = fields(line)
        elif "[qw3-server-accounting]" in line:
            row = {"request_index": len(self.rows) + 1,
                   "observed_elapsed_sec": elapsed, **self.pending,
                   "accounting": fields(line)}
            gen = row.get("generation", {})
            if "prompt_tokens" in gen and "decoded" in gen:
                row["output_context_tokens"] = gen["prompt_tokens"] + gen["decoded"]
            self.rows.append(row)
            self.pending = {}
            return row
        return None

    def poll(self, path, elapsed=None):
        if not path.exists():
            return []
        with path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
            self.offset = stream.tell()
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        new = []
        for line in lines:
            row = self.line(line.decode(errors="replace"), elapsed)
            if row is not None:
                new.append(row)
        return new


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def request_csv(rows):
    result = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for section in ("pool", "generation", "ttft", "accounting", "prefix_cache", "prefix_capture"):
            flat.update({f"{section}_{key}": value for key, value in row.get(section, {}).items()})
        for event in row.get("compactions", []):
            flat.update({f"thinking_{key}": value for key, value in event.items()})
        result.append(flat)
    return result


def http_error_audit(text, artifact):
    """Recognize only evidenced, recovered unsupported WebSearch requests.

    These requests fail tool-schema validation before entering the engine. The
    client returns the error as a tool result and continues the coding session.
    Unknown 400s, primary inference failures and all 5xx remain fatal.
    """
    errors = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r'\[qw3-serve\] POST (\S+) -> ([45]\d\d)', line)
        if match:
            errors.append({'server_line': number, 'path': match[1], 'status': int(match[2])})
    audit = {'http_errors': errors, 'recovered_web_search': [], 'recovered_server_lines': []}
    candidates = [e for e in errors if e['path'] == '/v1/messages' and e['status'] == 400]
    debug_path, stream_path = artifact / 'claude.debug.log', artifact / 'claude.stream.jsonl'
    if not candidates or not debug_path.is_file() or not stream_path.is_file():
        return audit
    message = 'each tool requires string name and object input_schema'
    expected = {'error': {'message': message, 'type': 'invalid_request_error'}, 'type': 'error'}
    proofs, active, pending, api_errors = [], None, None, 0
    for number, line in enumerate(debug_path.read_text(errors='replace').splitlines(), 1):
        start = re.search(r'\[Stall\] tool_dispatch_start tool=(\S+) toolUseId=(\S+)', line)
        if start:
            pending = None
            active = {'tool_use_id': start[2], 'debug_start_line': number, 'sources': [], 'errors': []} if start[1] == 'WebSearch' else None
        source = re.search(r'\[API REQUEST\] /v1/messages source=(\S+)', line)
        if source:
            if active is not None:
                active['sources'].append(source[1])
            if pending is not None:
                pending['resume_source'] = source[1]
        error = re.search(r'\[ERROR\] API error \(attempt \d+/\d+\): (.*)', line)
        if error:
            api_errors += 1
            pending = None
            if active is not None:
                active['errors'].append(error[1])
        end = re.search(r'\[Stall\] tool_dispatch_end tool=WebSearch toolUseId=(\S+) outcome=(\S+)', line)
        if end and active is not None and end[1] == active['tool_use_id']:
            if end[2] == 'ok' and active['sources'] == ['web_search_tool'] and len(active['errors']) == 1:
                status, _, body = active['errors'][0].partition('400 400 ')
                try:
                    if not status and json.loads(body) == expected:
                        pending = {'tool_use_id': active['tool_use_id'],
                                   'debug_start_line': active['debug_start_line'], 'debug_end_line': number}
                except ValueError:
                    pass
            active = None
        if pending and pending.get('resume_source') == 'sdk' and '[DEBUG] Stream started - received first chunk' in line:
            pending['debug_resume_line'] = number
            proofs.append(pending)
            pending = None
    # Require one-to-one corroboration, including the tool result actually
    # delivered to the main agent. Never subtract a guessed number of errors.
    if len(proofs) != len(candidates) or api_errors != len(proofs):
        return audit
    by_id = {proof['tool_use_id']: proof for proof in proofs}
    if len(by_id) != len(proofs):
        return audit
    with stream_path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not any(tool_id in line for tool_id in by_id):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                return audit
            for block in event.get('message', {}).get('content', []):
                if not isinstance(block, dict):
                    continue
                if event.get('type') == 'assistant' and block.get('type') == 'tool_use' and block.get('name') == 'WebSearch' and block.get('id') in by_id:
                    by_id[block['id']]['stream_call_line'] = number
                if event.get('type') == 'user' and block.get('type') == 'tool_result' and block.get('tool_use_id') in by_id:
                    proof = by_id[block['tool_use_id']]
                    content = block.get('content')
                    if isinstance(content, str) and content.startswith('Web search results for query: ') and f'\n\nAPI Error: 400 {message}\n' in content and proof.get('stream_call_line', number) < number:
                        proof['stream_result_line'] = number
    if all('stream_result_line' in proof for proof in proofs):
        audit['recovered_web_search'] = proofs
        audit['recovered_server_lines'] = [e['server_line'] for e in candidates]
    return audit


def invalid_reasons(metrics, text, *, http_audit=None):
    recovered = set((http_audit or {}).get('recovered_server_lines', []))
    text = '\n'.join(line for number, line in enumerate(text.splitlines(), 1)
                     if '[thinking-summary-output]' not in line and number not in recovered)
    reasons = []
    if not metrics:
        return ["missing_metrics"]
    for key in ("agent_exit", "verifier_exit"):
        if metrics.get(key) != 0:
            reasons.append(key)
    claude = metrics.get("claude", {})
    if claude.get("terminal_subtype") != "success":
        reasons.append("agent_abnormal_termination")
    for key in ("api_errors", "private_query_leak_count", "subagent_tool_calls"):
        if claude.get(key, 0):
            reasons.append(key)
    for key in ("network_isolation", "dependency_preflight"):
        if not metrics.get(key, {}).get("passed"):
            reasons.append(key)
    server = metrics.get("server", {})
    if server.get("stream_terminal_errors") or re.search(r"terminal_status=error|POST .* -> [45]\d\d", text):
        reasons.append("inference_or_http_error")
    reward = metrics.get("reward") or {}
    if reward.get("reward") not in (0, 1) or any(type(reward.get(key)) is not int
            for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")):
        reasons.append("missing_verifier_counts")
    return reasons


def trial_metrics(metrics, requests):
    pools = [row["pool"] for row in requests if "pool" in row]
    events = [event for row in requests for event in row.get("compactions", [])]
    generations = [row["generation"] for row in requests if "generation" in row]
    final = requests[-1] if requests else {}
    after_compaction = [current for previous, current in zip(requests, requests[1:])
        if any(e.get("status") == "compacted" for e in previous.get("compactions", []))]
    reward = metrics.get("reward") or {}
    server = metrics.get("server") or {}
    result = {
        "passed": reward.get("reward") == 1 and all(
            reward.get(a) == reward.get(b) for a, b in
            (("f2p_passed", "f2p_total"), ("p2p_passed", "p2p_total"))),
        "reward": reward.get("reward"),
        "agent_elapsed_sec": metrics.get("agent_elapsed_sec"),
        "total_elapsed_sec": metrics.get("total_elapsed_sec"),
        "request_wall_sec": server.get("request_wall_sec"),
        "prefill_sec": server.get("prefill_sec"),
        "decode_sec": server.get("decode_sec"),
        "non_server_wall_sec": server.get("non_server_wall_sec"),
        "request_count": len(requests),
        "tool_calls": metrics.get("claude", {}).get("tool_calls"),
        "prefix_cache_hits": server.get("prefix_cache_hits"),
        "post_compaction_requests": len(after_compaction),
        "post_compaction_cache_hits": sum(bool(r.get("prefix_cache", {}).get("hit")) for r in after_compaction),
        "post_compaction_query_snapshot_reuse": sum(
            bool(r.get("prefix_cache", {}).get("hit")) and r["prefix_cache"].get("query_snapshot") == 1
            for r in after_compaction),
        "post_compaction_prefilled_tokens": sum(r.get("generation", {}).get("prefilled", 0) for r in after_compaction),
        "guided_query_sec": server.get("guided_query_sec"),
        "final_prompt_tokens": final.get("generation", {}).get("prompt_tokens"),
        "final_context_tokens": final.get("output_context_tokens"),
        "final_committed_context_tokens": final.get("pool", {}).get("history"),
        "peak_context_tokens": max((row.get("output_context_tokens", 0) for row in requests), default=None),
        "visible_decoded_tokens": sum(row.get("decoded", 0) for row in generations),
        "compacted_requests": sum(e.get("status") == "compacted" for e in events),
        "compaction_status_counts": dict(Counter(e.get("status") for e in events)),
        "discarded_thinking_tokens": sum(e.get("original_tokens", 0) - e.get("retained_tokens", 0)
            for e in events if e.get("status") == "compacted"),
        "summary_decode_tokens": sum(e.get("summary_decode_tokens", 0) for e in events),
        "summary_masked_steps": sum(e.get("summary_masked_steps", 0) for e in events),
        "pool_snapshots": len(pools),
    }
    result.update({key: reward.get(key) for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")})
    for key in ("summary_ms", "rollback_ms", "replay_ms"):
        result[key.removesuffix("_ms") + "_sec"] = sum(e.get(key, 0) for e in events) / 1000
    result["original_target_decoded_tokens"] = result["visible_decoded_tokens"] + result["discarded_thinking_tokens"]
    for key in ("slots", "live", "ready", "writing", "retired", "releasing", "cpu_bytes", "gpu_kv_bytes",
                "index_bytes", "prefix_bytes", "retained_tokens", "gc_metadata_bytes"):
        result["final_" + key] = final.get("pool", {}).get(key)
        result["peak_" + key] = max((p[key] for p in pools if key in p), default=None)
    result["minimum_ready_slots"] = min((p["ready"] for p in pools if "ready" in p), default=None)
    times = [r["ttft"]["response_ttft_ms"] for r in requests
             if r.get("ttft", {}).get("response_ttft_ms", -1) >= 0]
    result["response_ttft_median_ms"] = statistics.median(times) if times else None
    return result


def summarize_rows(rows, arms):
    samples, pairs = {}, []
    for row in rows:
        samples.setdefault(row["task"], {}).setdefault(row["arm"], []).append(row)
    for task, groups in samples.items():
        if not {"baseline", "summary"} <= groups.keys():
            continue
        baseline = {r["round"]: r for r in groups["baseline"]}
        for summary in groups["summary"]:
            base = baseline.get(summary["round"])
            if base is None:
                continue
            pair = {"task": task, "round": summary["round"], "seed": summary["seed"],
                    "baseline_passed": base["passed"], "summary_passed": summary["passed"],
                    "both_passed": base["passed"] and summary["passed"],
                    "same_binary": base["binary_sha256"] == summary["binary_sha256"]}
            for key in ("agent_elapsed_sec", "request_wall_sec", "final_context_tokens", "final_live",
                        "final_ready", "final_gpu_kv_bytes", "final_cpu_bytes", "peak_live"):
                a, b = base.get(key), summary.get(key)
                pair[key + "_delta"] = b - a if a is not None and b is not None else None
                pair[key + "_change_pct"] = (b / a - 1) * 100 if a and b is not None else None
            pairs.append(pair)
    per_sample = []
    for task, groups in samples.items():
        for arm in arms:
            entries = groups.get(arm, [])
            row = {"task": task, "arm": arm, "completed_rounds": len(entries),
                   "successes": sum(e["passed"] for e in entries),
                   "pass_at_4": any(e["passed"] for e in entries) if len(entries) == 4 else None}
            for key in ("agent_elapsed_sec", "final_context_tokens", "final_live", "final_ready",
                        "final_gpu_kv_bytes", "final_cpu_bytes"):
                values = [e[key] for e in entries if e.get(key) is not None]
                row[key + "_mean"] = statistics.mean(values) if values else None
                row[key + "_median"] = statistics.median(values) if values else None
            per_sample.append(row)
    return per_sample, pairs
