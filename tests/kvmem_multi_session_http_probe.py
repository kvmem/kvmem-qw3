#!/usr/bin/env python3
"""Manual GPU qualification for bounded KVMem histories exceeding active slots.

Launches only its own two sequential localhost servers. The reference keeps
each session hot with the new feature disabled; the candidate interleaves more
histories than GPU slots. This checks exact text/usage, real resume and prefix
reuse, conflicting writers, cancellation, invalidation and explicit deletion.
It is a functional probe, not a throughput benchmark. Use --dry-run without a GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def json_call(base, path, body=None, *, method=None, request_id=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["x-qw3-request-id"] = request_id
    call = urllib.request.Request(
        base + path, data=None if body is None else json.dumps(body).encode(),
        headers=headers, method=method)
    try:
        with opener().open(call, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        return error.code, payload


def make_messages(session):
    color, animal = [("amber", "otter"), ("violet", "heron"),
                     ("indigo", "lynx"), ("silver", "ibis")][session % 4]
    notes = "\n".join(
        f"Entry {i}: session {session} uses {color}; the animal is {animal}; count {i * 7 + session}."
        for i in range(24))
    return [{"role": "system", "content": "Read this session's records. Answer briefly and accurately."},
            {"role": "user", "content": notes +
             f"\nWhat color and animal belong to session {session}? Explain in one sentence."}]


def body_for(key, messages, max_tokens=32, stream=False):
    return {"model": "test", "messages": messages,
            "kvmem_prefix_cache_key": key, "temperature": 0,
            "seed": 1907, "max_tokens": max_tokens, "enable_thinking": False,
            "stream": stream, **({"stream_options": {"include_usage": True}} if stream else {})}


def result_identity(response):
    require("error" not in response, f"generation failed: {response}")
    choices = response.get("choices", [])
    require(len(choices) == 1, f"missing completion: {response}")
    message = choices[0].get("message", {})
    usage = response.get("usage", {})
    require(bool(message.get("content")), f"empty completion: {response}")
    require(usage.get("completion_tokens", 0) > 0, f"missing token usage: {response}")
    return {"message": message, "usage": usage, "finish_reason": choices[0].get("finish_reason")}


class Server:
    def __init__(self, args, directory, enabled):
        self.args, self.directory, self.enabled = args, directory, enabled
        self.log_path = directory / "server.log"
        self.process = None
        self.log = None
        if args.dry_run:
            self.port = 18081 + int(enabled)
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
        self.env.update(
            QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS="1",
            QW3_CONTINUOUS_KVMEM_CONTROLLER="1",
            QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL="1",
            QW3_CONTINUOUS_KVMEM_SESSION_REUSE="1",
            QW3_CONTINUOUS_BATCHING_TRACE="1",
            QW3_KVMEM_PREFIX_CACHE_TRACE="1",
            QW3_KVMEM_PROBE_RESERVE_TOKENS="2048",
            QW3_CONTINUOUS_BATCHING_ADMISSION_WAIT_US="50000",
            QW3_CONTINUOUS_BATCHING_SAMPLING="1",
            QW3_CONTINUOUS_BATCHING_KVMEM_BODY="1")
        self.command = [str(args.binary.resolve()), "serve", "--model", str(args.model.resolve()),
            "--host", "127.0.0.1", "--port", str(self.port), "--ctx", "8192",
            "--prefill-chunk", "128", "--paged-kv", "--kv-page-size", "16", "--kv-dtype", "fp8",
            "--max-active", str(args.active), "--continuous-batching", "--kvmem", "--kvmem-prefix-cache",
            "--kvmem-query-conditioned", "--kvmem-update-mode", "step",
            "--kvmem-repository-blocks", "256", "--kvmem-repository-gc", "delete",
            "--kvmem-block-tokens", "32", "--kvmem-budget", "1024", "--kvmem-prefill-budget", "1024",
            "--kvmem-gen-budget", "2048", "--kvmem-sink-blocks", "1", "--kvmem-recent-blocks", "4",
            "--kvmem-index-placement", "cpu", "--kvmem-cpu-gb", str(args.sessions + 3),
            "--kvmem-gpu-memory-ratio", "1", "--speculative-method", args.route]
        if args.route == "mtp":
            self.command += ["--mtp-chain", "4"]
        if args.route == "dflash2":
            self.env["QW3_CONTINUOUS_DFLASH"] = "1"
            self.command += ["--dflash-model", str(args.dflash.resolve())]
        if enabled:
            self.command += ["--kvmem-session-capacity", str(args.sessions + 2),
                             "--kvmem-session-swap-gb", str(args.swap_gb)]

    def plan(self):
        return {"command": self.command,
                "environment": {k: v for k, v in self.env.items() if k.startswith("QW3_")}}

    def __enter__(self):
        self.directory.mkdir()
        write_json(self.directory / "launch.json", self.plan())
        self.log = self.log_path.open("x")
        try:
            self.process = subprocess.Popen(self.command, env=self.env, stdout=self.log,
                                            stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.args.timeout
            while time.monotonic() < deadline:
                require(self.process.poll() is None, f"server exited; inspect {self.log_path}")
                try:
                    status, _ = json_call(self.base, "/health", timeout=2)
                    if status == 200:
                        return self
                except OSError:
                    pass
                time.sleep(.25)
            raise TimeoutError(f"server readiness timeout: {self.log_path}")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()

    def __exit__(self, *_):
        self.close()

    def request(self, label, body, barrier=None, expected=200):
        body = {**body, "kvmem_trace_tag": "multi-session-" + label}
        if barrier is not None:
            barrier.wait(timeout=30)
        began = time.monotonic()
        status, response = json_call(self.base, "/v1/chat/completions", body, timeout=self.args.timeout)
        write_json(self.directory / (label + ".json"),
                   {"request": body, "status": status, "response": response,
                    "wall_s": time.monotonic() - began})
        require(status == expected, f"{label}: expected HTTP {expected}, got {status}: {response}")
        return result_identity(response) if expected == 200 else response

    def session(self, key, *, method="GET", expected=200, label=None):
        path = "/v1/kvmem/sessions/" + urllib.parse.quote(key, safe="")
        status, response = json_call(self.base, path, method=method, timeout=30)
        write_json(self.directory / ((label or (method.lower() + "-" + key)) + ".json"),
                   {"status": status, "response": response})
        require(status == expected, f"{method} {key}: expected {expected}, got {status}: {response}")
        return response

    def stream(self, label, body, first_text, path="/v1/chat/completions"):
        request_id = "multi-session-" + label
        call = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-qw3-request-id": request_id})
        result = {"request_id": request_id, "content": "", "events": [], "done": False}
        try:
            with opener().open(call, timeout=self.args.timeout) as response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(data)
                    result["events"].append(event)
                    for choice in event.get("choices", []):
                        piece = choice.get("delta", {}).get("content", choice.get("text", ""))
                        result["content"] += piece
                        if piece:
                            first_text.set()
                    if event.get("error"):
                        result["error"] = event["error"]
        finally:
            write_json(self.directory / (label + ".json"), {"request": body, "response": result})
        return result


def prefix_reuse_evidence(directory, sessions, *, require_trace_tags=True):
    """Associate native reuse accounting and checkpoint decisions per request.

    Accounting tags identify requests even when their completions interleave.
    HIT/MISS traces currently lack tags, so their prompt length must uniquely
    identify a saved request; ambiguity fails rather than relying on log order.
    Legacy untagged artifacts can be checked offline only if (prompt, decoded)
    accounting fingerprints are unique among all saved lifecycle requests.
    """
    requests = {}
    for i in range(sessions):
        for turn in (0, 1):
            label = f"session-{i}-turn-{turn}"
            record = json.loads((directory / (label + ".json")).read_text())
            require(record["status"] == 200, f"{label}: request failed")
            requests[label] = {"key": record["request"]["kvmem_prefix_cache_key"],
                               "tag": record["request"].get("kvmem_trace_tag"),
                               "usage": record["response"]["usage"], "turn": turn}
    native, decisions = [], []
    current_tag = None
    for line in (directory / "server.log").read_text().splitlines():
        fields = dict(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line))
        if "[qw3-native-accounting]" in line:
            require(current_tag is None, "native accounting was not followed by its generate record")
            current_tag = fields["trace_tag"]
        elif "native generate:" in line:
            require(current_tag is not None, "native generate record is missing accounting")
            native.append({"tag": current_tag, "prompt": int(fields["prompt_tokens"]),
                           "decoded": int(fields["decoded"]),
                           "reuse": int(fields.get("kvmem_reuse", "0"))})
            current_tag = None
        match = re.search(r"kvmem prefix-cache (HIT|MISS) \([^)]+\):", line)
        if match:
            decisions.append({"result": match.group(1), "fields": fields})

    evidence = {}
    for label, request in requests.items():
        usage = request["usage"]
        if require_trace_tags:
            require(request["tag"] is not None, f"{label}: missing request trace tag")
            matches = [row for row in native if row["tag"] == request["tag"]]
        else:
            signature = (usage["prompt_tokens"], usage["completion_tokens"])
            require(sum((other["usage"]["prompt_tokens"], other["usage"]["completion_tokens"])
                        == signature for other in requests.values()) == 1,
                    f"{label}: ambiguous legacy accounting fingerprint {signature}")
            matches = [row for row in native if (row["prompt"], row["decoded"]) == signature]
        require(len(matches) == 1, f"{label}: expected one native accounting record, got {matches}")
        row = matches[0]
        require(row["prompt"] == usage["prompt_tokens"] and row["decoded"] == usage["completion_tokens"],
                f"{label}: native accounting does not match saved response usage")
        evidence[label] = {"key": request["key"], "prompt_tokens": row["prompt"],
                           "decoded": row["decoded"], "reuse": row["reuse"], "decision": None}
    for decision in decisions:
        prompt = int(decision["fields"]["prompt"])
        labels = [label for label, entry in evidence.items() if entry["prompt_tokens"] == prompt]
        require(len(labels) == 1,
                f"prefix decision prompt={prompt} is ambiguous among {labels}; use distinct fixture lengths")
        label = labels[0]
        require(evidence[label]["decision"] is None, f"{label}: duplicate prefix decision")
        evidence[label]["decision"] = decision
        require(evidence[label]["reuse"] == int(decision["fields"].get("reuse", "0")),
                f"{label}: HIT/MISS decision disagrees with native reuse accounting")
    for label, request in requests.items():
        entry = evidence[label]
        if request["turn"] == 1:
            require(entry["decision"] is not None, f"{label}: missing warm-prefix decision")
        else:
            require(entry["reuse"] == 0 and entry["decision"] is None,
                    f"{label}: fresh key unexpectedly reused history")
    return evidence


def compare_prefix_reuse(reference, candidate, sessions, *, require_trace_tags=True):
    reference = prefix_reuse_evidence(reference, sessions, require_trace_tags=require_trace_tags)
    candidate = prefix_reuse_evidence(candidate, sessions, require_trace_tags=require_trace_tags)
    for label in reference:
        require(reference[label] == candidate[label],
                f"{label}: resumed prefix behavior differs from warm reference: "
                f"reference={reference[label]}, candidate={candidate[label]}")
    positive = [label for label, entry in candidate.items() if entry["reuse"] > 0]
    require(bool(positive), "fixture contains no positive prefix reuse after resume")
    return {"requests": candidate, "positive_reuse_requests": positive,
            "accounting_identity": "trace_tag" if require_trace_tags else "unique_legacy_token_counts",
            "decision_identity": "unique_prompt_token_count"}


def exercise_normal_stop(server, initial, reference):
    key = "probe-normal-stop-chat"
    body = body_for(key, initial, max_tokens=128, stream=True)
    body["stop"] = [reference["message"]["content"][:12]]
    result = server.stream("normal-stop-chat", body, threading.Event())
    require(result["done"] and not result.get("error"), f"normal chat stop failed: {result}")
    require(any(choice.get("finish_reason") == "stop" for event in result["events"]
                for choice in event.get("choices", [])), "chat fixture did not reach its stop sequence")
    require(len(result["content"]) < len(reference["message"]["content"]),
            "chat fixture reached ordinary EOS instead of truncating at its stop sequence")
    state = server.session(key, label="normal-stop-chat-state")
    require(state.get("state") in ("warm", "suspended"), f"normal stop poisoned chat history: {state}")
    later = initial + [{"role": "assistant", "content": result["content"]},
                       {"role": "user", "content": "Repeat the project's color in one word."}]
    server.request("normal-stop-chat-followup", body_for(key, later))
    server.session(key, method="DELETE")

    key = "probe-normal-stop-completion"
    path = "/v1/completions"
    prompt = "Write one short sentence about an amber otter.\nSentence:"
    base = {"model": "test", "prompt": prompt, "kvmem_prefix_cache_key": key,
            "temperature": 0, "seed": 1907, "max_tokens": 32, "enable_thinking": False}
    status, clean = json_call(server.base, path, base, timeout=server.args.timeout)
    write_json(server.directory / "normal-stop-completion-reference.json", {"status": status, "response": clean})
    require(status == 200, f"completion stop reference failed: {status}, {clean}")
    text = clean.get("choices", [{}])[0].get("text", "")
    require(bool(text), "completion stop fixture generated no text")
    body = {**base, "max_tokens": 128, "stream": True, "stream_options": {"include_usage": True},
            "stop": [text[:12]]}
    result = server.stream("normal-stop-completion", body, threading.Event(), path)
    require(result["done"] and not result.get("error"), f"normal completion stop failed: {result}")
    require(any(choice.get("finish_reason") == "stop" for event in result["events"]
                for choice in event.get("choices", [])), "completion fixture did not reach its stop sequence")
    require(len(result["content"]) < len(text),
            "completion fixture reached ordinary EOS instead of truncating at its stop sequence")
    state = server.session(key, label="normal-stop-completion-state")
    require(state.get("state") in ("warm", "suspended"), f"normal stop poisoned completion history: {state}")
    later = {**base, "prompt": prompt + result["content"] + "\nName the color again:"}
    status, continued = json_call(server.base, path, later, timeout=server.args.timeout)
    write_json(server.directory / "normal-stop-completion-followup.json", {"status": status, "response": continued})
    require(status == 200 and not continued.get("error"), f"completion stop lost continuation: {status}, {continued}")
    server.session(key, method="DELETE")


def exercise_cancel(server, initial, reference):
    key = "probe-cancel"
    first = threading.Event()
    long_body = body_for(key, initial, max_tokens=1024, stream=True)
    long_body["ignore_eos"] = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(server.stream, "cancel-running", long_body, first)
        try:
            require(first.wait(timeout=server.args.timeout), "cancel fixture produced no text")
            require(not running.done(), "cancel fixture finished before conflict check")
            server.request("same-key-conflict", body_for(key, initial), expected=409)
            server.session(key, expected=409, label="get-active")
            server.session(key, method="DELETE", expected=409, label="delete-active")
            peer = pool.submit(server.request, "cancel-peer", body_for("probe-peer", initial))
            peer.result(timeout=server.args.timeout)
            require(not running.done(), "cancel fixture ended before cancellation")
        finally:
            status, response = json_call(server.base, "/v1/requests/multi-session-cancel-running/cancel",
                                         {}, timeout=30)
            write_json(server.directory / "cancel.json", {"status": status, "response": response})
            require(status in (200, 202), f"cancel failed: {status}, {response}")
        running.result(timeout=server.args.timeout)
    server.request("invalid-after-cancel", body_for(key, initial), expected=409)
    invalid = server.session(key, label="get-invalid")
    require(invalid.get("state") == "invalid", f"cancelled session lost its invalid state: {invalid}")
    server.session(key, method="DELETE", label="delete-invalid")
    recovered = server.request("recreated-after-delete", body_for(key, initial))
    require(recovered == reference, "explicit recreation changed the clean reference output")
    server.request("capacity-rejected", body_for("probe-over-capacity", initial), expected=429)
    for i in range(server.args.sessions):
        server.session(f"probe-session-{i}", label=f"preserved-after-capacity-{i}")
    server.session("probe-peer", method="DELETE", label="delete-peer")
    reused = server.request("capacity-reused", body_for("probe-capacity-reused", initial))
    require(reused == reference, "history capacity was not reusable after explicit delete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("none", "mtp", "dflash2"), default="none")
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--swap-gb", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.active < 2 or args.sessions <= args.active or args.swap_gb <= 0 or args.timeout <= 0:
        parser.error("require active >= 2, sessions > active, swap-gb > 0 and timeout > 0")
    if args.route == "dflash2" and args.dflash is None:
        parser.error("dflash2 requires --dflash")
    if not args.binary.is_file() or not args.model.is_file():
        parser.error("binary and model must be existing files")
    baseline = Server(args, args.output / "reference", False)
    candidate = Server(args, args.output / "candidate", True)
    if args.dry_run:
        print(json.dumps({"reference": baseline.plan(), "candidate": candidate.plan()}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "active": args.active, "sessions": args.sessions,
              "route": args.route, "performance_qualified": False}
    write_json(args.output / "validation.json", report)
    try:
        initial = [make_messages(i) for i in range(args.sessions)]
        followup = []
        references = {}
        with baseline:
            for i, messages in enumerate(initial):
                key = f"probe-session-{i}"
                first = baseline.request(f"session-{i}-turn-0", body_for(key, messages))
                references[i, 0] = first
                later = messages + [first["message"], {"role": "user", "content":
                    f"For session {i}, repeat the color and animal, then give the count from Entry 23."}]
                followup.append(later)
                references[i, 1] = baseline.request(f"session-{i}-turn-1", body_for(key, later))
            require("prefix-cache HIT" in baseline.log_path.read_text(), "reference never reused a hot prefix")
        with candidate:
            for turn, sessions in enumerate((initial, followup)):
                for start in range(0, args.sessions, args.active):
                    indices = list(range(start, min(args.sessions, start + args.active)))
                    barrier = threading.Barrier(len(indices))
                    with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                        requests = {i: pool.submit(candidate.request, f"session-{i}-turn-{turn}",
                                    body_for(f"probe-session-{i}", sessions[i]), barrier) for i in indices}
                        for i, result in requests.items():
                            require(result.result() == references[i, turn],
                                    f"session {i} turn {turn}: interleaving changed text, usage or finish reason")
                states = [candidate.session(f"probe-session-{i}", label=f"state-turn-{turn}-{i}")
                          for i in range(args.sessions)]
                require(all(state.get("history_tokens", 0) > 0 for state in states),
                        f"retained histories disappeared: {states}")
                require(sum(state.get("state") == "suspended" for state in states) >= args.sessions - args.active,
                        f"histories exceeding active slots were not suspended: {states}")
                require(all(state.get("swap_bytes", 0) > 0 for state in states if state.get("state") == "suspended"),
                        f"suspended histories have no retained swap backing: {states}")
            lines = candidate.log_path.read_text().splitlines()
            for i in range(args.sessions):
                key = f"probe-session-{i}"
                owned = [line for line in lines if re.search(rf"\bkey={re.escape(key)}(?:\s|$)", line)]
                for event in ("park", "resume"):
                    require(any("kvmem_session_memory:" in line and f"event={event}" in line for line in owned),
                            f"{key}: no actual session {event} evidence")
            prefix_comparison = compare_prefix_reuse(baseline.directory, candidate.directory, args.sessions)
            write_json(args.output / "prefix-reuse-comparison.json", prefix_comparison)
            exercise_normal_stop(candidate, initial[0], references[0, 0])
            exercise_cancel(candidate, initial[0], references[0, 0])
        report.update(status="passed", exact_reference_match=True, resume_evidence=True,
                      prefix_reuse_matches_reference=True,
                      positive_reuse_requests=prefix_comparison["positive_reuse_requests"],
                      conflicting_writer=True, cancelled_session_invalid=True, explicit_delete_recovery=True,
                      capacity_rejection_preserves_histories=True, normal_stream_stop_continues=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "validation.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
