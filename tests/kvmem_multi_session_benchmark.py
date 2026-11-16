#!/usr/bin/env python3
"""Benchmark an already running server with independent, full-history sessions.

Example (does not start a server or delete its sessions):
  python3 tests/kvmem_multi_session_benchmark.py --base-url http://127.0.0.1:8080 \
      --output /tmp/kvmem-bench --users 4 --concurrency 2 --rounds 3 \
      --warmup 1 --max-tokens 128 --ignore-eos --inspect-sessions

Use --dry-run for deterministic parser/payload/scheduler self-tests without sockets.
Warmup is per user, using the same histories as measurement. Within each phase a
FIFO scheduler keeps up to concurrency users running, never overlapping a key.
TTFT is arrival of the first nonempty content/reasoning delta, not first SSE data.
Output throughput uses server completion-token usage, never SSE chunk counts.
"""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def initial_messages(user, context_lines):
    notes = "\n".join(
        f"Record {i}: user {user} owns item {user * 1000 + i}; its value is {i * 7 + user}."
        for i in range(context_lines))
    return [
        {"role": "system", "content": "Answer accurately using this user's records. Do not reason aloud."},
        {"role": "user", "content": notes + "\nSummarize the records and explain the numbering pattern."},
    ]


def payload_for(args, user):
    payload = {"model": args.model, "messages": deepcopy(user["messages"]),
               "kvmem_prefix_cache_key": user["key"], "temperature": 0,
               "seed": args.seed, "max_tokens": args.max_tokens, "enable_thinking": False,
               "stream": True, "stream_options": {"include_usage": True}}
    if args.ignore_eos:
        payload["ignore_eos"] = True
    return payload


class StreamResult:
    """SSE collector with strict completion and server-usage validation."""
    def __init__(self):
        self.data = []
        self.content = []
        self.reasoning = []
        self.events = []
        self.usage = None
        self.finish_reason = None
        self.ttft_s = None
        self.done = False

    def feed(self, line, elapsed_s):
        line = line.rstrip("\r\n")
        if not line:
            self.dispatch(elapsed_s)
        elif line.startswith("data:"):
            self.data.append(line[5:].lstrip(" "))
        # SSE comments, event names and IDs do not contain generated tokens.

    def dispatch(self, elapsed_s):
        if not self.data:
            return
        raw = "\n".join(self.data)
        self.data.clear()
        require(not self.done, "received SSE event after [DONE]")
        if raw == "[DONE]":
            self.done = True
            return
        event = json.loads(raw)
        require(isinstance(event, dict), "SSE event is not an object")
        self.events.append({"elapsed_s": elapsed_s, "data": event})
        require("error" not in event, f"stream error: {event.get('error')}")
        if event.get("usage") is not None:
            require(self.usage is None, "duplicate usage event")
            self.usage = event["usage"]
        choices = event.get("choices", [])
        require(isinstance(choices, list) and len(choices) <= 1, "expected one stream choice")
        for choice in choices:
            require(choice.get("index", 0) == 0, "unexpected choice index")
            delta = choice.get("delta", {})
            require(isinstance(delta, dict), "invalid stream delta")
            require(not delta.get("tool_calls") and not delta.get("function_call"),
                    "unexpected tool/function output")
            for name, target in (("content", self.content), ("reasoning_content", self.reasoning)):
                part = delta.get(name)
                require(part is None or isinstance(part, str), f"invalid {name}")
                if part:
                    require(self.finish_reason is None, "text received after finish reason")
                    if self.ttft_s is None:
                        self.ttft_s = elapsed_s
                    target.append(part)
            reason = choice.get("finish_reason")
            if reason is not None:
                require(self.finish_reason is None, "duplicate finish reason")
                require(reason in ("stop", "length"), f"unsuccessful finish reason: {reason}")
                self.finish_reason = reason

    def finish(self, max_tokens, fixed_output):
        require(not self.data, "unterminated SSE event")
        require(self.done, "stream ended without [DONE]")
        require(self.finish_reason is not None, "missing finish reason")
        require(isinstance(self.usage, dict), "missing usage")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            require(type(self.usage.get(name)) is int and self.usage[name] >= 0,
                    f"invalid usage.{name}")
        require(self.usage["prompt_tokens"] > 0, "empty prompt usage")
        count = self.usage["completion_tokens"]
        require(0 < count <= max_tokens, f"completion usage {count} outside (0, {max_tokens}]")
        require(self.usage["total_tokens"] == self.usage["prompt_tokens"] + count,
                "usage total does not match prompt + completion")
        require(self.ttft_s is not None, "no generated text")
        require(bool("".join(self.content)), "empty assistant content with thinking disabled")
        if fixed_output:
            require(count == max_tokens, f"ignore_eos did not produce {max_tokens} tokens: {count}")
            require(self.finish_reason == "length", "ignore_eos did not finish with length")
        message = {"role": "assistant", "content": "".join(self.content)}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        return {"message": message, "usage": self.usage, "finish_reason": self.finish_reason,
                "ttft_s": self.ttft_s}


def request_stream(args, payload, label):
    result = {"id": label, "request": payload, "ok": False}
    stream = StreamResult()
    began = time.perf_counter()
    try:
        request = urllib.request.Request(
            args.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     "x-qw3-request-id": label}, method="POST")
        with opener().open(request, timeout=args.timeout) as response:
            result["http_status"] = response.status
            require(response.status == 200, f"unexpected HTTP status: {response.status}")
            require("text/event-stream" in response.headers.get("Content-Type", ""),
                    "response is not an SSE stream")
            for line in response:
                stream.feed(line.decode("utf-8"), time.perf_counter() - began)
                if stream.done:
                    break
        result.update(stream.finish(args.max_tokens, args.ignore_eos))
        result["ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, urllib.error.HTTPError):
            result["http_status"] = error.code
            result["error_body"] = error.read().decode(errors="replace")
    result["latency_s"] = time.perf_counter() - began
    result["events"] = stream.events
    return result


def run_phase(args, users, rounds, phase, request_fn=request_stream):
    """Rolling FIFO: each user's next turn becomes eligible after its last ends."""
    if not rounds:
        return [], 0.0
    pending = deque((user, 0) for user in users)
    records = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(users))) as pool:
        began = time.perf_counter()
        active = {}
        while pending or active:
            while pending and len(active) < args.concurrency:
                user, turn = pending.popleft()
                label = f"{args.key_prefix}-{phase}-u{user['id']:03d}-r{turn:03d}"
                future = pool.submit(request_fn, args, payload_for(args, user), label)
                active[future] = (user, turn)
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            # Sorting simultaneous completions stabilizes otherwise arbitrary set order.
            for future in sorted(completed, key=lambda item: active[item][0]["id"]):
                user, turn = active.pop(future)
                record = future.result()
                record.update(phase=phase, user=user["id"], round=turn, key=user["key"])
                records.append(record)
                if record["ok"]:
                    user["messages"].append(record["message"])
                    user["turn"] += 1
                    user["messages"].append({"role": "user", "content":
                        f"Follow-up {user['turn']}: explain another pattern in my records and give examples."})
                    if turn + 1 < rounds:
                        pending.append((user, turn + 1))
        wall_s = time.perf_counter() - began
    return records, wall_s


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * percent / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values):
    return {"mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95),
            "min": min(values) if values else None, "max": max(values) if values else None}


def metrics(records, wall_s, expected):
    good = [record for record in records if record["ok"]]
    tokens = sum(record["usage"]["completion_tokens"] for record in good)
    return {"expected_requests": expected, "completed_requests": len(records),
            "successful_requests": len(good), "failed_requests": len(records) - len(good),
            "skipped_requests": expected - len(records), "wall_s": wall_s,
            "completion_tokens": tokens,
            "prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in good),
            "output_tokens_per_s": tokens / wall_s if wall_s else None,
            "requests_per_s": len(good) / wall_s if wall_s else None,
            "ttft_s": distribution([record["ttft_s"] for record in good]),
            "latency_s": distribution([record["latency_s"] for record in good]),
            "finish_reasons": {reason: sum(record["finish_reason"] == reason for record in good)
                               for reason in ("stop", "length")}}


def json_get(args, path):
    with opener().open(args.base_url + path, timeout=args.timeout) as response:
        require(response.status == 200, f"GET {path}: HTTP {response.status}")
        return json.load(response)


def self_test(args):
    # Tests exercise full-history ownership and protocol failures without a socket.
    user = {"id": 0, "key": "self-test-a", "turn": 0, "messages": initial_messages(0, 2)}
    request = payload_for(args, user)
    request["messages"][0]["content"] = "mutated"
    require(user["messages"][0]["content"] != "mutated", "payload aliases mutable history")
    require(request["kvmem_prefix_cache_key"] == user["key"], "missing user key")

    def sample(usage=None, reason="length", done=True):
        events = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                  {"choices": [{"index": 0, "delta": {"content": "abc"}}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
                  {"choices": [], "usage": usage or
                   {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}]
        parser = StreamResult()
        for index, event in enumerate(events):
            parser.feed("data: " + json.dumps(event), index + 0.25)
            parser.feed("", index + 0.5)
        if done:
            parser.feed("data: [DONE]", 4.25)
            parser.feed("", 4.5)
        return parser

    parsed = sample().finish(2, True)
    require(parsed["ttft_s"] == 1.5 and parsed["message"]["content"] == "abc",
            "TTFT counted a role event, or content was lost")
    failed_cases = [lambda: sample(done=False).finish(2, True),
                    lambda: sample(reason="error"),
                    lambda: sample(reason="stop").finish(2, True),
                    lambda: sample().finish(3, True),
                    lambda: sample(usage={"prompt_tokens": 3, "completion_tokens": 2,
                                          "total_tokens": 99}).finish(2, True),
                    lambda: sample(usage={"prompt_tokens": 3, "completion_tokens": True,
                                          "total_tokens": 4}).finish(2, True)]
    for case in failed_cases:
        try:
            case()
        except ValueError:
            pass
        else:
            raise AssertionError("malformed/incomplete stream unexpectedly passed")
    fake_args = argparse.Namespace(**vars(args))
    fake_args.concurrency = 2
    users = [{"id": i, "key": f"self-test-{i}", "turn": 0, "messages": initial_messages(i, 2)}
             for i in range(3)]

    def fake_request(_args, payload, label):
        key = payload["kvmem_prefix_cache_key"]
        for message in payload["messages"]:
            if message["role"] == "assistant":
                require(message["content"] == key, "cross-user history contamination")
        result = deepcopy(parsed)
        result.update(id=label, request=payload, ok=True, latency_s=2.0, events=[])
        result["message"]["content"] = key
        return result

    records, _ = run_phase(fake_args, users, 3, "self-test", fake_request)
    require(len(records) == 9 and all(len(u["messages"]) == 8 for u in users),
            "scheduler lost or overlapped a user turn")
    for record in records:
        require(len(record["request"]["messages"]) == 2 + record["round"] * 2,
                "request did not include complete preceding history")
    report = metrics(records, 10.0, 9)
    require(report["output_tokens_per_s"] == 1.8 and report["requests_per_s"] == .9,
            "throughput did not use total wall time")
    require(percentile([1, 2, 3], 95) == 2.9, "incorrect percentile interpolation")

    # Warmup history remains part of each measured request. Failed users stop
    # advancing, while unrelated users still complete their own queued turns.
    warm_records, _ = run_phase(fake_args, users, 1, "after-warmup", fake_request)
    require(all(len(record["request"]["messages"]) == 8 for record in warm_records),
            "phase transition discarded user history")

    def failed_request(_args, payload, label):
        if payload["kvmem_prefix_cache_key"] == "self-test-0":
            return {"id": label, "request": payload, "ok": False, "error": "injected failure"}
        return fake_request(_args, payload, label)

    failed, _ = run_phase(fake_args, users, 2, "failure", failed_request)
    failed_report = metrics(failed, 1.0, 6)
    require(failed_report["failed_requests"] == 1 and failed_report["skipped_requests"] == 1
            and failed_report["successful_requests"] == 4,
            "failed key advanced or prevented independent users from completing")
    print(json.dumps({"ok": True, "dry_run": True, "self_tests": "passed",
                      "sample_request": payload_for(args, user),
                      "measured_requests": args.users * args.rounds,
                      "warmup_requests": args.users * args.warmup}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Server origin, without /v1")
    parser.add_argument("--model", help="Default: discover the first /v1/models entry")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite existing results")
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3, help="Measured turns per user")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup turns per user")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context-lines", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--timeout", type=float, default=600, help="Per-socket-operation timeout in seconds")
    parser.add_argument("--ignore-eos", "--fixed-output", action="store_true",
                        help="Require exactly max-tokens output and finish_reason=length")
    parser.add_argument("--inspect-sessions", action="store_true", help="GET and validate final session states outside timing")
    parser.add_argument("--key-prefix", default="bench-" + uuid.uuid4().hex[:12])
    parser.add_argument("--label", default="", help="Matrix metadata, e.g. mtp-u4-c2")
    parser.add_argument("--dry-run", action="store_true", help="Run self-tests and show sample request, without sockets or files")
    args = parser.parse_args()
    for name in ("users", "concurrency", "rounds", "max_tokens", "context_lines"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--warmup must be nonnegative and --timeout finite and positive")
    if not args.key_prefix:
        parser.error("--key-prefix must not be empty")
    args.base_url = args.base_url.rstrip("/")
    parsed_url = urllib.parse.urlsplit(args.base_url)
    if (parsed_url.scheme not in ("http", "https") or not parsed_url.netloc
            or parsed_url.path not in ("", "/") or parsed_url.query or parsed_url.fragment):
        parser.error("--base-url must be an HTTP(S) server origin")
    if args.dry_run:
        args.model = args.model or "dry-run-model"
        self_test(args)
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "ok": False,
               "config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "notes": ["Client TTFT includes server queueing, ending at first nonempty text delta.",
                         "Measured wall excludes model discovery, warmup, session GETs and result writes.",
                         "Full-history sessions remain in the server after this benchmark.",
                         "Throughput on a failed run is diagnostic only."]}
    records = []
    try:
        if args.model is None:
            models = json_get(args, "/v1/models")
            args.model = models["data"][0]["id"]
            summary["config"]["model"] = args.model
        users = [{"id": i, "key": f"{args.key_prefix}-u{i:03d}", "turn": 0,
                  "messages": initial_messages(i, args.context_lines)} for i in range(args.users)]
        summary["keys"] = [user["key"] for user in users]
        warmup, wall = run_phase(args, users, args.warmup, "warmup")
        records.extend(warmup)
        summary["warmup"] = metrics(warmup, wall, args.users * args.warmup)
        require(all(record["ok"] for record in warmup) and len(warmup) == args.users * args.warmup,
                "warmup failed; measurement was not started")
        measured, wall = run_phase(args, users, args.rounds, "measured")
        records.extend(measured)
        summary["measured"] = metrics(measured, wall, args.users * args.rounds)
        if args.inspect_sessions:
            states = {}
            summary["sessions"] = states
            for user in users:
                state = json_get(args, "/v1/kvmem/sessions/" + urllib.parse.quote(user["key"], safe=""))
                states[user["key"]] = state
                require(state.get("object") == "kvmem.session" and state.get("id") == user["key"],
                        f"incorrect session identity: {state}")
                require(state.get("state") in ("warm", "suspended") and state.get("history_tokens", 0) > 0,
                        f"invalid final session state: {state}")
        require(all(record["ok"] for record in measured) and len(measured) == args.users * args.rounds,
                "measured requests failed or were skipped")
        summary["ok"] = True
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        summary["errors"] = [{key: record[key] for key in ("id", "user", "round", "phase", "error")}
                             for record in records if not record["ok"]]
        write_json(args.output / "requests.json", records)
        write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
