#!/usr/bin/env python3
"""Verify that a cancelled stream preserves the prior KVMem warm checkpoint."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import struct
import time
from pathlib import Path
from typing import Any

from kvmem_prefix_cache_above_budget_canary import (
    fit_content_below_budget,
    free_port,
    payload,
    post_chat,
    response_text,
    start_server,
    stop_server,
    wait_health,
)


def cancel_stream(port: int, body: dict[str, Any], timeout_s: float) -> int:
    request = dict(body)
    request["stream"] = True
    request["max_tokens"] = max(int(request.get("max_tokens", 0)), 512)
    request["ignore_eos"] = True

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(request),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        raw = response.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(
            f"stream request failed: status={response.status} body={raw}")

    events = 0
    while events < 2:
        line = response.fp.readline()
        if not line:
            break
        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
            events += 1

    # Force an immediate RST so the next server write observes cancellation.
    if conn.sock is not None:
        conn.sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
    conn.close()
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qw3", type=Path, default=Path("./build/qw3"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--gen-budget", type=int, default=1024)
    parser.add_argument("--cpu-gb", type=int, default=2)
    parser.add_argument("--filler-repeats", type=int, default=4096)
    parser.add_argument("--threshold-margin", type=int, default=8)
    parser.add_argument("--crossing-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/tmp/qw3_kvmem_cancel_canary.json"),
    )
    parser.add_argument(
        "--warm-log",
        type=Path,
        default=Path("/tmp/qw3_kvmem_cancel_canary.log"),
    )
    args = parser.parse_args()
    args.qw3 = args.qw3.resolve()
    args.model = args.model.resolve()

    query = "Remember that the cancellation codename is SILVER-PINE."
    lead = " ".join([
        "Stable prefix material for the cancellation checkpoint test."
    ] * 64) + "\n\n"
    first_prefix = lead + query + "\n\n"
    query_start = len(lead.encode())
    query_end = query_start + len(query.encode())

    port = free_port()
    proc, log_file = start_server(args, port, args.warm_log)
    status_a = status_retry = 0
    body_a: dict[str, Any] = {}
    body_retry: dict[str, Any] = {}
    events = 0
    try:
        wait_health(port, args.timeout, proc)
        first_content, _, _ = fit_content_below_budget(
            port,
            first_prefix,
            args.budget,
            args.threshold_margin,
            args.filler_repeats,
            add_generation_prompt=True,
            timeout_s=args.timeout,
        )
        messages_a = [{"role": "user", "content": first_content}]
        status_a, body_a = post_chat(
            port,
            payload(
                messages_a,
                query_start,
                query_end,
                args.crossing_tokens,
                ignore_eos=True,
            ),
            args.timeout,
        )
        if status_a != 200:
            raise RuntimeError(f"turn A failed: {status_a} {body_a}")

        followup = "Start a long draft, but preserve the prior checkpoint."
        messages_b = [
            *messages_a,
            {"role": "assistant", "content": response_text(body_a)},
            {"role": "user", "content": followup},
        ]
        request_b = payload(
            messages_b, query_start, query_end, 512, ignore_eos=True)
        request_b["kvmem_query_span"] = {
            "message_index": len(messages_b) - 1,
            "content_start": 0,
            "content_end": len(followup.encode()),
        }
        events = cancel_stream(port, request_b, args.timeout)

        # The server detects the reset on its next stream write. A buffered
        # retry naturally waits behind the single-request lock until cleanup.
        time.sleep(0.25)
        retry_request = dict(request_b)
        retry_request["stream"] = False
        retry_request["max_tokens"] = args.max_tokens
        retry_request.pop("ignore_eos", None)
        status_retry, body_retry = post_chat(
            port, retry_request, args.timeout)
    finally:
        stop_server(proc, log_file)

    log_text = args.warm_log.read_text(encoding="utf-8", errors="replace")
    route = "plain" if args.plain else "mtp"
    cancel_offset = log_text.find("client_closed=true")
    retry_hit = ""
    if cancel_offset >= 0:
        match = re.search(
            rf"kvmem prefix-cache HIT \({route}\):[^\n]+",
            log_text[cancel_offset:],
        )
        retry_hit = match.group(0) if match else ""

    checks = {
        "http_ok": status_a == 200 and status_retry == 200,
        "stream_started": events >= 1,
        "client_cancelled": cancel_offset >= 0,
        "suspended": "kvmem prefix-cache SUSPEND generation=1" in log_text,
        "resumed": "kvmem prefix-cache RESUME generation=1" in log_text,
        "staging_aborted":
            "ABORT keep_generation=1 reason=stream_cancelled_" in log_text,
        "retry_hit": bool(retry_hit),
        "retry_has_output": bool(response_text(body_retry)),
        "no_server_error": "POST /v1/chat/completions -> 5" not in log_text,
    }
    report = {
        "ok": all(checks.values()),
        "route": route,
        "checks": checks,
        "events_before_cancel": events,
        "retry_hit": retry_hit,
        "log": str(args.warm_log),
    }
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
