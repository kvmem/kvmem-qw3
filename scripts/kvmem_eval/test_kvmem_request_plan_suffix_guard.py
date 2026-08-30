#!/usr/bin/env python3
"""GPU regression for the final RequestPlan A+B suffix safety gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from test_harness_long_trace import make_ballast, post


def chat(base_url: str, messages: list[dict], timeout: int):
    status, body = post(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": "Qwen3.8-27B-Q8_0.gguf",
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 151,
            "enable_thinking": False,
            "max_tokens": 4,
            "stream": False,
            "kvmem_reselect": "off",
            "kvmem_prefill_window": "keep_selected",
        },
        {"User-Agent": "request-plan-suffix-regression/1"},
        timeout,
    )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"raw": body}
    return status, parsed, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output", required=True)
    # Start above A so the first turn builds a durable source index; a tiny
    # below-A prompt intentionally has no above-budget source-index checkpoint
    # to reuse on the next turn.
    parser.add_argument("--base-tokens", type=int, default=70000)
    parser.add_argument("--suffix-tokens", type=int, default=70000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    base_ballast, measured_base = make_ballast(
        args.base_url, args.base_tokens, args.timeout
    )
    messages: list[dict] = [
        {"role": "system", "content": "You are a concise coding agent."},
        {
            "role": "user",
            "content": base_ballast + "\nRemember checkpoint marker ALPHA.",
        },
    ]
    first_status, first, first_body = chat(
        args.base_url, messages, args.timeout
    )
    first_message = first.get("choices", [{}])[0].get("message", {})
    ballast, measured = make_ballast(
        args.base_url, args.suffix_tokens, args.timeout
    )
    start = log_path.stat().st_size if log_path.exists() else 0
    if first_status == 200 and first_message:
        messages.append(first_message)
        messages.append({
            "role": "user",
            "content": ballast + "\nReturn the marker in one word.",
        })
        second_status, _, second_body = chat(
            args.base_url, messages, args.timeout
        )
    else:
        second_status, second_body = 0, "first request failed"
    time.sleep(0.2)
    log = log_path.read_bytes()[start:].decode("utf-8", errors="replace")
    match = re.search(
        r"stage=final action=pressure-prefill"
        r"[^\n]*pressure_ingest=1[^\n]*checkpoint=1"
        r"[^\n]*suffix=(\d+)",
        log,
    )
    suffix = int(match.group(1)) if match else 0
    forbidden = any(text in log for text in (
        "keep-selected prefill exhausted",
        "global KV page pool exhausted",
        "mandatory selection plus sink blocks exceeds",
    ))
    report = {
        "schema": "qw3.kvmem_request_plan_suffix_guard.v1",
        "base_tokens": measured_base,
        "suffix_tokens": measured,
        "first_http_status": first_status,
        "second_http_status": second_status,
        "final_action": "pressure-prefill" if match else "missing",
        "checkpoint_available": bool(match),
        "incremental_suffix_tokens": suffix,
        "forbidden_capacity_error": forbidden,
        "first_body_prefix": first_body[:300],
        "second_body_prefix": second_body[:300],
        "passed": (
            first_status == 200
            and second_status == 200
            and suffix > 32768
            and not forbidden
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
