#!/usr/bin/env python3
"""GPU regression for one tool callback that jumps beyond A+B headroom."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from test_harness_long_trace import make_ballast, post


def chat(base_url: str, messages: list[dict], max_tokens: int, timeout: int):
    status, body = post(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": "Qwen3.8-27B-Q8_0.gguf",
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 117,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "stream": False,
        },
        {
            "User-Agent": "opencode/1.18.18",
            "x-opencode-session": "kvmem-oversized-tool-epoch-regression",
        },
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
    parser.add_argument("--history-tokens", type=int, default=58000)
    parser.add_argument("--tool-tokens", type=int, default=70000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    history, history_tokens = make_ballast(
        args.base_url, args.history_tokens, args.timeout
    )
    tool_result, tool_tokens = make_ballast(
        args.base_url, args.tool_tokens, args.timeout
    )
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": (
                "Continue diagnosing epoch safety and preserve the exact final "
                "status from subsequent tools."
            ),
        },
        {
            "role": "assistant",
            "content": "I will read the historical trace.",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "history.log"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": history},
        {
            "role": "assistant",
            "content": "The historical trace is complete.",
        },
        {
            "role": "user",
            "content": "State the next validation briefly.",
        },
    ]
    first_status, first, first_body = chat(
        args.base_url, messages, 8, args.timeout
    )
    first_message = first.get("choices", [{}])[0].get("message", {})
    second_start = log_path.stat().st_size if log_path.exists() else 0
    if first_status == 200 and first_message:
        messages.append(first_message)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "I will inspect the new oversized result.",
                    "tool_calls": [
                        {
                            "id": "call_large_result",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {"path": "large-result.log"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_large_result",
                    "content": tool_result,
                },
            ]
        )
        second_status, second, second_body = chat(
            args.base_url, messages, 8, args.timeout
        )
    else:
        second_status, second, second_body = 0, {}, "first request failed"
    time.sleep(0.2)
    log = log_path.read_bytes()[second_start:].decode(
        "utf-8", errors="replace"
    )
    private_pressure_plan = bool(re.search(
        r"stage=draft action=generate-private-query-then-reselect"
        r"[^\n]*capacity_fitted=1[^\n]*pressure_ingest=1",
        log,
    ))
    guided_query = "native kvmem guided query:" in log
    forbidden = any(text in log for text in (
        "keep-selected prefill exhausted",
        "global KV page pool exhausted",
        "mandatory selection plus sink blocks exceeds",
    ))
    report = {
        "schema": "qw3.kvmem_oversized_tool_epoch.v1",
        "history_tokens": history_tokens,
        "tool_result_tokens": tool_tokens,
        "first_http_status": first_status,
        "second_http_status": second_status,
        "private_pressure_plan": private_pressure_plan,
        "guided_query_generated_or_fallback": guided_query,
        "forbidden_capacity_error": forbidden,
        "first_body_prefix": first_body[:300],
        "second_body_prefix": second_body[:300],
        "passed": (
            first_status == 200
            and second_status == 200
            and private_pressure_plan
            and guided_query
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
