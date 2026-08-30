#!/usr/bin/env python3
"""End-to-end KvmemRequestPlan regression on a short agent continuation."""

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
            "seed": 91,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "stream": False,
        },
        {
            "User-Agent": "opencode/1.18.18",
            "x-opencode-session": "kvmem-request-plan-gpu-regression",
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
    parser.add_argument("--ballast-tokens", type=int, default=70000)
    parser.add_argument("--large-ballast-tokens", type=int, default=140000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    start = log_path.stat().st_size if log_path.exists() else 0
    ballast, measured = make_ballast(
        args.base_url, args.ballast_tokens, args.timeout
    )
    oversized_messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": "Inspect this complete tool result and report its final marker.",
        },
        {
            "role": "assistant",
            "content": "I will read the complete result.",
            "tool_calls": [
                {
                    "id": "call_oversized",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "oversized.log"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_oversized",
            "content": ballast,
        },
    ]
    oversized_start = log_path.stat().st_size if log_path.exists() else 0
    oversized_status, oversized, oversized_body = chat(
        args.base_url, oversized_messages, 8, args.timeout
    )
    oversized_end = log_path.stat().st_size

    large_ballast, measured_large = make_ballast(
        args.base_url, args.large_ballast_tokens, args.timeout
    )
    large_messages = json.loads(json.dumps(oversized_messages))
    large_messages[1]["content"] = (
        "Inspect this larger complete tool result and report its final marker."
    )
    large_messages[2]["tool_calls"][0]["id"] = "call_larger"
    large_messages[3]["tool_call_id"] = "call_larger"
    large_messages[3]["content"] = large_ballast
    large_status, large, large_body = chat(
        args.base_url, large_messages, 4, args.timeout
    )
    large_end = log_path.stat().st_size

    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": (
                "Keep investigating the current failure. Preserve the exact "
                "task and use the newest tool result. Answer briefly."
            ),
        },
        {
            "role": "assistant",
            "content": "I will inspect the execution trace.",
            "tool_calls": [
                {
                    "id": "call_trace",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "trace.log"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_trace", "content": ballast},
        {
            "role": "assistant",
            "content": "The old trace is complete and is now retrieval history.",
        },
        {
            "role": "user",
            "content": (
                "Using that history, state the next checkpoint validation in "
                "one short sentence."
            ),
        },
    ]
    first_status, first, first_body = chat(
        args.base_url, messages, 13, args.timeout
    )
    first_message = first.get("choices", [{}])[0].get("message", {})
    first_end = log_path.stat().st_size
    if first_status == 200 and first_message:
        messages.append(first_message)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "I will read the newest status.",
                    "tool_calls": [
                        {
                            "id": "call_status",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {"path": "status.log"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_status",
                    "content": "STATUS=ready; identify the next check briefly.",
                },
            ]
        )
        second_status, second, second_body = chat(
            args.base_url, messages, 16, args.timeout
        )
    else:
        second_status, second, second_body = 0, {}, "first request failed"
    second_end = log_path.stat().st_size
    if second_status == 200:
        # Simulate an upstream harness rewriting an old tool payload without
        # changing the stable task identity. This intentionally invalidates
        # both P and M and exercises the transactional recovery plan.
        recovery_messages = json.loads(json.dumps(messages))
        old_tool = recovery_messages[3]["content"]
        recovery_messages[3]["content"] = (
            ("X" if not old_tool.startswith("X") else "Y") + old_tool[1:]
        )
        third_status, third, third_body = chat(
            args.base_url, recovery_messages, 8, args.timeout
        )
    else:
        third_status, third, third_body = 0, {}, "second request failed"
    time.sleep(0.2)
    log_bytes = log_path.read_bytes()
    oversized_log = log_bytes[oversized_start:oversized_end].decode(
        "utf-8", errors="replace"
    )
    large_log = log_bytes[oversized_end:large_end].decode(
        "utf-8", errors="replace"
    )
    log = log_bytes[start:].decode("utf-8", errors="replace")
    second_log = log_bytes[first_end:second_end].decode(
        "utf-8", errors="replace"
    )
    third_log = log_bytes[second_end:].decode("utf-8", errors="replace")
    draft_keep = "stage=draft action=keep-selected-append" in second_log
    finals = re.findall(
        r"KVMem request plan stage=final action=([^ ]+)", second_log
    )
    final_action = finals[-1] if finals else "missing"
    safe_final = final_action in {
        "resume-m-then-append",
        "resume-p-then-replay-tail",
        "pressure-prefill",
    }
    forbidden = any(
        text in second_log + third_log
        for text in (
            "keep-selected prefill exhausted",
            "global KV page pool exhausted",
            "HTTP 500",
        )
    )
    recovery_draft_keep = (
        "stage=draft action=keep-selected-append" in third_log
    )
    recovery_final = (
        "KVMem request plan stage=final action=pressure-prefill"
        in third_log
        and "semantic=keep-selected-append pressure_ingest=1" in third_log
    )
    oversized_fitted = bool(re.search(
        r"live_suffix_blocks_raw=(\d+).*live_suffix_blocks_fitted=(\d+)"
        r".*capacity_fitted=1",
        oversized_log,
    ))
    large_fitted = bool(re.search(
        r"live_suffix_blocks_raw=(\d+).*live_suffix_blocks_fitted=(\d+)"
        r".*capacity_fitted=1",
        large_log,
    ))
    report = {
        "schema": "qw3.kvmem_request_plan_gpu.v1",
        "ballast_tokens": measured,
        "large_ballast_tokens": measured_large,
        "oversized_live_http_status": oversized_status,
        "oversized_live_capacity_fitted": oversized_fitted,
        "large_live_http_status": large_status,
        "large_live_capacity_fitted": large_fitted,
        "first_http_status": first_status,
        "second_http_status": second_status,
        "recovery_http_status": third_status,
        "draft_keep_selected": draft_keep,
        "final_action": final_action,
        "forbidden_capacity_error": forbidden,
        "recovery_draft_keep_selected": recovery_draft_keep,
        "recovery_final_action": (
            "pressure-prefill" if recovery_final else "missing"
        ),
        "first_body_prefix": first_body[:300],
        "oversized_live_body_prefix": oversized_body[:300],
        "large_live_body_prefix": large_body[:300],
        "second_body_prefix": second_body[:300],
        "recovery_body_prefix": third_body[:300],
        "passed": (
            oversized_status == 200
            and large_status == 200
            and oversized_fitted
            and large_fitted
            and first_status == 200
            and second_status == 200
            and third_status == 200
            and draft_keep
            and safe_final
            and recovery_draft_keep
            and recovery_final
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
