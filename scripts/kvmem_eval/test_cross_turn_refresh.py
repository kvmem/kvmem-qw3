#!/usr/bin/env python3
"""Verify that guided KVMem refreshes accumulate across tool HTTP turns.

Real coding harnesses usually stop an assistant response at every tool call.
Consequently, a mid-decode threshold alone rarely fires.  This regression sends
three requests for the same stable task:

1. establish the task and its initial retrieval query;
2. append a tiny completed tool round (must stay below the refresh threshold);
3. append a larger completed tool round (must cross the threshold once).

The server log is part of the evidence because the private retrieval query must
not be exposed through either the OpenAI or Anthropic response schema.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from test_harness_long_trace import (
    PRIVATE_MARKERS,
    anthropic_payload,
    make_ballast,
    openai_payload,
    post,
)


def append_tool_round(payload: dict, harness: str, ordinal: int, content: str) -> None:
    call_id = f"call_cross_turn_{ordinal}"
    if harness == "claude-code":
        payload["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I will inspect cross-turn artifact {ordinal}.",
                        },
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": "read_file",
                            "input": {"path": f"build/cross-turn-{ordinal}.log"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": content,
                        }
                    ],
                },
            ]
        )
        return
    payload["messages"].extend(
        [
            {
                "role": "assistant",
                "content": f"I will inspect cross-turn artifact {ordinal}.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {"path": f"build/cross-turn-{ordinal}.log"}
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ]
    )


def request_spec(harness: str, ballast: str) -> tuple[str, dict, dict[str, str]]:
    if harness == "claude-code":
        return (
            "/v1/messages",
            anthropic_payload(ballast),
            {"User-Agent": "claude-cli/2.1.148"},
        )
    payload = openai_payload(ballast)
    if harness == "opencode":
        headers = {
            "User-Agent": "opencode/1.18.18",
            "x-opencode-session": "cross-turn-refresh-regression",
        }
    elif harness == "deepseek-harness":
        headers = {
            "User-Agent": "deepseek-harness/0.9.0",
            "x-deepseek-harness-session-id": "cross-turn-refresh-regression",
        }
    else:
        raise ValueError(f"unknown harness: {harness}")
    return "/v1/chat/completions", payload, headers


def run_request(
    base_url: str,
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int,
) -> dict:
    payload["max_tokens"] = 1
    payload["stream"] = False
    started = time.monotonic()
    status, body = post(base_url + endpoint, payload, headers, timeout)
    elapsed = time.monotonic() - started
    private_leak = any(marker in body for marker in PRIVATE_MARKERS)
    try:
        response: object = json.loads(body)
    except json.JSONDecodeError:
        response = body
    return {
        "http_status": status,
        "elapsed_sec": elapsed,
        "private_query_leaked": private_leak,
        "passed": status == 200 and not private_leak,
        "response": response,
    }


def log_evidence(log_text: str, harness: str) -> dict:
    escaped = re.escape(harness)
    request_boundary = re.findall(
        rf"KVMem automatic retrieval harness={escaped}[^\n]*"
        rf"trigger=request-boundary",
        log_text,
    )
    suppressed = re.findall(
        rf"KVMem harness continuation gate harness={escaped}[^\n]*"
        rf"reselect=off trajectory_delta_tokens=(\d+)[^\n]*threshold=(\d+)",
        log_text,
    )
    cross_turn = re.findall(
        rf"KVMem automatic retrieval harness={escaped}[^\n]*"
        rf"trigger=cross-tool-turn[^\n]*trajectory_delta_tokens=(\d+)",
        log_text,
    )
    return {
        "request_boundary_refreshes": len(request_boundary),
        "suppressed_continuations": [
            {"delta_tokens": int(delta), "threshold_tokens": int(threshold)}
            for delta, threshold in suppressed
        ],
        "cross_tool_refreshes": [int(delta) for delta in cross_turn],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ballast-tokens", type=int, default=70000)
    parser.add_argument("--growth-repetitions", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--harnesses", default="claude-code,opencode,deepseek-harness"
    )
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    log_start = log_path.stat().st_size if log_path.exists() else 0
    ballast, ballast_tokens = make_ballast(
        args.base_url, args.ballast_tokens, args.timeout
    )
    harness_reports = []
    for harness in (name.strip() for name in args.harnesses.split(",")):
        if not harness:
            continue
        endpoint, payload, headers = request_spec(harness, ballast)
        phases = [
            {
                "phase": "establish-task",
                **run_request(
                    args.base_url, endpoint, payload, headers, args.timeout
                ),
            }
        ]
        append_tool_round(payload, harness, 1, "STATUS=unchanged\n")
        phases.append(
            {
                "phase": "below-threshold-tool-turn",
                **run_request(
                    args.base_url, endpoint, payload, headers, args.timeout
                ),
            }
        )
        growth = (
            "completed test telemetry changed=file-count stable=true; "
            * args.growth_repetitions
        )
        append_tool_round(payload, harness, 2, growth)
        phases.append(
            {
                "phase": "threshold-crossing-tool-turn",
                **run_request(
                    args.base_url, endpoint, payload, headers, args.timeout
                ),
            }
        )
        harness_reports.append(
            {"harness": harness, "endpoint": endpoint, "phases": phases}
        )

    # stderr is unbuffered, but tee may need a moment to publish its final line.
    time.sleep(0.2)
    log_slice = log_path.read_text(errors="replace")[log_start:]
    for item in harness_reports:
        evidence = log_evidence(log_slice, item["harness"])
        item["server_evidence"] = evidence
        item["passed"] = (
            all(phase["passed"] for phase in item["phases"])
            and evidence["request_boundary_refreshes"] >= 1
            and len(evidence["suppressed_continuations"]) >= 1
            and len(evidence["cross_tool_refreshes"]) >= 1
        )

    report = {
        "schema": "qw3.kvmem_cross_turn_refresh.v1",
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "ballast_tokens": ballast_tokens,
        "growth_repetitions": args.growth_repetitions,
        "harnesses": harness_reports,
        "passed": all(item["passed"] for item in harness_reports),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
