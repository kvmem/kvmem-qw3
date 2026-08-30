#!/usr/bin/env python3
"""Exercise Claude Code, OpenCode, and DSH-style requests above 64K.

The trace deliberately contains one large *completed* tool result followed by
one tiny live tool transaction.  A correct server may retrieve the former but
must pin only the latter; the private guided query must never appear in the API
response.  Results are written as one auditable JSON document.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request


SENTINEL = "MARIGOLD_7319"
PRIVATE_MARKERS = ("PRIVATE MEMORY RETRIEVAL TASK", "Retrieval query:")
CONTINUATION = (
    "Continue and finish the original task using the archived evidence; "
    "do not replace its exact acceptance criteria."
)


def post(url: str, payload: dict, headers: dict[str, str], timeout: int) -> tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def token_count(base_url: str, text: str, timeout: int) -> int:
    status, body = post(
        base_url + "/v1/tokenize", {"content": text}, {}, timeout
    )
    if status != 200:
        raise RuntimeError(f"tokenize failed: HTTP {status}: {body[:500]}")
    return int(json.loads(body)["count"])


def make_ballast(base_url: str, target_tokens: int, timeout: int) -> tuple[str, int]:
    line = (
        "historical build observation file=src/archive_worker.cpp "
        "symbol=restore_checkpoint test=archive_resume expected=stable; "
        "the completed tool round remains a retrieval candidate.\n"
    )
    lo, hi = 1, max(2, target_tokens // 4)
    while token_count(base_url, line * hi, timeout) < target_tokens:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(base_url, line * mid, timeout) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    text = (
        "<system-reminder>\n"
        "Instructions from: AGENTS.md\n"
        "Run focused tests before reporting completion.\n"
        "</system-reminder>\n"
        f"CRITICAL_EVIDENCE={SENTINEL} file=src/critical.cpp line=417\n"
        + line * lo
    )
    return text, token_count(base_url, text, timeout)


def openai_payload(ballast: str) -> dict:
    return {
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent. Preserve project policy, use the "
                    "provided tools, and continue unfinished work accurately."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Inspect the repository history and finish the task. The "
                    f"final answer must recover the critical evidence {SENTINEL}."
                ),
            },
            {
                "role": "assistant",
                "content": "I will inspect the archived build output.",
                "tool_calls": [
                    {
                        "id": "call_archive",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "build/archive.log"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_archive", "content": ballast},
            {
                "role": "assistant",
                "content": "The archived output is now part of the history.",
            },
            {"role": "user", "content": CONTINUATION},
            {
                "role": "assistant",
                "content": "The archive is recorded. I will check the current status.",
                "tool_calls": [
                    {
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "status.txt"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_status",
                "content": "STATUS=ready. Continue from the archived evidence.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one repository file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }


def anthropic_payload(ballast: str) -> dict:
    return {
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "system": (
            "You are Claude Code. Preserve project policy, use tools, and "
            "continue unfinished repository work accurately."
        ),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Inspect the repository history and finish the task. The "
                    f"final answer must recover the critical evidence {SENTINEL}."
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect the archived build output."},
                    {
                        "type": "tool_use",
                        "id": "call_archive",
                        "name": "read_file",
                        "input": {"path": "build/archive.log"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_archive",
                        "content": ballast,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The archived output is now part of the history.",
            },
            {"role": "user", "content": CONTINUATION},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will check the current status."},
                    {
                        "type": "tool_use",
                        "id": "call_status",
                        "name": "read_file",
                        "input": {"path": "status.txt"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_status",
                        "content": "STATUS=ready. Continue from archived evidence.",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Read one repository file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }


def run_case(
    base_url: str,
    harness: str,
    ballast: str,
    timeout: int,
    max_tokens: int,
    ignore_eos: bool,
) -> dict:
    if harness == "claude-code":
        endpoint = "/v1/messages"
        payload = anthropic_payload(ballast)
        headers = {"User-Agent": "claude-cli/2.1.148"}
    else:
        endpoint = "/v1/chat/completions"
        payload = openai_payload(ballast)
        if harness == "opencode":
            headers = {
                "User-Agent": "opencode/1.18.18",
                "x-opencode-session": "long-trace-regression",
            }
        else:
            headers = {
                "User-Agent": "deepseek-harness/0.9.0",
                "x-deepseek-harness-session-id": "long-trace-regression",
            }
    payload["max_tokens"] = max_tokens
    if ignore_eos:
        payload["ignore_eos"] = True
    started = time.monotonic()
    status, body = post(base_url + endpoint, payload, headers, timeout)
    elapsed = time.monotonic() - started
    private_leak = any(marker in body for marker in PRIVATE_MARKERS)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    return {
        "harness": harness,
        "endpoint": endpoint,
        "http_status": status,
        "elapsed_sec": elapsed,
        "private_query_leaked": private_leak,
        "passed": status == 200 and not private_leak,
        "response": parsed if parsed is not None else body,
    }


def run_oversize_exact_case(
    base_url: str, ballast: str, timeout: int
) -> dict:
    """A genuinely oversized exact system span must fail as 413, not 500."""
    payload = openai_payload(ballast)
    payload["messages"][0]["content"] = (
        "SYSTEM POLICY MUST REMAIN EXACT.\n" + ballast
    )
    payload["max_tokens"] = 8
    started = time.monotonic()
    status, body = post(
        base_url + "/v1/chat/completions",
        payload,
        {
            "User-Agent": "opencode/1.18.18",
            "x-opencode-session": "oversize-exact-regression",
        },
        timeout,
    )
    elapsed = time.monotonic() - started
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    return {
        "harness": "opencode-oversize-exact",
        "endpoint": "/v1/chat/completions",
        "http_status": status,
        "elapsed_sec": elapsed,
        "expected_http_status": 413,
        "passed": status == 413,
        "response": parsed if parsed is not None else body,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ballast-tokens", type=int, default=70000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--include-overflow-case", action="store_true")
    parser.add_argument(
        "--harnesses",
        default="claude-code,opencode,deepseek-harness",
    )
    args = parser.parse_args()
    ballast, ballast_tokens = make_ballast(
        args.base_url, args.ballast_tokens, args.timeout
    )
    results = [
        run_case(
            args.base_url,
            name.strip(),
            ballast,
            args.timeout,
            args.max_tokens,
            args.ignore_eos,
        )
        for name in args.harnesses.split(",")
        if name.strip()
    ]
    if args.include_overflow_case:
        results.append(
            run_oversize_exact_case(
                args.base_url, ballast, args.timeout
            )
        )
    report = {
        "schema": "qw3.kvmem_harness_long_trace.v1",
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "selection_budget_tokens": 65536,
        "generation_budget_tokens": 32768,
        "block_tokens": 32,
        "ballast_tokens": ballast_tokens,
        "sentinel": SENTINEL,
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
