#!/usr/bin/env python3
"""Three-request GPU probe for durable guided-query prefix reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def post(endpoint: str, payload: dict) -> dict:
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def assistant_text(response: dict) -> str:
    content = response["choices"][0]["message"].get("content", "")
    return content if isinstance(content, str) else json.dumps(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--short-public-query",
        action="store_true",
        help=(
            "Use a deliberately tiny public query while leaving detailed "
            "trajectory evidence in history. This exercises restoration of a "
            "private guided query that is longer than the public query span."
        ),
    )
    args = parser.parse_args()

    telemetry = args.telemetry.read_text()
    task = (
        "Continue the unfinished investigation."
        if args.short_public_query
        else " ".join(
            [
                "Inspect the repository and preserve exact file names, symbols, tests,"
                " failure messages, protocol constraints, unfinished edits, and the next"
                " verification step while diagnosing durable guided-query snapshot reuse."
            ]
            * 16
        )
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "Archived trajectory:\n" + telemetry},
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "content": " ".join(
                [
                    "Historical tool progress inspected repository files and"
                    " recorded exact symbols tests errors constraints and"
                    " unfinished verification steps."
                ]
                * 512
            ),
        },
    ]
    common = {
        "model": "qw3/Qwen3.8-27B-Q8_0.gguf",
        "temperature": 0.0,
        "seed": 73,
        "max_tokens": 1,
        "enable_thinking": False,
        "kvmem_reselect": "force",
        "kvmem_query_span": {
            "message_index": 1,
            "content_start": 0,
            "content_end": len(task.encode()),
        },
    }
    responses = []
    first = post(
        args.endpoint,
        {
            **common,
            "messages": messages,
            "kvmem_query_guided_thinking_max_tokens": 1,
            "kvmem_query_guided_query_max_tokens": 256,
        },
    )
    responses.append(first)
    messages.append({"role": "assistant", "content": assistant_text(first)})
    second = post(args.endpoint, {**common, "messages": messages})
    responses.append(second)
    messages.append({"role": "assistant", "content": assistant_text(second)})
    third = post(args.endpoint, {**common, "messages": messages})
    responses.append(third)

    report = {
        "short_public_query": args.short_public_query,
        "requests": [
            {
                "usage": response.get("usage", {}),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "content": assistant_text(response),
            }
            for response in responses
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
