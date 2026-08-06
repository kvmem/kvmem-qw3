#!/usr/bin/env python3
"""Exercise the request-level, process-local KVMem checkpoint API.

The server must already be running with KVMem and query-conditioned mean-k.
This script verifies save, repeated frozen restore, append with version CAS,
status lookup, stale-version rejection, and deletion without resending history.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw}
        return exc.code, body, time.perf_counter() - started


def chat(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    max_tokens: int,
    cache: dict[str, Any],
    *,
    reselect: str,
    semantic_budget: int | None = None,
) -> tuple[int, dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
        "kvmem_reselect": reselect,
        "kvmem_cache": cache,
    }
    if semantic_budget is not None:
        payload["kvmem_semantic_budget"] = semantic_budget
    if reselect == "force":
        payload["kvmem_query_message_range"] = {
            "message_begin": 0,
            "message_end": len(messages),
        }
    return request_json(
        args.base_url, "POST", "/chat/completions", payload, args.timeout
    )


def cache_info(body: dict[str, Any]) -> dict[str, Any]:
    info = body.get("kvmem_cache")
    if not isinstance(info, dict):
        raise AssertionError(f"response has no kvmem_cache metadata: {body}")
    return info


def answer(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content") or "")


def require_status(status: int, expected: int, body: dict[str, Any]) -> None:
    if status != expected:
        raise AssertionError(f"expected HTTP {expected}, got {status}: {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qw3")
    parser.add_argument("--cache-id", default="local-cache-smoke")
    parser.add_argument("--history-repeats", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument(
        "--semantic-budgets",
        default="",
        help=(
            "comma-separated request-local semantic budgets; when set, the "
            "same frozen history is queried once per budget"
        ),
    )
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if args.history_repeats < 1:
        parser.error("--history-repeats must be positive")
    semantic_budgets: list[int | None]
    if args.semantic_budgets:
        try:
            semantic_budgets = [
                int(value.strip())
                for value in args.semantic_budgets.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            parser.error(f"invalid --semantic-budgets: {exc}")
        if not semantic_budgets or any(
            value is None or value <= 0 for value in semantic_budgets
        ):
            parser.error("--semantic-budgets must contain positive integers")
    else:
        semantic_budgets = [None, None]

    fact = "The archival access code is CERULEAN-7319."
    filler = (
        "This is an unrelated laboratory note about storage scheduling, "
        "weather, and routine maintenance."
    )
    history = "\n".join([fact] + [filler] * args.history_repeats)

    status, body, elapsed = chat(
        args,
        [{"role": "user", "content": history}],
        0,
        {
            "save": {
                "id": args.cache_id,
                "scope": "local",
                "when": "after_request",
                "ttl_seconds": args.ttl_seconds,
            }
        },
        reselect="off",
    )
    require_status(status, 200, body)
    info_v1 = cache_info(body)
    assert info_v1["status"] == "ready" and info_v1["version"] == 1
    assert body["choices"][0]["finish_reason"] == "prefill_only"
    print(
        f"save: {elapsed:.3f}s v={info_v1['version']} "
        f"position={info_v1['position']} fingerprint={info_v1['fingerprint']}"
    )

    frozen_answers: list[str] = []
    for index, semantic_budget in enumerate(semantic_budgets):
        status, body, elapsed = chat(
            args,
            [{"role": "user", "content": "What is the archival access code?"}],
            64,
            {
                "load": {
                    "id": args.cache_id,
                    "mode": "frozen",
                    "required": True,
                    "expected_version": 1,
                }
            },
            reselect="force",
            semantic_budget=semantic_budget,
        )
        require_status(status, 200, body)
        info = cache_info(body)
        assert info["version"] == 1 and info["position"] == info_v1["position"]
        frozen_answers.append(answer(body))
        budget_label = "server-default" if semantic_budget is None else str(
            semantic_budget
        )
        print(
            f"frozen[{index + 1}] budget={budget_label}: "
            f"{elapsed:.3f}s answer={frozen_answers[-1]!r}"
        )
    for text in frozen_answers:
        assert "CERULEAN-7319" in text, f"frozen restore lost saved fact: {text!r}"

    status, body, elapsed = chat(
        args,
        [{"role": "user", "content": "Remember that the new color is amber."}],
        0,
        {
            "load": {
                "id": args.cache_id,
                "mode": "append",
                "required": True,
                "expected_version": 1,
            }
        },
        reselect="force",
    )
    require_status(status, 200, body)
    info_v2 = cache_info(body)
    assert info_v2["version"] == 2
    assert info_v2["position"] > info_v1["position"]
    print(f"append: {elapsed:.3f}s v=2 position={info_v2['position']}")

    status, body, elapsed = chat(
        args,
        [{"role": "user", "content": "What is the new color? Reply with one word."}],
        32,
        {
            "load": {
                "id": args.cache_id,
                "mode": "frozen",
                "required": True,
                "expected_version": 2,
            }
        },
        reselect="force",
    )
    require_status(status, 200, body)
    color_answer = answer(body)
    assert "amber" in color_answer.lower(), color_answer
    print(f"frozen-v2: {elapsed:.3f}s answer={color_answer!r}")

    status, body, _ = chat(
        args,
        [{"role": "user", "content": "This stale append must fail."}],
        0,
        {
            "load": {
                "id": args.cache_id,
                "mode": "append",
                "required": True,
                "expected_version": 1,
            }
        },
        reselect="force",
    )
    require_status(status, 409, body)
    print("stale append: HTTP 409 (expected)")

    status, body, _ = request_json(
        args.base_url, "GET", f"/kvmem/caches/{args.cache_id}", None, args.timeout
    )
    require_status(status, 200, body)
    assert cache_info(body)["version"] == 2
    print("status: ready v=2")

    if not args.keep_cache:
        status, body, _ = request_json(
            args.base_url,
            "DELETE",
            f"/kvmem/caches/{args.cache_id}",
            None,
            args.timeout,
        )
        require_status(status, 200, body)
        print("delete: evicted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
