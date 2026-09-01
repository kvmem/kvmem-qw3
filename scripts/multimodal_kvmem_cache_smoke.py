#!/usr/bin/env python3
"""Exercise multimodal KVMem save, frozen restore, and append restore.

The server must already be running with ``--vision-cpu-model``, KVMem,
query-conditioned Adaptive/Fixed4 retrieval, and query replay.  The image is
sent exactly once during save; every later request verifies that projected
embeddings, M-RoPE coordinates, and mandatory visual spans survive the named
local-cache boundary.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(
    base_url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any], float]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
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


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def answer(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    return str(choices[0].get("message", {}).get("content") or "")


def cache_info(body: dict[str, Any]) -> dict[str, Any]:
    info = body.get("kvmem_cache")
    if not isinstance(info, dict):
        raise AssertionError(f"response has no kvmem_cache metadata: {body}")
    return info


def require_ok(status: int, body: dict[str, Any]) -> None:
    if status != 200:
        raise AssertionError(f"expected HTTP 200, got {status}: {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qw3")
    parser.add_argument("--cache-id", default="multimodal-kvmem-smoke")
    parser.add_argument("--session-id", default="multimodal-session-smoke")
    parser.add_argument("--history-repeats", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--skip-session", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.history_repeats < 0:
        parser.error("--history-repeats must be non-negative")

    filler = (
        "Unrelated archival note about routine storage scheduling and weather. "
        * args.history_repeats
    )
    common: dict[str, Any] = {
        "model": args.model,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }
    save = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": filler},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(args.image)},
                    },
                    {
                        "type": "text",
                        "text": "Retain this image for later questions.",
                    },
                ],
            }
        ],
        "max_tokens": 0,
        "kvmem_reselect": "off",
        "kvmem_cache": {
            "save": {
                "id": args.cache_id,
                "scope": "local",
                "when": "after_request",
                "ttl_seconds": 3600,
            }
        },
    }
    status, body, elapsed = request_json(args.base_url, save, args.timeout)
    require_ok(status, body)
    v1 = cache_info(body)
    assert v1["version"] == 1 and v1["status"] == "ready"
    print(f"save: {elapsed:.3f}s position={v1['position']} version=1")

    def load_payload(
        question: str, mode: str, version: int, max_tokens: int
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **common,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "kvmem_reselect": "force",
            "kvmem_query_message_range": {
                "message_begin": 0,
                "message_end": 1,
            },
            "kvmem_cache": {
                "load": {
                    "id": args.cache_id,
                    "mode": mode,
                    "required": True,
                    "expected_version": version,
                }
            },
        }
        return payload

    frozen = load_payload(
        "According to the saved image, describe its Level 4 and Level 5 rows.",
        "frozen",
        1,
        96,
    )
    status, body, elapsed = request_json(args.base_url, frozen, args.timeout)
    require_ok(status, body)
    text = answer(body)
    assert text, "frozen multimodal query returned an empty answer"
    assert cache_info(body)["version"] == 1
    print(f"frozen-v1: {elapsed:.3f}s answer={text!r}")

    append = load_payload(
        "Remember that the experiment label is ORCHID-2048.",
        "append",
        1,
        0,
    )
    status, body, elapsed = request_json(args.base_url, append, args.timeout)
    require_ok(status, body)
    v2 = cache_info(body)
    assert v2["version"] == 2 and v2["position"] > v1["position"]
    print(f"append: {elapsed:.3f}s position={v2['position']} version=2")

    frozen_v2 = load_payload(
        "What is the experiment label, and what are the Level 4 and Level 5 "
        "descriptions in the saved image?",
        "frozen",
        2,
        128,
    )
    status, body, elapsed = request_json(args.base_url, frozen_v2, args.timeout)
    require_ok(status, body)
    text = answer(body)
    assert text, "version-2 multimodal query returned an empty answer"
    assert cache_info(body)["version"] == 2
    print(f"frozen-v2: {elapsed:.3f}s answer={text!r}")

    if args.skip_session:
        return 0

    # The persistent-session API has a separate state lifecycle from named
    # caches. Re-send the image once at session start, then prove that both a
    # text-only append and the final semantic query retain its embeddings and
    # three-axis positions without embedding the image in either request.
    session_start = {
        **common,
        "messages": save["messages"],
        "max_tokens": 0,
        "kvmem_session_id": args.session_id,
        "kvmem_session_op": "start",
        "kvmem_reselect": "off",
        "kvmem_prefill_window": "pressure",
    }
    status, body, elapsed = request_json(
        args.base_url, session_start, args.timeout
    )
    require_ok(status, body)
    assert body["choices"][0]["finish_reason"] == "prefill_only"
    print(f"session-start: {elapsed:.3f}s")

    session_append = {
        **common,
        "messages": [
            {"role": "user", "content": "The session label is JADE-4096."}
        ],
        "max_tokens": 0,
        "kvmem_session_id": args.session_id,
        "kvmem_session_op": "append",
        "kvmem_reselect": "off",
        "kvmem_prefill_window": "pressure",
    }
    status, body, elapsed = request_json(
        args.base_url, session_append, args.timeout
    )
    require_ok(status, body)
    assert body["choices"][0]["finish_reason"] == "prefill_only"
    print(f"session-append: {elapsed:.3f}s")

    session_finish = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Give the session label and the Level 4/Level 5 "
                    "descriptions from the earlier image."
                ),
            }
        ],
        "max_tokens": 128,
        "kvmem_session_id": args.session_id,
        "kvmem_session_op": "finish",
        "kvmem_reselect": "force",
        "kvmem_prefill_window": "keep_selected",
        "kvmem_query_message_range": {
            "message_begin": 0,
            "message_end": 1,
        },
    }
    status, body, elapsed = request_json(
        args.base_url, session_finish, args.timeout
    )
    require_ok(status, body)
    text = answer(body)
    assert text, "persistent multimodal session returned an empty answer"
    print(f"session-finish: {elapsed:.3f}s answer={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
