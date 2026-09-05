#!/usr/bin/env python3
"""Verify warm LM-prefix reuse when a full-transcript request adds an image.

Start ``qw3 serve`` with image input, KVMem, ``--kvmem-prefix-cache``, and
``QW3_KVMEM_PREFIX_CACHE_TRACE=1``. The first turn contains image A. The second
turn resends A in the accumulated transcript and appends image B. A passing
server log contains a KVMem prefix-cache HIT on the second request even though
the ordered request-level image fingerprint changed from [A] to [A, B]. The
optional third append also covers immutable raw-K publication across two warm
resumes.
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


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_json(
    base_url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            if response.status != 200:
                raise AssertionError(f"HTTP {response.status}: {body}")
            return body
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise AssertionError(f"HTTP {error.code}: {body}") from error


def answer(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--replacement-image",
        type=Path,
        help=(
            "optional same-grid image used to replace the first image in a "
            "third request; the server log must report a prefix-cache MISS"
        ),
    )
    parser.add_argument(
        "--append-third",
        action="store_true",
        help="append a third image turn and require another warm resume",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--model", default="qw3")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.replacement_image and not args.replacement_image.is_file():
        parser.error(
            f"replacement image does not exist: {args.replacement_image}")

    url = image_data_url(args.image)
    first_user: dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Remember this first image."},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Reply with a short acknowledgement."},
        ],
    }
    common: dict[str, Any] = {
        "model": args.model,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }

    started = time.perf_counter()
    first = request_json(
        args.base_url,
        {**common, "messages": [first_user], "max_tokens": 8},
        args.timeout,
    )
    first_elapsed = time.perf_counter() - started
    first_answer = answer(first)
    if not first_answer:
        raise AssertionError(f"first request returned no answer: {first}")

    second_user: dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Now compare it with this new image."},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
    started = time.perf_counter()
    second = request_json(
        args.base_url,
        {
            **common,
            "messages": [
                first_user,
                {"role": "assistant", "content": first_answer},
                second_user,
            ],
            "max_tokens": 8,
        },
        args.timeout,
    )
    second_elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "first_seconds": round(first_elapsed, 3),
        "second_seconds": round(second_elapsed, 3),
        "first_answer": first_answer,
        "second_answer": answer(second),
        "instruction": (
            "Confirm the server log reports 'kvmem prefix-cache HIT' for the "
            "second request and vision_cache_hits=2 vision_cache_misses=0."
        ),
    }

    second_answer = answer(second)
    if not second_answer:
        raise AssertionError(f"second request returned no answer: {second}")
    if args.append_third:
        third_user: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare one more observation."},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
        started = time.perf_counter()
        third = request_json(
            args.base_url,
            {
                **common,
                "messages": [
                    first_user,
                    {"role": "assistant", "content": first_answer},
                    second_user,
                    {"role": "assistant", "content": second_answer},
                    third_user,
                ],
                "max_tokens": 1,
            },
            args.timeout,
        )
        result["third_seconds"] = round(time.perf_counter() - started, 3)
        result["third_answer"] = answer(third)
        result["third_instruction"] = (
            "Confirm the third append reports another prefix-cache HIT and no "
            "immutable raw-K uncaptured-token error."
        )

    if args.replacement_image:
        replacement_url = image_data_url(args.replacement_image)
        replaced_first_user = json.loads(json.dumps(first_user))
        replaced_first_user["content"][1]["image_url"]["url"] = replacement_url
        started = time.perf_counter()
        replaced = request_json(
            args.base_url,
            {
                **common,
                "messages": [
                    replaced_first_user,
                    {"role": "assistant", "content": first_answer},
                    second_user,
                ],
                "max_tokens": 1,
            },
            args.timeout,
        )
        result["replacement_seconds"] = round(
            time.perf_counter() - started, 3)
        result["replacement_answer"] = answer(replaced)
        result["replacement_instruction"] = (
            "Confirm the third request reports prefix-cache MISS with visual_D "
            "at the first changed visual token."
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
