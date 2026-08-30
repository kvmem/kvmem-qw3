#!/usr/bin/env python3
"""GPU regression for stable tool continuations after guided reselection.

The first request is an above-budget stable tool trajectory that remains inside
the initial A+B grace window, then decodes across the configured mid-decode
threshold.  The second and third requests extend that exact transcript with
short tool results but remain below the next A+B epoch threshold.  Both must
resume the current-generation checkpoint without causing another semantic
selection.  This covers both the historical ``page table too small`` failure
and the partial raw-K staging tail that used to leave M non-resumable, while
matching the production policy (one refresh per large generation epoch rather
than the invalid stress pattern of refreshing every few tokens).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from test_harness_long_trace import make_ballast, post


def read_log_from(
    path: pathlib.Path, byte_offset: int, byte_end: int | None = None
) -> str:
    if not path.exists():
        return ""
    return path.read_bytes()[byte_offset:byte_end].decode(
        "utf-8", errors="replace"
    )


def request(base_url: str, payload: dict, timeout: int) -> tuple[int, dict, str]:
    status, body = post(
        base_url.rstrip("/") + "/v1/chat/completions",
        payload,
        {
            "User-Agent": "opencode/1.18.18",
            "x-opencode-session": "middecode-checkpoint-resume-regression",
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ballast-tokens", type=int, default=5820)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    log_start = log_path.stat().st_size if log_path.exists() else 0
    ballast, measured_ballast = make_ballast(
        args.base_url, args.ballast_tokens, args.timeout
    )
    messages: list[dict] = [
        {
            "role": "system",
            "content": "You are a coding agent. Continue the stable tool trajectory.",
        },
        {
            "role": "user",
            "content": (
                "Diagnose checkpoint restore safety. Preserve exact failures, "
                "symbols, and the next validation step."
            ),
        },
        {
            "role": "assistant",
            "content": "I will inspect the accumulated execution record.",
            "tool_calls": [
                {
                    "id": "call_history",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "build/history.log"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_history",
            "content": ballast,
        },
        {
            "role": "assistant",
            "content": "The historical record is loaded; I will check its final status.",
            "tool_calls": [
                {
                    "id": "call_status",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "build/status.log"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_status",
            "content": "CHECKPOINT_STATUS=ready; no failure was recorded.",
        },
        {
            "role": "user",
            "content": (
                "This is a deterministic decode-length regression, not a tool "
                "task. Do not call or mention any tool. Produce exactly 700 "
                "numbered checkpoint observations in plain text before "
                "stopping; every observation must repeat the words CHECKPOINT "
                "STATE STABLE."
            ),
        },
    ]
    common = {
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 73,
        # Keep the generated suffix visible when replayed as history so the
        # second request has an exact LCP through the post-selection checkpoint.
        "enable_thinking": False,
        "stream": False,
    }
    first_status, first, first_body = request(
        args.base_url,
        {**common, "messages": messages, "max_tokens": args.max_tokens},
        args.timeout,
    )
    first_message = (
        first.get("choices", [{}])[0].get("message", {})
        if isinstance(first, dict)
        else {}
    )
    if first_status != 200 or not first_message:
        report = {
            "schema": "qw3.kvmem_middecode_checkpoint_resume.v2",
            "ballast_tokens": measured_ballast,
            "first_http_status": first_status,
            "first_body_prefix": first_body[:1000],
            "passed": False,
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    # Preserve the exact server-returned assistant message, then add a stable
    # tool callback.  There is no new user task, so request-boundary semantic
    # reselection is not permitted on this continuation.
    messages.append(first_message)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "I will now verify the checkpoint state.",
                "tool_calls": [
                    {
                        "id": "call_verify",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {"path": "build/checkpoint-state.log"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_verify",
                "content": "CHECKPOINT_STATUS=ready; report the next step briefly.",
            },
        ]
    )
    first_log_end = log_path.stat().st_size
    second_status, second, second_body = request(
        args.base_url,
        {**common, "messages": messages, "max_tokens": 32},
        args.timeout,
    )
    second_message = (
        second.get("choices", [{}])[0].get("message", {})
        if isinstance(second, dict)
        else {}
    )
    time.sleep(0.2)
    second_log_end = log_path.stat().st_size
    if second_status == 200 and second_message:
        messages.append(second_message)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "I will confirm the final persisted checkpoint marker.",
                    "tool_calls": [
                        {
                            "id": "call_final_marker",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps(
                                    {"path": "build/final-checkpoint.log"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_final_marker",
                    "content": "FINAL_CHECKPOINT=durable; answer OK only.",
                },
            ]
        )
        third_status, third, third_body = request(
            args.base_url,
            {**common, "messages": messages, "max_tokens": 8},
            args.timeout,
        )
    else:
        third_status, third, third_body = 0, {}, "second request failed"
    time.sleep(0.2)
    log_slice = read_log_from(log_path, log_start)
    second_log = read_log_from(log_path, first_log_end, second_log_end)
    third_log = read_log_from(log_path, second_log_end)
    guided = len(re.findall(
        r"native kvmem middecode guided reselect: refresh=", log_slice
    ))
    second_guided = len(re.findall(
        r"native kvmem middecode guided reselect: refresh=", second_log
    ))
    third_guided = len(re.findall(
        r"native kvmem middecode guided reselect: refresh=", third_log
    ))
    tail_finalize = len(re.findall(r"\[bs-tail-index\]", log_slice))
    postselection_p_ready = bool(re.search(
        r"prefix-cache CAPTURE \(mtp\):[^\n]*P_resumable=1[^\n]*"
        r"P_selection_current=1[^\n]*P_rebased_after_selection=1",
        log_slice,
    ))
    # The one-token MTP bridge may immediately publish a newer M checkpoint
    # instead of retaining the boundary P as the current selection.  That M
    # is the preferred result as long as both its source index and MTP payload
    # passed checkpoint admission.
    postselection_m_ready = bool(re.search(
        r"prefix-cache CAPTURE \(mtp\):[^\n]*M_resumable=1[^\n]*"
        r"M_source_index=1",
        log_slice,
    ))
    postselection_checkpoint_ready = (
        postselection_p_ready or postselection_m_ready
    )
    # A decode-end M checkpoint is newer than the post-selection P checkpoint
    # and is the preferred resume point after MTP rebase.  Accept either: the
    # invariant is that the second request resumes a selection-current,
    # MTP-complete checkpoint instead of rebuilding the whole prefix.
    hit_postselection_checkpoint = bool(re.search(
        r"prefix-cache HIT \(mtp\):[^\n]*ckpt=(?:P|M)", second_log
    ))
    second_current_m_ready = bool(re.search(
        r"prefix-cache CAPTURE \(mtp\):[^\n]*M_selection_generation=(\d+)"
        r"[^\n]*M_resumable=1[^\n]*M_source_index=1",
        second_log,
    ))
    third_hit_current = bool(re.search(
        r"prefix-cache HIT \(mtp\):[^\n]*ckpt=(?:M|P)", third_log
    ))
    third_miss = "prefix-cache MISS (mtp)" in third_log
    page_error = "page table too small" in log_slice
    passed = (
        first_status == 200
        and second_status == 200
        and third_status == 200
        and guided >= 1
        and second_guided == 0
        and third_guided == 0
        and tail_finalize >= 1
        and postselection_checkpoint_ready
        and hit_postselection_checkpoint
        and second_current_m_ready
        and third_hit_current
        and not third_miss
        and not page_error
    )
    report = {
        "schema": "qw3.kvmem_middecode_checkpoint_resume.v2",
        "ballast_tokens": measured_ballast,
        "first_http_status": first_status,
        "second_http_status": second_status,
        "third_http_status": third_status,
        "guided_middecode_reselects": guided,
        "second_request_guided_reselects": second_guided,
        "third_request_guided_reselects": third_guided,
        "tail_index_finalizations": tail_finalize,
        "postselection_checkpoint_ready": postselection_checkpoint_ready,
        "postselection_p_ready": postselection_p_ready,
        "postselection_m_ready": postselection_m_ready,
        "second_request_hit_postselection_checkpoint": (
            hit_postselection_checkpoint
        ),
        "second_request_current_m_ready": second_current_m_ready,
        "third_request_hit_current_checkpoint": third_hit_current,
        "third_request_prefix_miss": third_miss,
        "page_table_error": page_error,
        "passed": passed,
        "first_response": first,
        "second_response": second,
        "third_response": third,
        "first_body_prefix": first_body[:500] if first_status != 200 else "",
        "second_body_prefix": second_body[:500] if second_status != 200 else "",
        "third_body_prefix": third_body[:500] if third_status != 200 else "",
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
