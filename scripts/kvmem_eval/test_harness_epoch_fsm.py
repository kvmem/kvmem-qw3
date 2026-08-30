#!/usr/bin/env python3
"""End-to-end regression for the A/B/T harness reselection state machine.

For each supported coding harness this sends one stable logical task through:

1. a prompt below A (remember the task, no selection),
2. the same task just above A (retain the initial dense A+B epoch),
3. the same task above A+T (private guided refresh), and
4. a short tool continuation (retain the refreshed A+B epoch), and
5. a genuinely new user instruction above A (immediate real-query refresh).

The response and server log are both audited. Private retrieval text must never
appear on the wire, and the expected reason-coded transitions must occur once.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from test_cross_turn_refresh import append_tool_round, request_spec, run_request
from test_harness_long_trace import PRIVATE_MARKERS, make_ballast


def read_log_since(path: pathlib.Path, byte_offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(byte_offset)
        return stream.read().decode("utf-8", "replace")


def append_real_user(payload: dict, harness: str, ordinal: int) -> None:
    text = (
        f"NEW USER QUERY {ordinal}: inspect src/critical.cpp and verify the "
        "MARIGOLD_7319 acceptance condition before continuing."
    )
    if harness == "claude-code":
        payload["messages"].extend(
            [
                {"role": "assistant", "content": "The previous tool round is complete."},
                {"role": "user", "content": text},
            ]
        )
    else:
        payload["messages"].extend(
            [
                {"role": "assistant", "content": "The previous tool round is complete."},
                {"role": "user", "content": text},
            ]
        )


def prompt_tokens_from_log(log_text: str, harness: str) -> list[int]:
    escaped = re.escape(harness)
    return [
        int(value)
        for value in re.findall(
            rf"KVMem automatic retrieval harness={escaped}[^\n]*"
            rf"prompt_tokens=(\d+)",
            log_text,
        )
    ]


def phase_evidence(log_text: str, harness: str) -> dict:
    """Return request-local evidence for one FSM phase.

    In particular, a block-aligned prefix checkpoint is not allowed to turn a
    KeepSelected A+B grace request into an A-sized pressure window.  That bug
    lives below the server FSM, so checking only the reason-coded server log is
    insufficient: reject any explicit selection that actually drops blocks.
    """
    escaped = re.escape(harness)
    explicit = [
        (int(source), int(selected))
        for source, selected in re.findall(
            r"\[kvmem-reselect-perf\] kind=explicit[^\n]*"
            r"source_blocks=(\d+) selected_blocks=(\d+)",
            log_text,
        )
    ]
    return {
        "prompt_tokens": prompt_tokens_from_log(log_text, harness),
        "grace_gates": len(
            re.findall(
                rf"KVMem harness continuation gate harness={escaped}[^\n]*"
                r"\sheadroom_grace=1",
                log_text,
            )
        ),
        "initial_grace_gates": len(
            re.findall(
                rf"KVMem harness continuation gate harness={escaped}[^\n]*"
                r"initial_headroom_grace=1",
                log_text,
            )
        ),
        "initial_headroom_refreshes": len(
            re.findall(
                rf"KVMem automatic retrieval harness={escaped}[^\n]*"
                r"trigger=initial-headroom",
                log_text,
            )
        ),
        "semantic_reselections": len(
            re.findall(
                r"\[kvmem-reselect-perf\] kind=semantic\b", log_text
            )
        ),
        "guided_queries": len(
            re.findall(r"native kvmem guided query:", log_text)
        ),
        "prefix_cache_hits": len(
            re.findall(r"kvmem prefix-cache HIT", log_text)
        ),
        "explicit_selections": [
            {"source_blocks": source, "selected_blocks": selected}
            for source, selected in explicit
        ],
        "explicit_pressure_reductions": sum(
            selected < source for source, selected in explicit
        ),
    }


def evidence(log_text: str, harness: str) -> dict:
    escaped = re.escape(harness)
    grace = re.findall(
        rf"KVMem harness continuation gate harness={escaped}[^\n]*"
        rf"initial_headroom_grace=1[^\n]*middecode_until_refresh=(\d+)",
        log_text,
    )
    initial = re.findall(
        rf"KVMem automatic retrieval harness={escaped}[^\n]*"
        rf"trigger=initial-headroom",
        log_text,
    )
    new_user = re.findall(
        rf"KVMem automatic retrieval harness={escaped}[^\n]*"
        rf"trigger=new-user-query",
        log_text,
    )
    guided = re.findall(
        r"native kvmem guided query:[^\n]*private_prompt_tokens=(\d+)"
        r"[^\n]*private_prompt_ms=([0-9.]+)"
        r"[^\n]*generated_query_ms=([0-9.]+)"
        r"[^\n]*state_ms=([0-9.]+)",
        log_text,
    )
    pin_lines = re.findall(
        rf"KVMem harness pins harness={escaped}[^\n]*", log_text
    )
    pin_rows = [
        {
            key: int(value)
            for key, value in re.findall(r"\b([a-z_]+)=(\d+)\b", line)
        }
        for line in pin_lines
    ]
    system_block_counts = [row.get("blocks_system", -1) for row in pin_rows]
    return {
        "grace_remaining_tokens": [int(value) for value in grace],
        "initial_headroom_refreshes": len(initial),
        "new_user_refreshes": len(new_user),
        "guided_query_breakdown": [
            {
                "private_prompt_tokens": int(tokens),
                "private_prompt_ms": float(prompt_ms),
                "generated_query_ms": float(query_ms),
                "state_ms": float(state_ms),
            }
            for tokens, prompt_ms, query_ms, state_ms in guided
        ],
        "automatic_prompt_tokens": prompt_tokens_from_log(log_text, harness),
        "mandatory_pin_rows": pin_rows,
        "system_control_blocks_fixed": (
            bool(system_block_counts)
            and len(set(system_block_counts)) == 1
            and system_block_counts[0] > 0
        ),
        "score_query_and_live_suffix_independent": any(
            row.get("score_query_message") != row.get("live_suffix_message")
            for row in pin_rows
        ),
        "old_trajectory_returns_to_retrieval_pool": any(
            row.get("live_suffix_blocks", 1 << 30) <= 8
            and row.get("mandatory_tokens", 1 << 30) <= 4096
            and row.get("score_query_message", -1) >= 0
            for row in pin_rows[-1:]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-ballast-tokens", type=int, default=58000)
    parser.add_argument("--cross-a-tokens", type=int, default=12000)
    parser.add_argument("--cross-at-tokens", type=int, default=26000)
    parser.add_argument("--post-refresh-tokens", type=int, default=1000)
    parser.add_argument("--selection-budget", type=int, default=65536)
    parser.add_argument("--generation-budget", type=int, default=32768)
    parser.add_argument("--trigger-tokens", type=int, default=28672)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--harnesses", default="claude-code,opencode,deepseek-harness"
    )
    args = parser.parse_args()

    log_path = pathlib.Path(args.server_log)
    log_start = log_path.stat().st_size if log_path.exists() else 0
    initial, initial_tokens = make_ballast(
        args.base_url, args.initial_ballast_tokens, args.timeout
    )
    cross_a, cross_a_tokens = make_ballast(
        args.base_url, args.cross_a_tokens, args.timeout
    )
    cross_at, cross_at_tokens = make_ballast(
        args.base_url, args.cross_at_tokens, args.timeout
    )
    post_refresh, post_refresh_tokens = make_ballast(
        args.base_url, args.post_refresh_tokens, args.timeout
    )

    reports: list[dict] = []
    for harness in (item.strip() for item in args.harnesses.split(",")):
        if not harness:
            continue
        endpoint, payload, headers = request_spec(harness, initial)
        def run_phase(name: str) -> dict:
            phase_log_start = log_path.stat().st_size if log_path.exists() else 0
            result = run_request(
                args.base_url, endpoint, payload, headers, args.timeout
            )
            time.sleep(0.1)
            phase_log = read_log_since(log_path, phase_log_start)
            return {
                "phase": name,
                **result,
                "server_evidence": phase_evidence(phase_log, harness),
            }

        phases = [run_phase("below-A-establish")]
        append_tool_round(payload, harness, 1, cross_a)
        phases.append(run_phase("above-A-grace"))
        append_tool_round(payload, harness, 2, cross_at)
        phases.append(run_phase("above-A-plus-T-refresh"))
        append_tool_round(payload, harness, 3, post_refresh)
        phases.append(run_phase("post-refresh-headroom-grace"))
        append_real_user(payload, harness, 4)
        phases.append(run_phase("new-user-immediate-refresh"))
        reports.append({"harness": harness, "phases": phases})

    time.sleep(0.5)
    log_slice = read_log_since(log_path, log_start)
    for report in reports:
        report["server_evidence"] = evidence(log_slice, report["harness"])
        ev = report["server_evidence"]
        phase_ev = {
            phase["phase"]: phase["server_evidence"]
            for phase in report["phases"]
        }
        grace_ev = phase_ev["above-A-grace"]
        refresh_ev = phase_ev["above-A-plus-T-refresh"]
        post_refresh_ev = phase_ev["post-refresh-headroom-grace"]
        report["passed"] = (
            all(phase["passed"] for phase in report["phases"])
            and not any(
                any(marker in json.dumps(phase["response"], ensure_ascii=False)
                    for marker in PRIVATE_MARKERS)
                for phase in report["phases"]
            )
            and len(ev["grace_remaining_tokens"]) >= 1
            and ev["initial_headroom_refreshes"] >= 1
            and ev["new_user_refreshes"] >= 1
            and len(ev["mandatory_pin_rows"]) >= 4
            and ev["system_control_blocks_fixed"]
            and ev["score_query_and_live_suffix_independent"]
            and ev["old_trajectory_returns_to_retrieval_pool"]
            and grace_ev["grace_gates"] >= 1
            and grace_ev["prefix_cache_hits"] >= 1
            and grace_ev["initial_headroom_refreshes"] == 0
            and grace_ev["semantic_reselections"] == 0
            and grace_ev["guided_queries"] == 0
            and grace_ev["explicit_pressure_reductions"] == 0
            and refresh_ev["initial_headroom_refreshes"] >= 1
            and refresh_ev["semantic_reselections"] >= 1
            and refresh_ev["guided_queries"] >= 1
            and refresh_ev["explicit_pressure_reductions"] == 0
            and post_refresh_ev["grace_gates"] >= 1
            and post_refresh_ev["prefix_cache_hits"] >= 1
            and post_refresh_ev["initial_headroom_refreshes"] == 0
            and post_refresh_ev["semantic_reselections"] == 0
            and post_refresh_ev["guided_queries"] == 0
            and post_refresh_ev["explicit_pressure_reductions"] == 0
            and all(
                row.get("spans_system") == 1
                and row.get("mandatory_tokens", 1 << 30)
                <= row.get("budget_tokens", 0)
                for row in ev["mandatory_pin_rows"]
            )
        )

    output = {
        "schema": "qw3.kvmem_harness_epoch_fsm.v1",
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "selection_budget_tokens": args.selection_budget,
        "generation_budget_tokens": args.generation_budget,
        "trigger_tokens": args.trigger_tokens,
        "block_tokens": 32,
        "input_content_tokens": {
            "initial": initial_tokens,
            "cross_a": cross_a_tokens,
            "cross_a_plus_t": cross_at_tokens,
            "post_refresh": post_refresh_tokens,
        },
        "harnesses": reports,
        "passed": all(report["passed"] for report in reports),
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
