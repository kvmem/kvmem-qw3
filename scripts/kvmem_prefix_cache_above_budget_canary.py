#!/usr/bin/env python3
"""Real-model canary for the KVMem selection-budget threshold crossing.

The default route constructs turn A just below the selection budget and forces
enough deterministic decode tokens to move its warm state above the threshold.
Turn B is a strict extension and must reuse the prepared P/M source index plus
the durable Query snapshot instead of pre-filling the complete history twice.

``--prefill-only-first`` exercises the second capture boundary: turn A commits
the same below-budget preparation with ``max_tokens=0`` and turn B's appended
suffix crosses the threshold. Both routes verify that preparation did not run a
semantic scorer before the threshold and that the first above-budget request is
a warm suffix-only hit.

This is the small-context analogue of a long OpenCode conversation whose warm
state moves from 229,282 tokens to 229,477 tokens during decode, then receives a
new query-conditioned request at 229,574 tokens.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(port: int, timeout_s: float,
                proc: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    last: BaseException | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"server exited before health check (exit={proc.returncode})")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = response.read()
            conn.close()
            if response.status == 200 and b"ok" in body:
                return
        except BaseException as exc:  # noqa: BLE001 - report final connection error
            last = exc
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy: {last}")


def post_chat(port: int, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    conn.request(
        "POST", "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    raw = response.read().decode("utf-8", errors="replace")
    status = response.status
    conn.close()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {"raw": raw}
    return status, body


def count_tokens(port: int, text: str, timeout_s: float) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    conn.request(
        "POST", "/v1/tokenize",
        body=json.dumps({"content": text}),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    raw = response.read().decode("utf-8", errors="replace")
    status = response.status
    conn.close()
    if status != 200:
        raise RuntimeError(f"tokenize failed: status={status} body={raw}")
    body = json.loads(raw)
    count = body.get("count")
    if not isinstance(count, int):
        raise RuntimeError(f"tokenize returned no integer count: {body}")
    return count


def render_single_user(content: str, add_generation_prompt: bool) -> str:
    rendered = f"<|im_start|>user\n{content.strip()}<|im_end|>\n"
    if add_generation_prompt:
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return rendered


def fit_content_below_budget(
    port: int,
    prefix: str,
    budget: int,
    margin: int,
    initial_repeats: int,
    add_generation_prompt: bool,
    timeout_s: float,
) -> tuple[str, int, int]:
    """Return content whose exact rendered prompt is near but below budget."""
    target = budget - margin
    if target <= 0:
        raise ValueError("threshold margin must be smaller than the budget")

    def candidate(repeats: int) -> str:
        return prefix + " routine" * repeats

    base_count = count_tokens(
        port, render_single_user(candidate(0), add_generation_prompt), timeout_s)
    if base_count > target:
        raise RuntimeError(
            f"canary base prompt ({base_count}) exceeds target ({target})")

    low = 0
    high = max(1, initial_repeats)
    while count_tokens(
        port, render_single_user(candidate(high), add_generation_prompt), timeout_s
    ) <= target:
        low = high
        high *= 2
        if high > budget * 8:
            raise RuntimeError("could not bracket the threshold prompt length")

    while low + 1 < high:
        mid = (low + high) // 2
        measured = count_tokens(
            port, render_single_user(candidate(mid), add_generation_prompt), timeout_s
        )
        if measured <= target:
            low = mid
        else:
            high = mid

    content = candidate(low)
    measured = count_tokens(
        port, render_single_user(content, add_generation_prompt), timeout_s)
    if not (0 < measured < budget):
        raise RuntimeError(
            f"failed to construct a below-budget prompt: {measured}/{budget}")
    return content, measured, low


def response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    return content if isinstance(content, str) else ""


def common_prefix_chars(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def start_server(args: argparse.Namespace, port: int, log_path: Path) -> tuple[subprocess.Popen[str], Any]:
    cmd = [
        str(args.qw3), "serve",
        "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx", str(args.ctx),
        "-n", str(max(args.max_tokens, args.crossing_tokens)),
        "--temp", "0",
        "--kv-dtype", "fp8",
        "--prefill-chunk", "512",
        "--kvmem", "--kvmem-prefix-cache",
        "--kvmem-block-tokens", str(args.block_tokens),
        "--kvmem-budget", str(args.budget),
        "--kvmem-prefill-budget", str(args.budget),
        "--kvmem-gen-budget", str(args.gen_budget),
        "--kvmem-sink-tokens", "128",
        "--kvmem-recent-tokens", "0",
        "--kvmem-method", "retrieval",
        "--kvmem-retrieval-method", "key-direction-adaptive",
        "--kvmem-index-placement", "gpu",
        "--kvmem-update-mode", "step",
        "--kvmem-query-conditioned",
        "--kvmem-immutable-k",
        "--kvmem-gpu-memory-ratio", "1.0",
        "--kvmem-cpu-gb", str(args.cpu_gb),
        "--kvmem-opt-stage-out", "off",
        "--kvmem-opt-stage-in", "on",
        "--kvmem-opt-pack", "on",
    ]
    if not args.plain:
        cmd.extend(["--native-mtp-speculate", "--mtp-chain", "4"])
    env = os.environ.copy()
    env.update({
        "QW3_KVMEM_PREFIX_CACHE_TRACE": "1",
        "QW3_KVMEM_TRACE": "1",
        "QW3_FATTN_NSPLIT": "1",
        "QW3_PREFILL_FA2_NSPLIT": "1",
    })
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    return proc, log_file


def stop_server(proc: subprocess.Popen[str], log_file: Any) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    log_file.close()


def payload(messages: list[dict[str, str]], query_start: int,
            query_end: int, max_tokens: int,
            ignore_eos: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": "qw3",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
        "kvmem_query_span": {
            "message_index": 0,
            "content_start": query_start,
            "content_end": query_end,
        },
    }
    if ignore_eos:
        out["ignore_eos"] = True
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qw3", type=Path, default=Path("./build/qw3"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--gen-budget", type=int, default=1024)
    parser.add_argument("--cpu-gb", type=int, default=2)
    parser.add_argument(
        "--filler-repeats", type=int, default=4096,
        help="initial upper bound for automatic threshold fitting")
    parser.add_argument("--threshold-margin", type=int, default=8)
    parser.add_argument("--crossing-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-only-first", action="store_true",
        help="capture turn A with max_tokens=0; turn B's suffix crosses budget")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--plain", action="store_true",
                        help="exercise plain decode instead of MTP=4")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out-json", type=Path,
                        default=Path("/tmp/qw3_kvmem_above_budget_mtp.json"))
    parser.add_argument("--warm-log", type=Path,
                        default=Path("/tmp/qw3_kvmem_above_budget_mtp_warm.log"))
    parser.add_argument("--cold-log", type=Path,
                        default=Path("/tmp/qw3_kvmem_above_budget_mtp_cold.log"))
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()
    args.qw3 = args.qw3.resolve()
    args.model = args.model.resolve()

    query = (
        "Which codename is assigned to the resilient cache protocol? "
        "Answer using the codename and one short sentence."
    )
    lead = " ".join([
        "Calibration preamble: preserve exact token ordering for this test."
    ] * 64) + "\n\n"
    first_prefix = lead + query + "\n\nThe codename is ORCHID-DELTA.\n\n"
    query_start = len(lead.encode())
    query_end = query_start + len(query.encode())

    warm_port = free_port()
    warm_proc, warm_file = start_server(args, warm_port, args.warm_log)
    status_a = status_b = 0
    body_a: dict[str, Any] = {}
    body_b: dict[str, Any] = {}
    messages_b: list[dict[str, str]] = []
    fitted_prompt_tokens = 0
    fitted_repeats = 0
    try:
        wait_health(warm_port, args.timeout, warm_proc)
        first_content, fitted_prompt_tokens, fitted_repeats = (
            fit_content_below_budget(
                warm_port,
                first_prefix,
                args.budget,
                args.threshold_margin,
                args.filler_repeats,
                add_generation_prompt=not args.prefill_only_first,
                timeout_s=args.timeout,
            )
        )
        messages_a = [{"role": "user", "content": first_content}]
        first_max_tokens = 0 if args.prefill_only_first else args.crossing_tokens
        status_a, body_a = post_chat(
            warm_port,
            payload(
                messages_a, query_start, query_end, first_max_tokens,
                ignore_eos=not args.prefill_only_first,
            ),
            args.timeout)
        answer_a = response_text(body_a)
        if status_a != 200:
            raise RuntimeError(f"turn A failed: status={status_a} body={body_a}")
        if args.prefill_only_first:
            crossing_suffix = (
                "Continue from the prepared memory and perform the final "
                "consistency check. " + "suffix " *
                max(args.crossing_tokens, args.threshold_margin + 8)
            )
            messages_b = [
                *messages_a,
                {"role": "user", "content": crossing_suffix},
            ]
        else:
            messages_b = [
                *messages_a,
                {"role": "assistant", "content": answer_a},
                {"role": "user", "content": "Now perform one final consistency check."},
            ]
        status_b, body_b = post_chat(
            warm_port,
            payload(messages_b, query_start, query_end, args.max_tokens),
            args.timeout,
        )
    finally:
        stop_server(warm_proc, warm_file)

    warm_log = args.warm_log.read_text(encoding="utf-8", errors="replace")
    warm_text = response_text(body_b)
    generate_lines = [line for line in warm_log.splitlines()
                      if "native generate:" in line]
    second_generate = generate_lines[-1] if generate_lines else ""
    first_complete_marker = (
        "native prefill-only:" if args.prefill_only_first
        else "native generate:"
    )
    first_complete_offset = warm_log.find(first_complete_marker)
    first_request_log = (
        warm_log[:first_complete_offset]
        if first_complete_offset >= 0 else warm_log
    )
    capture_lines = [
        line for line in warm_log.splitlines()
        if f"CAPTURE ({'plain' if args.plain else 'mtp'}" in line
    ]
    first_capture = capture_lines[0] if capture_lines else ""
    first_pos_match = re.search(r"\bpos=(\d+)", first_capture)
    first_end_pos = int(first_pos_match.group(1)) if first_pos_match else 0
    match = re.search(r"kvmem_reuse=(\d+)\s+prefilled=(\d+)", second_generate)
    prompt_match = re.search(r"prompt_tokens=(\d+)", second_generate)
    reuse = int(match.group(1)) if match else 0
    prefilled = int(match.group(2)) if match else 0
    prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0

    cold_status = 0
    cold_body: dict[str, Any] = {}
    cold_text = ""
    if not args.skip_cold:
        cold_port = free_port()
        cold_proc, cold_file = start_server(args, cold_port, args.cold_log)
        try:
            wait_health(cold_port, args.timeout, cold_proc)
            cold_status, cold_body = post_chat(
                cold_port,
                payload(messages_b, query_start, query_end, args.max_tokens),
                args.timeout,
            )
        finally:
            stop_server(cold_proc, cold_file)
        cold_text = response_text(cold_body)

    prefix_chars = common_prefix_chars(warm_text, cold_text) if cold_text else 0
    route = "plain" if args.plain else "mtp"
    first_capture_kind = (
        f"CAPTURE ({route} prefill-only):"
        if args.prefill_only_first else f"CAPTURE ({route}):"
    )
    checks = {
        "http_ok": status_a == 200 and status_b == 200 and
                   (args.skip_cold or cold_status == 200),
        "first_prompt_below_budget":
            0 < fitted_prompt_tokens < args.budget,
        "first_capture_kind": first_capture_kind in first_capture,
        "capture_source_index": "P_source_index=1" in first_capture,
        "capture_query_snapshot": "query_snapshot=1" in first_capture,
        "threshold_crossed": (
            first_end_pos <= args.budget and prompt_tokens > args.budget
            if args.prefill_only_first
            else first_end_pos > args.budget
        ),
        "prepare_did_not_score": "[kvmem-scorer]" not in first_request_log,
        "post_query_hit": bool(re.search(
            rf"HIT \({route}\):[^\n]*query_snapshot=1", warm_log)),
        "no_cold_rebuild": "reason=no_reusable_checkpoint" not in warm_log,
        "reuse_reported": reuse > 0 and prefilled > 0,
        "suffix_only": prompt_tokens > 0 and prefilled < prompt_tokens // 4,
        "semantic_scorer": "used=key-direction-adaptive-packed-gpu fallback=0" in warm_log,
        "mtp_active": args.plain or
                      "native mtp_spec_summary: enabled=true" in warm_log,
        "no_server_error": "POST /v1/chat/completions -> 5" not in warm_log,
        "warm_output": bool(warm_text),
        "cold_agreement": args.skip_cold or (
            bool(cold_text) and (warm_text == cold_text or prefix_chars >= 24)
        ),
    }
    report = {
        "ok": all(checks.values()),
        "route": route,
        "checks": checks,
        "status": {"turn_a": status_a, "turn_b": status_b, "cold": cold_status},
        "metrics": {
            "budget": args.budget,
            "fitted_prompt_tokens": fitted_prompt_tokens,
            "fitted_repeats": fitted_repeats,
            "first_end_pos": first_end_pos,
            "reuse": reuse,
            "prefilled": prefilled,
            "prompt_tokens": prompt_tokens,
            "warm_cold_common_prefix_chars": prefix_chars,
            "warm_cold_exact": bool(cold_text) and warm_text == cold_text,
        },
        "turn_a_text": response_text(body_a),
        "turn_b_warm_text": warm_text,
        "turn_b_cold_text": cold_text,
        "second_generate": second_generate,
        "warm_log": str(args.warm_log),
        "cold_log": str(args.cold_log),
    }
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for line in warm_log.splitlines():
        if any(marker in line for marker in (
            "kvmem prefix-cache", "native kvmem query replay:",
            "[kvmem-scorer]", "native generate:",
            "native mtp_spec_summary:",
        )):
            print(line)
    print(json.dumps({"ok": report["ok"], "checks": checks,
                      "metrics": report["metrics"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
