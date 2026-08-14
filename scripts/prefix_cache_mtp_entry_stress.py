#!/usr/bin/env python3
"""Stress the ordinary MTP prefix cache's independent entry LRU.

The prompts form one strict-extension chain, so their physical KV pages are
mostly shared while every commit still owns a distinct recurrent/conv/MTP
state snapshot. This is the shape that previously exhausted device memory even
though the KV page pool itself remained below capacity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from prefix_cache_canary import post_completion, wait_for_server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18139)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=96)
    ap.add_argument("--max-entries", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--server-log", default="/tmp/qw3_prefix_mtp_entry_stress.log")
    ap.add_argument("--out-json", default="/tmp/qw3_prefix_mtp_entry_stress.json")
    args = ap.parse_args()

    env = dict(os.environ)
    env.update({
        "QW3_PREFIX_CACHE_TRACE": "1",
        "QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES": "1",
        "QW3_PREFIX_CACHE_MAX_ENTRIES": str(args.max_entries),
        "QW3_FATTN_NSPLIT": "1",
        "QW3_PREFILL_FA2_NSPLIT": "1",
        "QW3_CONTINUOUS_MTP_PHASE_SYNC": "1",
    })
    command = [
        args.qw3, "serve", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--ctx", str(args.ctx), "--continuous-batching", "--max-active", "1",
        "--prefix-cache", "--native-mtp-speculate", "--mtp-chain", "4",
        "--kv-dtype", "fp8", "--prefill-chunk", "512",
        "--temp", "0", "-n", str(args.max_tokens),
    ]
    log_path = Path(args.server_log)
    out_path = Path(args.out_json)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True)
    statuses: list[int] = []
    outputs: list[str] = []
    try:
        base_url = f"http://127.0.0.1:{args.port}"
        if not wait_for_server(base_url, args.timeout):
            raise RuntimeError("server did not become healthy")
        prompt = (
            "You are checking a deterministic state machine. Preserve every "
            "previous ledger row exactly and answer only with the word OK.\n"
            + ("stable-prefix-ledger alpha beta gamma delta epsilon\n" * 24)
        )
        for index in range(args.rounds):
            prompt += (
                f"row-{index:04d}: strict extension payload alpha beta gamma "
                "delta epsilon zeta eta theta.\n"
            )
            status, output, error = post_completion(
                base_url, prompt, args.max_tokens, args.timeout)
            statuses.append(status)
            outputs.append(output)
            if status != 200:
                raise RuntimeError(
                    f"request {index} failed: status={status} error={error}")
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited after request {index}: rc={proc.returncode}")
    except Exception as exc:  # noqa: BLE001
        failure = str(exc)
    else:
        failure = ""
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log_file.close()

    log = log_path.read_text(encoding="utf-8", errors="replace")
    hits = len(re.findall(r"prefix_cache hit ", log))
    commits = len(re.findall(r"prefix_cache commit ", log))
    evictions = len(re.findall(r"prefix_cache evict ", log))
    entries = [int(value) for value in re.findall(r"\bentries=(\d+)", log)]
    mtp_active = len(re.findall(
        r"native mtp_spec_summary: enabled=true[^\n]*drafted=[1-9][0-9]*", log))
    no_oom = "out of memory" not in log.lower() and "cudamalloc" not in log.lower()
    checks = {
        "all_http_ok": len(statuses) == args.rounds and all(s == 200 for s in statuses),
        "server_completed": not failure,
        "prefix_hits": hits >= max(1, args.rounds // 2),
        "commits": commits >= max(1, args.rounds // 2),
        "entry_evictions": evictions >= max(1, args.rounds - args.max_entries - 2),
        "entry_peak_bounded": bool(entries) and max(entries) <= args.max_entries + 1,
        "entry_final_bounded": bool(entries) and entries[-1] <= args.max_entries,
        "mtp_active": mtp_active >= max(1, args.rounds // 2),
        "no_oom": no_oom,
    }
    report = {
        "ok": all(checks.values()),
        "failure": failure,
        "checks": checks,
        "metrics": {
            "rounds": args.rounds,
            "max_entries": args.max_entries,
            "hits": hits,
            "commits": commits,
            "evictions": evictions,
            "entry_peak": max(entries) if entries else None,
            "entry_final": entries[-1] if entries else None,
            "mtp_active_requests": mtp_active,
            "nonempty_outputs": sum(bool(x) for x in outputs),
        },
        "server_log": str(log_path),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
