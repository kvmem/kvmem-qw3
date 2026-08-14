#!/usr/bin/env python3
"""Focused lossless ordinary-prefix-cache + MTP canary.

One server receives the same prompt twice. The first request must commit both
main and MTP KV pages; the second must reuse them, retain active speculative
drafting, and produce byte-identical greedy output with less prefill work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from kvmem_cb_mtp_gate import run_prefix_probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-json", default="/tmp/qw3_prefix_mtp_canary.json")
    ap.add_argument("--server-log", default="/tmp/qw3_prefix_mtp_canary.log")
    args = ap.parse_args()

    result = run_prefix_probe(
        Path(args.qw3), Path(args.model), args.ctx, args.max_tokens,
        args.timeout, mtp=True)
    log = result["log"]
    Path(args.server_log).write_text(log, encoding="utf-8")

    prefills = [float(v) for v in re.findall(
        r"native continuous_batch:[^\n]*prefill=([0-9.]+)s", log)]
    prefill_ops = [int(v) for v in re.findall(
        r"native continuous_batch:[^\n]*prefill_ops=([0-9]+)", log)]
    result["prefill_seconds"] = prefills
    result["prefill_ops"] = prefill_ops
    result["prefill_reduced"] = (
        len(prefills) >= 2 and prefills[-1] < prefills[0] and
        len(prefill_ops) >= 2 and prefill_ops[-1] < prefill_ops[0])

    failures = []
    required = {
        "commit": "first request produced no prefix-cache commit",
        "hit": "second request produced no prefix-cache hit",
        "reused": "hit did not report reused_tokens",
        "commit_has_mtp_pages": "commit did not pin MTP KV pages",
        "hit_has_mtp_pages": "hit did not restore MTP KV pages",
        "mtp_active_after_hit": "hit request did not execute MTP drafts",
        "identical": "warm greedy output differs from cold output",
        "ok_status": "one or both requests returned no completion",
        "prefill_reduced": "cache hit did not reduce prefill time",
    }
    for key, message in required.items():
        if not result.get(key):
            failures.append(message)

    report = {"ok": not failures, "failures": failures, "result": result}
    Path(args.out_json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for line in log.splitlines():
        if any(marker in line for marker in (
                "prefix_cache ", "native mtp_spec_summary:",
                "native continuous_batch:")):
            print(line)
    print(json.dumps({
        "ok": not failures,
        "failures": failures,
        "prefill_seconds": prefills,
        "prefill_ops": prefill_ops,
        "identical": result["identical"],
        "commit_has_mtp_pages": result["commit_has_mtp_pages"],
        "hit_has_mtp_pages": result["hit_has_mtp_pages"],
        "mtp_active_after_hit": result["mtp_active_after_hit"],
    }, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
