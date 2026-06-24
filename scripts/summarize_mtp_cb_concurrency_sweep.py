#!/usr/bin/env python3
"""Summarize MTP continuous-batching concurrency sweep JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def accept_rate(run: Dict[str, Any]) -> str:
    accepted = sum(s.get("accepted", 0) for s in run.get("mtp_summaries", []))
    drafted = sum(s.get("drafted", 0) for s in run.get("mtp_summaries", []))
    if drafted <= 0:
        return "n/a"
    return f"{100.0 * accepted / drafted:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = data.get("runs", [])
    cfg = data.get("config", {})
    chain = cfg.get("chain", "?")

    by_max_active: Dict[int, List[Dict[str, Any]]] = {}
    for run in runs:
        ma = int(run.get("max_active", 0))
        by_max_active.setdefault(ma, []).append(run)
    for ma in by_max_active:
        by_max_active[ma].sort(key=lambda r: int(r.get("concurrency", 0)))

    lines = [
        "# MTP × Continuous Batching Concurrency Sweep",
        "",
        f"MTP chain={chain}. CB on (paged KV auto-enabled). "
        "Throughput = total output tokens / batch wall time.",
        "",
    ]

    for ma in sorted(by_max_active):
        lines.extend([f"## max_active={ma}", ""])
        lines.append(
            "| concurrency | output tok/s | total tokens | mean latency (s) | "
            "mtp accept | batched verify |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for run in by_max_active[ma]:
            batched = sum(
                s.get("batched_verify_batches", 0)
                for s in run.get("mtp_summaries", [])
            )
            lines.append(
                f"| {run.get('concurrency', 0)} | "
                f"{float(run.get('output_tokens_per_s', 0)):.2f} | "
                f"{run.get('output_tokens', 0)} | "
                f"{float(run.get('mean_latency_s', 0)):.3f} | "
                f"{accept_rate(run)} | {batched} |"
            )
        lines.append("")

    lines.extend(["## Scaling vs concurrency=1 (output tok/s)", ""])
    lines.append("| max_active | C=1 | C=2 | C=3 | C=4 | C=5 | C=6 | C=7 | C=8 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ma in sorted(by_max_active):
        cells = [str(ma)]
        lookup = {int(r.get("concurrency", 0)): r for r in by_max_active[ma]}
        base = float(lookup.get(1, {}).get("output_tokens_per_s", 0))
        for c in range(1, 9):
            run = lookup.get(c)
            if run is None:
                cells.append("-")
            else:
                tps = float(run.get("output_tokens_per_s", 0))
                if c == 1 or base <= 0:
                    cells.append(f"{tps:.1f}")
                else:
                    cells.append(f"{tps:.1f} ({100.0 * tps / base:.0f}%)")
        lines.append("| " + " | ".join(cells) + " |")

    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
