#!/usr/bin/env python3
"""Summarize MTP5 CB + paged-KV compatibility study JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_runs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("runs", [])


def accept_rate(run: Dict[str, Any]) -> str:
    accepted = sum(s.get("accepted", 0) for s in run.get("mtp_summaries", []))
    drafted = sum(s.get("drafted", 0) for s in run.get("mtp_summaries", []))
    if drafted <= 0:
        return "n/a"
    return f"{100.0 * accepted / drafted:.1f}%"


def config_label(run: Dict[str, Any], paged_kv: bool) -> str:
    mode = run.get("mode", "")
    if mode == "legacy_mtp":
        cb = "CB off"
        ma = "-"
    else:
        cb = "CB on"
        ma = str(run.get("max_active", "-"))
    pk = "paged" if paged_kv else "plain"
    return f"{cb} | max_active={ma} | {pk}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    plain_runs = load_runs(input_dir / "plain_kv.json")
    paged_runs = load_runs(input_dir / "paged_kv.json")

    lines = [
        "# MTP5 × Continuous Batching × Paged KV Study",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "Single request (concurrency=1). MTP chain=5.",
        "",
        "## Throughput (output tok/s)",
        "",
        "| config | output tok/s | latency (s) | mtp accept | continuous_mtp logs |",
        "|---|---:|---:|---:|---:|",
    ]

    for paged_kv, runs in ((False, plain_runs), (True, paged_runs)):
        for run in runs:
            if run.get("concurrency", 0) != 1:
                continue
            lines.append(
                f"| {config_label(run, paged_kv)} | "
                f"{float(run.get('output_tokens_per_s', 0)):.2f} | "
                f"{float(run.get('mean_latency_s', 0)):.3f} | "
                f"{accept_rate(run)} | "
                f"{run.get('continuous_mtp_count', 0)} |"
            )

    lines.extend(["", "## CB on: max_active sweep (plain vs paged)", ""])
    lines.append("| max_active | plain tok/s | paged tok/s | delta |")
    lines.append("|---:|---:|---:|---:|")

    plain_cb = {
        r.get("max_active"): r
        for r in plain_runs
        if r.get("mode") == "continuous_mtp" and r.get("concurrency") == 1
    }
    paged_cb = {
        r.get("max_active"): r
        for r in paged_runs
        if r.get("mode") == "continuous_mtp" and r.get("concurrency") == 1
    }
    for ma in sorted(set(plain_cb) | set(paged_cb)):
        p_tps = float(plain_cb.get(ma, {}).get("output_tokens_per_s", 0))
        g_tps = float(paged_cb.get(ma, {}).get("output_tokens_per_s", 0))
        if p_tps > 0 and g_tps > 0:
            delta = f"{100.0 * g_tps / p_tps:.1f}%"
        else:
            delta = "n/a"
        lines.append(f"| {ma} | {p_tps:.2f} | {g_tps:.2f} | {delta} |")

    lines.extend(["", "## CB off baseline (legacy_mtp)", ""])
    for paged_kv, runs in ((False, plain_runs), (True, paged_runs)):
        for run in runs:
            if run.get("mode") != "legacy_mtp":
                continue
            pk = "paged" if paged_kv else "plain"
            lines.append(
                f"- {pk}: {float(run.get('output_tokens_per_s', 0)):.2f} tok/s, "
                f"accept {accept_rate(run)}"
            )

    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
