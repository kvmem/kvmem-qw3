#!/usr/bin/env python3
"""Summarize the MTP x continuous-batching sweep into pivot tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    accepted = sum(s.get("accepted", 0) for s in run.get("mtp_summaries", []))
    drafted = sum(s.get("drafted", 0) for s in run.get("mtp_summaries", []))
    acc = (accepted / drafted) if drafted else 0.0
    return {
        "tok_s": float(run.get("output_tokens_per_s", 0.0)),
        "accepted": accepted,
        "drafted": drafted,
        "acc": acc,
    }


def collect(input_dir: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[int]]:
    # key: (label, concurrency) -> metrics
    table: Dict[Tuple[str, int], Dict[str, Any]] = {}
    concs: set[int] = set()

    baseline = input_dir / "sweep_baseline_nomtp.json"
    if baseline.exists():
        for run in load(baseline).get("runs", []):
            mode = run.get("mode", "")
            label = "CB on (no MTP)" if mode == "continuous" else "CB off (no MTP)"
            c = int(run.get("concurrency", 0))
            concs.add(c)
            table[(label, c)] = run_metrics(run)

    for path in sorted(input_dir.glob("sweep_chain*.json")):
        chain = path.stem.replace("sweep_chain", "")
        for run in load(path).get("runs", []):
            mode = run.get("mode", "")
            if mode == "continuous_mtp":
                label = f"CB on  + MTP{chain}"
            elif mode == "legacy_mtp":
                label = f"CB off + MTP{chain}"
            else:
                label = f"{mode} (chain {chain})"
            c = int(run.get("concurrency", 0))
            concs.add(c)
            table[(label, c)] = run_metrics(run)

    return table, sorted(concs)


def label_order(label: str) -> Tuple[int, int]:
    # group: 0 baseline, 1 MTP; within, sort by chain then CB off/on
    if "no MTP" in label:
        return (0, 0 if "off" in label else 1)
    chain = 0
    for tok in label.split():
        if tok.startswith("MTP"):
            chain = int(tok[3:])
    return (1 + chain, 0 if "off" in label else 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    table, concs = collect(args.input_dir)
    labels = sorted({k[0] for k in table}, key=label_order)

    lines: List[str] = [
        "# MTP x Continuous Batching Sweep",
        "",
        f"Input directory: `{args.input_dir}`",
        "",
        "ctx=4096, kv_dtype=fp16. Values = output tok/s "
        "(single request uses concurrency=1).",
        "",
    ]

    # Throughput pivot
    header = "| config | " + " | ".join(f"C={c}" for c in concs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(concs)) + "|"
    lines.append("## Throughput (output tok/s)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for label in labels:
        cells = []
        for c in concs:
            m = table.get((label, c))
            cells.append(f"{m['tok_s']:.2f}" if m else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Acceptance pivot (MTP rows only)
    lines.append("## MTP acceptance rate (accepted/drafted)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for label in labels:
        if "MTP" not in label or "no MTP" in label:
            continue
        cells = []
        for c in concs:
            m = table.get((label, c))
            if m and m["drafted"]:
                cells.append(f"{100*m['acc']:.1f}%")
            else:
                cells.append("-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
