#!/usr/bin/env python3
"""Run the factorial KVMem scaling/optimization ablation.

Each cell grows one persistent context through the same cumulative ladder.
This avoids re-prefilling shorter prefixes from scratch and models a real
long-lived agent session. The eight cells cover every combination of the three
paper-facing storage optimizations while keeping model, precision, budgets,
query replay, retrieval, CPU capacity, and NVMe capacity fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


CELLS: Dict[str, tuple[str, str, str]] = {
    "all-off": ("off", "off", "off"),
    "stage-out-only": ("on", "off", "off"),
    "stage-in-only": ("off", "on", "off"),
    "pack-only": ("off", "off", "on"),
    "no-stage-out": ("off", "on", "on"),
    "no-stage-in": ("on", "off", "on"),
    "no-pack": ("on", "on", "off"),
    "all-on": ("on", "on", "on"),
}


def write_summary(out_root: Path, results: Dict[str, dict]) -> None:
    summary: dict = {"cells": {}, "comparisons": {}}
    for cell, payload in results.items():
        turns = payload.get("turns", [])
        totals = {
            key: sum(float(t.get(key, 0.0)) for t in turns)
            for key in (
                "total_ms", "setup_ms", "prefill_ms", "postprefill_ms",
                "decode_ms", "finalize_ms", "semantic_ms",
                "query_replay_ms", "post_other_ms")
        }
        phase_sum_ms = sum(totals[k] for k in (
            "setup_ms", "prefill_ms", "postprefill_ms",
            "decode_ms", "finalize_ms"))
        post_sum_ms = sum(totals[k] for k in (
            "semantic_ms", "query_replay_ms", "post_other_ms"))
        totals["phase_sum_ms"] = phase_sum_ms
        totals["phase_error_ms"] = totals["total_ms"] - phase_sum_ms
        totals["post_sum_ms"] = post_sum_ms
        totals["post_error_ms"] = totals["postprefill_ms"] - post_sum_ms
        summary["cells"][cell] = {
            "optimization_switches": {
                "stage_out": CELLS[cell][0],
                "stage_in": CELLS[cell][1],
                "pack": CELLS[cell][2],
            },
            "returncode": payload.get("returncode"),
            "wall_s": payload.get("wall_s"),
            "peak_gpu_proc_mib": payload.get("peak_gpu_proc_mib"),
            "peak_host_rss_mib": payload.get("peak_host_rss_mib"),
            "totals": totals,
            "turns": turns,
        }

    if "all-off" in summary["cells"] and "all-on" in summary["cells"]:
        off = summary["cells"]["all-off"]["totals"]
        on = summary["cells"]["all-on"]["totals"]
        summary["comparisons"]["all_on_vs_all_off"] = {
            "total_speedup": (
                off["total_ms"] / on["total_ms"]
                if on["total_ms"] else 0.0),
            "total_reduction_pct": (
                100.0 * (off["total_ms"] - on["total_ms"]) /
                off["total_ms"] if off["total_ms"] else 0.0),
            "postprefill_speedup": (
                off["postprefill_ms"] / on["postprefill_ms"]
                if on["postprefill_ms"] else 0.0),
            "postprefill_reduction_pct": (
                100.0 * (off["postprefill_ms"] -
                         on["postprefill_ms"]) /
                off["postprefill_ms"]
                if off["postprefill_ms"] else 0.0),
        }

    # Ordered, cumulative marginal contributions. Each adjacent pair differs
    # by exactly one optimization while the workload and model stay fixed.
    marginal_pairs = (
        ("pack",
         "all-on", "no-pack"),
        ("stage-in",
         "no-pack", "stage-out-only"),
        ("stage-out",
         "stage-out-only", "all-off"),
    )
    marginal: dict = {}
    for name, enabled_cell, disabled_cell in marginal_pairs:
        if (enabled_cell not in summary["cells"] or
                disabled_cell not in summary["cells"]):
            continue
        enabled = summary["cells"][enabled_cell]["totals"]
        disabled = summary["cells"][disabled_cell]["totals"]
        marginal[name] = {
            "enabled_cell": enabled_cell,
            "disabled_cell": disabled_cell,
            "total_speedup": (
                disabled["total_ms"] / enabled["total_ms"]
                if enabled["total_ms"] else 0.0),
            "total_reduction_pct": (
                100.0 * (disabled["total_ms"] - enabled["total_ms"]) /
                disabled["total_ms"] if disabled["total_ms"] else 0.0),
            "prefill_speedup": (
                disabled["prefill_ms"] / enabled["prefill_ms"]
                if enabled["prefill_ms"] else 0.0),
            "postprefill_speedup": (
                disabled["postprefill_ms"] / enabled["postprefill_ms"]
                if enabled["postprefill_ms"] else 0.0),
        }
    summary["comparisons"]["ordered_marginal"] = marginal

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# KVMem scaling and optimization ablation",
        "",
        "Top-level timing is directly instrumented and additive:",
        "",
        "`total = setup + prefill + post-prefill + decode + finalize`",
        "",
        "`post-prefill = semantic selection + query replay + other`",
        "",
        "Nested selection/stage-in/stage-out/assembly counters are not added "
        "because their I/O and compute may overlap.",
        "",
        "## Cumulative result",
        "",
        "| cell | total (s) | setup | prefill | post | semantic | query replay "
        "| decode | finalize | sum error (ms) | peak GPU MiB | peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        if cell not in summary["cells"]:
            continue
        c = summary["cells"][cell]
        t = c["totals"]
        lines.append(
            f"| {cell} | {t['total_ms']/1000:.3f} "
            f"| {t['setup_ms']/1000:.3f} | {t['prefill_ms']/1000:.3f} "
            f"| {t['postprefill_ms']/1000:.3f} "
            f"| {t['semantic_ms']/1000:.3f} "
            f"| {t['query_replay_ms']/1000:.3f} "
            f"| {t['decode_ms']/1000:.3f} "
            f"| {t['finalize_ms']/1000:.3f} "
            f"| {t['phase_error_ms']:.6f} "
            f"| {c.get('peak_gpu_proc_mib', 0)} "
            f"| {c.get('peak_host_rss_mib', 0)} |")

    lines += ["", "## Per context length", ""]
    for cell in CELLS:
        if cell not in summary["cells"]:
            continue
        lines += [
            f"### {cell}",
            "",
            "| context | delta | total (s) | setup | prefill | post | semantic "
            "| query replay | decode | finalize | error ms | prefill tok/s "
            "| GPU MiB | RSS MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in summary["cells"][cell]["turns"]:
            lines.append(
                f"| {t.get('ctx_tokens', 0)} | {t.get('delta_tokens', 0)} "
                f"| {t.get('total_ms', 0)/1000:.3f} "
                f"| {t.get('setup_ms', 0)/1000:.3f} "
                f"| {t.get('prefill_ms', 0)/1000:.3f} "
                f"| {t.get('postprefill_ms', 0)/1000:.3f} "
                f"| {t.get('semantic_ms', 0)/1000:.3f} "
                f"| {t.get('query_replay_ms', 0)/1000:.3f} "
                f"| {t.get('decode_ms', 0)/1000:.3f} "
                f"| {t.get('finalize_ms', 0)/1000:.3f} "
                f"| {t.get('phase_error_ms', 0):.6f} "
                f"| {t.get('prefill_tps', 0):.1f} "
                f"| {t.get('gpu_mib', 0)} | {t.get('rss_mib', 0)} |")
        lines.append("")

        lines += [
            "Nested KVMem diagnostics (not additive because asynchronous I/O "
            "may overlap foreground compute):",
            "",
            "| context | retrieval | stage-in | stage-out | assembly | pages "
            "| re-RoPE | k-bar | in-prefill KVMem |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in summary["cells"][cell]["turns"]:
            lines.append(
                f"| {t.get('ctx_tokens', 0)} "
                f"| {t.get('sel_ms', 0)/1000:.6f} "
                f"| {t.get('stage_in_ms', 0)/1000:.6f} "
                f"| {t.get('stage_out_ms', 0)/1000:.6f} "
                f"| {t.get('assemble_ms', 0)/1000:.6f} "
                f"| {t.get('asm_pages_ms', 0)/1000:.6f} "
                f"| {t.get('asm_rerope_ms', 0)/1000:.6f} "
                f"| {t.get('asm_kbar_ms', 0)/1000:.6f} "
                f"| {t.get('inprefill_offload_ms', 0)/1000:.6f} |")
        lines.append("")

    if marginal:
        lines += [
            "## Ordered marginal ablation",
            "",
            "Each row compares two adjacent cumulative cells differing by "
            "one optimization. Results are order-dependent when optimizations "
            "interact.",
            "",
            "| optimization restored | enabled cell | disabled cell | total "
            "speedup | total reduction | prefill speedup | post speedup |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for name, values in marginal.items():
            lines.append(
                f"| {name} | {values['enabled_cell']} "
                f"| {values['disabled_cell']} "
                f"| {values['total_speedup']:.4f}x "
                f"| {values['total_reduction_pct']:.3f}% "
                f"| {values['prefill_speedup']:.4f}x "
                f"| {values['postprefill_speedup']:.4f}x |")
        lines.append("")

    (out_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(
        Path(__file__).resolve().parents[2]))
    ap.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--ladder", default="128K,256K,1M,2M,5M,10M")
    ap.add_argument("--timeout", type=int, default=43200)
    ap.add_argument("--cpu-gb", type=float, default=64.0)
    ap.add_argument("--nvme-gb", type=float, default=520.0)
    ap.add_argument("--index-placement", choices=("gpu", "cpu"),
                    default="cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    model = Path(args.model)
    if not model.is_absolute():
        model = root / model
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (
        Path("/data/chaidi/kvmem_eval/performance") /
        f"kvmem_scaling_ablation_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}")

    manifest = {
        "binary": str(root / "build/qw3"),
        "model": str(model),
        "ladder": args.ladder,
        "cells": selected,
        "common": {
            "context_budget": 229376,
            "generation_budget": 32768,
            "block_tokens": 512,
            "query_tokens": 32,
            "decode_tokens": 1,
            "retrieval": "key-direction-adaptive",
            "adaptive_gains": [0.10, 0.06],
            "index_placement": args.index_placement,
            "kv_dtype": "fp8",
            "index_query_dtype": "fp16",
            "prefill_chunk": 2048,
            "gpu_memory_ratio": 0.5,
            "cpu_gb": args.cpu_gb,
            "nvme_gb": args.nvme_gb,
            "raw_k_nvme": True,
            "immutable_k": True,
            "mtp_chain": 4,
        },
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    results: Dict[str, dict] = {}
    for cell in selected:
        cell_dir = out_root / cell
        cell_dir.mkdir(exist_ok=True)
        out_json = cell_dir / "profile.json"
        if args.resume and out_json.exists():
            old = json.loads(out_json.read_text(encoding="utf-8"))
            if old.get("returncode") == 0 and len(old.get("turns", [])) == 6:
                print(f"SKIP completed cell={cell}", flush=True)
                results[cell] = old
                continue
        cmd = [
            sys.executable, str(root / "scripts/kvmem_session_profile.py"),
            "--qw3", str(root / "build/qw3"),
            "--model", str(model),
            "--ladder", args.ladder,
            "--decode-tokens", "1",
            "--query-tokens", "32",
            "--window", "229376",
            "--gen-budget", "32768",
            "--block-tokens", "512",
            "--method", "retrieval",
            "--retrieval-method", "key-direction-adaptive",
            "--adaptive-gain-1to2", "0.10",
            "--adaptive-gain-2to4", "0.06",
            "--index-placement", args.index_placement,
            "--index-staging-mb", "64",
            "--gpu-ratio", "0.5",
            "--cpu-gb", str(args.cpu_gb),
            "--nvme-gb", str(args.nvme_gb),
            "--kv-dtype", "fp8",
            "--prefill-chunk", "2048",
            "--raw-k-nvme",
            "--timeout", str(args.timeout),
            "--out-json", str(out_json),
        ]
        stage_out, stage_in, pack = CELLS[cell]
        cmd += [
            "--opt-stage-out", stage_out,
            "--opt-stage-in", stage_in,
            "--opt-pack", pack,
        ]
        (cell_dir / "command.txt").write_text(
            " ".join(cmd) + "\n", encoding="utf-8")
        print(f"[{time.strftime('%F %T')}] START {cell}", flush=True)
        with (cell_dir / "driver.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd, cwd=root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env=os.environ.copy())
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            rc = proc.wait()
        if rc != 0 or not out_json.exists():
            print(f"cell={cell} failed rc={rc}", file=sys.stderr)
            write_summary(out_root, results)
            return rc or 1
        results[cell] = json.loads(out_json.read_text(encoding="utf-8"))
        write_summary(out_root, results)
        print(f"[{time.strftime('%F %T')}] DONE {cell}", flush=True)

    write_summary(out_root, results)
    print(f"results: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
