#!/usr/bin/env python3
"""Summarize benchmark JSON outputs into Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(numer: float, denom: float) -> str:
    if denom <= 0:
        return "n/a"
    return f"{100.0 * numer / denom:.1f}%"


def summarize_efficiency(input_dir: Path, output: Path) -> None:
    lines: List[str] = [
        "# Efficiency Benchmark Summary",
        "",
        f"Input directory: `{input_dir}`",
        "",
    ]

    sweep_files = sorted(input_dir.glob("llama_sweep*.json"))
    if sweep_files:
        lines.extend(["## llama.cpp vs qw3 (long prompt sweep)", ""])
        lines.append(
            "| variant | prompt tokens | qw3 prefill | llama prefill | prefill % | "
            "qw3 decode | llama decode | decode % | peak MiB |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for path in sweep_files:
            data = load_json(path)
            for cell in data.get("cells", []):
                variant = cell.get("variant", path.stem)
                actual = cell.get("actual_prompt_tokens_qw3") or cell.get("actual_prompt_tokens_llama", 0)
                qw3_p = float(cell.get("qw3_prefill_med", 0))
                llama_p = float(cell.get("llama_prefill_med", 0))
                qw3_d = float(cell.get("qw3_decode_med", 0))
                llama_d = float(cell.get("llama_decode_med", 0))
                peak = float(cell.get("qw3_peak_mib_med", 0))
                lines.append(
                    f"| {variant} | {actual} | {qw3_p:.1f} | {llama_p:.1f} | {pct(qw3_p, llama_p)} | "
                    f"{qw3_d:.2f} | {llama_d:.2f} | {pct(qw3_d, llama_d)} | {peak:.1f} |"
                )
        lines.append("")

    cb_paths = [input_dir / name for name in ("cb_matrix.json", "cb_matrix_4k.json", "cb_matrix_32k.json")]
    for cb_path in cb_paths:
        if not cb_path.exists():
            continue
        data = load_json(cb_path)
        lines.extend([f"## Continuous batching matrix (`{cb_path.name}`)", ""])
        lines.append(
            "| variant | ctx | input target | concurrency | output tok/s | decode tok/s | max batch |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for run in data.get("runs", []):
            lines.append(
                f"| {run.get('variant', '')} | {run.get('ctx_size', 0)} | "
                f"{run.get('input_target_tokens', 0)} | {run.get('concurrency', 0)} | "
                f"{float(run.get('tokens_per_s', 0)):.2f} | "
                f"{float(run.get('decode_tokens_per_s', 0)):.2f} | "
                f"{run.get('max_batch', 0)} |"
            )
        failures = data.get("status", {}).get("failures", [])
        if failures:
            lines.extend(["", "**Failures:**"] + [f"- {f}" for f in failures])
        lines.append("")

    mtp_paths = [input_dir / "mtp_legacy_4k.json", input_dir / "mtp_cb_4k.json"]
    mtp_runs: List[Dict[str, Any]] = []
    for mtp_path in mtp_paths:
        if mtp_path.exists():
            mtp_runs.extend(load_json(mtp_path).get("runs", []))
    if mtp_runs:
        lines.extend(["## MTP @ 4K serve", ""])
        lines.append("| mode | concurrency | output tok/s | mtp accepted/drafted | batched verify |")
        lines.append("|---|---:|---:|---|---:|")
        for run in mtp_runs:
            accepted = sum(s.get("accepted", 0) for s in run.get("mtp_summaries", []))
            drafted = sum(s.get("drafted", 0) for s in run.get("mtp_summaries", []))
            batched = sum(s.get("batched_verify_batches", 0) for s in run.get("mtp_summaries", []))
            lines.append(
                f"| {run.get('mode', '')} | {run.get('concurrency', 0)} | "
                f"{float(run.get('output_tokens_per_s', 0)):.2f} | "
                f"{accepted}/{drafted} | {batched} |"
            )
        lines.append("")

    for extra in sorted(input_dir.glob("mtp_longctx_*.json")):
        data = load_json(extra)
        lines.extend([f"## Long-context spot (`{extra.name}`)", ""])
        for cell in data.get("cells", []):
            variant = cell.get("variant", extra.stem)
            actual = cell.get("actual_prompt_tokens_qw3") or cell.get("actual_prompt_tokens_llama", 0)
            qw3_d = float(cell.get("qw3_decode_med", 0))
            llama_d = float(cell.get("llama_decode_med", 0))
            lines.append(
                f"- {variant} @ {actual} tokens: qw3 decode {qw3_d:.2f} tok/s, "
                f"llama {llama_d:.2f} tok/s ({pct(qw3_d, llama_d)})"
            )
        lines.append("")

    fp8_path = input_dir / "mtp_cb_4k_fp8.json"
    if fp8_path.exists():
        data = load_json(fp8_path)
        lines.extend(["## FP8 KV spot (4K)", ""])
        for run in data.get("runs", []):
            lines.append(
                f"- {run.get('mode', '')} C={run.get('concurrency', 0)}: "
                f"{float(run.get('output_tokens_per_s', 0)):.2f} tok/s"
            )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def find_metric(results: Dict[str, Any], task: str, metric_suffix: str) -> Optional[float]:
    task_results = results.get("results", {})
    if task in task_results:
        for key, value in task_results[task].items():
            if key.startswith(metric_suffix) and isinstance(value, (int, float)):
                return float(value)
    return None


def summarize_accuracy(input_dir: Path, output: Path, baseline_dir: Optional[Path]) -> None:
    lines: List[str] = [
        "# Accuracy Benchmark Summary",
        "",
        f"Input directory: `{input_dir}`",
        "",
    ]

    result_files = sorted(input_dir.glob("**/results_*.json"))
    if not result_files and (input_dir / "lm_eval_prod").exists():
        result_files = sorted((input_dir / "lm_eval_prod").glob("**/results_*.json"))

    eval_dirs = sorted({p.parent for p in result_files})
    if not eval_dirs:
        for sub in sorted(input_dir.iterdir()):
            if sub.is_dir() and list(sub.glob("**/results_*.json")):
                eval_dirs.append(sub)

    lines.extend(["## lm_eval runs", ""])
    lines.append("| run | task | metric | score | vs baseline |")
    lines.append("|---|---|---|---:|---|")

    for run_dir in eval_dirs:
        for result_path in sorted(run_dir.glob("**/results_*.json")):
            data = load_json(result_path)
            results = data.get("results", {})
            run_name = run_dir.name
            for task, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    if not any(
                        key.startswith(prefix)
                        for prefix in ("boxed-match", "exact_match", "acc", "acc_norm")
                    ):
                        continue
                    base_score: Optional[float] = None
                    if baseline_dir is not None:
                        for base_path in baseline_dir.glob("**/results_*.json"):
                            base_score = find_metric(load_json(base_path), task, key.split(",")[0])
                            if base_score is not None:
                                break
                    delta = ""
                    if base_score is not None:
                        delta = f"{(float(value) - base_score) * 100:+.2f} pp"
                    lines.append(
                        f"| {run_name} | {task} | {key} | {float(value) * 100:.2f}% | {delta or 'n/a'} |"
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("efficiency", "accuracy"), required=True)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory not found: {args.input_dir}")

    if args.kind == "efficiency":
        summarize_efficiency(args.input_dir, args.output)
    else:
        summarize_accuracy(args.input_dir, args.output, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
