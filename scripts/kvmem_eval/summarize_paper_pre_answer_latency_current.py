#!/usr/bin/env python3
"""Aggregate validated current-machine P50 pre-answer latency."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


def load_validator():
    path = Path(__file__).with_name("validate_paper_latency_current.py")
    spec = importlib.util.spec_from_file_location("paper_latency_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        raise RuntimeError("cannot average an empty latency list")
    return statistics.fmean(values)


def cell(
    values: Iterable[float], *, excluding: Iterable[float] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    raw = [float(value) for value in values]
    result: dict[str, Any] = {"n": len(raw), "per_sample_sec": raw, "mean_sec": mean(raw)}
    if excluding is not None:
        steady = [float(value) for value in excluding]
        if len(steady) != len(raw):
            raise RuntimeError("main/steady sample count mismatch")
        result.update({
            "excluding_compaction_per_sample_sec": steady,
            "excluding_compaction_mean_sec": mean(steady),
        })
    if note:
        result["note"] = note
    return result


def dict_components(rows: list[dict[str, float]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def build(root: Path) -> dict[str, Any]:
    validator = load_validator()
    baseline = validator.validate_baseline(root)
    kvmem = validator.validate_kvmem(root)

    lme = baseline["longmemeval_s"]
    mab = baseline["memoryagentbench_gt256k"]
    le = baseline["agentlongbench_le256k"]
    a512 = baseline["agentlongbench_512k"]
    a1m = baseline["agentlongbench_1m"]
    summary = {
        "schema_version": 2,
        "metric": (
            "final-query arrival from an already-maintained full active state "
            "through first non-empty model token; earlier history maintenance "
            "and final-answer decode excluded"
        ),
        "aggregation": "one frozen P50-length representative per benchmark slice",
        "machine": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "root": str(root.resolve()),
        "benchmarks": {
            "LongMemEval-S": {
                "Full Context": cell(lme["full_context"]["ttft_sec"]),
                "Sliding Window": cell(lme["sliding-window"]["ttft_sec"]),
                "Compact-only": cell(
                    dict_components(lme["compact-only"], "main_sec"),
                    excluding=dict_components(lme["compact-only"], "steady_sec"),
                ),
                "Compact+RAG": cell(
                    dict_components(lme["compact-rag"], "main_sec"),
                    excluding=dict_components(lme["compact-rag"], "steady_sec"),
                ),
                "KVMem": cell(kvmem["longmemeval_s"]["ttft_sec"]),
            },
            "MemoryAgentBench (>256K)": {
                "Full Context": None,
                "Sliding Window": cell(mab["sliding-window"]["ttft_sec"]),
                "Compact-only": cell(
                    mab["compact-only"]["main_sec"],
                    excluding=mab["compact-only"]["excluding_compaction_sec"],
                ),
                "Compact+RAG": cell(
                    mab["compact-rag"]["main_sec"],
                    excluding=mab["compact-rag"]["excluding_compaction_sec"],
                ),
                "KVMem": cell(kvmem["memoryagentbench_gt256k"]["ttft_sec"]),
            },
            "AgentLongBench (<=256K)": {
                "Full Context": cell(le["full_context"]["ttft_sec"]),
                "Sliding Window": cell(le["sliding-window"]["ttft_sec"]),
                "Compact-only": cell(
                    le["compact_only"]["main_sec"],
                    excluding=le["compact_only"]["excluding_compaction_sec"],
                ),
                "Compact+RAG": cell(
                    le["compact_rag_t30"]["main_sec"],
                    excluding=le["compact_rag_t30"]["excluding_compaction_sec"],
                ),
                "KVMem": cell(kvmem["agentlongbench_le256k"]["ttft_sec"]),
            },
            "AgentLongBench (512K)": {
                "Full Context": None,
                "Sliding Window": cell(a512["sliding-window"]["ttft_sec"]),
                "Compact-only": cell(
                    a512["compact-only"]["main_sec"],
                    excluding=a512["compact-only"]["excluding_compaction_sec"],
                ),
                "Compact+RAG": cell(
                    a512["compact-rag"]["main_sec"],
                    excluding=a512["compact-rag"]["excluding_compaction_sec"],
                ),
                "KVMem": cell(kvmem["agentlongbench_512k"]["ttft_sec"]),
            },
            "AgentLongBench (1M)": {
                "Full Context": None,
                "Sliding Window": cell(a1m["sliding-window"]["ttft_sec"]),
                "Compact-only": cell(
                    a1m["compact-only"]["main_sec"],
                    excluding=a1m["compact-only"]["excluding_compaction_sec"],
                ),
                "Compact+RAG": cell(
                    a1m["compact-rag"]["main_sec"],
                    excluding=a1m["compact-rag"]["excluding_compaction_sec"],
                ),
                "KVMem": cell(kvmem["agentlongbench_1m"]["ttft_sec"]),
            },
        },
    }
    return summary


METHODS = ("Full Context", "Sliding Window", "Compact-only", "Compact+RAG", "KVMem")


def format_cell(value: dict[str, Any] | None) -> str:
    if value is None:
        return "--"
    main = f"{float(value['mean_sec']):.2f}"
    steady = value.get("excluding_compaction_mean_sec")
    return f"{main} ({float(steady):.2f})" if steady is not None else main


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Benchmark | Full Context | Sliding Window | Compact-only | Compact+RAG | KVMem |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark, methods in summary["benchmarks"].items():
        lines.append(
            "| " + benchmark + " | "
            + " | ".join(format_cell(methods[method]) for method in METHODS) + " |"
        )
    return "\n".join(lines) + "\n"


def latex(summary: dict[str, Any]) -> str:
    lines = []
    for benchmark, methods in summary["benchmarks"].items():
        escaped = benchmark.replace("<=", r"$\leq$").replace(">", r"$>$")
        values = " & ".join(format_cell(methods[method]) for method in METHODS)
        lines.extend([
            rf"% {escaped}",
            rf"& Pre-answer Latency (s) $\downarrow$ & {values} \\",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()
    summary = build(args.root)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(json_text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    if args.latex_output:
        args.latex_output.write_text(latex(summary), encoding="utf-8")
    print(json_text, end="")


if __name__ == "__main__":
    main()
