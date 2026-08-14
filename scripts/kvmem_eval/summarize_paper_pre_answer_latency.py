#!/usr/bin/env python3
"""Reproduce the paper pre-answer-latency aggregates from frozen artifacts.

The script deliberately keeps utility generation out of scope.  It reads the
same sample IDs/artifacts used by the utility table and emits arithmetic means
for the revised first-token boundary.  LongMemEval-S values are registered as
legacy audited candidates because their original per-sample files use an older
runner.  AgentLongBench uses the deterministic P25/P50/P75 cohort.  The
MemoryAgentBench aggregate is question weighted over the exact >256K cohort.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


LATENCY_ROOT = Path(
    "/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803"
)
MAB_KVMEM_ROOT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "memoryagentbench_kvmem_archive_20260802_full"
)
MAB_BASELINE_ROOT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "memoryagentbench_plain_baselines_20260802"
)

# The historical AgentLongBench plain baselines ran on A40 instances.  A
# matched 32,768-token Qwen3.6-27B-Q8_0 llama.cpp cold-window gate on the
# current RTX PRO 6000 measured 16.504 s, versus 39.117871 s for the frozen
# A40 512K cohort.  Preserve raw measurements and expose a clearly labelled
# hardware-normalized estimate rather than silently replacing provenance.
ALB_A40_TO_PRO6000_SPEEDUP = 39.11787108161176 / 16.504
LLAMA_TO_QW3_PRO6000_SPEEDUP = 16.504 / 8.171
LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP = (
    ALB_A40_TO_PRO6000_SPEEDUP * LLAMA_TO_QW3_PRO6000_SPEEDUP
)

ALB_IDS = {
    "le256": [
        "eaf6372f9c37a999d2b88fa4c4a9373d212bb9e817b0c52fdd7d9b2356275a89",
        "52cdde276a122e3a13f69f3da271afefa59cf69bde72e924112329964f39c848",
        "febc5e457e02ecea1bc51e9f3955591ee48e573c44e15a435d15c0c474062fcd",
    ],
    "512k": [
        "fdd49faa1c2dc69a45de638c50948573f9b70cfdd2298973ee40fa52211d1b2b",
        "346db45888398d0cd09cf20734b719bc2d5cf6243a530727e1b9ac20c2afed78",
        "6cbf7656d6d2ac043b7b8d4483ade98b6f2b8f255e31c0067c559dcef4d2c150",
    ],
    "1m": [
        "94f775dc6576cb65cd30b031541fe8b47f1ba6b96825443a01308c6a633fbcca",
        "5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e",
        "7a825f6dde6347489a4c6d0a02aeb60520ed5680a857578f74ad4884053aa018",
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["stable_sample_id"]): row
        for row in load_jsonl(path)
        if "stable_sample_id" in row
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(values)


def first_token_s(row: dict[str, Any]) -> float:
    timing = row.get("timing") or row.get("answer_timing") or {}
    for key in ("first_reasoning_sec", "ttft_sec", "first_content_sec"):
        value = timing.get(key, row.get(key))
        if value is not None:
            return float(value)
    raise KeyError("record has no first-token timing")


def select(path: Path, ids: list[str]) -> list[dict[str, Any]]:
    rows = by_id(path)
    missing = [sample_id for sample_id in ids if sample_id not in rows]
    if missing:
        raise RuntimeError(f"{path}: missing stable IDs {missing}")
    return [rows[sample_id] for sample_id in ids]


def alb_summary() -> dict[str, Any]:
    motivation = Path("/home/chaidi/AgentLongBench_Motivation")
    long_root = Path("/home/chaidi/AgentLongBench-Long/results")
    paths = {
        "le256": {
            "full_context": LATENCY_ROOT / "alb_le256_full_context.jsonl",
            "sliding": motivation / "outputs/slidingwindow/answers.jsonl",
            "compact": motivation / "outputs/compactonly/answers.jsonl",
            "compact_rag": motivation / "outputs/compactrag/answers.jsonl",
            "kvmem": LATENCY_ROOT / "alb_le256_kvmem.jsonl",
        },
        "512k": {
            "sliding": long_root / (
                "agentlongbench_512k_normal100/"
                "sliding_window_normal100_qwen32k_totalprompt_budget10k/"
                "answers/sliding_window.answers.jsonl"
            ),
            "compact": long_root / (
                "agentlongbench_512k_normal100/compact_only_normal100/output/"
                "answers/compact_only.answers.jsonl"
            ),
            "compact_rag": long_root / (
                "agentlongbench_512k_normal100/"
                "compact_only_normal100_rag32_o8_t960/answers/final.answers.jsonl"
            ),
            "kvmem": LATENCY_ROOT / "alb_512k_kvmem.jsonl",
        },
        "1m": {
            "sliding": long_root / (
                "deepseek_million_qwen_q8_sliding_window_32k_50/"
                "answers/sliding_window.answers.jsonl"
            ),
            "compact": long_root / (
                "deepseek_million_qwen_q8_compact_only_50/"
                "answers/compact_only.answers.jsonl"
            ),
            "compact_rag": long_root / (
                "deepseek_million_qwen_q8_compact_rag_jina_t30_b1024_o128_50/"
                "answers/compact_rag.answers.jsonl"
            ),
            "kvmem": LATENCY_ROOT / "alb_1m_kvmem_v3.jsonl",
        },
    }
    result: dict[str, Any] = {}
    for slice_name, method_paths in paths.items():
        ids = ALB_IDS[slice_name]
        item: dict[str, Any] = {"n": len(ids), "sample_ids": ids}
        full_rows = load_jsonl(method_paths.get("full_context", Path("/__missing__")))
        if full_rows:
            full_by_id = {row["sample_id"]: row for row in full_rows}
            missing_full = [sid for sid in ids if sid not in full_by_id]
            if missing_full:
                item["full_context"] = {
                    "status": "incomplete", "missing_ids": missing_full
                }
            else:
                full_values = [
                    full_by_id[sid]["pre_answer_latency_ms"] / 1000
                    for sid in ids
                ]
                item["full_context"] = {
                    "per_sample_s": full_values,
                    "mean_s": mean(full_values),
                    "status": "complete",
                }
        else:
            item["full_context"] = {
                "status": "not_applicable" if slice_name != "le256" else "not_started"
            }
        for method in ("sliding", "compact", "compact_rag"):
            rows = select(method_paths[method], ids)
            values = [first_token_s(row) for row in rows]
            item[method] = {"per_sample_s": values, "steady_mean_s": mean(values)}
        sliding_raw_s = item["sliding"]["steady_mean_s"]
        item["sliding"]["pro6000_normalized_mean_s"] = (
            sliding_raw_s / ALB_A40_TO_PRO6000_SPEEDUP
        )
        item["sliding"]["pro6000_qw3_normalized_mean_s"] = (
            sliding_raw_s / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP
        )
        item["sliding"]["hardware_normalization"] = {
            "source_gpu": "NVIDIA A40",
            "target_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "speedup": ALB_A40_TO_PRO6000_SPEEDUP,
            "reference": "32,768-token Qwen3.6-27B-Q8_0 llama.cpp cold-window",
            "source_reference_s": 39.11787108161176,
            "target_reference_s": 16.504,
            "status": "estimate-not-rerun",
        }
        item["sliding"]["backend_normalization"] = {
            "source_backend": "llama.cpp",
            "target_backend": "qw3 native",
            "speedup": LLAMA_TO_QW3_PRO6000_SPEEDUP,
            "source_reference_s": 16.504,
            "target_reference_s": 8.171,
            "status": "estimate-not-rerun",
        }

        if slice_name == "le256":
            compact_rows = select(method_paths["compact"], ids)
            compaction = [float(row["summary_generation_total_sec"]) for row in compact_rows]
            rag_rows = select(method_paths["compact_rag"], ids)
            retrieval = [float(row["retrieval_time_sec"]) for row in rag_rows]
        elif slice_name == "512k":
            checkpoint_root = long_root / (
                "agentlongbench_512k_normal100/compact_only_normal100/"
                "output/checkpoints"
            )
            retrieval_root = long_root / (
                "agentlongbench_512k_normal100/"
                "compact_only_normal100_rag32_o8_t960/retrieval/final"
            )
            compaction = [
                float(json.loads((checkpoint_root / f"{sid}.json").read_text())
                      ["last_compaction_invocation_timing"]["invocation_total_sec"])
                for sid in ids
            ]
            retrieval = [
                float(json.loads((retrieval_root / f"{sid}.json").read_text())
                      ["retrieval_time_sec"])
                for sid in ids
            ]
        else:
            # The source DeepSeek 1M summary-tail generation artifacts were
            # not retained.  Keep the non-amortized main value explicitly NA;
            # do not invent it from answer-side timestamps.
            compaction = []
            retrieval_root = long_root / (
                "deepseek_million_qwen_q8_compact_rag_jina_t30_b1024_o128_50/"
                "retrieval"
            )
            retrieval = [
                float(json.loads((retrieval_root / f"{sid}.json").read_text())
                      ["retrieval_time_sec"])
                for sid in ids
            ]

        compact_steady = item["compact"]["per_sample_s"]
        rag_answer = item["compact_rag"]["per_sample_s"]
        rag_steady = [answer + retrieve for answer, retrieve in zip(rag_answer, retrieval)]
        item["compact"]["compaction_per_sample_s"] = compaction or None
        item["compact"]["main_mean_s"] = (
            mean(c + q for c, q in zip(compaction, compact_steady))
            if compaction else None
        )
        item["compact_rag"]["retrieval_per_sample_s"] = retrieval
        item["compact_rag"]["steady_per_sample_s"] = rag_steady
        item["compact_rag"]["steady_mean_s"] = mean(rag_steady)
        item["compact_rag"]["main_mean_s"] = (
            mean(c + q for c, q in zip(compaction, rag_steady))
            if compaction else None
        )

        # The plain AgentLongBench baselines above were collected on the same
        # historical A40 deployment.  Provide a simple whole-operation
        # normalization as requested for paper-table comparison.  This is an
        # estimate: it also scales the small CPU/tokenizer fraction, so raw
        # values remain the source of record.
        for method in ("compact", "compact_rag"):
            method_item = item[method]
            method_item["pro6000_normalized_steady_mean_s"] = (
                method_item["steady_mean_s"] / ALB_A40_TO_PRO6000_SPEEDUP
            )
            method_item["pro6000_qw3_normalized_steady_mean_s"] = (
                method_item["steady_mean_s"]
                / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP
            )
            raw_main = method_item.get("main_mean_s")
            method_item["pro6000_normalized_main_mean_s"] = (
                raw_main / ALB_A40_TO_PRO6000_SPEEDUP
                if raw_main is not None else None
            )
            method_item["pro6000_qw3_normalized_main_mean_s"] = (
                raw_main / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP
                if raw_main is not None else None
            )
            method_item["hardware_normalization"] = {
                "source_gpu": "NVIDIA A40",
                "target_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "speedup": ALB_A40_TO_PRO6000_SPEEDUP,
                "status": "estimate-not-rerun",
                "scope": "whole recorded operation",
            }

        kvmem_rows = load_jsonl(method_paths["kvmem"])
        if kvmem_rows:
            row_by_id = {row["sample_id"]: row for row in kvmem_rows}
            missing = [sid for sid in ids if sid not in row_by_id]
            if missing:
                item["kvmem"] = {"status": "incomplete", "missing_ids": missing}
            else:
                values = [row_by_id[sid]["pre_answer_latency_ms"] / 1000 for sid in ids]
                item["kvmem"] = {
                    "per_sample_s": values,
                    "mean_s": mean(values),
                    "status": "complete",
                }
        else:
            item["kvmem"] = {"status": "not_started"}
        result[slice_name] = item
    return result


def mab_summary() -> dict[str, Any]:
    selected_names: list[str] = []
    kvmem_values: list[float] = []
    for summary_path in sorted((MAB_KVMEM_ROOT / "rows").glob("*/row_summary.json")):
        summary = json.loads(summary_path.read_text())
        if int(summary.get("archive_tokens", 0)) <= 262_144:
            continue
        selected_names.append(summary_path.parent.name)
        for row in load_jsonl(summary_path.parent / "results.jsonl"):
            # The frozen archive result stored total request wall time and the
            # complete decode interval, not a client first-token timestamp.
            # Estimate the first decode step with its per-token mean interval.
            decoded = max(1, int(row.get("decoded_tokens", 1)))
            decode_s = float(row.get("decode_s", 0.0))
            kvmem_values.append(float(row["wall_s"]) - decode_s + decode_s / decoded)

    methods: dict[str, Any] = {}
    for method in ("sliding-window", "compact-only", "compact-rag"):
        values: list[float] = []
        main_values: list[float] = []
        for name in selected_names:
            row_dir = MAB_BASELINE_ROOT / "methods" / method / "rows" / name
            rows = load_jsonl(row_dir / "results.jsonl")
            if not rows:
                raise RuntimeError(f"missing baseline results: {row_dir}")
            compaction_s = 0.0
            cold_window_prefill_s = 0.0
            if method == "sliding-window":
                # The utility runner primed one fixed 32K recent window per
                # context, so row ttft_s only measures the warm-prefix query.
                # Reconstruct the cold-window boundary by charging that saved
                # cold prefill to every independently evaluated question, then
                # add the observed question-suffix TTFT.  Do not add
                # selection_sec: the fixed recent window is context-dependent,
                # not query-dependent, and the utility runner selects it once.
                row_summary = json.loads((row_dir / "row_summary.json").read_text())
                cold_window_prefill_s = float(
                    row_summary["prefix_cache"]["warmup_ttft_s"]
                )
            if method != "sliding-window":
                state_path = MAB_BASELINE_ROOT / "shared" / "compact" / f"{name}.json"
                state = json.loads(state_path.read_text())
                compaction_s = sum(
                    float(r.get("selection_sec", 0.0)) + float(r.get("latency_s", 0.0))
                    for r in state["rounds"]
                )
            for row in rows:
                query_s = float(row["ttft_s"]) + cold_window_prefill_s
                values.append(query_s)
                main_values.append(query_s + compaction_s)
        methods[method] = {
            "questions": len(values),
            "steady_mean_s": mean(values),
            "main_mean_s": mean(main_values),
            **(
                {
                    "boundary": "cold 32K window prefill + question-suffix TTFT",
                    "cache_state": "cold-window",
                }
                if method == "sliding-window"
                else {}
            ),
        }
    return {
        "contexts": len(selected_names),
        "questions": len(kvmem_values),
        "selection": "archive_tokens > 262144",
        "kvmem": {
            "mean_s": mean(kvmem_values),
            "boundary": "wall_s - decode_s + decode_s / decoded_tokens",
            "status": "reconstructed-first-token",
        },
        **methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {
        "metric": "query-dependent preparation through first non-empty model token",
        "aggregation": "arithmetic mean",
        "longmemeval_s": {
            "status": "legacy-candidate-audited-at-aggregate-level",
            "full_context_s": 0.81,
            "sliding_s": 42.29,
            "compact_main_s": 68.88,
            "compact_steady_s": 3.50,
            "compact_rag_main_s": 116.13,
            "compact_rag_steady_s": 50.75,
            "kvmem_s": 1.68,
            "pro6000_qw3_normalized_estimate": {
                "status": "legacy-estimate-not-rerun",
                "source_assumption": "A40 llama.cpp plain baselines",
                "speedup": LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "full_context_s": 0.81 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "sliding_s": 42.29 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "compact_main_s": 68.88 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "compact_steady_s": 3.50 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "compact_rag_main_s": 116.13 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "compact_rag_steady_s": 50.75 / LEGACY_A40_LLAMA_TO_PRO6000_QW3_SPEEDUP,
                "kvmem_s": 1.68,
            },
        },
        "memoryagentbench_gt256k": mab_summary(),
        "agentlongbench": alb_summary(),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
