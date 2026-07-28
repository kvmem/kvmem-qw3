#!/usr/bin/env python3
"""Compare per-layer mean-K retrieval on grounded AgentLongBench evidence.

The input is produced by ``QW3_KVMEM_DUMP_LAYER_SCORES``.  Each normal
attention layer is evaluated against the exact same query, mean-K index,
selection budget, and benchmark-derived evidence spans.  Besides individual
layers, the report includes early/middle/late layer groups and the all-layer
production-equivalent mean.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

import analyze_agentlongbench_gold_block_coverage as gold
import run_agentlongbench_kvmem as runner


DEFAULT_TOKENIZER = gold.DEFAULT_TOKENIZER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--layer-dump", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    return parser.parse_args()


def load_layer_dump(path: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in runner.read_jsonl(path):
        kind = row.get("type")
        if kind == "sample":
            if current is not None:
                snapshots.append(current)
            current = {"meta": row, "layers": []}
        elif kind == "layer":
            if current is None:
                raise RuntimeError("layer dump starts with a layer row")
            if int(row.get("seq", -1)) != int(current["meta"].get("seq", -2)):
                raise RuntimeError("layer row sequence does not match sample row")
            current["layers"].append(row)
        else:
            raise RuntimeError(f"unsupported layer dump row type: {kind!r}")
    if current is not None:
        snapshots.append(current)

    seen: set[str] = set()
    for snapshot in snapshots:
        meta = snapshot["meta"]
        sid = str(meta.get("trace_tag") or "")
        if not sid or sid in seen:
            raise RuntimeError(f"missing/duplicate layer dump trace tag: {sid!r}")
        seen.add(sid)
        if meta.get("schema_version") != "qw3_kvmem_layer_score_dump.v1":
            raise RuntimeError(f"unsupported layer dump schema for {sid}")
        blocks = meta.get("blocks") or []
        layers = snapshot["layers"]
        if len(blocks) != int(meta["block_count"]):
            raise RuntimeError(f"block count mismatch for {sid}")
        if len(layers) != int(meta["layer_count"]):
            raise RuntimeError(f"layer count mismatch for {sid}")
        for expected_slot, layer in enumerate(layers):
            if int(layer["slot"]) != expected_slot:
                raise RuntimeError(f"non-contiguous layer slots for {sid}")
            if len(layer.get("scores") or []) != len(blocks):
                raise RuntimeError(
                    f"score count mismatch for {sid} slot={expected_slot}"
                )
    return snapshots


def strategy_slots(layer_count: int) -> dict[str, list[int]]:
    if layer_count <= 0:
        raise RuntimeError("layer dump contains no layers")
    strategies = {f"slot_{slot:02d}": [slot] for slot in range(layer_count)}
    if layer_count >= 4:
        quartile = max(1, layer_count // 4)
        strategies.update(
            {
                "early_quarter": list(range(0, quartile)),
                "middle_early_quarter": list(
                    range(quartile, min(2 * quartile, layer_count))
                ),
                "middle_late_quarter": list(
                    range(2 * quartile, min(3 * quartile, layer_count))
                ),
                "late_quarter": list(range(3 * quartile, layer_count)),
            }
        )
    half = max(1, layer_count // 2)
    strategies["first_half"] = list(range(0, half))
    strategies["last_half"] = list(range(half, layer_count))
    strategies["all_layers"] = list(range(layer_count))
    return strategies


def mean_scores(
    layers: list[dict[str, Any]], slots: Iterable[int]
) -> list[float]:
    selected = [layers[slot]["scores"] for slot in slots]
    if not selected:
        raise RuntimeError("strategy has no layers")
    inv = 1.0 / len(selected)
    return [
        sum(float(layer_scores[block]) for layer_scores in selected) * inv
        for block in range(len(selected[0]))
    ]


def select_blocks(meta: dict[str, Any], scores: list[float]) -> set[int]:
    n = int(meta["block_count"])
    budget = int(meta["budget_blocks"])
    if budget == 0 or n <= budget:
        return set(range(n))

    mandatory: set[int] = set()
    sink = min(int(meta.get("sink") or 0), n)
    mandatory.update(range(sink))
    recent = min(int(meta.get("recent") or 0), n)
    mandatory.update(range(n - recent, n))
    pin_from = int(meta.get("pin_from_block", 0xFFFFFFFF))
    if pin_from < n:
        mandatory.update(range(pin_from, n))
    if len(mandatory) > budget:
        raise RuntimeError(
            f"mandatory blocks ({len(mandatory)}) exceed budget ({budget})"
        )
    candidates = [block for block in range(n) if block not in mandatory]
    candidates.sort(key=lambda block: (scores[block], block), reverse=True)
    mandatory.update(candidates[: budget - len(mandatory)])
    return mandatory


def eligible_ranks(
    meta: dict[str, Any], scores: list[float], gold_ids: set[int]
) -> dict[str, Any]:
    n = int(meta["block_count"])
    forced: set[int] = set(range(min(int(meta.get("sink") or 0), n)))
    recent = min(int(meta.get("recent") or 0), n)
    forced.update(range(n - recent, n))
    pin_from = int(meta.get("pin_from_block", 0xFFFFFFFF))
    if pin_from < n:
        forced.update(range(pin_from, n))
    candidates = [block for block in range(n) if block not in forced]
    candidates.sort(key=lambda block: (scores[block], block), reverse=True)
    rank_by_id = {block: rank for rank, block in enumerate(candidates, start=1)}
    ranks = [rank_by_id[block] for block in gold_ids if block in rank_by_id]
    forced_gold = len(gold_ids & forced)
    best = min(ranks) if ranks else (0 if forced_gold else None)
    return {
        "eligible_blocks": len(candidates),
        "forced_gold_blocks": forced_gold,
        "best_gold_rank": best,
        "mrr": (1.0 / best) if best and best > 0 else (1.0 if best == 0 else 0.0),
    }


def scope_gold_ids(scope: dict[str, Any]) -> set[int]:
    return {
        int(block)
        for unit in scope["units"]
        for block in unit["gold_block_ids"]
    }


def aggregate_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer = gold.aggregate(rows, "answer_evidence")
    verification = gold.aggregate(rows, "verification_scope")
    n = len(rows)
    return {
        "samples": n,
        "answer_evidence": answer,
        "verification_scope": verification,
        "answer_hit_samples": sum(
            row["answer_evidence"]["metrics"]["selected_gold_blocks"] > 0
            for row in rows
        ),
        "answer_full_samples": sum(
            row["answer_evidence"]["metrics"]["selected_gold_blocks"]
            == row["answer_evidence"]["metrics"]["gold_blocks"]
            for row in rows
        ),
        "mean_reciprocal_rank": (
            sum(row["answer_rank"]["mrr"] for row in rows) / n if n else None
        ),
        "mean_best_gold_rank": (
            sum(
                row["answer_rank"]["best_gold_rank"]
                for row in rows
                if row["answer_rank"]["best_gold_rank"] is not None
            )
            / sum(
                row["answer_rank"]["best_gold_rank"] is not None for row in rows
            )
            if any(
                row["answer_rank"]["best_gold_rank"] is not None for row in rows
            )
            else None
        ),
    }


def render_markdown(
    summary: dict[str, Any], strategy_meta: dict[str, dict[str, Any]]
) -> str:
    aggregate = summary["strategies"]
    ordered = sorted(
        aggregate,
        key=lambda name: (
            aggregate[name]["answer_evidence"]["macro_block_coverage"],
            aggregate[name]["mean_reciprocal_rank"],
        ),
        reverse=True,
    )
    lines = [
        "# AgentLongBench per-layer mean-K recall",
        "",
        f"Samples: {summary['sample_count']}",
        "",
        "| Strategy | Model layers | Answer block recall | Full evidence | Hit | MRR | Verification recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        row = aggregate[name]
        meta = strategy_meta[name]
        layer_text = ",".join(str(layer) for layer in meta["model_layers"])
        lines.append(
            "| "
            f"{name} | {layer_text} | "
            f"{row['answer_evidence']['macro_block_coverage']:.4f} | "
            f"{row['answer_full_samples']}/{row['samples']} | "
            f"{row['answer_hit_samples']}/{row['samples']} | "
            f"{row['mean_reciprocal_rank']:.6f} | "
            f"{row['verification_scope']['macro_block_coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Recall is measured after applying the production block budget, "
            "sink/recent policy, and pinned question tail. MRR ranks benchmark "
            "answer-evidence blocks among non-forced retrieval candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("layer recall analysis requires a fast tokenizer")
    canonical = runner.load_canonical_module(args.benchmark_repo)
    samples = {
        str(row.get("stable_sample_id")): row
        for row in runner.read_jsonl(args.dataset)
    }
    snapshots = load_layer_dump(args.layer_dump)
    if not snapshots:
        raise RuntimeError("layer dump contains no samples")

    layer_count = int(snapshots[0]["meta"]["layer_count"])
    strategies = strategy_slots(layer_count)
    first_layers = snapshots[0]["layers"]
    strategy_meta = {
        name: {
            "slots": slots,
            "model_layers": [int(first_layers[slot]["layer"]) for slot in slots],
        }
        for name, slots in strategies.items()
    }
    by_strategy: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    per_sample: list[dict[str, Any]] = []

    for sample_index, snapshot in enumerate(snapshots, start=1):
        meta = snapshot["meta"]
        layers = snapshot["layers"]
        if int(meta["layer_count"]) != layer_count:
            raise RuntimeError("all snapshots must use the same layer count")
        sid = str(meta["trace_tag"])
        sample = samples.get(sid)
        if sample is None:
            raise RuntimeError(f"dataset is missing dump sample {sid}")
        prompt = canonical.full_context_prompt(sample)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_start = rendered.find(prompt)
        if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
            raise RuntimeError(f"rendered prompt embedding is ambiguous for {sid}")
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        if len(encoded["input_ids"]) != int(meta["prompt_tokens"]):
            raise RuntimeError(
                f"prompt token parity failure for {sid}: "
                f"local={len(encoded['input_ids'])} dump={meta['prompt_tokens']}"
            )
        base_blocks = [
            {"b": int(block), "p0": int(p0), "nt": int(nt), "sel": int(sel)}
            for block, p0, nt, sel in meta["blocks"]
        ]
        records = gold.message_records(canonical, sample, prompt)
        answer_units, verification_units, evidence_policy = (
            gold.evidence_for_sample(sample, records)
        )
        sample_result: dict[str, Any] = {
            "stable_sample_id": sid,
            "task_type": str(
                sample.get("task_type") or sample.get("question_type") or ""
            ),
            "prompt_tokens": int(meta["prompt_tokens"]),
            "source_blocks": int(meta["block_count"]),
            "budget_blocks": int(meta["budget_blocks"]),
            "evidence_policy": evidence_policy,
            "strategies": {},
        }
        for name, slots in strategies.items():
            scores = mean_scores(layers, slots)
            selected = select_blocks(meta, scores)
            blocks = [
                {**block, "sel": 1 if int(block["b"]) in selected else 0}
                for block in base_blocks
            ]
            answer_rows, answer_metrics = gold.decorate_scope(
                answer_units, prompt_start, offsets, blocks
            )
            verification_rows, verification_metrics = gold.decorate_scope(
                verification_units, prompt_start, offsets, blocks
            )
            answer_scope = {
                "metrics": answer_metrics,
                "units": answer_rows,
            }
            verification_scope = {
                "metrics": verification_metrics,
                "units": verification_rows,
            }
            result = {
                "answer_evidence": answer_scope,
                "verification_scope": verification_scope,
                "answer_rank": eligible_ranks(
                    meta, scores, scope_gold_ids(answer_scope)
                ),
            }
            sample_result["strategies"][name] = result
            by_strategy[name].append(result)
        per_sample.append(sample_result)
        print(
            f"[layer-recall] {sample_index}/{len(snapshots)} "
            f"{sid} {sample_result['task_type']}",
            flush=True,
        )

    aggregate = {
        name: aggregate_strategy(rows)
        for name, rows in by_strategy.items()
    }
    all_selected_matches = []
    for snapshot, sample_result in zip(snapshots, per_sample):
        meta = snapshot["meta"]
        all_scores = mean_scores(
            snapshot["layers"], strategies["all_layers"]
        )
        recomputed = select_blocks(meta, all_scores)
        as_run = {
            int(block)
            for block, _p0, _nt, selected in meta["blocks"]
            if int(selected) == 1
        }
        all_selected_matches.append(
            {
                "stable_sample_id": sample_result["stable_sample_id"],
                "exact": recomputed == as_run,
                "overlap": len(recomputed & as_run),
                "union": len(recomputed | as_run),
                "jaccard": (
                    len(recomputed & as_run) / len(recomputed | as_run)
                    if recomputed or as_run
                    else 1.0
                ),
            }
        )
    summary = {
        "schema_version": "agentlongbench_kvmem_layer_recall.v1",
        "sample_count": len(per_sample),
        "layer_dump": str(args.layer_dump),
        "strategy_layers": strategy_meta,
        "strategies": aggregate,
        "all_layer_selection_parity": {
            "exact_samples": sum(row["exact"] for row in all_selected_matches),
            "mean_jaccard": sum(
                row["jaccard"] for row in all_selected_matches
            )
            / len(all_selected_matches),
            "per_sample": all_selected_matches,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "per_sample_layer_recall.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in per_sample:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    runner.write_json(args.output_root / "layer_recall_summary.json", summary)
    (args.output_root / "layer_recall_report.md").write_text(
        render_markdown(summary, strategy_meta), encoding="utf-8"
    )
    print(f"[complete] layer-recall={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
