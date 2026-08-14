#!/usr/bin/env python3
"""Validate the frozen utility artifacts used by the revised paper table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULTS = {
    "mab": Path(
        "/data/chaidi/kvmem_eval/results/"
        "memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812"
    ),
    "alb512_cr": Path(
        "/data/chaidi/kvmem_eval/results/"
        "agentlongbench_512k_compact_rag_notail_cap64k_vllm_fp8_mtp4_20260811"
    ),
    "alb1m_cr": Path(
        "/data/chaidi/kvmem_eval/results/"
        "agentlongbench_1m_ds_summary_rag_notail_cap100k_vllm022_fp8_mtp4_seq2_20260811"
    ),
    "alb512_sw": Path(
        "/data/chaidi/kvmem_eval/results/"
        "agentlongbench_512k_sliding_window_cap64k_vllm022_fp8_mtp4_20260812"
    ),
    "alb1m_sw": Path(
        "/data/chaidi/kvmem_eval/results/"
        "agentlongbench_1m_sliding_window_cap100k_vllm022_fp8_mtp4_20260812"
    ),
}
DATASETS = {
    "alb512": Path(
        "/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl"
    ),
    "alb1m": Path("/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl"),
}
MAB_KVMEM_REFERENCE = Path(
    "/data/chaidi/kvmem_eval/results/"
    "memoryagentbench_infbench_r078_k64_semantic_chunk_b64_20260811_102509"
)


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"missing JSONL: {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answers(root: Path) -> list[dict[str, Any]]:
    directory = root / "answers"
    if not directory.is_dir():
        raise ValidationError(f"missing answer directory: {directory}")
    return [load_json(path) for path in sorted(directory.glob("*.json"))]


def require_unique(rows: list[dict[str, Any]], expected: int, label: str) -> None:
    ids = [str(row.get("stable_sample_id") or "") for row in rows]
    if len(rows) != expected or len(set(ids)) != expected or "" in ids:
        raise ValidationError(
            f"{label}: rows={len(rows)} unique_ids={len(set(ids))} expected={expected}"
        )


def cohort_ids(path: Path, count: int) -> set[str]:
    rows = load_jsonl(path)
    if len(rows) < count:
        raise ValidationError(f"dataset {path} has {len(rows)} rows, need {count}")
    identities = {str(row.get("stable_sample_id") or "") for row in rows[:count]}
    if len(identities) != count or "" in identities:
        raise ValidationError(f"dataset {path}: invalid first-{count} stable IDs")
    return identities


def validate_compact_rag(
    root: Path, *, expected: int, cap: int, deepseek: bool,
    expected_ids: set[str],
) -> dict[str, Any]:
    label = f"compact-rag:{root.name}"
    rows = answers(root)
    require_unique(rows, expected, label)
    if {str(row["stable_sample_id"]) for row in rows} != expected_ids:
        raise ValidationError(f"{label}: sample cohort differs from canonical IDs")
    config = load_json(root / "summary" / "run_config.json")
    summary = load_json(root / "summary" / "final_summary.json")
    if config.get("final_input_cap") != cap or config.get("raw_tail_tokens") != 0:
        raise ValidationError(f"{label}: incorrect cap/tail config")
    if config.get("rag_prompt_metadata") != "none":
        raise ValidationError(f"{label}: RAG metadata must be disabled")
    if config.get("rag_block_size") != 32 or config.get("rag_overlap") != 8:
        raise ValidationError(f"{label}: expected RAG B32/O8")
    if config.get("compact_question_visible") is not False:
        raise ValidationError(f"{label}: compaction leaked final question")
    if summary.get("completed") != expected or summary.get("expected") != expected:
        raise ValidationError(f"{label}: incomplete final summary")
    for row in rows:
        tokens = row.get("actual_prompt_tokens")
        if not isinstance(tokens, int) or not 0 < tokens <= cap:
            raise ValidationError(f"{label}: invalid prompt tokens {tokens!r}")
        if row.get("metadata_headers") is not False:
            raise ValidationError(f"{label}: answer used metadata headers")
        if row.get("answer_finish_reason") not in {"stop", "length"}:
            raise ValidationError(f"{label}: invalid finish reason")
        if row.get("config_sha256") != config.get("config_sha256"):
            raise ValidationError(f"{label}: config fingerprint mismatch")
    if deepseek:
        required = {
            "summary_provider": "deepseek",
            "summary_model": "deepseek-v4-pro",
            "summary_tail_used": False,
            "combined_context_used": False,
            "final_boundary_compaction": "reuse_question_blind_deepseek_summary_without_tail",
        }
        for key, expected_value in required.items():
            if config.get(key) != expected_value:
                raise ValidationError(
                    f"{label}: {key}={config.get(key)!r}, expected={expected_value!r}"
                )
        if any(
            row.get("precomputed_summary_used") is not True
            or row.get("precomputed_summary_provider") != "deepseek"
            for row in rows
        ):
            raise ValidationError(f"{label}: not every answer used the DeepSeek summary")
    scores = [float((row.get("evaluation") or {}).get("score") or 0.0) for row in rows]
    return {
        "samples": expected,
        "score_percent": 100.0 * sum(scores) / expected,
        "exact_percent": 100.0 * float(summary["exact_fixed_denominator"]),
        "mean_prompt_tokens": sum(int(row["actual_prompt_tokens"]) for row in rows) / expected,
        "prompt_cap": cap,
        "config_sha256": config["config_sha256"],
        "deepseek_summary": deepseek,
    }


def validate_sliding(
    root: Path, *, expected: int, cap: int, expected_ids: set[str],
) -> dict[str, Any]:
    label = f"sliding:{root.name}"
    rows = load_jsonl(root / "answers" / "sliding_window.answers.jsonl")
    require_unique(rows, expected, label)
    if {str(row["stable_sample_id"]) for row in rows} != expected_ids:
        raise ValidationError(f"{label}: sample cohort differs from canonical IDs")
    evaluations = load_jsonl(root / "eval" / "sliding_window.eval.jsonl")
    require_unique(evaluations, expected, label + "/eval")
    if {
        str(row.get("stable_sample_id") or "") for row in rows
    } != {
        str(row.get("stable_sample_id") or "") for row in evaluations
    }:
        raise ValidationError(f"{label}: answer/evaluation sample IDs differ")
    config = load_json(root / "summary" / "run_config.json")
    summary = load_json(root / "summary" / "evaluation_report.json")
    environment = load_json(root / "summary" / "serving_environment.json")
    if environment.get("vllm_version") != "0.22.0":
        raise ValidationError(
            f"{label}: expected vLLM 0.22.0, got "
            f"{environment.get('vllm_version')!r}"
        )
    if environment.get("transformers_version") != "5.10.2":
        raise ValidationError(
            f"{label}: expected Transformers 5.10.2, got "
            f"{environment.get('transformers_version')!r}"
        )
    configured_cap = config.get("window_prompt_tokens")
    if configured_cap != cap:
        raise ValidationError(f"{label}: cap={configured_cap!r}, expected={cap}")
    if (
        summary.get("expected_samples") != expected
        or summary.get("answered_samples") != expected
        or summary.get("answer_failures") != 0
    ):
        raise ValidationError(f"{label}: incomplete summary")
    controlled = config.get("controlled_sampling") or {}
    controlled_checks = {
        "api_backend": "vllm",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking_enabled": True,
        "thinking_budget_tokens": 8192,
        "answer_max_tokens": 32768,
    }
    for field, wanted in controlled_checks.items():
        if controlled.get(field) != wanted:
            raise ValidationError(
                f"{label}/sampling/{field}: expected {wanted!r}, "
                f"got {controlled.get(field)!r}"
            )
    if config.get("prompt_contains_compact") is not False:
        raise ValidationError(f"{label}: prompt unexpectedly contains compaction")
    if config.get("prompt_contains_retrieval") is not False:
        raise ValidationError(f"{label}: prompt unexpectedly contains retrieval")
    for row in rows:
        tokens = row.get("prompt_tokens")
        if not isinstance(tokens, int) or not 0 < tokens <= cap:
            raise ValidationError(f"{label}: invalid prompt tokens {tokens!r}")
        server_tokens = row.get("server_prompt_tokens")
        if not isinstance(server_tokens, int) or not 0 < server_tokens <= cap:
            raise ValidationError(
                f"{label}: invalid server prompt tokens {server_tokens!r}"
            )
        if row.get("prompt_mode") != "recent_history_window_plus_question":
            raise ValidationError(f"{label}: unexpected prompt mode")
        if row.get("window_limit_tokens") != cap:
            raise ValidationError(f"{label}: row cap mismatch")
        if row.get("thinking_budget_tokens") != 8192:
            raise ValidationError(f"{label}: row thinking budget mismatch")
    scores = [float(row.get("score") or 0.0) for row in evaluations]
    report_score = float(summary.get("mean_score_end_to_end") or 0.0)
    computed_score = sum(scores) / expected
    if abs(report_score - computed_score) > 5e-5:
        raise ValidationError(
            f"{label}: evaluation report score {report_score} != {computed_score}"
        )
    return {
        "samples": expected,
        "score_percent": 100.0 * computed_score,
        "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows) / expected,
        "prompt_cap": cap,
    }


def validate_mab(root: Path) -> dict[str, Any]:
    reference_summaries = [
        load_json(path)
        for path in sorted((MAB_KVMEM_REFERENCE / "rows").glob("*/row_summary.json"))
    ]
    expected_contexts = {
        (row.get("split"), row.get("source"), row.get("dataset_row"))
        for row in reference_summaries
    }
    if len(reference_summaries) != 30 or len(expected_contexts) != 30:
        raise ValidationError(
            "MAB K64/B64 reference does not contain 30 unique contexts"
        )
    result = {}
    for method in ("compact-rag", "sliding-window"):
        method_root = root / "methods" / method
        summaries = sorted((method_root / "rows").glob("*/row_summary.json"))
        paths = sorted((method_root / "rows").glob("*/results.jsonl"))
        rows = [row for path in paths for row in load_jsonl(path)]
        identities = {
            (row.get("split"), row.get("source"), row.get("dataset_row"), row.get("question_index"))
            for row in rows
        }
        actual_contexts = {
            (row.get("split"), row.get("source"), row.get("dataset_row"))
            for row in rows
        }
        if len(summaries) != 30 or len(rows) != 1316 or len(identities) != 1316:
            raise ValidationError(
                f"MAB/{method}: contexts={len(summaries)} rows={len(rows)} unique={len(identities)}"
            )
        if actual_contexts != expected_contexts:
            raise ValidationError(
                f"MAB/{method}: context cohort differs from K64/B64 reference"
            )
        config = load_json(method_root / "run_config_over256k.json")
        compact_cfg = config.get("compact") or {}
        rag_cfg = config.get("rag") or {}
        sliding_cfg = config.get("sliding") or {}
        if method == "compact-rag":
            required = {
                "full_history": (compact_cfg.get("full_history"), True),
                "raw_tail_tokens": (compact_cfg.get("raw_tail_tokens"), 0),
                "question_seen": (compact_cfg.get("question_seen"), False),
                "retrieval_seen": (compact_cfg.get("retrieval_seen"), False),
                "rag_block_size": (rag_cfg.get("block_size"), 32),
                "rag_overlap": (rag_cfg.get("overlap"), 8),
                "dynamic_prompt_budget": (
                    rag_cfg.get("dynamic_prompt_budget"), True
                ),
                "prompt_metadata": (rag_cfg.get("prompt_metadata"), False),
            }
        else:
            required = {
                "complete_prompt_tokens": (
                    sliding_cfg.get("complete_prompt_tokens"), 65_536
                ),
                "selection": (
                    sliding_cfg.get("selection"),
                    "one fixed recent suffix fitting every row question",
                ),
            }
        for field, (actual, wanted) in required.items():
            if actual != wanted:
                raise ValidationError(
                    f"MAB/{method}/{field}: expected {wanted!r}, got {actual!r}"
                )
        for row in rows:
            tokens = row.get("prompt_tokens_local")
            if not isinstance(tokens, int) or not 0 < tokens <= 65_536:
                raise ValidationError(f"MAB/{method}: prompt overflow {tokens!r}")
            server_tokens = row.get("prompt_tokens")
            if server_tokens is not None and (
                not isinstance(server_tokens, int)
                or not 0 < server_tokens <= 65_536
            ):
                raise ValidationError(
                    f"MAB/{method}: server prompt overflow {server_tokens!r}"
                )
            if row.get("finish_reason") not in {"stop", "length"}:
                raise ValidationError(f"MAB/{method}: invalid finish reason")
            if row.get("previous_question_or_answer_appended") is not False:
                raise ValidationError(
                    f"MAB/{method}: question was not an independent frozen branch"
                )
            if method == "compact-rag":
                retrieval = row.get("retrieval") or {}
                if retrieval.get("metadata_headers") is not False:
                    raise ValidationError("MAB/compact-rag: metadata enabled")
        prompt_cap_audit = None
        if method == "compact-rag":
            audit_path = (
                root / "compact_rag_prompt_token_audit_vllm022_after_repair.json"
            )
            prompt_cap_audit = load_json(audit_path)
            if (
                prompt_cap_audit.get("cap") != 65_536
                or prompt_cap_audit.get("questions") != 1316
                or prompt_cap_audit.get("over_cap_rows") != 0
                or prompt_cap_audit.get("over_cap_questions") != 0
                or int(prompt_cap_audit.get("max_serving_prompt_tokens") or 0)
                > 65_536
            ):
                raise ValidationError(
                    "MAB/compact-rag: serving-tokenizer cap audit failed"
                )
        local_path = method_root / "official_local_summary.json"
        special_path = method_root / "special_judge_summary.json"
        final_path = method_root / "final_summary.json"
        local = load_json(local_path) if local_path.is_file() else None
        special = load_json(special_path) if special_path.is_file() else None
        final = load_json(final_path) if final_path.is_file() else None
        judge_complete = bool(
            local is not None
            and special is not None
            and special.get("completed") == special.get("expected")
            and special.get("judge_model") == "deepseek-v4-pro"
            and final is not None
            and isinstance(final.get("overall_macro_category_mean"), (int, float))
        )
        result[method] = {
            "contexts": 30,
            "questions": 1316,
            "overall_score_percent": (
                100.0 * float(final["overall_macro_category_mean"])
                if judge_complete else None
            ),
            "mean_prompt_tokens": sum(int(row["prompt_tokens_local"]) for row in rows) / len(rows),
            "prompt_cap": 65_536,
            "special_judge_pending": not judge_complete,
            "special_judge_completed": special.get("completed") if special else 0,
            "special_judge_expected": special.get("expected") if special else None,
            # Keep the completion marker compact; the per-question audit stays
            # in its authoritative standalone JSON artifact.
            "serving_prompt_cap_audit": (
                {
                    key: prompt_cap_audit.get(key)
                    for key in (
                        "cap", "questions", "over_cap_rows",
                        "over_cap_questions", "max_serving_prompt_tokens",
                        "delta_min", "delta_max", "delta_mean",
                    )
                }
                if prompt_cap_audit is not None else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("compact-rag", "all"), default="all")
    parser.add_argument("--mab", type=Path, default=DEFAULTS["mab"])
    parser.add_argument("--alb512-compact-rag", type=Path, default=DEFAULTS["alb512_cr"])
    parser.add_argument("--alb1m-compact-rag", type=Path, default=DEFAULTS["alb1m_cr"])
    parser.add_argument("--alb512-sliding", type=Path, default=DEFAULTS["alb512_sw"])
    parser.add_argument("--alb1m-sliding", type=Path, default=DEFAULTS["alb1m_sw"])
    parser.add_argument(
        "--require-special-judge", action="store_true",
        help="fail unless every MemoryAgentBench API-judged metric is complete",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alb512_ids = cohort_ids(DATASETS["alb512"], 100)
    # The frozen 1M utility cohort is the first 50 DeepseekMillion rows
    # (setting=ki-c), exactly the population covered by summary_plus_tail.jsonl.
    alb1m_ids = cohort_ids(DATASETS["alb1m"], 50)
    report: dict[str, Any] = {
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "agentlongbench_512k_compact_rag": validate_compact_rag(
            args.alb512_compact_rag, expected=100, cap=65_536, deepseek=False,
            expected_ids=alb512_ids,
        ),
        "agentlongbench_1m_compact_rag": validate_compact_rag(
            args.alb1m_compact_rag, expected=50, cap=102_400, deepseek=True,
            expected_ids=alb1m_ids,
        ),
    }
    if args.stage == "all":
        report.update({
            "memoryagentbench_gt256k": validate_mab(args.mab),
            "agentlongbench_512k_sliding": validate_sliding(
                args.alb512_sliding, expected=100, cap=65_536,
                expected_ids=alb512_ids,
            ),
            "agentlongbench_1m_sliding": validate_sliding(
                args.alb1m_sliding, expected=50, cap=102_400,
                expected_ids=alb1m_ids,
            ),
        })
        if args.require_special_judge:
            pending = [
                method for method, values in report["memoryagentbench_gt256k"].items()
                if values["special_judge_pending"]
            ]
            if pending:
                raise ValidationError(
                    "MemoryAgentBench special judge incomplete for: " + ", ".join(pending)
                )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
