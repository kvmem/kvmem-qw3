#!/usr/bin/env python3
"""Validate the current-machine paper latency artifacts before publishing.

The latency supervisors are intentionally resumable.  A clean process exit is
therefore not sufficient evidence that every requested P50 sample was
written: a runner may skip, retain a partial status file, or terminate after a
server error.  This verifier checks the frozen sample identities and the
method-specific first-token fields before a completion marker is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


ALB_IDS = {
    "le256": {
        "febc5e457e02ecea1bc51e9f3955591ee48e573c44e15a435d15c0c474062fcd",
    },
    "512k": {
        "346db45888398d0cd09cf20734b719bc2d5cf6243a530727e1b9ac20c2afed78",
    },
    "1m": {
        "5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e",
    },
}
LME_IDS = {"1d4e3b97"}
MAB_SPECS = {
    "mab_p50": ("Long_Range_Understanding", "infbench_sum_eng_shots2", 10),
}
ALB_SELECTION_MANIFEST = Path(__file__).with_name(
    "paper_latency_agentlongbench_samples_20260812.jsonl"
)
DEEPSEEK_1M_METHOD_SHA256 = (
    "4ecc2b81718b599951b01781dc3e071e68a69b9b4389eb64034d011d0b222c63"
)
DEEPSEEK_1M_TOKENIZER_LABEL = (
    "huggingface:deepseek-ai/DeepSeek-V4-Pro@"
    "b5968e9190ef611bbf34a7229255be88a0e937c1"
)
DEEPSEEK_1M_REPLAY_IDENTITY = {
    "5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e": {
        "history_sha256": "f400a55f4acd3e016c5ee134e9d2b15b144acbe5abfa3e321a1a1ec7715f89d2",
        "compacted_cursor": 2_517_398,
        "server_prompt_tokens": 1_022_416,
    },
    "7a825f6dde6347489a4c6d0a02aeb60520ed5680a857578f74ad4884053aa018": {
        "history_sha256": "b1bff973cb73f5104fba92050472cdefb98fe37b47bcb5b6d4b1dcfc05951c46",
        "compacted_cursor": 2_476_365,
        "server_prompt_tokens": 1_022_340,
    },
    "94f775dc6576cb65cd30b031541fe8b47f1ba6b96825443a01308c6a633fbcca": {
        "history_sha256": "c99ebeafd23d4c029051e2b96ad50203f414dc3ff2663d3ee4d15d548d36e03e",
        "compacted_cursor": 2_460_409,
        "server_prompt_tokens": 1_022_180,
    },
}


class ValidationError(RuntimeError):
    pass


def validate_selection_manifests() -> dict[str, Any]:
    rows = load_jsonl(ALB_SELECTION_MANIFEST)
    slice_keys = {
        "agentlongbench_le256k": "le256",
        "agentlongbench_512k": "512k",
        "agentlongbench_1m": "1m",
    }
    result: dict[str, Any] = {}
    for slice_name, key in slice_keys.items():
        selected = [
            row for row in rows
            if row.get("slice") == slice_name and row.get("quantile") == "p50"
        ]
        if [row.get("quantile") for row in selected] != ["p50"]:
            raise ValidationError(
                f"{slice_name}: selection manifest lacks exactly one P50 row"
            )
        identities = {str(row.get("stable_sample_id") or "") for row in selected}
        if identities != ALB_IDS[key]:
            raise ValidationError(
                f"{slice_name}: manifest IDs differ from frozen evaluator IDs"
            )
        history = [int(row.get("history_tokens") or 0) for row in selected]
        query = [int(row.get("query_tokens") or 0) for row in selected]
        if any(value <= 0 for value in history + query):
            raise ValidationError(
                f"{slice_name}: invalid or unordered token counts in manifest"
            )
        result[slice_name] = {
            "sample_ids": [row["stable_sample_id"] for row in selected],
            "quantiles": [row["quantile"] for row in selected],
            "history_tokens": history,
            "query_tokens": query,
        }
    if len(rows) != 9:
        raise ValidationError(
            f"AgentLongBench selection manifest has {len(rows)} rows, expected 9"
        )
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"missing JSONL artifact: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValidationError(f"expected object at {path}:{number}")
        rows.append(value)
    return rows


def validate_vllm_long_warmup(root: Path) -> None:
    """Require shape-representative prefill warm-up before paper samples.

    A one-token synthetic request only warms decode.  vLLM/Triton can still JIT
    chunked-prefill kernels on the first 64K request, which would make the first
    method absorb one-time compilation overhead.  The supervisor records two
    unrelated synthetic requests close to the 64K and 100K paper caps.
    """
    path = root / "logs" / "warmup_long.json"
    if not path.is_file():
        raise ValidationError(f"missing long-prefill warm-up artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 2:
        raise ValidationError("long-prefill warm-up must contain exactly two rows")
    for row, target in zip(value, (65_536, 102_400)):
        if row.get("synthetic_repetitions") != target:
            raise ValidationError(
                f"long warm-up target mismatch: {row.get('synthetic_repetitions')!r}"
            )
        elapsed = positive(row.get("elapsed_sec"), f"long-warmup/{target}")
        prompt_tokens = nested(row, "usage", "prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or prompt_tokens < target
            or prompt_tokens > target + 128
        ):
            raise ValidationError(
                f"long-warmup/{target}: unexpected prompt tokens {prompt_tokens!r}"
            )
        if elapsed <= 0:
            raise ValidationError(f"long-warmup/{target}: invalid elapsed time")


def validate_environment(root: Path, backend: str) -> dict[str, Any]:
    path = root / f"environment_{backend}.json"
    value = load_json(path)
    if value.get("schema_version") != 1:
        raise ValidationError(f"{backend} environment schema mismatch")
    gpu = str(value.get("gpu") or "")
    if "RTX PRO 6000 Blackwell Server Edition" not in gpu:
        raise ValidationError(f"{backend}: unexpected measurement GPU {gpu!r}")
    if not str(value.get("recorded_at") or ""):
        raise ValidationError(f"{backend}: missing environment timestamp")
    if backend == "qw3_plain":
        if value.get("backend") != "qw3":
            raise ValidationError("plain baseline backend is not qw3")
        digest = str(value.get("qw3_binary_sha256") or "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValidationError("plain qw3 environment lacks a binary SHA256")
        if not isinstance(value.get("qw3_binary_size"), int) or value["qw3_binary_size"] <= 0:
            raise ValidationError("plain qw3 environment has invalid binary size")
        if not str(value.get("model_path") or "").endswith("Qwen3.6-27B-Q8_0.gguf"):
            raise ValidationError("plain qw3 environment model-path mismatch")
        contract = value.get("server_contract") or {}
        required = {
            "kv_cache_dtype": "fp8",
            "max_model_len": 262_144,
            "max_num_seqs": 1,
            "prefill_chunk": 2_048,
            "chunked_prefill": True,
            "prefix_caching": True,
            "mtp_speculative_tokens": 0,
        }
    else:
        digest = str(value.get("binary_sha256") or "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValidationError("KVMem environment lacks a binary SHA256")
        if not isinstance(value.get("binary_size"), int) or value["binary_size"] <= 0:
            raise ValidationError("KVMem environment has invalid binary size")
        if not str(value.get("model_path") or "").endswith(
            "Qwen3.6-27B-Q8_0.gguf"
        ):
            raise ValidationError("KVMem environment model-path mismatch")
        contract = value.get("common_contract") or {}
        required = {
            "generation_reserve": 32_768,
            "prefill_chunk": 2_048,
            "query_replay": True,
            "immutable_k": True,
            "immutable_refresh_tokens": 8,
            "stage_out": True,
            "stage_in": True,
            "pack": True,
            "mtp_chain": 4,
        }
    for field, wanted in required.items():
        if contract.get(field) != wanted:
            raise ValidationError(
                f"{backend}/environment/{field}: expected {wanted!r}, "
                f"got {contract.get(field)!r}"
            )
    return value


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValidationError(f"{label}: expected a positive latency, got {value!r}")
    return float(value)


def require_close(actual: Any, expected: float, label: str) -> float:
    value = positive(actual, label)
    tolerance = max(1e-6, abs(expected) * 1e-8)
    if abs(value - expected) > tolerance:
        raise ValidationError(
            f"{label}: expected {expected:.9f}, got {value:.9f}"
        )
    return value


def unique_rows(
    rows: Iterable[dict[str, Any]], expected: set[str], label: str,
    key: str = "stable_sample_id",
) -> list[dict[str, Any]]:
    rows = list(rows)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if identity:
            mapping[identity] = row
    # Resumable output directories may retain earlier P25/P75 pilot rows.
    # Formal aggregation selects only the frozen P50 identity.
    if not expected.issubset(mapping):
        raise ValidationError(
            f"{label}: sample IDs differ; missing={sorted(expected - set(mapping))}"
        )
    return [mapping[identity] for identity in sorted(expected)]


def per_sample_answers(root: Path) -> list[dict[str, Any]]:
    answer_dir = root / "answers"
    if not answer_dir.is_dir():
        raise ValidationError(f"missing answer directory: {answer_dir}")
    return [load_json(path) for path in sorted(answer_dir.glob("*.json"))]


def check_current_deepseek_1m_compaction(root: Path) -> dict[str, Any]:
    """Validate current-run DeepSeek compaction timing for the 1M cohort.

    The strict utility prompt deliberately reuses a frozen question-blind
    DeepSeek summary, but its July timing is not a current-machine latency
    measurement.  These three calls replay the identical compaction policy on
    the frozen P25/P50/P75 histories.  Only their measured model-call wall time
    replaces the historical compaction charge; final RAG and query TTFT still
    come from the current local vLLM run.
    """
    sample_root = root / "alb1m_deepseek_compaction_current"
    config = load_json(sample_root / "summary" / "run_config.json")
    checks = {
        "method": "deepseek_1m_one_summary_plus_tail",
        "method_config_sha256": DEEPSEEK_1M_METHOD_SHA256,
        "model": "deepseek-v4-pro",
        "api_base": "https://api.deepseek.com/v1",
        "api_provider": "deepseek_official_openai_compatible",
        "tokenizer_label": DEEPSEEK_1M_TOKENIZER_LABEL,
        "prompt_tokenizer": DEEPSEEK_1M_TOKENIZER_LABEL,
        "sample_count": 100,
        "enable_thinking": True,
        "compression_rounds": 1,
        "compact_prompt_question": False,
        "compact_prompt_retrieval": False,
    }
    for field, wanted in checks.items():
        if config.get(field) != wanted:
            raise ValidationError(
                f"ALB1M/current-compaction/config/{field}: expected "
                f"{wanted!r}, got {config.get(field)!r}"
            )
    if not str(config.get("started_at") or "").startswith("2026-08-12"):
        raise ValidationError("ALB1M/current-compaction was not run on 2026-08-12")

    rows = unique_rows(
        load_jsonl(sample_root / "contexts" / "summary_plus_tail.jsonl"),
        ALB_IDS["1m"],
        "ALB1M/current-compaction",
    )
    timing_by_id: dict[str, float] = {}
    for row in rows:
        sid = str(row.get("stable_sample_id") or "")
        identity = DEEPSEEK_1M_REPLAY_IDENTITY[sid]
        if row.get("method_config_sha256") != DEEPSEEK_1M_METHOD_SHA256:
            raise ValidationError(
                f"ALB1M/current-compaction/{sid}: method SHA mismatch"
            )
        if row.get("summary_tokenizer") != DEEPSEEK_1M_TOKENIZER_LABEL:
            raise ValidationError(
                f"ALB1M/current-compaction/{sid}: tokenizer mismatch"
            )
        if row.get("history_sha256") != identity["history_sha256"]:
            raise ValidationError(
                f"ALB1M/current-compaction/{sid}: history hash mismatch"
            )
        if row.get("compacted_cursor") != identity["compacted_cursor"]:
            raise ValidationError(
                f"ALB1M/current-compaction/{sid}: compacted cursor mismatch"
            )
        server_prompt_tokens = nested(
            row, "summary_timing", "server_reported_prompt_tokens"
        )
        if server_prompt_tokens != identity["server_prompt_tokens"]:
            raise ValidationError(
                f"ALB1M/current-compaction/{sid}: unexpected server prompt "
                f"tokens {server_prompt_tokens!r}; expected "
                f"{identity['server_prompt_tokens']}"
            )
        timing_by_id[sid] = positive(
            nested(row, "summary_timing", "total_sec"),
            f"ALB1M/current-compaction/{sid}",
        )
    return {
        "samples": len(rows),
        "total_sec_by_id": timing_by_id,
        "source": "current DeepSeek API wall-time replay",
    }


def check_long_sliding(
    root: Path, expected: set[str], cap: int, label: str,
) -> dict[str, Any]:
    config = load_json(root / "summary" / "run_config.json")
    config_checks = {
        "method": "sliding_window_recent_history_qwen_total_prompt",
        "window_prompt_tokens": cap,
        "window_budget_scope": "complete_final_prompt",
        "prompt_contains_compact": False,
        "prompt_contains_retrieval": False,
    }
    for field, wanted in config_checks.items():
        if config.get(field) != wanted:
            raise ValidationError(
                f"{label}/config/{field}: expected {wanted!r}, "
                f"got {config.get(field)!r}"
            )
    sampling = config.get("controlled_sampling") or {}
    sampling_checks = {
        "api_backend": "qw3",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking_enabled": True,
        "thinking_budget_tokens": 8192,
        "answer_max_tokens": 1,
    }
    for field, wanted in sampling_checks.items():
        if sampling.get(field) != wanted:
            raise ValidationError(
                f"{label}/sampling/{field}: expected {wanted!r}, "
                f"got {sampling.get(field)!r}"
            )
    # The canonical SlidingWindow runner writes one resumable JSONL, whereas
    # the strict Compact+RAG runners use one JSON object per sample.  Reading
    # this through per_sample_answers() would silently produce an empty set.
    rows = unique_rows(
        load_jsonl(root / "answers" / "sliding_window.answers.jsonl"),
        expected,
        label,
    )
    values = []
    selection_values = []
    query_ttft_values = []
    for row in rows:
        selection = positive(
            nested(row, "window_selection_timing", "selection_total_sec"),
            f"{label}/window-selection-diagnostic",
        )
        query_ttft = positive(
            nested(row, "answer_timing", "server_ttft_sec")
            or nested(row, "answer_timing", "ttft_sec"),
            f"{label}/query-ttft",
        )
        selection_values.append(selection)
        query_ttft_values.append(query_ttft)
        # Window maintenance and suffix selection happened before the final
        # query.  The publication value is the continuation TTFT from the
        # already-resident selected window.
        values.append(query_ttft)
        row_checks = {
            "prompt_mode": "recent_history_window_plus_question",
            "window_limit_tokens": cap,
            "answer_max_tokens_requested": 1,
            "thinking_enabled": True,
            "thinking_budget_tokens": 8192,
            "latency_protocol": "final_query_from_resident_window_v1",
            "history_window_preparation_excluded": True,
        }
        for field, wanted in row_checks.items():
            if row.get(field) != wanted:
                raise ValidationError(
                    f"{label}/{field}: expected {wanted!r}, got {row.get(field)!r}"
                )
        tokens = row.get("prompt_tokens")
        if not isinstance(tokens, int) or tokens > cap:
            raise ValidationError(f"{label}: invalid prompt tokens {tokens!r}>{cap}")
        server_tokens = row.get("server_prompt_tokens")
        if isinstance(server_tokens, int) and server_tokens > cap:
            raise ValidationError(
                f"{label}: server prompt tokens {server_tokens}>{cap}"
            )
    return {
        "samples": len(rows),
        "ttft_sec": values,
        "selection_sec": selection_values,
        "query_ttft_sec": query_ttft_values,
        "prompt_cap": cap,
    }


def check_long_compact_rag(
    root: Path, expected: set[str], cap: int, label: str,
    *, compaction_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    config = load_json(root / "summary" / "run_config.json")
    config_checks = {
        "final_input_cap": cap,
        "raw_tail_tokens": 0,
        "compact_question_visible": False,
        "compact_retrieval_visible": False,
        "rag_block_size": 32,
        "rag_overlap": 8,
        "rag_prompt_order": "source_chronological",
        "rag_prompt_metadata": "none",
        "max_tokens": 1,
        "answer_temperature": 0.6,
        "answer_top_p": 0.95,
        "answer_thinking_budget": 8192,
    }
    for field, wanted in config_checks.items():
        if config.get(field) != wanted:
            raise ValidationError(
                f"{label}/config/{field}: expected {wanted!r}, "
                f"got {config.get(field)!r}"
            )
    rows = unique_rows(per_sample_answers(root), expected, label)
    main, steady = [], []
    for row in rows:
        if row.get("metadata_headers") is not False:
            raise ValidationError(f"{label}: RAG metadata headers were not disabled")
        if row.get("final_input_cap") != cap:
            raise ValidationError(
                f"{label}: row final cap {row.get('final_input_cap')!r}!={cap}"
            )
        tokens = row.get("actual_prompt_tokens")
        if not isinstance(tokens, int) or tokens > cap:
            raise ValidationError(f"{label}: invalid actual prompt tokens {tokens!r}>{cap}")
        query_ttft = positive(
            nested(row, "answer_timing", "server_ttft_sec")
            or nested(row, "answer_timing", "ttft_sec"), label
        )
        retrieval = positive(row.get("rag_query_path_sec"), f"{label}/query-retrieval")
        materialization = positive(
            row.get("final_prompt_materialization_sec"),
            f"{label}/prompt-materialization",
        )
        stored_compaction = positive(
            row.get("boundary_compaction_total_sec"), f"{label}/compaction"
        )
        expected_steady = retrieval + materialization + query_ttft
        stored_main = stored_compaction + expected_steady
        steady.append(require_close(
            row.get("pre_answer_excluding_compaction_sec"), expected_steady,
            f"{label}/steady-composition",
        ))
        require_close(
            row.get("pre_answer_boundary_sec"), stored_main,
            f"{label}/main-composition",
        )
        compaction = stored_compaction
        main.append(compaction + expected_steady)
    return {
        "samples": len(rows), "main_sec": main, "excluding_compaction_sec": steady,
        "prompt_cap": cap,
    }


def check_compact_only_jsonl(
    path: Path, expected: set[str], label: str, *, cap: int, slice_name: str,
    compaction_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = unique_rows(load_jsonl(path), expected, label)
    main, steady = [], []
    for row in rows:
        checks = {
            "status": "completed",
            "slice": slice_name,
            "method": "strict_full_history_compact_only_no_tail",
            "active_context_cap": cap,
            "raw_tail_tokens": 0,
            "retrieved_tokens": 0,
            "max_tokens": 1,
            "thinking_budget": 8192,
        }
        for field, wanted in checks.items():
            if row.get(field) != wanted:
                raise ValidationError(
                    f"{label}/{field}: expected {wanted!r}, got {row.get(field)!r}"
                )
        prompt_tokens = row.get("active_prompt_tokens_local")
        if not isinstance(prompt_tokens, int) or prompt_tokens > cap:
            raise ValidationError(
                f"{label}: invalid prompt tokens {prompt_tokens!r}>{cap}"
            )
        query = positive(row.get("query_ttft_sec"), f"{label}/query")
        provenance = row.get("summary_provenance") or {}
        stored_compaction = positive(
            provenance.get("source_compaction_total_sec"),
            f"{label}/compaction",
        )
        steady.append(require_close(
            row.get("steady_pre_answer_latency_sec"), query,
            f"{label}/steady-composition",
        ))
        require_close(
            row.get("main_pre_answer_latency_sec"), stored_compaction + query,
            f"{label}/main-composition",
        )
        compaction = (
            positive(
                compaction_override.get(str(row.get("stable_sample_id"))),
                f"{label}/current-compaction",
            )
            if compaction_override is not None
            else stored_compaction
        )
        main.append(compaction + query)
    return {"samples": len(rows), "main_sec": main, "excluding_compaction_sec": steady}


def check_mab_vllm(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("compact-rag", "compact-only", "sliding-window"):
        ttft, main, steady = [], [], []
        sliding_selection, sliding_query_ttft = [], []
        for label, expected in MAB_SPECS.items():
            workspace = root / f"{label}_boundary_v2"
            config = load_json(
                workspace / "methods" / method / "run_config_over256k.json"
            )
            server = config.get("server_required") or {}
            sampling = config.get("final_sampling") or {}
            prompt_limits = config.get("prompt_limits") or {}
            common_checks = {
                "method": (config.get("method"), method),
                "server/kv_dtype": (server.get("kv_dtype"), "fp8"),
                "server/mtp_chain": (server.get("mtp_chain"), 0),
                "server/prefix_cache": (server.get("prefix_cache"), True),
                "sampling/temperature": (sampling.get("temperature"), 0.6),
                "sampling/top_p": (sampling.get("top_p"), 0.95),
                "sampling/top_k": (sampling.get("top_k"), 20),
                "sampling/thinking": (sampling.get("thinking"), False),
                "sampling/max_tokens_override": (
                    sampling.get("max_tokens_override"), 1
                ),
                "prompt/final_input": (
                    prompt_limits.get("compact_rag_final_input"), 65_536
                ),
            }
            for field, (actual, wanted) in common_checks.items():
                if actual != wanted:
                    raise ValidationError(
                        f"{label}/{method}/{field}: expected {wanted!r}, "
                        f"got {actual!r}"
                    )
            compact_cfg = config.get("compact") or {}
            rag_cfg = config.get("rag") or {}
            sliding_cfg = config.get("sliding") or {}
            if method in ("compact-rag", "compact-only"):
                for field, wanted in {
                    "full_history": True,
                    "raw_tail_tokens": 0,
                    "question_seen": False,
                    "retrieval_seen": False,
                }.items():
                    if compact_cfg.get(field) != wanted:
                        raise ValidationError(
                            f"{label}/{method}/compact/{field}: expected "
                            f"{wanted!r}, got {compact_cfg.get(field)!r}"
                        )
            if method == "compact-rag":
                for field, wanted in {
                    "block_size": 32,
                    "overlap": 8,
                    "dynamic_prompt_budget": True,
                    "prompt_metadata": False,
                }.items():
                    if rag_cfg.get(field) != wanted:
                        raise ValidationError(
                            f"{label}/{method}/rag/{field}: expected {wanted!r}, "
                            f"got {rag_cfg.get(field)!r}"
                        )
            if method == "sliding-window" and (
                sliding_cfg.get("complete_prompt_tokens") != 65_536
            ):
                raise ValidationError(
                    f"{label}/{method}: Sliding Window cap is not 65536"
                )
            paths = sorted((workspace / "methods" / method / "rows").glob("*/results.jsonl"))
            matches: list[tuple[dict[str, Any], Path]] = []
            for path in paths:
                for candidate in load_jsonl(path):
                    if (
                        candidate.get("split"), candidate.get("source"),
                        candidate.get("dataset_row"),
                    ) == expected:
                        matches.append((candidate, path))
            if len(matches) != 1 or int(matches[0][0].get("question_index", -1)) != 0:
                raise ValidationError(
                    f"{label}/{method}: expected exactly question_index=0, got {len(matches)}"
                )
            row, result_path = matches[0]
            if row.get("previous_question_or_answer_appended") is not False:
                raise ValidationError(
                    f"{label}/{method}: question was not an independent branch"
                )
            prompt_tokens = row.get("prompt_tokens")
            if not isinstance(prompt_tokens, int) or prompt_tokens > 65_536:
                raise ValidationError(
                    f"{label}/{method}: invalid prompt tokens {prompt_tokens!r}"
                )
            query = positive(
                row.get("server_ttft_s") or row.get("ttft_s"),
                f"{label}/{method}/query-ttft",
            )
            summary_path = result_path.parent / "row_summary.json"
            if not summary_path.is_file():
                # Latency sampling deliberately runs one independent question
                # from a multi-question context.  The baseline runner records
                # that resumable subset as ``partial_summary.json`` and only
                # creates ``row_summary.json`` after every official question.
                summary_path = result_path.parent / "partial_summary.json"
            summary = load_json(summary_path)
            selection = summary.get("selection") or {}
            compact = float(selection.get("compact_boundary_total_sec") or 0.0)
            retrieval_row = row.get("retrieval") or {}
            retrieval = (
                float(retrieval_row.get("query_embedding_sec") or 0.0)
                + float(retrieval_row.get("ranking_sec") or 0.0)
                + float(retrieval_row.get("prompt_materialization_sec") or 0.0)
            )
            reprefill = float(
                selection.get("final_context_reprefill_total_sec") or 0.0
            )
            if method == "compact-only":
                ttft.append(query)
                positive(reprefill, f"{label}/{method}/reprefill")
                steady.append(reprefill + query)
                main.append(query + reprefill + compact)
            elif method == "compact-rag":
                ttft.append(query)
                retrieval_row = row.get("retrieval") or {}
                if retrieval_row.get("metadata_headers") is not False:
                    raise ValidationError(
                        f"{label}/{method}: RAG metadata headers were not disabled"
                    )
                if retrieval_row.get("dynamic_prompt_budget") is not True:
                    raise ValidationError(
                        f"{label}/{method}: dynamic RAG prompt budget is disabled"
                    )
                positive(retrieval, f"{label}/{method}/retrieval")
                positive(reprefill, f"{label}/{method}/reprefill")
                steady.append(query + retrieval + reprefill)
                main.append(query + retrieval + reprefill + compact)
            else:
                selection_sec = positive(
                    selection.get("selection_sec"),
                    f"{label}/{method}/window-selection",
                )
                sliding_selection.append(selection_sec)
                sliding_query_ttft.append(query)
                # Sliding has a resident active window at query arrival.
                ttft.append(query)
        result[method] = {
            "samples": len(ttft), "ttft_sec": ttft,
            **({"main_sec": main, "excluding_compaction_sec": steady}
               if method != "sliding-window" else {
                   "selection_sec": sliding_selection,
                   "query_ttft_sec": sliding_query_ttft,
               }),
        }
    return result


def check_alb_le_vllm(root: Path) -> dict[str, Any]:
    expected = ALB_IDS["le256"]
    # The dedicated resident-full-context probe predates the AgentLongBench
    # workers' ``stable_sample_id`` convention and writes the same frozen
    # SHA-256 identity under ``sample_id``.  Validate that field explicitly;
    # treating it as a missing row would incorrectly reject a completed run.
    full = unique_rows(
        load_jsonl(root / "alb_le256_full_cached_boundary_v2.jsonl"),
        expected,
        "ALB<=256/full",
        key="sample_id",
    )
    sliding = unique_rows(
        load_jsonl(root / "alb_le256_sliding_boundary_v2/answers/sliding_window_r32k_raw500.jsonl"),
        expected, "ALB<=256/sliding",
    )
    method_rows: dict[str, list[dict[str, Any]]] = {}
    for method in ("compact_only", "compact_rag_t30"):
        path = (
            root / "alb_le256_compact_eval_boundary_v2" / method / "answers" /
            f"{method}_paperlat_v2.answers.jsonl"
        )
        rows = unique_rows(load_jsonl(path), expected, f"ALB<=256/{method}")
        method_rows[method] = rows
    summary_rows = unique_rows(
        load_jsonl(root / "alb_le256_summary_boundary_v2/summaries/full_history_s_all_paperlat_v2.jsonl"),
        expected, "ALB<=256/summary",
    )
    summary_sec = [
        positive(
            row.get("summary_generation_total_sec")
            or nested(row, "summary_timing", "total_sec"),
            "ALB<=256/summary",
        ) for row in summary_rows
    ]
    methods = {}
    for method, rows in method_rows.items():
        ttft = [
            positive(
                nested(row, "timing", "server_ttft_sec")
                or nested(row, "timing", "ttft_sec"),
                method,
            )
            for row in rows
        ]
        retrieval = (
            [positive(row.get("rag_query_path_sec"), method) for row in rows]
            if method == "compact_rag_t30" else [0.0] * len(rows)
        )
        materialization = [
            positive(row.get("prompt_build_elapsed_sec"), f"{method}/prompt-build")
            for row in rows
        ]
        steady = [
            query + rag + build
            for query, rag, build in zip(ttft, retrieval, materialization)
        ]
        methods[method] = {
            "samples": len(rows),
            "ttft_sec": ttft,
            "retrieval_sec": retrieval,
            "prompt_materialization_sec": materialization,
            "excluding_compaction_sec": steady,
            "main_sec": [summary + value for summary, value in zip(summary_sec, steady)],
        }
    return {
        "full_context": {
            "samples": len(full),
            "ttft_sec": [
                positive(row.get("pre_answer_latency_sec"), "ALB<=256/full")
                for row in full
                if row.get("latency_protocol")
                == "final_query_from_resident_full_context_v1"
            ],
        },
        "sliding-window": {
            "samples": len(sliding),
            "ttft_sec": [
                positive(
                    nested(row, "timing", "server_ttft_sec")
                    or nested(row, "timing", "ttft_sec"),
                    "ALB<=256/sliding/query-ttft",
                )
                for row in sliding
                if row.get("latency_protocol")
                == "final_query_from_resident_window_v1"
            ],
            "selection_sec": [
                positive(
                    row.get("window_selection_sec"),
                    "ALB<=256/sliding/window-selection",
                )
                for row in sliding
            ],
            "query_ttft_sec": [
                positive(
                    nested(row, "timing", "server_ttft_sec")
                    or nested(row, "timing", "ttft_sec"),
                    "ALB<=256/sliding/query-ttft",
                )
                for row in sliding
            ],
        },
        "summary_generation_sec": summary_sec,
        **methods,
    }


def check_lme_vllm(root: Path) -> dict[str, Any]:
    full, compact, rag = [], [], []
    for qid in sorted(LME_IDS):
        row = load_json(root / "lme_s_full_context" / f"{qid}_cache_hit_timing.json")
        continuation = nested(row, "requests", "cached_continuation") or {}
        if continuation.get("finish_reason") != "length":
            raise ValidationError(
                f"LME/full/{qid}: final one-token request did not terminate by length"
            )
        full.append(positive(
            nested(continuation, "timing", "server_ttft_sec")
            or nested(continuation, "timing", "ttft_sec"),
            f"LME/full/{qid}",
        ))
        row = load_json(root / "lme_s_compact_rag" / f"{qid}_compact_rag_cache_hit_smoke.json")
        summary = positive(
            nested(row, "compact_only", "cached_summary_generation", "timing", "total_stream_elapsed_sec")
            or nested(row, "compact_only", "cached_summary_generation", "timing", "total_sec"),
            f"LME/summary/{qid}",
        )
        compact_ttft = positive(
            nested(row, "compact_only", "cached_final_answer", "timing", "ttft_sec"),
            f"LME/compact/{qid}",
        )
        reprefill = positive(
            nested(
                row, "compact_only", "time_components",
                "final_context_reprefill_total_sec",
            ),
            f"LME/reprefill/{qid}",
        )
        retrieval = positive(
            nested(row, "rag", "retrieval", "query_path_sec"),
            f"LME/retrieval/{qid}",
        )
        rag_ttft = positive(
            nested(row, "rag", "cached_final_answer", "timing", "ttft_sec"),
            f"LME/rag/{qid}",
        )
        compact.append({
            "main_sec": summary + reprefill + compact_ttft,
            "steady_sec": reprefill + compact_ttft,
        })
        rag.append({
            "main_sec": summary + retrieval + reprefill + rag_ttft,
            "steady_sec": retrieval + reprefill + rag_ttft,
        })
    sliding_rows = unique_rows(
        load_jsonl(root / "lme_s_sliding_boundary_v2.jsonl"), LME_IDS, "LME/sliding", "question_id"
    )
    for row in sliding_rows:
        checks = {
            "method": "sliding_window",
            "active_cap_tokens": 32_768,
            "active_prompt_tokens_local": 32_768,
            "truncated": True,
            "latency_protocol": "final_query_from_resident_window_v1",
            "history_window_preparation_excluded": True,
            "status": "completed",
        }
        for field, wanted in checks.items():
            if row.get(field) != wanted:
                raise ValidationError(
                    f"LME/sliding/{field}: expected {wanted!r}, "
                    f"got {row.get(field)!r}"
                )
        prompt_tokens = row.get("server_reported_prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens > 32_800:
            raise ValidationError(
                f"LME/sliding: invalid server prompt tokens {prompt_tokens!r}"
            )
        if int(row.get("selected_history_tokens_local") or 0) >= int(
            row.get("full_history_tokens_local") or 0
        ):
            raise ValidationError("LME/sliding did not truncate the full history")
    return {
        "full_context": {"samples": len(full), "ttft_sec": full},
        "sliding-window": {
            "samples": len(sliding_rows),
            "ttft_sec": [positive(row.get("pre_answer_latency_sec"), "LME/sliding") for row in sliding_rows],
        },
        "compact-only": compact,
        "compact-rag": rag,
    }


def validate_baseline(root: Path) -> dict[str, Any]:
    environment = validate_environment(root, "qw3_plain")
    validate_vllm_long_warmup(root)
    return {
        "environment": environment,
        "agentlongbench_512k": {
            "sliding-window": check_long_sliding(
                root / "alb512_sliding_boundary_v2", ALB_IDS["512k"], 65_536, "ALB512/sliding"
            ),
            "compact-rag": check_long_compact_rag(
                root / "alb512_compact_rag", ALB_IDS["512k"], 65_536, "ALB512/compact-rag"
            ),
            "compact-only": check_compact_only_jsonl(
                root / "alb512_compact_only_boundary_v2.jsonl", ALB_IDS["512k"],
                "ALB512/compact-only", cap=65_536, slice_name="512k",
            ),
        },
        "agentlongbench_1m": {
            "sliding-window": check_long_sliding(
                root / "alb1m_sliding_boundary_v2", ALB_IDS["1m"], 102_400, "ALB1M/sliding"
            ),
            "compact-rag": check_long_compact_rag(
                root / "alb1m_compact_rag", ALB_IDS["1m"], 102_400,
                "ALB1M/compact-rag",
            ),
            "compact-only": check_compact_only_jsonl(
                root / "alb1m_compact_only_boundary_v2.jsonl", ALB_IDS["1m"],
                "ALB1M/compact-only", cap=102_400, slice_name="1m",
            ),
        },
        "memoryagentbench_gt256k": check_mab_vllm(root),
        "agentlongbench_le256k": check_alb_le_vllm(root),
        "longmemeval_s": check_lme_vllm(root),
    }


def check_kvmem_jsonl(
    path: Path, expected: set[str], label: str, key: str = "sample_id",
    *, budget: int, reserve: int, block_tokens: int,
    prefill_window: str,
) -> dict[str, Any]:
    rows = unique_rows(load_jsonl(path), expected, label, key)
    for row in rows:
        checks = {
            "method": (row.get("method"), "kvmem"),
            "status": (row.get("status"), "completed"),
            "active_context_budget": (row.get("active_context_budget"), budget),
            "generation_reserve": (row.get("generation_reserve"), reserve),
            "block_tokens": (
                row.get("kvmem_block_tokens", row.get("block_tokens")),
                block_tokens,
            ),
            "prefill_window": (row.get("prefill_window"), prefill_window),
        }
        for field, (actual, wanted) in checks.items():
            if actual != wanted:
                raise ValidationError(
                    f"{label}/{field}: expected {wanted!r}, got {actual!r}"
                )
        if row.get("final_reselect") != "force":
            raise ValidationError(
                f"{label}: final semantic reselection was not forced"
            )
        if row.get("cache_state") != "clean_frozen_raw_token_session":
            raise ValidationError(
                f"{label}: unexpected cache state {row.get('cache_state')!r}"
            )
        if row.get("latency_protocol") != "final_query_boundary_v2":
            raise ValidationError(f"{label}: stale final-query latency protocol")
        if row.get("history_maintenance_excluded") is not True:
            raise ValidationError(f"{label}: history maintenance was not excluded")
        replay_prefix = row.get("alignment_replay_prefix_tokens")
        if not isinstance(replay_prefix, int) or not (
            0 <= replay_prefix < block_tokens
        ):
            raise ValidationError(
                f"{label}: final request replays {replay_prefix!r} history "
                f"tokens; exact block-boundary continuation requires "
                f"0 <= replay < {block_tokens}"
            )
        finish_tokens = row.get("finish_fragment_tokens")
        query_tokens = row.get("query_tokens")
        if (
            not isinstance(finish_tokens, int)
            or not isinstance(query_tokens, int)
            or finish_tokens > query_tokens + block_tokens + 32
        ):
            raise ValidationError(
                f"{label}: measured final fragment is unexpectedly large: "
                f"finish={finish_tokens!r}, query={query_tokens!r}, "
                f"block={block_tokens}"
            )
        paper_ms = positive(row.get("pre_answer_latency_ms"), label)
        server_ms = positive(row.get("server_ttft_ms"), f"{label}/server_ttft")
        if abs(paper_ms - server_ms) > 1e-6:
            raise ValidationError(
                f"{label}: paper latency {paper_ms} != server TTFT {server_ms}"
            )
    return {
        "samples": len(rows),
        "ttft_sec": [
            positive(row.get("pre_answer_latency_ms"), label) / 1000.0 for row in rows
        ],
    }


def validate_kvmem(root: Path) -> dict[str, Any]:
    environment = validate_environment(root, "kvmem")
    mab_root = root / "mab_kvmem_k64_b64"
    mab_config = load_json(mab_root / "run_config.json")
    expected_config = {
        "archive_storage": "cpu-only",
        "question_limit": 1,
        "answer_max_tokens_override": 1,
        "block_tokens": 64,
        "budget": 65_536,
        "gen_budget": 32_768,
        "sink_tokens": 512,
        "recent_tokens": 0,
        "prefill_chunk": 2_048,
        "prefill_window": "semantic_chunk",
        "prefill_semantic_start_tokens": 65_536,
        "prefill_semantic_query_tokens": 0,
        "immutable_refresh_tokens": 8,
        "index_placement": "gpu",
        "retrieval": "key-direction-adaptive",
        "thinking": False,
        "kv_dtype": "fp8 archive; FP16 index/query",
        "contexts_selected": 1,
    }
    for field, wanted in expected_config.items():
        if mab_config.get(field) != wanted:
            raise ValidationError(
                f"MAB/KVMem config {field}: expected {wanted!r}, "
                f"got {mab_config.get(field)!r}"
            )
    mab_paths = sorted((mab_root / "rows").glob("*/results.jsonl"))
    mab_rows = [
        row for path in mab_paths
        for row in load_jsonl(path)
    ]
    if len(mab_rows) != 1:
        raise ValidationError(f"MAB/KVMem: expected 1 sampled answer, got {len(mab_rows)}")
    actual_mab = {
        (row.get("split"), row.get("source"), row.get("dataset_row"))
        for row in mab_rows
    }
    expected_mab = set(MAB_SPECS.values())
    if actual_mab != expected_mab:
        raise ValidationError(
            f"MAB/KVMem sample identities differ: expected={expected_mab}, "
            f"got={actual_mab}"
        )
    if any(int(row.get("question_index", -1)) != 0 for row in mab_rows):
        raise ValidationError("MAB/KVMem must measure question_index=0 per context")
    for path in mab_paths:
        summary = load_json(path.parent / "row_summary.json")
        row_checks = {
            "effective_kvmem_budget": 65_536,
            "effective_sink_tokens": 512,
            "effective_recent_tokens": 0,
            "archive_storage": "tmpfs",
            "questions_expected": 1,
            "questions_completed": 1,
            "max_generation_tokens": 1,
            "scorer_fallbacks": 0,
        }
        for field, wanted in row_checks.items():
            if summary.get(field) != wanted:
                raise ValidationError(
                    f"MAB/KVMem {path.parent.name}/{field}: expected "
                    f"{wanted!r}, got {summary.get(field)!r}"
                )
    # Archive queries start their timer before query rendering/tokenization and
    # stop it at the first non-empty native callback.  Do not substitute the
    # one-token request wall time: even with max_tokens=1 it additionally
    # includes post-token bookkeeping and result I/O.
    mab_values = [positive(row.get("ttft_s"), "MAB/KVMem") for row in mab_rows]
    return {
        "environment": environment,
        "memoryagentbench_gt256k": {"samples": 1, "ttft_sec": mab_values},
        "agentlongbench_512k": check_kvmem_jsonl(
            root / "alb512_kvmem.jsonl", ALB_IDS["512k"], "ALB512/KVMem",
            budget=65_536, reserve=32_768, block_tokens=32,
            prefill_window="semantic_chunk",
        ),
        "agentlongbench_1m": check_kvmem_jsonl(
            root / "alb1m_kvmem.jsonl", ALB_IDS["1m"], "ALB1M/KVMem",
            budget=102_400, reserve=32_768, block_tokens=128,
            prefill_window="semantic_chunk",
        ),
        "agentlongbench_le256k": check_kvmem_jsonl(
            root / "alb_le256_kvmem.jsonl", ALB_IDS["le256"], "ALB<=256/KVMem",
            budget=32_768, reserve=32_768, block_tokens=32,
            prefill_window="pressure",
        ),
        "longmemeval_s": check_kvmem_jsonl(
            root / "longmemeval_s_kvmem.jsonl", LME_IDS, "LME/KVMem", "question_id",
            budget=32_768, reserve=32_768, block_tokens=32,
            prefill_window="pressure",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("baseline", "kvmem", "all"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "status": "complete",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "selection_manifests": validate_selection_manifests(),
    }
    if args.stage in ("baseline", "all"):
        report["baseline"] = validate_baseline(args.root)
    if args.stage in ("kvmem", "all"):
        report["kvmem"] = validate_kvmem(args.root)
    if args.stage == "all":
        baseline_env = report["baseline"]["environment"]
        kvmem_env = report["kvmem"]["environment"]
        if baseline_env.get("qw3_binary_sha256") != kvmem_env.get("binary_sha256"):
            raise ValidationError("plain baseline and KVMem used different qw3 binaries")
        if baseline_env.get("model_path") != kvmem_env.get("model_path"):
            raise ValidationError("plain baseline and KVMem used different model files")
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
