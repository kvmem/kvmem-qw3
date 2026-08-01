#!/usr/bin/env python3
"""Build the KVMem accuracy-test ledger from preserved evaluation artifacts.

The generated Markdown is the human-readable index.  The JSON registry is the
source of truth for exact sample IDs, artifact paths, parameters, and notes.
Only whitelisted configuration fields are copied; secrets and arbitrary log
contents are never included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path("/data/chaidi/kvmem_eval/results")
DEFAULT_DOC = REPO / "docs/kvmem_utility_evaluation.md"
DEFAULT_REGISTRY = REPO / "docs/kvmem_utility_evaluation_registry.json"
DEFAULT_OVERRIDES = Path(__file__).with_name("utility_eval_overrides.json")
DEFAULT_AGENT_BASELINES = Path("/home/chaidi/AgentLongBench_Motivation/outputs")
DEFAULT_LONGMEM_BASELINES = Path("/home/chaidi/kvmem_eval/KVMem_Motivation/outputs")

ID_KEYS = (
    "stable_sample_id",
    "question_id",
    "sample_id",
    "id",
    "index",
    "instance_id",
    "task_id",
)
CORRECT_KEYS = ("correct", "is_correct", "exact_correct")
PARAM_KEYS = (
    "model",
    "context_window",
    "ctx_size",
    "kvmem_budget",
    "kvmem_gen_budget",
    "kvmem_block_tokens",
    "retrieval_method",
    "kv_dtype",
    "prefill_chunk",
    "optimization_level",
    "opt_stage_out",
    "opt_stage_in",
    "opt_pack",
    "gpu_memory_utilization",
    "enable_thinking",
    "thinking_budget",
    "temperature",
    "top_p",
    "max_tokens",
    "query_replay",
    "immutable_kv",
    "mtp",
    "mtp_chain",
    "subblock",
    "round_retrieval",
    "round_padding",
    "recent_blocks",
    "sink_blocks",
    "transcript_replay",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def canonical_path(value: Any, base: Path | None = None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute() and base:
        path = base / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def iso_date(*values: Any) -> str:
    text = " ".join(str(v) for v in values if v)
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", text)
    return "-".join(match.groups()) if match else "unknown"


def normal_accuracy(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result > 1.0:
        result /= 100.0
    return result if 0.0 <= result <= 1.0 else None


def pick_id(row: dict[str, Any], fallback: int) -> str:
    for key in ID_KEYS:
        if row.get(key) is not None:
            return str(row[key])
    for container in ("sample", "reference", "metadata"):
        nested = row.get(container)
        if isinstance(nested, dict):
            for key in ID_KEYS:
                if nested.get(key) is not None:
                    return str(nested[key])
    return f"row:{fallback}"


def sample_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [pick_id(row, i) for i, row in enumerate(rows)]


def fingerprint(ids: list[str]) -> str:
    data = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12] if ids else ""


def infer_dataset(*values: Any) -> str:
    text = " ".join(str(v).lower() for v in values if v)
    if "agent" in text or "alfworld" in text or "agentlongbench" in text:
        return "AgentLongBench"
    if "longmemeval_m" in text or "longmemeval-m" in text or re.search(r"(^|[/_])m(?:10|102|_)", text):
        return "LongMemEval-M"
    if "longmemeval_s" in text or "longmemeval-s" in text:
        return "LongMemEval-S"
    # Almost all legacy flat results use abbreviated tags but the S reference.
    if any(tag in text for tag in ("s500", "s102", "subset102", "f102", "lme_s", "warm_query")):
        return "LongMemEval-S"
    return "Other/unknown"


def bool_correct(row: dict[str, Any]) -> bool | None:
    for key in CORRECT_KEYS:
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
    for key in ("autoeval_label", "label", "verdict", "judge_result"):
        value = row.get(key)
        if isinstance(value, dict):
            nested = value.get("label", value.get("correct"))
            if isinstance(nested, bool):
                return nested
            if isinstance(nested, (int, float)):
                return nested > 0
            value = value.get("response", value.get("verdict"))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"correct", "true", "1", "yes", "pass", "passed"}:
                return True
            if normalized in {"incorrect", "false", "0", "no", "fail", "failed"}:
                return False
    score = row.get("score")
    if isinstance(score, (int, float)):
        return score > 0
    return None


def extract_params(*configs: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    aliases = {
        "context_window": ("context_window", "ctx_size", "ctx"),
        "kvmem_budget": ("kvmem_budget", "budget", "block_budget", "kvmem_block_budget"),
        "kvmem_gen_budget": ("kvmem_gen_budget", "gen_budget", "generation_reserve"),
        "kvmem_block_tokens": ("kvmem_block_tokens", "block_tokens", "block_size"),
        "retrieval_method": ("retrieval_method", "retrieval", "kvmem_retrieval"),
        "kv_dtype": ("kv_dtype", "kvmem_kv_dtype"),
        "prefill_chunk": ("prefill_chunk", "prefill_chunk_size"),
        "optimization_level": ("optimization_level", "kvmem_optimization_level"),
        "opt_stage_out": ("opt_stage_out", "kvmem_opt_stage_out"),
        "opt_stage_in": ("opt_stage_in", "kvmem_opt_stage_in"),
        "opt_pack": ("opt_pack", "kvmem_opt_pack"),
        "enable_thinking": ("enable_thinking", "thinking"),
        "immutable_kv": ("immutable_kv", "immutable"),
        "mtp_chain": ("mtp_chain", "mtp_chain_length"),
        "round_retrieval": (
            "round_retrieval",
            "kvmem_round_retrieval",
            "kvmem_round_only",
        ),
        "round_padding": (
            "round_padding",
            "kvmem_round_padding",
        ),
    }
    for config in configs:
        if not isinstance(config, dict):
            continue
        sources = [config]
        for key in ("server", "evaluation", "config", "data", "kvmem", "parameters"):
            nested = config.get(key)
            if isinstance(nested, dict):
                sources.append(nested)
        for source in sources:
            for target, keys in aliases.items():
                for key in keys:
                    if source.get(key) is not None:
                        params[target] = source[key]
                        break
            for key in PARAM_KEYS:
                if source.get(key) is not None:
                    params[key] = source[key]
    return params


def merge_params(
    row: dict[str, Any], values: dict[str, Any], source: str, *, overwrite: bool = True
) -> None:
    for key, value in values.items():
        if value is None or (not overwrite and key in row["params"]):
            continue
        row["params"][key] = value
        row["param_sources"][key] = source


def infer_tag_params(run_id: str, params: dict[str, Any]) -> None:
    text = run_id.lower()
    patterns = {
        "kvmem_budget": (r"(?:^|_)k(\d+)k(?:_|$)", r"(?:^|_)(32k|64k|128k|200k|224k)(?:_|$)"),
        "kvmem_gen_budget": (r"(?:^|_)g(\d+)k(?:_|$)",),
        "kvmem_block_tokens": (r"(?:^|_)(?:bt|b)(16|32|64|128|256)(?:_|$)",),
        "prefill_chunk": (r"(?:^|_)(?:chunk|c)(2048|4096|8192)(?:_|$)",),
    }
    for key, regexes in patterns.items():
        if key in params:
            continue
        for regex in regexes:
            match = re.search(regex, text)
            if not match:
                continue
            value = match.group(1)
            if value.endswith("k"):
                params[key] = int(value[:-1]) * 1024
            elif key in ("kvmem_budget", "kvmem_gen_budget"):
                params[key] = int(value) * 1024
            else:
                params[key] = int(value)
            break
    if "retrieval_method" not in params:
        if "meank" in text or "mean_k" in text or "mean-k" in text:
            params["retrieval_method"] = "mean-k"
        elif "deltanet" in text:
            params["retrieval_method"] = "deltanet"
        elif "softmaxpages" in text:
            params["retrieval_method"] = "softmax-pages"
    if "kv_dtype" not in params:
        if "fp16" in text:
            params["kv_dtype"] = "fp16"
        elif "fp8" in text:
            params["kv_dtype"] = "fp8"
    if "query_replay" not in params and ("query_replay" in text or "qreplay" in text):
        params["query_replay"] = True
    if "immutable_kv" not in params and "immutable" in text:
        params["immutable_kv"] = True
    for level in ("kvmem_init", "opt_1", "opt_2", "opt_3"):
        if level in text:
            params.setdefault("optimization_level", level)


def infer_params_with_source(row: dict[str, Any], text: str) -> None:
    before = dict(row["params"])
    infer_tag_params(text, row["params"])
    for key in sorted(row["params"].keys() - before.keys()):
        row["param_sources"][key] = "run tag"


def log_index(results_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    roots = (results_root, results_root.parent, results_root.parent / "logs")
    for root in roots:
        if not root.exists():
            continue
        try:
            candidates = root.rglob("*.log") if root == results_root else root.glob("*.log")
            for path in candidates:
                index.setdefault(path.name, path.resolve())
        except OSError:
            continue
    return index


def likely_log(run_id: str, index: dict[str, Path]) -> Path | None:
    stems = [run_id]
    stems.append(re.sub(r"_(?:eval|rejudge|official_grade)_20\d{6}_\d{6}$", "", run_id))
    stems.append(re.sub(r"_20\d{6}(?:_\d{6})?$", "", run_id))
    suffixes = ("_server.log", "_serve.log", "_run.log", "_runner.log", ".log")
    for stem in stems:
        for suffix in suffixes:
            if stem + suffix in index:
                return index[stem + suffix]
    return None


def manifest_index(results_root: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in results_root.glob("*_manifest.json"):
        data = load_json(path)
        tag = data.get("tag")
        if isinstance(tag, str) and tag:
            manifests[tag] = path.resolve()
    return manifests


def likely_manifest(run_id: str, manifests: dict[str, Path]) -> Path | None:
    matches = [
        (tag, path)
        for tag, path in manifests.items()
        if run_id == tag or run_id.startswith(tag + "_")
    ]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else None


def command_params(command: Any) -> dict[str, Any]:
    if not isinstance(command, list):
        return {}
    value_flags = {
        "--model": ("model", str),
        "--ctx": ("context_window", int),
        "--kv-dtype": ("kv_dtype", str),
        "--kvmem-block-tokens": ("kvmem_block_tokens", int),
        "--kvmem-budget": ("kvmem_budget", int),
        "--kvmem-gen-budget": ("kvmem_gen_budget", int),
        "--kvmem-sink-blocks": ("sink_blocks", int),
        "--kvmem-recent-blocks": ("recent_blocks", int),
        "--kvmem-retrieval-method": ("retrieval_method", str),
        "--kvmem-gpu-memory-ratio": ("gpu_memory_utilization", float),
        "--thinking-budget": ("thinking_budget", int),
        "--prefill-chunk": ("prefill_chunk", int),
        "--temp": ("temperature", float),
        "--mtp-chain": ("mtp_chain", int),
        "--kvmem-optimization-level": ("optimization_level", str),
        "--kvmem-opt-stage-out": ("opt_stage_out", str),
        "--kvmem-opt-stage-in": ("opt_stage_in", str),
        "--kvmem-opt-pack": ("opt_pack", str),
    }
    bool_flags = {
        "--enable-thinking": ("enable_thinking", True),
        "--native-mtp-speculate": ("mtp", True),
        "--kvmem-immutable-k": ("immutable_kv", True),
    }
    params: dict[str, Any] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if token in value_flags and index + 1 < len(command):
            key, converter = value_flags[token]
            try:
                params[key] = converter(command[index + 1])
            except (TypeError, ValueError):
                pass
            index += 2
            continue
        if token in bool_flags:
            key, value = bool_flags[token]
            params[key] = value
        index += 1
    return params


def attach_manifest(
    row: dict[str, Any], run_id: str, manifests: dict[str, Path]
) -> dict[str, Any]:
    path = likely_manifest(run_id, manifests)
    if path is None:
        return {}
    data = load_json(path)
    row["artifacts"]["manifest"] = str(path)
    merge_params(row, command_params(data.get("server_command")), "manifest", overwrite=True)
    judge = data.get("judge")
    if judge and not row.get("judge"):
        row["judge"] = judge
    eval_command = data.get("eval_command")
    if isinstance(eval_command, list):
        try:
            data_index = eval_command.index("--data")
            row["sample_source"] = eval_command[data_index + 1]
        except (ValueError, IndexError):
            pass
    safe_environment = {
        key: value
        for key, value in (data.get("environment_flags") or {}).items()
        if isinstance(key, str) and key.startswith("QW3_")
    }
    if safe_environment:
        row["environment_flags"] = safe_environment
    return data


def parse_log_params(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    params: dict[str, Any] = {}
    patterns: dict[str, re.Pattern[str]] = {
        "context_window": re.compile(r"(?:ctx(?:_size)?|n_ctx)=(\d+)"),
        "kvmem_budget": re.compile(r"kvmem_budget=(\d+)"),
        "kvmem_gen_budget": re.compile(r"kvmem_gen_budget=(\d+)"),
        "kvmem_block_tokens": re.compile(r"kvmem_block_tokens=(\d+)"),
        "prefill_chunk": re.compile(r"prefill_chunk=(\d+)"),
        "kv_dtype": re.compile(r"kv_dtype=([A-Za-z0-9_-]+)"),
        "retrieval_method": re.compile(r"kvmem_retrieval_method=([A-Za-z0-9_-]+)"),
        "optimization_level": re.compile(r"kvmem_optimization_level=([A-Za-z0-9_-]+)"),
        "opt_stage_out": re.compile(r"kvmem_opt_stage_out=([A-Za-z0-9_-]+)"),
        "opt_stage_in": re.compile(r"kvmem_opt_stage_in=([A-Za-z0-9_-]+)"),
        "opt_pack": re.compile(r"kvmem_opt_pack=([A-Za-z0-9_-]+)"),
        "gpu_memory_utilization": re.compile(r"gpu_(?:memory_)?(?:utilization|ratio)=([0-9.]+)"),
        "round_retrieval": re.compile(r"kvmem_round_retrieval=(\d+)"),
        "subblock": re.compile(r"\[bs-subblock\].*n_subblocks=(\d+)"),
    }
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream):
                if line_no > 4000:
                    break
                for key, regex in patterns.items():
                    match = regex.search(line)
                    if match:
                        raw = match.group(1)
                        if key == "round_retrieval":
                            params[key] = raw == "1"
                        else:
                            params[key] = (
                                float(raw)
                                if "." in raw
                                and raw.replace(".", "", 1).isdigit()
                                else int(raw) if raw.isdigit() else raw
                            )
    except OSError:
        return {}
    return params


def new_row(run_id: str, dataset: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "label": "",
        "date": iso_date(run_id),
        "dataset": dataset,
        "kind": "generation",
        "status": "unknown",
        "metric": "unknown",
        "judge": None,
        "judged": False,
        "sample_count": 0,
        "evaluated_count": 0,
        "correct": None,
        "accuracy": None,
        "accuracy_denominator": None,
        "sample_ids": [],
        "sample_id_sha256_12": "",
        "sample_source": None,
        "params": {},
        "param_sources": {},
        "environment_flags": {},
        "artifacts": {},
        "errors": None,
        "truncated": None,
        "finish_reason_length": None,
        "notes": [],
    }


def set_status(row: dict[str, Any]) -> None:
    run_id = row["run_id"].lower()
    n = int(row.get("sample_count") or 0)
    dataset = row["dataset"]
    if "smoke" in run_id or n <= 3:
        status = "smoke"
    elif dataset == "LongMemEval-S" and n == 500:
        status = "full"
    elif dataset == "LongMemEval-M" and n == 102:
        status = "full"
    elif dataset == "AgentLongBench" and n in (100, 250):
        status = "full"
    else:
        status = "subset"
    if not row.get("judged"):
        status += "/unjudged"
    elif row.get("evaluated_count", n) < n:
        status += "/partial"
    row["status"] = status


def populate_artifact_rows(row: dict[str, Any], result_path: Path | None) -> None:
    rows = load_jsonl(result_path) if result_path and result_path.exists() else []
    ids = sample_ids(rows)
    if ids:
        row["sample_ids"] = ids
        row["sample_id_sha256_12"] = fingerprint(ids)
        declared_count = int(row.get("sample_count") or 0)
        if declared_count and declared_count != len(rows):
            row["notes"].append(
                f"summary 样本数为 {declared_count}，逐样本文件有 {len(rows)} 行；"
                "registry 以逐样本文件作为 cohort。"
            )
        row["sample_count"] = len(rows)
        observed_errors = sum(bool(item.get("client_error")) for item in rows)
        if observed_errors:
            row["errors"] = max(int(row.get("errors") or 0), observed_errors)
    if rows and not row["judged"]:
        values = [bool_correct(item) for item in rows]
        observed = [value for value in values if value is not None]
        if observed:
            row["judged"] = True
            row["evaluated_count"] = len(observed)
            row["correct"] = sum(observed)
            row["accuracy"] = sum(observed) / len(rows)


def parse_flat_summary(
    path: Path, logs: dict[str, Path], manifests: dict[str, Path]
) -> dict[str, Any]:
    data = load_json(path)
    run_id = path.name[: -len("_summary.json")]
    dataset = infer_dataset(run_id, data.get("ref_path"), data.get("dataset"))
    row = new_row(run_id, dataset)
    row["date"] = iso_date(data.get("timestamp_utc"), run_id)
    row["artifacts"]["summary"] = str(path.resolve())
    row["sample_source"] = data.get("ref_path") or data.get("dataset")
    merge_params(row, extract_params(data), "summary")
    if "source_jsonl" in data:
        row["kind"] = "rejudge"
        row["metric"] = "LLM judge"
        result = canonical_path(data.get("graded_jsonl_path"), path.parent)
        row["sample_count"] = int(data.get("n_graded") or data.get("n_samples") or 0)
        row["evaluated_count"] = row["sample_count"]
        row["accuracy_denominator"] = row["evaluated_count"]
        row["correct"] = data.get("n_correct")
        row["accuracy"] = normal_accuracy(data.get("overall_accuracy"))
        row["judged"] = row["accuracy"] is not None
        row["judge"] = data.get("judge_model")
        source = canonical_path(data.get("source_jsonl"), path.parent)
        if source:
            row["artifacts"]["source_results"] = str(source)
    else:
        result = canonical_path(data.get("jsonl_path") or data.get("results"), path.parent)
        row["metric"] = "LLM judge"
        row["sample_count"] = int(data.get("n_samples") or 0)
        row["evaluated_count"] = row["sample_count"] if data.get("judged") else 0
        row["accuracy_denominator"] = row["sample_count"]
        row["correct"] = data.get("n_correct")
        row["accuracy"] = normal_accuracy(data.get("overall_accuracy"))
        row["judged"] = bool(data.get("judged")) and row["accuracy"] is not None
        row["judge"] = data.get("judge_model")
        row["errors"] = data.get("n_error")
        row["truncated"] = data.get("n_truncated")
    if result:
        row["artifacts"]["results"] = str(result)
    hyp = canonical_path(data.get("hyp_path"), path.parent)
    if hyp:
        row["artifacts"]["hypotheses"] = str(hyp)
    log = likely_log(run_id, logs)
    if log:
        row["artifacts"]["server_log"] = str(log)
        merge_params(row, parse_log_params(log), "server log")
    attach_manifest(row, run_id, manifests)
    infer_params_with_source(row, run_id)
    populate_artifact_rows(row, result)
    set_status(row)
    return row


def parse_agent_dir(
    directory: Path,
    logs: dict[str, Path],
    manifests: dict[str, Path],
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or directory.name
    config_path = directory / "run_config.json"
    summary_path = directory / "accuracy_summary.json"
    eval_path = directory / "eval.jsonl"
    config = load_json(config_path)
    summary = load_json(summary_path)
    rows = load_jsonl(eval_path)
    row = new_row(run_id, "AgentLongBench")
    row["date"] = iso_date(run_id, config.get("started_at"))
    row["metric"] = "official score + exact"
    row["kind"] = "derived" if config.get("derived") or "merged" in run_id else "generation"
    row["sample_source"] = config.get("dataset")
    merge_params(row, extract_params(config), "run_config")
    infer_params_with_source(row, run_id + "_" + str(config.get("method") or ""))
    row["sample_count"] = int(summary.get("total") or config.get("selected_samples") or len(rows))
    evaluated = summary.get("evaluated")
    row["evaluated_count"] = int(len(rows) if evaluated is None else evaluated)
    row["accuracy_denominator"] = row["sample_count"]
    row["correct"] = summary.get("exact_correct")
    row["accuracy"] = normal_accuracy(summary.get("exact_accuracy_total"))
    row["judged"] = row["accuracy"] is not None and row["evaluated_count"] > 0
    row["judge"] = "AgentLongBench official evaluator"
    official = normal_accuracy(
        summary.get("strict_score_total", summary.get("official_score_evaluated"))
    )
    if official is not None:
        row["official_score"] = official
    for name, artifact in (
        ("run_config", config_path),
        ("summary", summary_path),
        ("results", eval_path),
        ("answers", directory / "answers.jsonl"),
        ("validation", directory / "validation_report.json"),
        ("manifest", directory / "manifest.jsonl"),
    ):
        if artifact.exists():
            row["artifacts"][name] = str(artifact.resolve())
    validation = load_json(directory / "validation_report.json")
    finish_counts = validation.get("finish_reason_counts") or validation.get("finish_reasons")
    if isinstance(finish_counts, dict):
        row["finish_reason_length"] = int(finish_counts.get("length") or 0)
    else:
        row["finish_reason_length"] = validation.get("finish_reason_length")
    log = likely_log(run_id, logs)
    if log:
        row["artifacts"]["server_log"] = str(log)
        merge_params(row, parse_log_params(log), "server log")
    attach_manifest(row, run_id, manifests)
    populate_artifact_rows(row, eval_path)
    set_status(row)
    return row


def parse_special_summary(
    path: Path, logs: dict[str, Path], manifests: dict[str, Path]
) -> dict[str, Any]:
    data = load_json(path)
    directory = path.parent
    run_id = directory.name
    row = new_row(run_id, infer_dataset(run_id, data.get("dataset")))
    row["date"] = iso_date(run_id)
    row["metric"] = "LLM judge"
    row["artifacts"]["summary"] = str(path.resolve())
    config_paths = [directory / name for name in ("config.json", "replay_config.json", "run_config.json")]
    configs = [load_json(item) for item in config_paths if item.exists()]
    merge_params(row, extract_params(data, *configs), "special config")
    infer_params_with_source(row, run_id)
    for config_path in config_paths:
        if config_path.exists():
            row["artifacts"][config_path.stem] = str(config_path.resolve())
    result = canonical_path(data.get("results") or data.get("result_path"), directory)
    if result is None:
        candidate = directory / "results.jsonl"
        result = candidate if candidate.exists() else None
    if result:
        row["artifacts"]["results"] = str(result.resolve())
    row["sample_count"] = int(data.get("n_samples") or 0)
    if "replay_accuracy" in data:
        row["kind"] = "derived"
        row["correct"] = data.get("replay_correct")
        row["accuracy"] = normal_accuracy(data.get("replay_accuracy"))
        row["evaluated_count"] = row["sample_count"]
        row["accuracy_denominator"] = row["sample_count"]
    else:
        row["correct"] = data.get("n_correct")
        row["accuracy"] = normal_accuracy(data.get("accuracy"))
        n_judged = data.get("n_judged")
        row["evaluated_count"] = int(row["sample_count"] if n_judged is None else n_judged)
        row["accuracy_denominator"] = row["sample_count"]
    row["judged"] = row["accuracy"] is not None and row["evaluated_count"] > 0
    row["judge"] = data.get("judge_model") or "DeepSeek judge"
    row["errors"] = data.get("n_errors", data.get("judge_errors"))
    log = likely_log(run_id, logs)
    if log:
        row["artifacts"]["server_log"] = str(log)
        merge_params(row, parse_log_params(log), "server log")
    attach_manifest(row, run_id, manifests)
    populate_artifact_rows(row, result)
    set_status(row)
    return row


def accuracy_bearing(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    keys = set().union(*(row.keys() for row in rows[:20]))
    return bool(keys.intersection((*CORRECT_KEYS, "autoeval_label", "verdict", "score")))


def parse_orphan_jsonl(
    path: Path, logs: dict[str, Path], manifests: dict[str, Path]
) -> dict[str, Any] | None:
    lowered = path.name.lower()
    if any(token in lowered for token in ("_hyp", "request", "audit", "dump", "selection", "score_trace")):
        return None
    rows = load_jsonl(path)
    if not accuracy_bearing(rows):
        return None
    run_id = path.stem
    row = new_row(run_id, infer_dataset(run_id, str(path)))
    row["date"] = iso_date(run_id)
    row["kind"] = "rejudge" if any(token in lowered for token in ("grade", "judge", "rejudge")) else "generation"
    row["metric"] = "artifact-derived"
    row["artifacts"]["results"] = str(path.resolve())
    row["sample_count"] = len(rows)
    values = [bool_correct(item) for item in rows]
    observed = [value for value in values if value is not None]
    row["evaluated_count"] = len(observed)
    row["accuracy_denominator"] = len(rows)
    row["judged"] = bool(observed)
    if observed:
        row["correct"] = sum(observed)
        row["accuracy"] = sum(observed) / len(rows)
    infer_params_with_source(row, run_id)
    populate_artifact_rows(row, path)
    log = likely_log(run_id, logs)
    if log:
        row["artifacts"]["server_log"] = str(log)
        merge_params(row, parse_log_params(log), "server log")
    attach_manifest(row, run_id, manifests)
    set_status(row)
    return row


def parse_agent_baseline(directory: Path) -> dict[str, Any]:
    summary_path = directory / "accuracy_summary.json"
    eval_path = directory / "eval.jsonl"
    summary = load_json(summary_path)
    rows = load_jsonl(eval_path)
    run_id = f"agentlongbench_baseline_{directory.name}"
    row = new_row(run_id, "AgentLongBench")
    row["date"] = iso_date(summary.get("source_snapshot"))
    row["kind"] = "baseline"
    row["metric"] = "official score + exact"
    row["judge"] = "AgentLongBench official evaluator"
    row["sample_source"] = summary.get("source_snapshot")
    row["params"]["retrieval_method"] = summary.get("method") or directory.name
    row["param_sources"]["retrieval_method"] = "accuracy_summary"
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    row["sample_count"] = int(
        selection.get("selected_total") or summary.get("eval") or len(rows)
    )
    row["evaluated_count"] = int(summary.get("eval") or len(rows))
    row["correct"] = summary.get("exact_correct")
    row["accuracy"] = normal_accuracy(
        summary.get("exact_accuracy_selected250", summary.get("exact_accuracy_total"))
    )
    row["accuracy_denominator"] = row["sample_count"]
    row["official_score"] = normal_accuracy(
        summary.get("strict_score_selected250", summary.get("official_score_evaluated"))
    )
    row["judged"] = row["accuracy"] is not None
    row["finish_reason_length"] = summary.get("finish_reason_length")
    row["artifacts"]["summary"] = str(summary_path.resolve())
    if eval_path.exists():
        row["artifacts"]["results"] = str(eval_path.resolve())
    answers = directory / "answers.jsonl"
    if answers.exists():
        row["artifacts"]["answers"] = str(answers.resolve())
    populate_artifact_rows(row, eval_path)
    set_status(row)
    return row


def baseline_method(relative: Path) -> str:
    text = str(relative).lower()
    if "full_context" in text:
        return "full-context"
    if "compact_only" in text or "compact_dsv4pro" in text:
        return "compact-only"
    match = re.search(r"retrieval_k(\d+)", text)
    if match:
        return f"compact+TF-IDF retrieval, top-{match.group(1)} sessions"
    if "retrieval" in text:
        return "compact+TF-IDF retrieval, top-6 sessions"
    return relative.stem


def parse_longmem_baseline(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    safe_name = re.sub(r"[^a-z0-9]+", "_", str(relative).lower()).strip("_")
    run_id = "longmemeval_baseline_" + re.sub(r"_dsv4pro_eval_jsonl$", "", safe_name)
    rows = load_jsonl(path)
    row = new_row(run_id, "LongMemEval-S")
    row["date"] = "unknown"
    row["kind"] = "baseline"
    row["metric"] = "LLM judge"
    row["judge"] = "deepseek-v4-pro"
    row["sample_count"] = len(rows)
    values = [bool_correct(item) for item in rows]
    observed = [value for value in values if value is not None]
    row["evaluated_count"] = len(observed)
    row["correct"] = sum(observed)
    row["accuracy"] = sum(observed) / len(rows) if rows else None
    row["accuracy_denominator"] = len(rows)
    row["judged"] = bool(rows) and len(observed) == len(rows)
    row["params"]["retrieval_method"] = baseline_method(relative)
    row["param_sources"]["retrieval_method"] = "artifact path"
    row["artifacts"]["results"] = str(path.resolve())
    output_name = path.name.replace("_dsv4pro_eval.jsonl", "_outputs.jsonl")
    output_path = path.with_name(output_name)
    if output_path.exists():
        row["artifacts"]["source_results"] = str(output_path.resolve())
    populate_artifact_rows(row, path)
    set_status(row)
    return row


def collect(
    results_root: Path,
    agent_baseline_root: Path | None,
    longmem_baseline_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logs = log_index(results_root)
    manifests = manifest_index(results_root)
    rows: list[dict[str, Any]] = []
    covered: set[Path] = set()
    flat_summaries = sorted(results_root.glob("*_summary.json"))
    for path in flat_summaries:
        row = parse_flat_summary(path, logs, manifests)
        rows.append(row)
        for value in row["artifacts"].values():
            artifact = canonical_path(value)
            if artifact:
                covered.add(artifact)
    # A/B launchers may keep each independently valid arm one directory below
    # a shared experiment root.  Discover exactly one extra level while
    # deliberately excluding deeper per-sample replay trees.
    agent_dirs = sorted({
        path.parent
        for pattern in ("*/accuracy_summary.json", "*/*/accuracy_summary.json")
        for path in results_root.glob(pattern)
    })
    for directory in agent_dirs:
        relative_parts = directory.relative_to(results_root).parts
        run_id = "_".join(relative_parts)
        row = parse_agent_dir(directory, logs, manifests, run_id)
        rows.append(row)
        for value in row["artifacts"].values():
            artifact = canonical_path(value)
            if artifact:
                covered.add(artifact)
    special_summaries = [
        path for path in sorted(results_root.glob("*/summary.json"))
        if not (path.parent / "accuracy_summary.json").exists()
    ]
    for path in special_summaries:
        row = parse_special_summary(path, logs, manifests)
        rows.append(row)
        for value in row["artifacts"].values():
            artifact = canonical_path(value)
            if artifact:
                covered.add(artifact)
    orphan_count = 0
    for path in sorted(results_root.glob("*.jsonl")):
        if path.resolve() in covered:
            continue
        row = parse_orphan_jsonl(path, logs, manifests)
        if row is not None:
            rows.append(row)
            orphan_count += 1
    agent_baseline_count = 0
    if agent_baseline_root and agent_baseline_root.exists():
        for summary_path in sorted(agent_baseline_root.glob("*/accuracy_summary.json")):
            rows.append(parse_agent_baseline(summary_path.parent))
            agent_baseline_count += 1
    longmem_baseline_count = 0
    if longmem_baseline_root and longmem_baseline_root.exists():
        for path in sorted(longmem_baseline_root.rglob("*_dsv4pro_eval.jsonl")):
            rows.append(parse_longmem_baseline(path, longmem_baseline_root))
            longmem_baseline_count += 1
    # The canonical LongMemEval-M source contains very large histories.  Avoid
    # loading it merely to classify legacy tags: full S/M result artifacts
    # already give us the canonical question-ID sets at negligible cost.
    known_ids = {
        dataset: {
            sample_id
            for item in rows
            if item["dataset"] == dataset
            for sample_id in item["sample_ids"]
            if not sample_id.startswith("row:")
        }
        for dataset in ("LongMemEval-S", "LongMemEval-M")
    }
    for row in rows:
        if row["dataset"] != "Other/unknown":
            continue
        ids = {item for item in row["sample_ids"] if not item.startswith("row:")}
        overlaps = {
            dataset: len(ids.intersection(reference_ids))
            for dataset, reference_ids in known_ids.items()
        }
        if overlaps and max(overlaps.values()) > 0:
            row["dataset"] = max(overlaps, key=overlaps.get)
            set_status(row)
    rows.sort(key=lambda item: (item["date"], item["dataset"], item["run_id"], item["kind"]))
    counts = {
        "flat_summaries": len(flat_summaries),
        "nested_accuracy_runs": len(agent_dirs),
        "special_nested_runs": len(special_summaries),
        "standalone_accuracy_jsonl": orphan_count,
        "external_agent_baselines": agent_baseline_count,
        "external_longmemeval_baselines": longmem_baseline_count,
        "total_registry_rows": len(rows),
    }
    return rows, counts


def apply_overrides(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    overrides = load_json(path)
    mappings = overrides.get("runs", {})
    for row in rows:
        override = mappings.get(row["run_id"], {}) if isinstance(mappings, dict) else {}
        if not isinstance(override, dict):
            continue
        if override.get("label"):
            row["label"] = override["label"]
        if override.get("note"):
            row["notes"].append(override["note"])
        for key in ("status", "metric", "judge", "kind", "sample_source"):
            if override.get(key) is not None:
                row[key] = override[key]
        if isinstance(override.get("params"), dict):
            merge_params(row, override["params"], "curated override")
    return overrides


def human_tokens(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    number = int(value)
    if number % 1024 == 0 and number >= 1024:
        return f"{number // 1024}K"
    return str(number)


def compact_params(params: dict[str, Any]) -> str:
    names = (
        ("context_window", "ctx"),
        ("kvmem_budget", "K"),
        ("kvmem_gen_budget", "G"),
        ("kvmem_block_tokens", "block"),
        ("retrieval_method", "retrieval"),
        ("kv_dtype", "KV"),
        ("prefill_chunk", "chunk"),
        ("optimization_level", "opt"),
        ("opt_stage_out", "stage-out"),
        ("opt_stage_in", "stage-in"),
        ("opt_pack", "pack"),
        ("thinking_budget", "think"),
        ("max_tokens", "max_out"),
    )
    parts = []
    for key, label in names:
        value = params.get(key)
        if value is not None:
            parts.append(f"{label}={human_tokens(value)}")
    for key, label in (
        ("enable_thinking", "thinking"),
        ("query_replay", "query-replay"),
        ("immutable_kv", "immutable"),
        ("mtp", "MTP"),
        ("subblock", "subblock"),
        ("round_retrieval", "round-only"),
        ("round_padding", "round-pad"),
    ):
        value = params.get(key)
        if value is True:
            parts.append(label)
        elif value not in (None, False):
            parts.append(f"{label}={value}")
    return "; ".join(parts) if parts else "—"


def result_text(row: dict[str, Any]) -> str:
    n = row.get("sample_count") or 0
    if not row.get("judged"):
        result = f"未评分（生成 {n}）"
    else:
        correct = row.get("correct")
        accuracy = row.get("accuracy")
        denominator = row.get("accuracy_denominator") or n
        result = f"{correct}/{denominator} = {accuracy * 100:.2f}%"
        if denominator != n:
            result += f"（cohort={n}）"
        official = row.get("official_score")
        if official is not None:
            result += f"；official={official * 100:.2f}"
    length = row.get("finish_reason_length")
    if length:
        result += f"；length={length}"
    return result


def md_escape(value: Any) -> str:
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def artifact_links(row: dict[str, Any]) -> str:
    labels = {
        "summary": "summary",
        "results": "逐样本",
        "source_results": "源结果",
        "answers": "answers",
        "run_config": "参数",
        "config": "参数",
        "replay_config": "参数",
        "validation": "校验",
        "manifest": "manifest",
        "server_log": "日志",
    }
    links = []
    for key, label in labels.items():
        value = row["artifacts"].get(key)
        if value:
            links.append(f"[{label}]({value})")
    return " · ".join(links) if links else "—"


def row_table(entries: Iterable[dict[str, Any]], headline: bool = False) -> list[str]:
    lines = [
        "| 日期 | 测试 | 样本 | 参数 | 结果 | 原始记录 |",
        "|---|---|---:|---|---|---|",
    ]
    for row in entries:
        name = row.get("label") if headline and row.get("label") else row["run_id"]
        sample = f'{row["sample_count"]} / {row["status"]}'
        if row.get("sample_id_sha256_12"):
            sample += f' / ids `{row["sample_id_sha256_12"]}`'
        result = result_text(row)
        if row.get("kind") != "generation":
            result += f'；{row["kind"]}'
        if row.get("notes"):
            result += "；" + " ".join(row["notes"])
        lines.append(
            "| "
            + " | ".join(
                (
                    md_escape(row["date"]),
                    md_escape(name),
                    md_escape(sample),
                    md_escape(compact_params(row["params"])),
                    md_escape(result),
                    artifact_links(row),
                )
            )
            + " |"
        )
    return lines


def render_markdown(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    overrides: dict[str, Any],
    results_root: Path,
    external_roots: list[Path],
) -> str:
    headline_ids = overrides.get("headline_runs", [])
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["run_id"], []).append(row)
    headlines: list[dict[str, Any]] = []
    missing_headlines = []
    for run_id in headline_ids:
        matches = by_id.get(run_id, [])
        if matches:
            headlines.append(matches[-1])
        else:
            missing_headlines.append(run_id)
    datasets = ("LongMemEval-S", "LongMemEval-M", "AgentLongBench", "Other/unknown")
    known_dates = [row["date"] for row in rows if row["date"] != "unknown"]
    cutoff = max(known_dates) if known_dates else "unknown"
    lines = [
        "# KVMem Utility Evaluation 总账",
        "",
        "<!-- 由 scripts/kvmem_eval/update_utility_evaluation.py 生成；请勿手工编辑。 -->",
        "",
        f"本文件汇总 `{results_root}` 以及原始基线仓库"
        f"（{', '.join(f'`{root}`' for root in external_roots)}）中截至 **{cutoff}** 的准确性测试。"
        "它同时保留正式全量、子集、smoke、失败/未评分、重评和派生控制实验，"
        "避免只留下表现较好的结果。逐样本 ID、精确路径和完整结构化参数见"
        " [JSON registry](./kvmem_utility_evaluation_registry.json)。",
        "",
        "## 口径",
        "",
        "- `generation` 是一次 KVMem 模型生成；`baseline` 是原测试仓库的 FullContext、RAG 等"
        "对照；`rejudge` 只对已有回答重新评分；`derived` 是重试合并、selected-text replay 等"
        "派生结果。这几类不能当作独立同分布运行混在一起统计。",
        "- LongMemEval 的历史主指标是 DeepSeek LLM judge 的正确率。AgentLongBench 同时记录"
        "官方连续分数和 exact accuracy；表格中的百分比默认是 exact accuracy。",
        "- `未评分` 不等于 0% 准确率。缺失的旧参数显示为 `—`，不会根据印象补写。"
        "能从 `run_config.json` 或 server log 恢复的参数才进入结构化记录。",
        "- `full/subset/smoke/partial` 描述覆盖范围；每组样本 ID 的短哈希用于确认两次测试是否"
        "真的使用同一批、同一顺序的样本。",
        "",
        "## 主要可比较结果",
        "",
    ]
    lines.extend(row_table(headlines, headline=True))
    if missing_headlines:
        lines.extend(
            (
                "",
                "> 注意：以下人工指定的 headline artifact 当前缺失："
                + ", ".join(f"`{item}`" for item in missing_headlines),
            )
        )
    lines.extend(
        (
            "",
            "## 全部历史准确性记录",
            "",
            f"当前 registry 共 **{counts['total_registry_rows']}** 行："
            f"{counts['flat_summaries']} 个旧式 summary、"
            f"{counts['nested_accuracy_runs']} 个目录式 accuracy run、"
            f"{counts['special_nested_runs']} 个专项控制实验、"
            f"{counts['standalone_accuracy_jsonl']} 个没有 summary 的独立准确性 JSONL、"
            f"{counts['external_agent_baselines']} 个 AgentLongBench 原仓库基线、"
            f"{counts['external_longmemeval_baselines']} 个 LongMemEval 原仓库基线。",
        )
    )
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        if not subset:
            continue
        lines.extend(("", f"### {dataset}", ""))
        lines.extend(row_table(subset))
    lines.extend(
        (
            "",
            "## 后续测试的记录规范",
            "",
            "每次准确性测试（包括 smoke 和失败测试）必须保留：",
            "",
            "1. `run_config.json`：模型、context/KVMem/generation budget、block、retrieval、"
            "KV dtype、prefill chunk、optimization、query replay、immutable KV、MTP、采样参数和 judge。",
            "2. manifest 或 dataset 路径，以及逐样本 ID；若使用子集，必须保留选择顺序。",
            "3. 原始 answers/hypotheses、逐样本 eval、summary、validation 和 server log。",
            "4. 重评不能覆盖原评分；使用新的 tag，并记录 `source_jsonl`。",
            "5. 重试合并或 selected-text replay 必须标成 `derived`，同时保留源运行。",
            "",
            "测试完成后从仓库根目录运行：",
            "",
            "```bash",
            "python3 scripts/kvmem_eval/update_utility_evaluation.py",
            "python3 scripts/kvmem_eval/update_utility_evaluation.py --check",
            "```",
            "",
            "人工标题和特殊说明维护在"
            " `scripts/kvmem_eval/utility_eval_overrides.json`；不要直接编辑本生成文件。"
            "`--check` 可用于提交前或 CI，发现结果目录已经变化但总账未更新时会失败。",
            "",
            "## 已知历史限制",
            "",
            "- 一部分 2026-07 早期 flat JSONL 没有独立 `run_config.json`，只能从 summary、tag 和仍存在的"
            " server log 恢复部分参数；registry 会明确保留未知项。",
            "- 同一回答可能同时存在原始 judge、V4 Pro rejudge 和合并结果。引用论文数字时必须使用"
            " `run_id + sample_id hash + kind` 三元组。",
            "- smoke/性能 A/B 只证明链路或正确性未回退，不能与完整数据集准确率直接比较。",
            "",
        )
    )
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            return False
        return current == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument(
        "--agent-baseline-root",
        type=Path,
        default=DEFAULT_AGENT_BASELINES,
        help="optional original AgentLongBench output root",
    )
    parser.add_argument(
        "--longmem-baseline-root",
        type=Path,
        default=DEFAULT_LONGMEM_BASELINES,
        help="optional original LongMemEval motivation output root",
    )
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    if not args.results_root.exists():
        parser.error(f"results root does not exist: {args.results_root}")
    agent_root = args.agent_baseline_root.resolve() if args.agent_baseline_root.exists() else None
    longmem_root = (
        args.longmem_baseline_root.resolve() if args.longmem_baseline_root.exists() else None
    )
    rows, counts = collect(args.results_root.resolve(), agent_root, longmem_root)
    overrides = apply_overrides(rows, args.overrides)
    registry = {
        "schema_version": 1,
        "results_root": str(args.results_root.resolve()),
        "external_roots": [
            str(root) for root in (agent_root, longmem_root) if root is not None
        ],
        "counts": counts,
        "runs": rows,
    }
    markdown = render_markdown(
        rows,
        counts,
        overrides,
        args.results_root.resolve(),
        [root for root in (agent_root, longmem_root) if root is not None],
    )
    registry_text = json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ok_doc = write_or_check(args.output, markdown, args.check)
    ok_registry = write_or_check(args.registry, registry_text, args.check)
    if args.check and not (ok_doc and ok_registry):
        stale = []
        if not ok_doc:
            stale.append(str(args.output))
        if not ok_registry:
            stale.append(str(args.registry))
        print("utility evaluation ledger is stale: " + ", ".join(stale), file=sys.stderr)
        return 1
    action = "checked" if args.check else "wrote"
    print(
        f"{action} {len(rows)} runs "
        f"({counts['flat_summaries']} flat summaries, "
        f"{counts['nested_accuracy_runs']} nested accuracy runs, "
        f"{counts['special_nested_runs']} special runs, "
        f"{counts['standalone_accuracy_jsonl']} standalone JSONLs, "
        f"{counts['external_agent_baselines'] + counts['external_longmemeval_baselines']} baselines)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
