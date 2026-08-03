#!/usr/bin/env python3
"""Run MemoryAgentBench's API-based LongMemEval and InfBench judge stages.

The judge endpoint is OpenAI-compatible. Credentials are read only from
``DEEPSEEK_API_KEY`` and never written to the result files. The official prompt
text is loaded from a pinned checkout of MemoryAgentBench; the selected judge
model is recorded because using DeepSeek instead of the paper's GPT-4 judge is
method-compatible but not model-identical.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any

import requests


PINNED_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"
FULL_QUESTION_COUNT = 3671
FULL_SPECIAL_TARGET_COUNT = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument(
        "--official-repo", type=Path,
        default=Path("/home/chaidi/MemoryAgentBench-official"),
    )
    p.add_argument("--base-url", default=os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--model", default=os.environ.get(
        "DEEPSEEK_MODEL", "deepseek-v4-pro"))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument(
        "--thinking", choices=("disabled", "enabled"), default="enabled",
        help=(
            "DeepSeek thinking mode for LongMemEval; enabled is the V4-Pro "
            "API default"
        ),
    )
    p.add_argument(
        "--infbench-thinking", choices=("disabled", "enabled"),
        default="disabled",
        help=(
            "DeepSeek thinking mode for InfBench's JSON rubrics. Disabled is "
            "the default because enabled mode can consume the entire output "
            "budget in reasoning and return no JSON final content."
        ),
    )
    p.add_argument(
        "--longmem-max-tokens", type=int, default=1024,
        help="judge output budget including DeepSeek reasoning tokens",
    )
    p.add_argument(
        "--allow-partial", action="store_true",
        help="permit diagnostic judging before the full benchmark is generated",
    )
    p.add_argument(
        "--limit", type=int,
        help="diagnostic cap applied after completeness checks",
    )
    return p.parse_args()


def load_string_constants(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            out[target.id] = ast.literal_eval(node.value)
    missing = names - set(out)
    if missing:
        raise RuntimeError(f"missing official prompt constants: {sorted(missing)}")
    return out


def longmem_prompt(task: str, question: str, answer: Any, response: str,
                   abstention: bool) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
            "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        ).format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response is equivalent to the correct "
            "answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a "
            "subset of the information required by the answer, answer no. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
    elif task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response is equivalent to the correct "
            "answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a "
            "subset of the information required by the answer, answer no. In "
            "addition, do not penalize off-by-one errors for the number of days. If "
            "the question asks for the number of days/weeks/months, etc., and the "
            "model makes off-by-one errors (e.g., predicting 19 days when the "
            "answer is 18), the model's response is still correct. \n\nQuestion: "
            "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
            "response correct? Answer yes or no only."
        )
    elif task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    return template.format(question, answer, response)


def call(session: requests.Session, args: argparse.Namespace,
         api_key: str, prompt: str, max_tokens: int = 512,
         thinking: str = "enabled") -> str:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": thinking},
    }
    last = ""
    for attempt in range(args.max_retries):
        try:
            response = session.post(
                args.base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                content = (choice["message"].get("content") or "").strip()
                if content:
                    return content
                usage = data.get("usage") or {}
                last = (
                    "empty final content: finish_reason="
                    f"{choice.get('finish_reason')} completion_tokens="
                    f"{usage.get('completion_tokens')} reasoning_tokens="
                    f"{(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}"
                )
                continue
            last = f"HTTP {response.status_code}: {response.text[:500]}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"judge call failed: {last}")


def parse_json(text: str) -> dict[str, Any]:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for match in reversed(matches):
        try:
            value = json.loads(match)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError(f"judge did not return JSON: {text[:500]}")


def json_repair_prompt(prompt: str, schema: str) -> str:
    """Request the same judgment again with an explicit machine-only output.

    The pinned official prompts permit reasoning before the final object.
    DeepSeek normally follows that contract, but occasionally returns only the
    prose reasoning and omits the JSON object.  Such a response cannot be
    scored.  A repair is a fresh evaluation of the identical rubric/input, not
    a parser heuristic over incomplete prose.
    """
    return (
        "Re-evaluate the task below. Return exactly one valid JSON object and "
        "nothing else: no analysis, no markdown, and no code fence. The "
        f"required schema is {schema}. All values must be JSON numbers.\n\n"
        + prompt
    )


def key(record: dict[str, Any]) -> str:
    return f"{record['split']}|{record['dataset_row']}|{record['question_index']}"


def call_cache_key(record_key: str, stage: str, args: argparse.Namespace,
                   prompt: str, max_tokens: int, thinking: str) -> str:
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return "|".join((
        record_key,
        stage,
        args.model,
        args.base_url.rstrip("/"),
        thinking,
        str(max_tokens),
        prompt_sha256,
    ))


def category(source: str) -> str:
    if source.startswith(("eventqa_", "ruler_", "longmemeval_")):
        return "Accurate_Retrieval"
    if source.startswith("factconsolidation_"):
        return "Conflict_Resolution"
    if source.startswith(("infbench_", "detective_")):
        return "Long_Range_Understanding"
    if source.startswith(("icl_", "recsys_")):
        return "Test_Time_Learning"
    raise ValueError(source)


def main() -> int:
    args = parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(
            f"official repository HEAD {head} != pinned {PINNED_COMMIT}"
        )
    # A fully cached run must remain usable as a deterministic merge/rebuild
    # step after local scoring is regenerated.  Require the credential only
    # if an actual network call is missing from the call cache.
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    prompts = load_string_constants(
        args.official_repo / "llm_based_eval/summarization_evaluate.py",
        {"fluency_prompt_book", "recall_prompt_book", "precision_prompt_book"},
    )
    output = args.results_dir / "special_judgments.jsonl"
    call_output = args.results_dir / "special_judge_calls.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    dropped_existing = 0
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                valid_output = (
                    bool(str(row.get("judge_output") or "").strip())
                    if row.get("metric") == "llm_judge_correct"
                    else all(
                        str(value or "").strip()
                        for value in row.get("judge_outputs", [])
                    ) and len(row.get("judge_outputs", [])) == 3
                )
                expected_thinking = (
                    args.thinking
                    if row.get("metric") == "llm_judge_correct"
                    else args.infbench_thinking
                )
                config_matches = (
                    row.get("judge_model") == args.model and
                    row.get("judge_thinking") == expected_thinking and
                    (
                        row.get("metric") != "llm_judge_correct" or
                        row.get("judge_max_tokens") == args.longmem_max_tokens
                    )
                )
                if valid_output and config_matches:
                    existing[row["key"]] = row
                else:
                    dropped_existing += 1
    if dropped_existing:
        temp = output.with_suffix(".jsonl.tmp")
        temp.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in existing.values()
            ),
            encoding="utf-8",
        )
        temp.replace(output)
        print(
            f"discarded {dropped_existing} stale/empty prior judgments",
            flush=True,
        )
    cached_calls: dict[str, str] = {}
    if call_output.exists():
        for line in call_output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached_calls[row["cache_key"]] = row["output"]
    records = []
    for path in sorted((args.results_dir / "rows").glob("*/results.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines() if line.strip())
    targets = [r for r in records if r["source"].startswith(
        ("longmemeval_", "infbench_"))]
    identities = [key(record) for record in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate benchmark question identities in results")
    if not args.allow_partial and (
        len(records) != FULL_QUESTION_COUNT or
        len(targets) != FULL_SPECIAL_TARGET_COUNT
    ):
        raise RuntimeError(
            "refusing partial special judge: found "
            f"{len(records)}/{FULL_QUESTION_COUNT} total questions and "
            f"{len(targets)}/{FULL_SPECIAL_TARGET_COUNT} API targets; use "
            "--allow-partial only for diagnostics"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        targets = targets[:args.limit]
    session = requests.Session()
    # Prior grading runs on this host were accidentally routed through stale
    # http_proxy/https_proxy settings. The configured API endpoint should be
    # contacted directly and deterministically.
    session.trust_env = False

    def resumable_call(record_key: str, stage: str, prompt: str,
                       max_tokens: int, thinking: str) -> str:
        cache_key = call_cache_key(
            record_key, stage, args, prompt, max_tokens, thinking
        )
        if cache_key in cached_calls:
            return cached_calls[cache_key]
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for uncached judge call "
                f"{record_key} stage={stage}"
            )
        raw = call(
            session, args, api_key, prompt, max_tokens=max_tokens,
            thinking=thinking,
        )
        row = {
            "cache_key": cache_key,
            "record_key": record_key,
            "stage": stage,
            "judge_model": args.model,
            "judge_base_url": args.base_url.rstrip("/"),
            "judge_thinking": thinking,
            "max_tokens": max_tokens,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "output": raw,
        }
        with call_output.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        cached_calls[cache_key] = raw
        return raw

    def resumable_json_call(record_key: str, stage: str, prompt: str,
                            schema: str,
                            required_keys: tuple[str, ...],
                            ) -> tuple[dict[str, Any], str]:
        """Use a valid cached response or repair a prose-only judge output."""
        def checked_json(value: str) -> dict[str, Any]:
            parsed = parse_json(value)
            missing = [name for name in required_keys if name not in parsed]
            non_numeric = [
                name for name in required_keys
                if name in parsed and (
                    isinstance(parsed[name], bool) or
                    not isinstance(parsed[name], (int, float))
                )
            ]
            if missing or non_numeric:
                raise ValueError(
                    f"invalid judge object missing={missing} "
                    f"non_numeric={non_numeric}: {value[:500]}"
                )
            return parsed

        raw = resumable_call(
            record_key, stage, prompt, max_tokens=4096,
            thinking=args.infbench_thinking,
        )
        try:
            return checked_json(raw), raw
        except ValueError as first_error:
            # Keep the original call cache immutable for auditability. The
            # versioned repair stage gets its own prompt hash/cache entry, so a
            # resumed run never loops forever on the same invalid response.
            repair = json_repair_prompt(prompt, schema)
            repaired = resumable_call(
                record_key, stage + "_json_repair_v1", repair,
                max_tokens=1024, thinking=args.infbench_thinking,
            )
            try:
                return checked_json(repaired), repaired
            except ValueError as repair_error:
                raise ValueError(
                    f"{record_key} {stage}: primary and JSON repair failed; "
                    f"primary={first_error}; repair={repair_error}"
                ) from repair_error

    for i, record in enumerate(targets, 1):
        record_key = key(record)
        if record_key in existing:
            continue
        if record["source"].startswith("longmemeval_"):
            task = record.get("question_type") or ""
            qa_id = record.get("qa_pair_id") or ""
            prompt = longmem_prompt(
                task,
                record["raw_question"],
                record["gold_answer"],
                record["answer"],
                "_abs" in qa_id,
            )
            raw = resumable_call(
                record_key, "longmemeval", prompt,
                max_tokens=args.longmem_max_tokens,
                thinking=args.thinking,
            )
            judged = {
                "key": record_key,
                "source": record["source"],
                "qa_pair_id": qa_id,
                "metric": "llm_judge_correct",
                # Match the pinned official evaluator exactly. It treats any
                # judge response containing "yes" as a positive label rather
                # than requiring the response to begin with that word.
                "score": bool("yes" in raw.lower()),
                "judge_output": raw,
                "judge_model": args.model,
                "judge_thinking": args.thinking,
                "judge_max_tokens": args.longmem_max_tokens,
            }
        else:
            keypoints = record.get("keypoints") or []
            expert = record["gold_answer"]
            if isinstance(expert, list):
                expert = expert[0] if expert else ""
            summary = record["answer"].strip()
            fp = prompts["fluency_prompt_book"].format(text=summary)
            rp = prompts["recall_prompt_book"].format(
                keypoints="\n".join(
                    f"{j + 1}. {point}" for j, point in enumerate(keypoints)
                ),
                summary=summary,
            )
            pp = prompts["precision_prompt_book"].format(
                expert_summary=expert, summary=summary
            )
            f, raw_f = resumable_json_call(
                record_key, "infbench_fluency", fp,
                schema='{\"fluency\": 0_or_1}',
                required_keys=("fluency",),
            )
            r, raw_r = resumable_json_call(
                record_key, "infbench_recall", rp,
                schema='{\"recall\": integer}',
                required_keys=("recall",),
            )
            p, raw_p = resumable_json_call(
                record_key, "infbench_precision", pp,
                schema='{\"precision\": integer, \"sentence_count\": integer}',
                required_keys=("precision", "sentence_count"),
            )
            recall = r["recall"] / len(keypoints) if keypoints else 0.0
            precision = (
                p["precision"] / p["sentence_count"]
                if p.get("sentence_count", 0) > 0 else 0.0
            )
            f1 = (
                f["fluency"] * 2 * recall * precision / (recall + precision)
                if recall + precision > 0 else 0.0
            )
            judged = {
                "key": record_key,
                "source": record["source"],
                "qa_pair_id": record.get("qa_pair_id"),
                "metric": "gpt4_summary_f1",
                "score": f1,
                "fluency": f["fluency"],
                "recall": recall,
                "precision": precision,
                "judge_outputs": [raw_f, raw_r, raw_p],
                "judge_model": args.model,
                "judge_thinking": args.infbench_thinking,
                "note": "official GPT-4 rubric evaluated by configured DeepSeek model",
            }
        with output.open("a", encoding="utf-8") as out:
            out.write(json.dumps(judged, ensure_ascii=False) + "\n")
        existing[record_key] = judged
        print(f"[{i}/{len(targets)}] {record_key} score={judged['score']}",
              flush=True)

    by_source: dict[str, list[float]] = defaultdict(list)
    for row in existing.values():
        by_source[row["source"]].append(float(row["score"]))
    summary = {
        "judge_model": args.model,
        "longmem_judge_thinking": args.thinking,
        "infbench_judge_thinking": args.infbench_thinking,
        "judge_base_url": args.base_url,
        "completed": len(existing),
        "expected": len(targets),
        "by_source": {
            source: {
                "n": len(values),
                "mean": statistics.fmean(values),
            }
            for source, values in sorted(by_source.items())
        },
    }
    (args.results_dir / "special_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    local_path = args.results_dir / "official_local_summary.json"
    if local_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        combined_sources = local["by_source"]
        missing_special_sources = [
            source for source in summary["by_source"]
            if source not in combined_sources
        ]
        if missing_special_sources and args.allow_partial:
            print(
                "partial local summary does not yet contain special sources; "
                "skipping final_summary merge: " +
                ", ".join(missing_special_sources),
                flush=True,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        for source, values in summary["by_source"].items():
            if source not in combined_sources:
                raise RuntimeError(
                    f"special judge source absent from local summary: {source}"
                )
            combined_sources[source]["headline_score"] = values["mean"]
            combined_sources[source]["headline_judge_model"] = args.model
        category_rows: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for source, values in combined_sources.items():
            score = values.get("headline_score")
            if score is None:
                continue
            category_rows[category(source)].append(
                (float(score), int(values.get("questions", 0)))
            )
        categories = {}
        for name, values in sorted(category_rows.items()):
            denom = sum(n for _, n in values)
            categories[name] = {
                "sources": len(values),
                "questions": denom,
                "macro_source_mean": statistics.fmean(v for v, _ in values),
                "question_weighted_mean": (
                    sum(v * n for v, n in values) / denom if denom else None
                ),
            }
        final = {
            "questions": local["questions_scored"],
            "judge_model_for_special_metrics": args.model,
            "by_source": combined_sources,
            "by_category": categories,
            "overall_macro_category_mean": statistics.fmean(
                row["macro_source_mean"] for row in categories.values()
            ) if categories else None,
            "metric_contract": (
                "EventQA/Ruler/Fact=substring EM; Detective/ICL=exact EM; "
                "Recsys=Recall@5; LongMemEval=LLM judge; InfBench=summary "
                "rubric F1"
            ),
        }
        (args.results_dir / "final_summary.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
