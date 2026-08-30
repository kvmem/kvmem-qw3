#!/usr/bin/env python3
"""Probe task-routing behavior across prompt distributions on a local 27B model.

The upstream dsh-router-standard phase scan quantizes its numeric input into
three fixed prompts before sending requests.  This probe keeps that exact
three-template audit, then adds a genuinely distinct 21-point scalar prompt
family.  The primary outcome is the first tool action: explore for a repair
task and produce for a greenfield task.  Reasoning text is reduced to lexical
counts and is never written to the result file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_DEFAULT = "Qwen3.8-27B"
ENDPOINT_DEFAULT = "http://127.0.0.1:18314/v1"

MINIMAL = "You are a helpful software engineer assistant."
MIXED = (
    "You are a helpful software engineer assistant.\n"
    "Work directly: prefer writing or editing code over describing plans. "
    "Verify your changes by reading and running them."
)
REACT = (
    "You are a hands-on software engineer who delivers working output fast.\n"
    "Work directly: write or edit code, then verify it by reading and running. "
    "Keep the loop tight — produce, verify, fix — and do not build test "
    "harnesses, scaffolding, or ceremony the user did not ask for. "
    "Finish with a usable deliverable and a short summary."
)

ROUTER_V2 = (
    "You are a software engineer. Match your working style to the task type.\n"
    'Example 1: "fix the broken login flow" → inspect first, plan, then edit carefully.\n'
    'Example 2: "write a new CSV processing script" → write the code directly and verify it runs.\n'
    "Follow the same rule for the actual request."
)

WEAK_PERSONAS = {
    "w1-neutral": "You are a helpful assistant.",
    "w2-neutral-task": "You are a helpful assistant.\nComplete the user's request.",
    "w3-router-v2": ROUTER_V2,
    "w4-router-v2-strong": (
        ROUTER_V2
        + "\nThis rule is important: choose the style FIRST, then act consistently."
    ),
    "w5-spec+route": (
        MINIMAL
        + "\nAdapt your working style to the task type: build → direct production; "
        "fix → inspect and plan first."
    ),
    "w6-spec+fewshot": MINIMAL + "\n" + "\n".join(ROUTER_V2.splitlines()[1:]),
    "w7-neutral+explicit": (
        "You are a helpful assistant.\nBefore acting, decide the task type "
        "(build or fix) and adopt the matching style: build → hands-on "
        "production; fix → inspect-and-plan."
    ),
    "w8-neutral+role": (
        "You are a helpful assistant.\nFor build tasks act as a hands-on "
        "engineer who produces code fast; for fix tasks act as a careful "
        "maintainer who inspects first."
    ),
    "w9-neutral+lean-react": (
        "You are a helpful assistant.\nPrefer direct action: write or edit "
        "code, then verify by running. Keep planning brief."
    ),
}

TASKS = {
    "micro": (
        "Inspect the current repository before answering.\n"
        "First determine its top-level structure, then locate and read the "
        "project README.\nDo not guess from prior knowledge. Use the available "
        "tools first."
    ),
    "fix": (
        "The login endpoint in this repository returns 500 on valid "
        "credentials. Fix the regression."
    ),
    "build": (
        "Create a standalone JavaScript module solve.js exporting gcd(a,b) "
        "and csvSum(path)."
    ),
    "fix-login": (
        "The login endpoint in this repository returns 500 on valid "
        "credentials. Fix the regression."
    ),
    "fix-csv": (
        "The CSV parser crashes on rows with quoted commas. Fix the bug in "
        "parser.js."
    ),
    "fix-worker": (
        "The worker loop leaks memory and the heap grows on every tick. Fix "
        "worker.js."
    ),
    "fix-config": (
        "The config loader ignores the last line of its input. Fix config.js."
    ),
    "build-gcd": (
        "Create a standalone JavaScript module gcd.js exporting gcd(a,b) "
        "without libraries."
    ),
    "build-cart": (
        "Create a standalone shopping-cart module cart.js with addItem, "
        "removeItem, and total."
    ),
    "build-links": (
        "Create a standalone markdown-link extractor links.js exporting "
        "extractLinks(text)."
    ),
    "build-json": (
        "Create a standalone JSON formatter fmt.js exporting formatJson(text)."
    ),
}

SUITE_TASKS = (
    "fix-login",
    "fix-csv",
    "fix-worker",
    "fix-config",
    "build-gcd",
    "build-cart",
    "build-links",
    "build-json",
)

EXPLORE_TOOLS = {"bash", "read", "glob", "grep"}
PRODUCE_TOOLS = {"write", "edit"}


def function_tool(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    function_tool(
        "bash",
        "Run a command in a persistent shell.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    function_tool(
        "read",
        "Read a text file.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    function_tool(
        "edit",
        "Edit a text file by replacing a literal string.",
        {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        ["file_path", "old_string", "new_string"],
    ),
    function_tool(
        "write",
        "Write or fully replace a UTF-8 text file.",
        {"file_path": {"type": "string"}, "content": {"type": "string"}},
        ["file_path", "content"],
    ),
    function_tool(
        "glob",
        "Find files by path pattern.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    function_tool(
        "grep",
        "Search file contents with a regular expression.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
]


def scalar_persona(mode: float, wording: str = "forward") -> str:
    """Return one distinct prompt per mode without threshold quantization."""
    value = f"{mode:.2f}"
    if wording == "reverse":
        definition = (
            "1.00 means produce or edit immediately and then verify; 0.00 means "
            "inspect the repository and reason before editing."
        )
    else:
        definition = (
            "0.00 means inspect the repository and reason before editing; 1.00 "
            "means produce or edit immediately and then verify."
        )
    return (
        f"{MINIMAL}\n"
        f"Working-style coordinate ACTION_BIAS = {value} on a continuous scale.\n"
        f"{definition} Intermediate values proportionally balance these "
        "tendencies. Follow this exact coordinate while still respecting the "
        "task's explicit requirements."
    )


def first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.strip())
    return match.group(0) if match else "(empty)"


def count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text, flags=re.I))


def lexical_summary(reasoning: str) -> dict[str, Any]:
    first_line = reasoning.strip().splitlines()[0] if reasoning.strip() else ""
    we = count_phrase(reasoning, "we")
    let_me = len(re.findall(r"\blet me\b", reasoning, flags=re.I))
    lets = len(re.findall(r"\blet['’]s\b", reasoning, flags=re.I))
    score = 0
    if re.match(r"^we need\b", first_line, flags=re.I):
        score += 3
    if re.match(r"^let me\b", first_line, flags=re.I):
        score -= 3
    if we and not let_me:
        score += 2
    if let_me:
        score -= 2
    label = "minimal-like" if score >= 4 else "standard-like" if score <= -4 else "ambiguous"
    return {
        "chars": len(reasoning),
        "first_word": first_word(reasoning),
        "we": we,
        "let_me": let_me,
        "lets": lets,
        "legacy_label": label,
    }


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_once(
    *,
    endpoint: str,
    model: str,
    system: str,
    task_name: str,
    seed: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": TASKS[task_name]},
        ],
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
            "preserve_thinking": True,
        },
    }
    started = time.monotonic()
    error: Exception | None = None
    for attempt in range(3):
        try:
            payload = post_json(
                f"{endpoint.rstrip('/')}/chat/completions", body, timeout
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    else:  # pragma: no cover - loop either breaks or raises
        raise error or RuntimeError("request failed")

    choice = payload["choices"][0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    tool_names = [
        call.get("function", {}).get("name", "")
        for call in (message.get("tool_calls") or [])
    ]
    first_tool = tool_names[0] if tool_names else "(none)"
    if first_tool in EXPLORE_TOOLS:
        first_action = "explore"
    elif first_tool in PRODUCE_TOOLS:
        first_action = "produce"
    else:
        first_action = "other"

    expected = "produce" if task_name.startswith("build") else "explore"
    usage = payload.get("usage") or {}
    return {
        "seed": seed,
        "task": task_name,
        "first_tool": first_tool,
        "first_action": first_action,
        "route_correct": first_action == expected,
        "tool_names": tool_names,
        "finish_reason": choice.get("finish_reason"),
        "visible_before_tool": bool((message.get("content") or "").strip()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(time.monotonic() - started, 4),
        "lexical": lexical_summary(reasoning),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "route_rate": round(sum(row["route_correct"] for row in rows) / n, 4),
        "first_actions": dict(Counter(row["first_action"] for row in rows)),
        "first_tools": dict(Counter(row["first_tool"] for row in rows)),
        "first_words": dict(Counter(row["lexical"]["first_word"] for row in rows)),
        "legacy_labels": dict(
            Counter(row["lexical"]["legacy_label"] for row in rows)
        ),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "avg_reasoning_chars": round(mean([row["lexical"]["chars"] for row in rows]), 1),
        "avg_completion_tokens": round(
            mean([row["completion_tokens"] for row in rows if row["completion_tokens"] is not None]),
            1,
        ),
        "avg_latency_s": round(mean([row["latency_s"] for row in rows]), 3),
    }


def run_batch(
    jobs: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(request_once, **job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: row["seed"])


def original_audit(args: argparse.Namespace) -> dict[str, Any]:
    templates = {"minimal": MINIMAL, "mixed": MIXED, "react": REACT}
    conditions: list[dict[str, Any]] = []
    for template_idx, (label, system) in enumerate(templates.items()):
        for task_idx, task in enumerate(("micro", "fix", "build")):
            jobs = [
                {
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "system": system,
                    "task_name": task,
                    "seed": args.seed + 10000 * template_idx + 1000 * task_idx + i,
                    "max_tokens": args.max_tokens,
                    "timeout": args.timeout,
                }
                for i in range(args.n)
            ]
            rows = run_batch(jobs, args.workers)
            summary = summarize_rows(rows)
            print(
                f"original {label:7s} {task:5s} "
                f"route={summary['route_rate']:.2f} actions={summary['first_actions']}",
                flush=True,
            )
            conditions.append(
                {"template": label, "task": task, "summary": summary, "rows": rows}
            )
    return {"name": "original_three_template_audit", "conditions": conditions}


def scalar_scan(args: argparse.Namespace) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    modes = args.modes if args.modes is not None else [point_idx / 20 for point_idx in range(21)]
    for point_idx, mode in enumerate(modes):
        by_task: dict[str, Any] = {}
        for task_idx, task in enumerate(("fix", "build")):
            jobs = [
                {
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "system": scalar_persona(mode, args.wording),
                    "task_name": task,
                    "seed": args.seed + 100000 + 10000 * point_idx + 1000 * task_idx + i,
                    "max_tokens": args.max_tokens,
                    "timeout": args.timeout,
                }
                for i in range(args.n)
            ]
            rows = run_batch(jobs, args.workers)
            by_task[task] = {"summary": summarize_rows(rows), "rows": rows}
        fix_rate = by_task["fix"]["summary"]["route_rate"]
        build_rate = by_task["build"]["summary"]["route_rate"]
        route_rate = round((fix_rate + build_rate) / 2, 4)
        worst_task_rate = round(min(fix_rate, build_rate), 4)
        point = {
            "mode": round(mode, 2),
            "route_rate": route_rate,
            "worst_task_rate": worst_task_rate,
            "advantage": worst_task_rate >= args.advantage_threshold,
            "tasks": by_task,
        }
        points.append(point)
        print(
            f"scalar {mode:0.2f} fix={fix_rate:.2f} build={build_rate:.2f} "
            f"joint={route_rate:.2f} worst={worst_task_rate:.2f}",
            flush=True,
        )
    return {
        "name": "unquantized_scalar_scan",
        "wording": args.wording,
        "threshold": args.advantage_threshold,
        "points": points,
        "advantage_points": [point["mode"] for point in points if point["advantage"]],
    }


def natural_persona_scan(args: argparse.Namespace) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    selected = args.personas or list(WEAK_PERSONAS)
    unknown = [label for label in selected if label not in WEAK_PERSONAS]
    if unknown:
        raise SystemExit(f"unknown --personas values: {', '.join(unknown)}")
    for persona_idx, label in enumerate(selected):
        system = WEAK_PERSONAS[label]
        by_task: dict[str, Any] = {}
        for task_idx, task in enumerate(("fix", "build")):
            jobs = [
                {
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "system": system,
                    "task_name": task,
                    "seed": args.seed + 500000 + 10000 * persona_idx + 1000 * task_idx + i,
                    "max_tokens": args.max_tokens,
                    "timeout": args.timeout,
                }
                for i in range(args.n)
            ]
            rows = run_batch(jobs, args.workers)
            by_task[task] = {"summary": summarize_rows(rows), "rows": rows}
        fix_rate = by_task["fix"]["summary"]["route_rate"]
        build_rate = by_task["build"]["summary"]["route_rate"]
        joint = round((fix_rate + build_rate) / 2, 4)
        worst = round(min(fix_rate, build_rate), 4)
        condition = {
            "persona": label,
            "route_rate": joint,
            "worst_task_rate": worst,
            "advantage": worst >= args.advantage_threshold,
            "tasks": by_task,
        }
        conditions.append(condition)
        print(
            f"natural {label:22s} fix={fix_rate:.2f} build={build_rate:.2f} "
            f"joint={joint:.2f} worst={worst:.2f}",
            flush=True,
        )
    return {
        "name": "natural_persona_scan",
        "threshold": args.advantage_threshold,
        "conditions": conditions,
        "advantage_personas": [
            condition["persona"] for condition in conditions if condition["advantage"]
        ],
    }


def multitask_persona_scan(args: argparse.Namespace) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    selected = args.personas or list(WEAK_PERSONAS)
    unknown = [label for label in selected if label not in WEAK_PERSONAS]
    if unknown:
        raise SystemExit(f"unknown --personas values: {', '.join(unknown)}")
    for label in selected:
        jobs: list[dict[str, Any]] = []
        for task_idx, task in enumerate(SUITE_TASKS):
            jobs.extend(
                [
                {
                    "endpoint": args.endpoint,
                    "model": args.model,
                    "system": WEAK_PERSONAS[label],
                    "task_name": task,
                    # Paired seeds: every persona sees the same seed per task.
                    "seed": args.seed + 900000 + 1000 * task_idx + i,
                    "max_tokens": args.max_tokens,
                    "timeout": args.timeout,
                }
                for i in range(args.n)
                ]
            )
        rows = run_batch(jobs, args.workers)
        fix_rows = [row for row in rows if row["task"].startswith("fix")]
        build_rows = [row for row in rows if row["task"].startswith("build")]
        fix_summary = summarize_rows(fix_rows)
        build_summary = summarize_rows(build_rows)
        fix_rate = fix_summary["route_rate"]
        build_rate = build_summary["route_rate"]
        joint = round((fix_rate + build_rate) / 2, 4)
        worst = round(min(fix_rate, build_rate), 4)
        condition = {
            "persona": label,
            "route_rate": joint,
            "worst_task_rate": worst,
            "advantage": worst >= args.advantage_threshold,
            "fix": fix_summary,
            "build": build_summary,
            "rows": rows,
        }
        conditions.append(condition)
        print(
            f"suite {label:22s} fix={fix_rate:.2f} build={build_rate:.2f} "
            f"joint={joint:.2f} worst={worst:.2f}",
            flush=True,
        )
    return {
        "name": "multitask_persona_scan",
        "tasks": list(SUITE_TASKS),
        "threshold": args.advantage_threshold,
        "conditions": conditions,
        "advantage_personas": [
            condition["persona"] for condition in conditions if condition["advantage"]
        ],
    }


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# 27B router interval probe",
        "",
        f"- Model: `{result['model']}`",
        f"- Endpoint: `{result['endpoint']}`",
        f"- Samples per task/condition: {result['n']}",
        f"- Sampling: temperature 1.0, top-p 0.95, top-k 20",
        "- Primary metric: first tool action (fix → explore; build → produce)",
        "",
    ]
    for experiment in result["experiments"]:
        if experiment["name"] == "original_three_template_audit":
            lines.extend(["## Original three-template audit", ""])
            lines.append("| Template | Task | Route rate | First actions | Legacy labels |")
            lines.append("|---|---:|---:|---|---|")
            for condition in experiment["conditions"]:
                summary = condition["summary"]
                lines.append(
                    f"| {condition['template']} | {condition['task']} | "
                    f"{summary['route_rate']:.0%} | `{summary['first_actions']}` | "
                    f"`{summary['legacy_labels']}` |"
                )
            lines.append("")
        elif experiment["name"] == "unquantized_scalar_scan":
            lines.extend(["## Unquantized 21-point scalar scan", ""])
            lines.append("| ACTION_BIAS | Fix inspect | Build produce | Joint | Worst task | Advantage |")
            lines.append("|---:|---:|---:|---:|---:|:---:|")
            for point in experiment["points"]:
                fix_rate = point["tasks"]["fix"]["summary"]["route_rate"]
                build_rate = point["tasks"]["build"]["summary"]["route_rate"]
                lines.append(
                    f"| {point['mode']:.2f} | {fix_rate:.0%} | {build_rate:.0%} | "
                    f"{point['route_rate']:.0%} | {point['worst_task_rate']:.0%} | "
                    f"{'yes' if point['advantage'] else 'no'} |"
                )
            lines.extend(
                [
                    "",
                    f"Observed advantage points (both tasks >= {experiment['threshold']:.0%}): "
                    + ", ".join(f"{value:.2f}" for value in experiment["advantage_points"]),
                    "",
                ]
            )
        elif experiment["name"] == "natural_persona_scan":
            lines.extend(["## Natural-language persona scan", ""])
            lines.append("| Persona | Fix inspect | Build produce | Joint | Worst task | Advantage |")
            lines.append("|---|---:|---:|---:|---:|:---:|")
            for condition in experiment["conditions"]:
                fix_rate = condition["tasks"]["fix"]["summary"]["route_rate"]
                build_rate = condition["tasks"]["build"]["summary"]["route_rate"]
                lines.append(
                    f"| {condition['persona']} | {fix_rate:.0%} | {build_rate:.0%} | "
                    f"{condition['route_rate']:.0%} | {condition['worst_task_rate']:.0%} | "
                    f"{'yes' if condition['advantage'] else 'no'} |"
                )
            lines.extend(
                [
                    "",
                    f"Observed advantage personas (both tasks >= {experiment['threshold']:.0%}): "
                    + ", ".join(experiment["advantage_personas"]),
                    "",
                ]
            )
        elif experiment["name"] == "multitask_persona_scan":
            lines.extend(["## Eight-task persona scan", ""])
            lines.append("| Persona | Fix inspect | Build produce | Joint | Worst task | Advantage |")
            lines.append("|---|---:|---:|---:|---:|:---:|")
            for condition in experiment["conditions"]:
                lines.append(
                    f"| {condition['persona']} | {condition['fix']['route_rate']:.0%} | "
                    f"{condition['build']['route_rate']:.0%} | {condition['route_rate']:.0%} | "
                    f"{condition['worst_task_rate']:.0%} | "
                    f"{'yes' if condition['advantage'] else 'no'} |"
                )
            lines.extend(
                [
                    "",
                    f"Observed advantage personas (both task families >= {experiment['threshold']:.0%}): "
                    + ", ".join(experiment["advantage_personas"]),
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=ENDPOINT_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1792)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--wording", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--advantage-threshold", type=float, default=0.8)
    parser.add_argument(
        "--modes",
        type=lambda value: [float(item) for item in value.split(",")],
        help="comma-separated scalar points; default is 0.00..1.00 by 0.05",
    )
    parser.add_argument(
        "--personas",
        type=lambda value: [item.strip() for item in value.split(",")],
        help="comma-separated weak persona labels; default is all nine",
    )
    parser.add_argument(
        "--experiment",
        choices=("original", "scalar", "natural", "suite", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if not 0 <= args.advantage_threshold <= 1:
        raise SystemExit("--advantage-threshold must be in [0, 1]")
    if args.modes is not None and any(mode < 0 or mode > 1 for mode in args.modes):
        raise SystemExit("--modes values must be in [0, 1]")
    started = time.time()
    experiments = []
    if args.experiment in ("original", "all"):
        experiments.append(original_audit(args))
    if args.experiment in ("scalar", "all"):
        experiments.append(scalar_scan(args))
    if args.experiment in ("natural", "all"):
        experiments.append(natural_persona_scan(args))
    if args.experiment in ("suite", "all"):
        experiments.append(multitask_persona_scan(args))
    result = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": args.endpoint,
        "n": args.n,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "elapsed_s": round(time.time() - started, 3),
        "experiments": experiments,
    }
    output = args.output or Path("results") / (
        "router_interval_qwen38_27b_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown_summary(result))
    print(f"saved {output}")
    print(f"saved {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
