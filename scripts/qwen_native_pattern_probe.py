#!/usr/bin/env python3
"""Probe Qwen-native behavioral gates without encoding a desired phase boundary.

The probe holds the task, system prompt, sampling parameters, and paired seeds
fixed while changing only the callable tool surface.  It records aggregate
reasoning size, tool use, termination, and latency; full hidden reasoning is
never persisted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ENDPOINT_DEFAULT = "http://127.0.0.1:18314/v1"
MODEL_DEFAULT = "Qwen3.8-27B"
SYSTEM = "You are a precise software engineering assistant."

TASKS = {
    "matched_read": (
        "Use the available file tool to read /workspace/input.txt and report its "
        "first line. Do not guess the file contents."
    ),
    "matched_write": (
        "Use the available file tool to create /workspace/output.txt containing "
        "exactly HELLO followed by one newline."
    ),
    "repo_fix": (
        "The login endpoint in this repository returns HTTP 500 for valid "
        "credentials. Find the regression and fix it."
    ),
    "repo_locate": (
        "Inspect this repository and report the exact file and line that define "
        "the package version. Base the answer on repository evidence."
    ),
    "greenfield": (
        "Create a standalone JavaScript module gcd.js exporting gcd(a, b), and "
        "verify it on gcd(48, 18)."
    ),
    "reasoning": (
        "Explain how to implement an LRU cache with O(1) get and put, including "
        "the invariant that proves the complexity. Do not read or modify files."
    ),
}

# Keep paired seeds stable when new tasks are added or task display order changes.
TASK_SEED_OFFSETS = {
    "repo_fix": 0,
    "repo_locate": 1_000,
    "greenfield": 2_000,
    "reasoning": 3_000,
    "matched_read": 10_000,
    "matched_write": 11_000,
}


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
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


DUMMY_TOOL = function_tool(
    "get_weather",
    "Return the weather for a city. This does not access files or source code.",
    {"city": {"type": "string"}},
    ["city"],
)

QWEN_TOOLS = [
    function_tool(
        "read_file",
        "Read a UTF-8 text file from the local filesystem.",
        {"file_path": {"type": "string"}},
        ["file_path"],
    ),
    function_tool(
        "grep_search",
        "Search file contents with a regular expression.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}},
        ["pattern"],
    ),
    function_tool(
        "glob",
        "Find files whose paths match a glob pattern.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    function_tool(
        "edit",
        "Edit a text file by replacing one literal string.",
        {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        ["file_path", "old_string", "new_string"],
    ),
    function_tool(
        "write_file",
        "Write complete UTF-8 content to a file.",
        {"file_path": {"type": "string"}, "content": {"type": "string"}},
        ["file_path", "content"],
    ),
    function_tool(
        "run_shell_command",
        "Run a shell command in the project directory.",
        {"command": {"type": "string"}},
        ["command"],
    ),
]

GENERIC_TOOLS = [
    function_tool(
        "read",
        "Read a text file.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    function_tool(
        "grep",
        "Search file contents with a regular expression.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    function_tool(
        "glob",
        "Find files by path pattern.",
        {"pattern": {"type": "string"}},
        ["pattern"],
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
        "Write or replace a text file.",
        {"file_path": {"type": "string"}, "content": {"type": "string"}},
        ["file_path", "content"],
    ),
    function_tool(
        "bash",
        "Run a command in a shell.",
        {"command": {"type": "string"}},
        ["command"],
    ),
]

SURFACES: dict[str, list[dict[str, Any]] | None] = {
    "none": None,
    "dummy1": [DUMMY_TOOL],
    "qwen_read1": [QWEN_TOOLS[0]],
    "qwen_write1": [QWEN_TOOLS[4]],
    "qwen6": QWEN_TOOLS,
    "generic6": GENERIC_TOOLS,
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
    surface: str,
    task: str,
    replicate: int,
    seed: int,
    max_tokens: int,
    timeout: float,
    effort: str,
) -> dict[str, Any]:
    tools = SURFACES[surface]
    thinking_enabled = effort != "off"
    kwargs: dict[str, Any] = {
        "enable_thinking": thinking_enabled,
        "preserve_thinking": thinking_enabled,
    }
    if thinking_enabled:
        kwargs["reasoning_effort"] = effort
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASKS[task]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "chat_template_kwargs": kwargs,
    }
    if tools is not None:
        body["tools"] = tools

    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            payload = post_json(
                f"{endpoint.rstrip('/')}/chat/completions", body, timeout
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    else:  # pragma: no cover
        raise last_error or RuntimeError("request failed")

    choice = payload["choices"][0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_names = [
        call.get("function", {}).get("name", "")
        for call in (message.get("tool_calls") or [])
    ]
    usage = payload.get("usage") or {}
    return {
        "surface": surface,
        "task": task,
        "effort": effort,
        "replicate": replicate,
        "seed": seed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "tool_names": tool_names,
        "tool_call_count": len(tool_names),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": round(time.monotonic() - started, 4),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completions = [float(row["completion_tokens"]) for row in rows]
    reasoning_chars = [float(row["reasoning_chars"]) for row in rows]
    latencies = [float(row["latency_s"]) for row in rows]
    return {
        "n": len(rows),
        "prompt_tokens_median": statistics.median(
            row["prompt_tokens"] for row in rows
        ),
        "completion_tokens_mean": round(mean(completions), 1),
        "completion_tokens_median": round(statistics.median(completions), 1),
        "completion_tokens_iqr": [
            round(quantile(completions, 0.25), 1),
            round(quantile(completions, 0.75), 1),
        ],
        "reasoning_chars_mean": round(mean(reasoning_chars), 1),
        "reasoning_chars_median": round(statistics.median(reasoning_chars), 1),
        "tool_call_rate": round(
            sum(bool(row["tool_call_count"]) for row in rows) / len(rows), 4
        ),
        "tool_names": dict(
            Counter(name for row in rows for name in row["tool_names"])
        ),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "latency_mean_s": round(mean(latencies), 3),
    }


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(request_once, **job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['surface']:10s} {row['task']:11s} {row['effort']:6s} "
                f"rep={row['replicate']} ctok={row['completion_tokens']} "
                f"tools={row['tool_names']} finish={row['finish_reason']}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            row["effort"], row["surface"], row["task"], row["replicate"]
        ),
    )


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    surfaces = args.surfaces or list(SURFACES)
    tasks = args.tasks or list(TASKS)
    unknown_surfaces = [item for item in surfaces if item not in SURFACES]
    unknown_tasks = [item for item in tasks if item not in TASKS]
    if unknown_surfaces:
        raise SystemExit(f"unknown surfaces: {', '.join(unknown_surfaces)}")
    if unknown_tasks:
        raise SystemExit(f"unknown tasks: {', '.join(unknown_tasks)}")

    for surface in surfaces:
        for task in tasks:
            for replicate in range(args.n):
                # Paired seed across surfaces for each task/replicate.
                seed = args.seed + TASK_SEED_OFFSETS[task] + replicate
                jobs.append(
                    {
                        "endpoint": args.endpoint,
                        "model": args.model,
                        "surface": surface,
                        "task": task,
                        "replicate": replicate,
                        "seed": seed,
                        "max_tokens": args.max_tokens,
                        "timeout": args.timeout,
                        "effort": args.effort,
                    }
                )
    return jobs


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen-native tool-surface probe",
        "",
        f"- Model: `{result['model']}`",
        f"- Effort: `{result['effort']}`",
        f"- Samples per task/surface: {result['n']}",
        f"- Max completion tokens: {result['max_tokens']}",
        "- Paired seeds across tool surfaces",
        "- Full hidden reasoning was not persisted",
        "",
        "| Surface | Task | Prompt tok | Completion tok mean | Completion tok median [IQR] | Tool call | Finish reasons | Latency mean |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for condition in result["conditions"]:
        summary = condition["summary"]
        lines.append(
            f"| {condition['surface']} | {condition['task']} | "
            f"{summary['prompt_tokens_median']} | "
            f"{summary['completion_tokens_mean']} | "
            f"{summary['completion_tokens_median']} "
            f"[{summary['completion_tokens_iqr'][0]}, {summary['completion_tokens_iqr'][1]}] | "
            f"{summary['tool_call_rate']:.0%} | "
            f"`{summary['finish_reasons']}` | {summary['latency_mean_s']:.2f}s |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=ENDPOINT_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=380000)
    parser.add_argument(
        "--effort", choices=("off", "low", "medium", "xhigh"), default="xhigh"
    )
    parser.add_argument(
        "--surfaces",
        type=lambda value: [item.strip() for item in value.split(",")],
    )
    parser.add_argument(
        "--tasks",
        type=lambda value: [item.strip() for item in value.split(",")],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    started = time.time()
    rows = run_jobs(build_jobs(args), args.workers)
    conditions: list[dict[str, Any]] = []
    for surface in args.surfaces or list(SURFACES):
        for task in args.tasks or list(TASKS):
            selected = [
                row for row in rows
                if row["surface"] == surface and row["task"] == task
            ]
            conditions.append(
                {"surface": surface, "task": task, "summary": summarize(selected)}
            )
    result = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": args.endpoint,
        "effort": args.effort,
        "base_seed": args.seed,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "elapsed_s": round(time.time() - started, 3),
        "conditions": conditions,
        "rows": rows,
    }
    output = args.output or Path("results") / (
        "qwen_native_tool_surface_"
        + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown(result))
    print(f"saved {output}")
    print(f"saved {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
