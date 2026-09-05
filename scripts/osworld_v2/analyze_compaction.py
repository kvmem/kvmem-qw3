#!/usr/bin/env python3
"""Merge OSWorld agent/action traces with QW3 server timing and compare compaction."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


MM_RE = re.compile(
    r"multimodal prompt images=(?P<images>\d+) visual_tokens=(?P<visual_tokens>\d+) "
    r"vision_cache_hit=(?P<vision_cache_hit>\d+) mandatory_spans=(?P<mandatory_spans>\d+)"
)
NATIVE_RE = re.compile(
    r"native generate: prompt_tokens=(?P<prompt_tokens>\d+) "
    r"prefill=(?P<prefill_s>[0-9.]+)s .*? decoded=(?P<decoded_tokens>\d+) "
    r"decode=(?P<decode_s>[0-9.]+)s"
)
CHAT_RE = re.compile(
    r"\[qw3-serve\] #(?P<rid>\d+) chat chars=(?P<chars>\d+) "
    r"completion_tokens=(?P<completion_tokens>\d+) prompt_tokens=(?P<prompt_tokens>\d+) "
    r"route=(?P<route>\S+)(?: fallback_reason=(?P<fallback_reason>\S+))? "
    r"(?P<engine_summary_ms>[0-9.]+)ms"
)
TTFT_RE = re.compile(r"\[qw3-server-ttft\]\s+(?P<fields>.*)")
ACCOUNT_RE = re.compile(r"\[qw3-server-accounting\]\s+(?P<fields>.*)")
FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\"(?:[^\"\\]|\\.)*\"|\S+)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_num}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def scalar(value: str) -> Any:
    if value.startswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"')
    if value in {"true", "false"}:
        return value == "true"
    try:
        return float(value) if any(char in value for char in ".eE") else int(value)
    except ValueError:
        return value


def parse_fields(text: str) -> dict[str, Any]:
    return {match.group("key"): scalar(match.group("value")) for match in FIELD_RE.finditer(text)}


def parse_server_log(path: Path) -> list[dict[str, Any]]:
    requests: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    pending_multimodal: dict[str, Any] | None = None
    pending_native: dict[str, Any] | None = None
    if not path.exists():
        return []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := MM_RE.search(line):
            pending_multimodal = {key: int(value) for key, value in match.groupdict().items()}
            continue
        if match := NATIVE_RE.search(line):
            pending_native = {
                "native_prompt_tokens": int(match.group("prompt_tokens")),
                "native_prefill_ms": float(match.group("prefill_s")) * 1000.0,
                "native_decoded_tokens": int(match.group("decoded_tokens")),
                "native_decode_ms": float(match.group("decode_s")) * 1000.0,
            }
            continue
        if match := CHAT_RE.search(line):
            rid = int(match.group("rid"))
            record = requests.setdefault(rid, {"rid": rid})
            if rid not in order:
                order.append(rid)
            record.update(
                {
                    "chars": int(match.group("chars")),
                    "completion_tokens": int(match.group("completion_tokens")),
                    "prompt_tokens": int(match.group("prompt_tokens")),
                    "route": match.group("route"),
                    "fallback_reason": match.group("fallback_reason"),
                    "engine_summary_ms": float(match.group("engine_summary_ms")),
                }
            )
            if pending_multimodal:
                record.update(pending_multimodal)
            if pending_native:
                record.update(pending_native)
            pending_multimodal = None
            pending_native = None
            continue
        if match := TTFT_RE.search(line):
            fields = parse_fields(match.group("fields"))
            rid = int(fields.pop("rid"))
            requests.setdefault(rid, {"rid": rid}).update(
                {f"ttft_{key}": value for key, value in fields.items()}
            )
            continue
        if match := ACCOUNT_RE.search(line):
            fields = parse_fields(match.group("fields"))
            rid = int(fields.pop("rid"))
            requests.setdefault(rid, {"rid": rid}).update(
                {f"account_{key}": value for key, value in fields.items()}
            )

    return [requests[rid] for rid in order]


def nested(mapping: Any, *keys: str, default: Any = None) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_task_trace(run_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = sorted(run_dir.glob("osworld/**/step_timing.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"expected one step_timing.jsonl under {run_dir}, found {len(candidates)}")
    return candidates[0].parent, read_jsonl(candidates[0])


def match_server_request(
    agent_trace: dict[str, Any],
    server_requests: list[dict[str, Any]],
    used: set[int],
) -> dict[str, Any]:
    prompt_tokens = nested(agent_trace, "api", "usage", "prompt_tokens")
    completion_tokens = nested(agent_trace, "api", "usage", "completion_tokens")
    expected_total_ms = numeric(
        nested(agent_trace, "api", "server_timing", "request_total_sec")
    )
    if expected_total_ms is not None:
        expected_total_ms *= 1000.0

    candidates = []
    for index, request in enumerate(server_requests):
        if index in used:
            continue
        if prompt_tokens is not None and request.get("prompt_tokens") != prompt_tokens:
            continue
        if completion_tokens is not None and request.get("completion_tokens") != completion_tokens:
            continue
        total_ms = numeric(request.get("account_total_ms"))
        distance = abs(total_ms - expected_total_ms) if total_ms is not None and expected_total_ms is not None else index
        candidates.append((distance, index, request))
    if not candidates:
        return {}
    _, index, request = min(candidates, key=lambda item: (item[0], item[1]))
    used.add(index)
    return request


def load_condition(condition: str, run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_dir, timings = find_task_trace(run_dir)
    server_requests = parse_server_log(run_dir / "qw3_server.log")
    used_requests: set[int] = set()
    rows = []
    for timing in timings:
        agent = timing.get("agent_trace") or {}
        compact = agent.get("compaction") or {}
        api_timing = nested(agent, "api", "server_timing", default={}) or {}
        usage = nested(agent, "api", "usage", default={}) or {}
        request = match_server_request(agent, server_requests, used_requests)
        actions = timing.get("actions") or []
        env_timings = [item.get("environment_timing_ms") or {} for item in actions]
        row = {
            "condition": condition,
            "task_id": task_dir.name,
            "step_num": timing.get("step_num"),
            "status": timing.get("status"),
            "compaction_triggered": bool(compact.get("triggered", False)),
            "newly_compacted": compact.get("newly_compacted", 0),
            "folded_prefix_after": compact.get("folded_prefix_after", 0),
            "history_start_step": compact.get("history_start_step"),
            "history_truncated_steps": compact.get("history_truncated_steps", 0),
            "active_image_count": compact.get("active_image_count"),
            "collapsed_steps": compact.get("collapsed_steps", []),
            "image_steps": compact.get("image_steps", []),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "predict_wall_ms": numeric(timing.get("predict_wall_time_ms")),
            "agent_predict_total_ms": numeric(nested(agent, "timing_ms", "predict_total")),
            "client_api_ms": numeric(nested(agent, "timing_ms", "api_call")),
            "client_image_prepare_ms": numeric(nested(agent, "timing_ms", "image_prepare")),
            "client_trace_dump_ms": sum(
                numeric(nested(agent, "timing_ms", key)) or 0.0
                for key in ("input_dump", "message_dump")
            ),
            "server_total_ms": (
                numeric(api_timing.get("request_total_sec")) * 1000.0
                if numeric(api_timing.get("request_total_sec")) is not None
                else numeric(request.get("account_total_ms"))
            ),
            "server_ttft_ms": (
                numeric(api_timing.get("server_ttft_sec")) * 1000.0
                if numeric(api_timing.get("server_ttft_sec")) is not None
                else numeric(request.get("ttft_server_ttft_ms"))
            ),
            "engine_ttft_ms": (
                numeric(api_timing.get("engine_ttft_sec")) * 1000.0
                if numeric(api_timing.get("engine_ttft_sec")) is not None
                else numeric(request.get("ttft_engine_ttft_ms"))
            ),
            "server_preprocess_ms": numeric(request.get("account_preprocess_ms")),
            "server_queue_ms": numeric(request.get("account_queue_ms")),
            "server_engine_ms": numeric(request.get("account_engine_ms")),
            "server_response_ms": numeric(request.get("account_response_ms")),
            "server_json_ms": numeric(request.get("account_pre_json_ms")),
            "server_validate_ms": numeric(request.get("account_pre_validate_ms")),
            "server_render_ms": numeric(request.get("account_pre_render_ms")),
            "server_tokenize_ms": numeric(request.get("account_pre_tokenize_ms")),
            "server_span_setup_ms": numeric(request.get("account_pre_span_setup_ms")),
            "native_prefill_ms": numeric(request.get("native_prefill_ms")),
            "native_decode_ms": numeric(request.get("native_decode_ms")),
            "server_rid": request.get("rid"),
            "server_route": request.get("route"),
            "server_kvmem_fallback_reason": request.get("fallback_reason"),
            "action_count": len(actions),
            "action_wall_ms": numeric(timing.get("action_wall_time_ms")),
            "action_dispatch_ms": sum(numeric(item.get("action_dispatch_ms")) or 0.0 for item in env_timings),
            "observation_ms": sum(numeric(item.get("observation_ms")) or 0.0 for item in env_timings),
            "turn_wall_ms": numeric(timing.get("turn_wall_time_ms")),
            "input_file": nested(agent, "request", "input_file"),
            "message_file": nested(agent, "request", "message_file"),
            "response_text": nested(agent, "response", "text", default=""),
            "actions": [item.get("action") for item in actions],
        }
        rows.append(row)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return rows, manifest


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [numeric(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [numeric(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_rows = [row for row in rows if row["compaction_triggered"]]
    return {
        "step_count": len(rows),
        "compaction_event_steps": [row["step_num"] for row in event_rows],
        "median_server_total_ms": median(rows, "server_total_ms"),
        "mean_server_total_ms": mean(rows, "server_total_ms"),
        "median_server_preprocess_ms": median(rows, "server_preprocess_ms"),
        "median_server_engine_ms": median(rows, "server_engine_ms"),
        "median_action_wall_ms": median(rows, "action_wall_ms"),
        "median_turn_wall_ms": median(rows, "turn_wall_ms"),
        "max_active_image_count": max((row.get("active_image_count") or 0 for row in rows), default=0),
        "max_prompt_tokens": max((row.get("prompt_tokens") or 0 for row in rows), default=0),
    }


def compare(compact_rows: list[dict[str, Any]], no_compact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left = {int(row["step_num"]): row for row in compact_rows}
    right = {int(row["step_num"]): row for row in no_compact_rows}
    comparisons = []
    for step in sorted(set(left) & set(right)):
        compact = left[step]
        no_compact = right[step]
        compact_ms = numeric(compact.get("server_total_ms"))
        no_compact_ms = numeric(no_compact.get("server_total_ms"))
        saved_ms = no_compact_ms - compact_ms if compact_ms is not None and no_compact_ms is not None else None
        speedup = no_compact_ms / compact_ms if compact_ms and no_compact_ms is not None else None
        comparisons.append(
            {
                "step_num": step,
                "compact_images": compact.get("active_image_count"),
                "no_compact_images": no_compact.get("active_image_count"),
                "compact_triggered": compact.get("compaction_triggered"),
                "compact_server_total_ms": compact_ms,
                "no_compact_server_total_ms": no_compact_ms,
                "saved_ms": saved_ms,
                "speedup": speedup,
                "compact_prompt_tokens": compact.get("prompt_tokens"),
                "no_compact_prompt_tokens": no_compact.get("prompt_tokens"),
            }
        )
    return comparisons


def paired_summary(
    compact_rows: list[dict[str, Any]],
    no_compact_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    event_steps = [int(row["step_num"]) for row in compact_rows if row["compaction_triggered"]]
    first_event = min(event_steps) if event_steps else None
    baseline = [
        row for row in comparisons if first_event is None or int(row["step_num"]) < first_event
    ]
    post = [
        row for row in comparisons if first_event is not None and int(row["step_num"]) >= first_event
    ]

    def average(items: list[dict[str, Any]], key: str) -> float | None:
        values = [numeric(item.get(key)) for item in items]
        clean = [value for value in values if value is not None]
        return statistics.mean(clean) if clean else None

    compact_by_step = {int(row["step_num"]): row for row in compact_rows}
    event_boundary_adjacent_changes = []
    for step in event_steps:
        current = compact_by_step.get(step)
        previous = compact_by_step.get(step - 1)
        if not current or not previous:
            continue
        current_ms = numeric(current.get("server_total_ms"))
        previous_ms = numeric(previous.get("server_total_ms"))
        delta_ms = current_ms - previous_ms if current_ms is not None and previous_ms is not None else None
        event_boundary_adjacent_changes.append(
            {
                "step_num": step,
                "active_images_before": previous.get("active_image_count"),
                "active_images_after": current.get("active_image_count"),
                "server_total_before_ms": previous_ms,
                "server_total_after_ms": current_ms,
                "delta_ms": delta_ms,
                "delta_pct": delta_ms * 100.0 / previous_ms if delta_ms is not None and previous_ms else None,
            }
        )

    compact_post = [row for row in compact_rows if first_event is not None and int(row["step_num"]) >= first_event]
    no_compact_post = [row for row in no_compact_rows if first_event is not None and int(row["step_num"]) >= first_event]
    compact_post_total = mean(compact_post, "server_total_ms")
    no_compact_post_total = mean(no_compact_post, "server_total_ms")
    saved_total = (
        no_compact_post_total - compact_post_total
        if compact_post_total is not None and no_compact_post_total is not None
        else None
    )
    preprocess_saved = None
    engine_saved = None
    if compact_post and no_compact_post:
        compact_preprocess = mean(compact_post, "server_preprocess_ms")
        no_compact_preprocess = mean(no_compact_post, "server_preprocess_ms")
        compact_engine = mean(compact_post, "server_engine_ms")
        no_compact_engine = mean(no_compact_post, "server_engine_ms")
        if compact_preprocess is not None and no_compact_preprocess is not None:
            preprocess_saved = no_compact_preprocess - compact_preprocess
        if compact_engine is not None and no_compact_engine is not None:
            engine_saved = no_compact_engine - compact_engine

    x_values = [numeric(row.get("active_image_count")) for row in no_compact_rows]
    y_values = [numeric(row.get("server_total_ms")) for row in no_compact_rows]
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    slope_ms_per_image = None
    if len(pairs) >= 2:
        mean_x = statistics.mean(x for x, _ in pairs)
        mean_y = statistics.mean(y for _, y in pairs)
        denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
        if denominator:
            slope_ms_per_image = sum(
                (x - mean_x) * (y - mean_y) for x, y in pairs
            ) / denominator

    return {
        "first_compaction_step": first_event,
        "baseline_mean_saved_ms": average(baseline, "saved_ms"),
        "post_compaction_mean_compact_ms": compact_post_total,
        "post_compaction_mean_no_compact_ms": no_compact_post_total,
        "post_compaction_mean_saved_ms": saved_total,
        "post_compaction_saved_pct": (
            saved_total * 100.0 / no_compact_post_total
            if saved_total is not None and no_compact_post_total
            else None
        ),
        "post_compaction_mean_speedup": average(post, "speedup"),
        "post_compaction_preprocess_saved_ms": preprocess_saved,
        "post_compaction_engine_saved_ms": engine_saved,
        "preprocess_share_of_saved_pct": (
            preprocess_saved * 100.0 / saved_total
            if preprocess_saved is not None and saved_total
            else None
        ),
        "engine_share_of_saved_pct": (
            engine_saved * 100.0 / saved_total
            if engine_saved is not None and saved_total
            else None
        ),
        "no_compact_ols_ms_per_extra_image": slope_ms_per_image,
        "event_boundary_adjacent_changes": event_boundary_adjacent_changes,
    }


def fmt_ms(value: Any) -> str:
    number = numeric(value)
    return "—" if number is None else f"{number / 1000.0:.3f}s"


def write_markdown(
    path: Path,
    compact_rows: list[dict[str, Any]],
    no_compact_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    summaries: dict[str, Any],
    paired: dict[str, Any],
) -> None:
    lines = [
        "# OSWorld-V2 screenshot-compaction latency",
        "",
        "This experiment isolates the Qwen GUI adapter's screenshot-history folding. ",
        "The QW3 server uses the plain route with KVMem disabled; `compaction` here does not mean KVMem retrieval.",
        "",
        "## Key findings",
        "",
        f"- After the first compaction boundary, compact requests average {fmt_ms(paired['post_compaction_mean_compact_ms'])} versus {fmt_ms(paired['post_compaction_mean_no_compact_ms'])} without compaction: {fmt_ms(paired['post_compaction_mean_saved_ms'])} saved per step ({paired['post_compaction_saved_pct']:.1f}%).",
        f"- Mean paired speedup after the boundary is {paired['post_compaction_mean_speedup']:.2f}x. The no-compaction fit adds about {fmt_ms(paired['no_compact_ols_ms_per_extra_image'])} per retained screenshot.",
        f"- Server preprocessing explains {paired['preprocess_share_of_saved_pct']:.1f}% of the post-boundary saving; engine generation explains {paired['engine_share_of_saved_pct']:.1f}%.",
        "- Within the compact run, adjacent changes at event boundaries: " + "; ".join(
            f"step {item['step_num']} {item['active_images_before']}→{item['active_images_after']} images, {fmt_ms(item['delta_ms'])} ({item['delta_pct']:.1f}%)"
            for item in paired["event_boundary_adjacent_changes"]
        ) + ".",
        "- This is one sequential live A/B. Screens and actions can diverge between conditions, so the paired values are engineering observations rather than a confidence interval.",
        "",
        "## Per-step trace",
        "",
        "| Condition | Step | Event | Images | Prompt tokens | Server total | Preprocess | Engine | Action | Turn wall |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in compact_rows + no_compact_rows:
        lines.append(
            f"| {row['condition']} | {row['step_num']} | "
            f"{'yes' if row['compaction_triggered'] else ''} | {row['active_image_count']} | "
            f"{row['prompt_tokens']} | {fmt_ms(row['server_total_ms'])} | "
            f"{fmt_ms(row['server_preprocess_ms'])} | {fmt_ms(row['server_engine_ms'])} | "
            f"{fmt_ms(row['action_wall_ms'])} | {fmt_ms(row['turn_wall_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            "| Step | Compact images | No-compact images | Compact total | No-compact total | Saved | Speedup |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        speedup = "—" if row["speedup"] is None else f"{row['speedup']:.2f}×"
        lines.append(
            f"| {row['step_num']} | {row['compact_images']} | {row['no_compact_images']} | "
            f"{fmt_ms(row['compact_server_total_ms'])} | {fmt_ms(row['no_compact_server_total_ms'])} | "
            f"{fmt_ms(row['saved_ms'])} | {speedup} |"
        )

    lines.extend(["", "## Aggregate", ""])
    for condition, summary in summaries.items():
        lines.append(
            f"- `{condition}`: {summary['step_count']} steps; compaction events "
            f"{summary['compaction_event_steps']}; median server total "
            f"{fmt_ms(summary['median_server_total_ms'])}; max active images "
            f"{summary['max_active_image_count']}; max prompt tokens {summary['max_prompt_tokens']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(path: Path, rows_by_condition: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 960, 540
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 45, 70
    all_rows = [row for rows in rows_by_condition.values() for row in rows]
    points = [
        (int(row["step_num"]), numeric(row.get("server_total_ms")) or 0.0)
        for row in all_rows
    ]
    max_step = max((step for step, _ in points), default=1)
    max_ms = max((value for _, value in points), default=1.0)
    max_ms = max(max_ms, 1000.0) * 1.1

    def x(step: int) -> float:
        return margin_left + (step - 1) * (width - margin_left - margin_right) / max(max_step - 1, 1)

    def y(value: float) -> float:
        return height - margin_bottom - value * (height - margin_top - margin_bottom) / max_ms

    colors = {"compact": "#2563eb", "no_compact": "#dc2626"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">OSWorld per-step QW3 request latency</text>',
    ]
    for tick in range(6):
        value = max_ms * tick / 5
        yy = y(value)
        svg.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width-margin_right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{margin_left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value/1000:.0f}s</text>')
    for step in range(1, max_step + 1):
        xx = x(step)
        svg.append(f'<text x="{xx:.1f}" y="{height-margin_bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{step}</text>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#111827"/>')
    svg.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#111827"/>')
    svg.append(f'<text x="480" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">OSWorld model step</text>')
    for condition, rows in rows_by_condition.items():
        color = colors.get(condition, "#111827")
        coords = []
        for row in rows:
            value = numeric(row.get("server_total_ms"))
            if value is None:
                continue
            coords.append((x(int(row["step_num"])), y(value), row))
        if coords:
            svg.append('<polyline fill="none" stroke="%s" stroke-width="3" points="%s"/>' % (
                color,
                " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy, _ in coords),
            ))
        for xx, yy, row in coords:
            fill = "#16a34a" if row.get("compaction_triggered") else color
            svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{fill}"/>')
    svg.extend(
        [
            '<line x1="720" y1="55" x2="755" y2="55" stroke="#2563eb" stroke-width="3"/><text x="765" y="60" font-family="sans-serif" font-size="13">compact</text>',
            '<line x1="720" y1="78" x2="755" y2="78" stroke="#dc2626" stroke-width="3"/><text x="765" y="83" font-family="sans-serif" font-size="13">no compact</text>',
            '<circle cx="737" cy="101" r="5" fill="#16a34a"/><text x="765" y="106" font-family="sans-serif" font-size="13">compaction event</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-run", type=Path, required=True)
    parser.add_argument("--no-compact-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    compact_rows, compact_manifest = load_condition("compact", args.compact_run.resolve())
    no_compact_rows, no_compact_manifest = load_condition(
        "no_compact", args.no_compact_run.resolve()
    )
    comparisons = compare(compact_rows, no_compact_rows)
    summaries = {
        "compact": summarize(compact_rows),
        "no_compact": summarize(no_compact_rows),
    }
    paired = paired_summary(compact_rows, no_compact_rows, comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = compact_rows + no_compact_rows
    csv_path = args.output_dir / "step_latency.csv"
    fieldnames = list(all_rows[0]) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )

    report = {
        "schema_version": 1,
        "scope": "Qwen35VLAgent screenshot-history folding; QW3 runtime route is plain",
        "runs": {
            "compact": str(args.compact_run.resolve()),
            "no_compact": str(args.no_compact_run.resolve()),
        },
        "manifests": {
            "compact": compact_manifest,
            "no_compact": no_compact_manifest,
        },
        "summary": summaries,
        "paired_summary": paired,
        "comparison": comparisons,
        "steps": all_rows,
    }
    (args.output_dir / "compaction_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "compaction_analysis.md",
        compact_rows,
        no_compact_rows,
        comparisons,
        summaries,
        paired,
    )
    write_svg(
        args.output_dir / "step_latency.svg",
        {"compact": compact_rows, "no_compact": no_compact_rows},
    )
    print(f"analysis={args.output_dir.resolve()}")
    print(json.dumps(summaries, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
