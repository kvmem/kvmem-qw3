#!/usr/bin/env python3
"""Reconstruct predict-to-first-action-token intervals from completed traces.

Historical non-streaming traces do not split request upload from response
transport/deserialization. The reconstruction therefore produces a bracket,
not a directly observed end-to-end timestamp. The upper edge is a conservative
estimate; only durations known to happen after the first token are subtracted.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


PRE_FIELDS = ("image_prepare", "input_dump", "message_build", "message_dump")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reconstruct(row):
    """Bracket first action-token time; summary preparation is already nested."""
    trace = row.get("agent_trace") or {}
    event = trace.get("summary_compaction")
    result = {"step": row["step_num"], "step_status": row["status"],
              "lower_sec": None, "upper_sec": None, "uncertainty_sec": None,
              "server_ttft_sec": None, "preparation_measured_sec": None,
              "summary_sec": event["wall_time_ms"] / 1000 if event else 0.0,
              "compaction_status": event.get("status") if event else None,
              "failure_wait_sec": None, "exclusion_reason": None}
    if trace.get("status") != "success":
        result.update(failure_wait_sec=row.get("predict_wall_time_ms", 0) / 1000,
                      exclusion_reason="Action request produced no recorded first token")
        return result
    api, timing = trace["api"], trace["timing_ms"]
    attempts = api["attempts"]
    if len(attempts) != 1 or attempts[0]["status"] != "success":
        result["exclusion_reason"] = "Retry attribution requires additional timestamps"
        return result
    server = api["server_timing"]
    ttft, request = server["server_ttft_sec"], server["request_total_sec"]
    prep = sum(timing[k] for k in PRE_FIELDS) / 1000
    # All four preparation durations occur before call_llm. message_build
    # already includes history_prepare and the COMPLETE summary call.
    # Lower: known preparation + server TTFT, omitting unassigned overhead.
    lower = prep + ttft
    # Upper: all predict time minus measured response parsing and server work
    # after its first token. This retains unresolved request/response transport,
    # SDK work and logging, some of which may occur after the first token.
    upper = (timing["predict_total"] - timing["parse"]) / 1000 - (request - ttft)
    assert np.isfinite(lower) and np.isfinite(upper)
    assert request >= ttft >= 0
    assert upper + 1e-5 >= lower >= 0, (row["step_num"], lower, upper)
    assert timing["api_call"] / 1000 + 1e-5 >= request
    assert not event or result["summary_sec"] <= prep
    result.update(lower_sec=lower, upper_sec=upper, uncertainty_sec=max(0, upper - lower),
                  server_ttft_sec=ttft, preparation_measured_sec=prep)
    return result


def analyze(task_dir):
    path = task_dir / "step_timing.jsonl"
    raw = read_rows(path)
    assert [r["step_num"] for r in raw] == list(range(1, len(raw) + 1)), "Phase-reset steps need separate plots"
    assert all(r.get("agent_trace", {}).get("step_num") == r["step_num"] for r in raw)
    rows = [reconstruct(row) for row in raw]
    valid = [r for r in rows if r["upper_sec"] is not None]
    compact = [r for r in valid if r["compaction_status"] == "success"]
    event_file = task_dir / "history_compaction.jsonl"
    events = read_rows(event_file) if event_file.exists() else []
    assert [(r["step"], r["compaction_status"]) for r in rows if r["compaction_status"]] == [
        (e["step_num"], e["status"]) for e in events]
    score_file = task_dir / "result.txt"
    return {"task_id": task_dir.name, "steps_recorded": len(rows), "valid_ttft_steps": len(valid),
            "score": float(score_file.read_text()) if score_file.exists() else None,
            "rows": rows, "compactions": compact,
            "excluded_steps": [r for r in rows if r["upper_sec"] is None],
            "summary_successes": len(compact),
            "summary_failures": sum(r["compaction_status"] == "error" for r in rows),
            "mean_lower_sec": float(np.mean([r["lower_sec"] for r in valid])),
            "mean_upper_sec": float(np.mean([r["upper_sec"] for r in valid])),
            "p95_upper_sec": float(np.percentile([r["upper_sec"] for r in valid], 95)),
            "max_upper_sec": max(r["upper_sec"] for r in valid),
            "median_uncertainty_sec": float(np.median([r["uncertainty_sec"] for r in valid])),
            "max_uncertainty_sec": max(r["uncertainty_sec"] for r in valid),
            "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


BLUE, ORANGE, RED, SHADE = "#2563a6", "#dc6c1b", "#c44444", "#93b9de"


def draw_panel(ax, item, ymax):
    rows = item["rows"]
    x = [r["step"] for r in rows]
    low = [r["lower_sec"] if r["lower_sec"] is not None else np.nan for r in rows]
    high = [r["upper_sec"] if r["upper_sec"] is not None else np.nan for r in rows]
    ax.fill_between(x, low, high, color=SHADE, alpha=0.6, zorder=1)
    ax.plot(x, low, color=BLUE, lw=0.65, alpha=0.5, zorder=2)
    ax.plot(x, high, color=BLUE, lw=1.25, zorder=3)
    for row in item["compactions"]:
        step, upper = row["step"], row["upper_sec"]
        ax.axvline(step, color=ORANGE, lw=0.8, ls="--", alpha=0.3)
        ax.scatter(step, upper, color=ORANGE, s=30, zorder=4, edgecolors="white", linewidths=0.6)
        ax.annotate(f"#{step}\n{upper:.1f}s", xy=(step, upper), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#a74708",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5})
    for row in item["excluded_steps"]:
        step = row["step"]
        ax.axvline(step, color=RED, lw=1, ls=":", alpha=0.7)
        # Position the failure label in axes coordinates, not a fake TTFT value.
        ax.text(step, 0.8, f"#{step} 摘要失败\n无动作首 token", transform=ax.get_xaxis_transform(),
                ha="right", va="center", fontsize=9, color=RED)
    ax.set_title(f"{item['task_id']}  ·  {item['steps_recorded']} steps  ·  摘要成功 {item['summary_successes']} 次"
                 + (f" / 失败 {item['summary_failures']} 次" if item["summary_failures"] else ""),
                 loc="left", fontsize=12, fontweight="bold", color="#253b52", pad=10)
    ax.text(0.02, 0.965, f"平均区间 {item['mean_lower_sec']:.2f}–{item['mean_upper_sec']:.2f}s",
            transform=ax.transAxes, va="top", fontsize=9, color="#536477")
    ax.set_xlim(0, item["steps_recorded"] * 1.015)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Step", labelpad=5)
    ax.set_ylabel("TTFT 重建值（秒）")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.16)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)


def configure_plot():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.fonttype": "none"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(p) for p in args.task_dir]
    assert len(results) == 6, "This overview uses a six-panel comparison"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "target": "agent.predict start to first generated token of the action request, including all preceding summary work",
        "measurement": "Retrospective interval reconstruction, not a directly observed end-to-end TTFT",
        "lower_formula": "(image_prepare + input_dump + message_build + message_dump) / 1000 + server_ttft_sec",
        "upper_formula": "(predict_total - parse) / 1000 - (request_total_sec - server_ttft_sec)",
        "double_counting_guard": "message_build already contains history_prepare and complete summary generation; do not add either again",
        "uncertainty": "Unassigned client request/response transport, SDK processing, hashing, and logging cannot be split around the server first token. Lower omits this residual; upper retains it. Timing rounding may affect endpoints at microsecond scale.",
        "excluded": "Desktop action execution, later model action-answer generation and evaluation; failed action requests have no TTFT",
        "results": results}
    (args.output_dir / "predict_ttft_comparison.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    combined = []
    for item in results:
        for row in item["rows"]:
            combined.append({"task_id": item["task_id"], **row})
    with (args.output_dir / "predict_ttft_comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    configure_plot()
    ymax = max(27, max(r["max_upper_sec"] for r in results) + 5)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.835, bottom=0.11, hspace=0.44, wspace=0.15)
    fig.suptitle("每步 TTFT · 从 predict 开始，计入完整 compaction 等待", x=0.065, ha="left",
                 y=0.972, fontsize=20, fontweight="bold", color="#172d43")
    fig.text(0.065, 0.932, "六个样本同一纵轴尺度  |  含图片准备、摘要、请求处理和 prefill  |  排除桌面执行及动作回答后续生成",
             fontsize=11, color="#536477")
    handles = [Line2D([0], [0], color=BLUE, lw=1.5, label="新口径 TTFT：保守估计（上界）"),
               Patch(facecolor=SHADE, alpha=0.6, label="重建区间"),
               Line2D([0], [0], marker="o", color=ORANGE, lw=0, label="摘要成功"),
               Line2D([0], [0], color=RED, lw=1, ls=":", label="摘要失败：TTFT 缺失")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.907), ncol=4,
               frameon=False, columnspacing=1.8, fontsize=10)
    for ax, item in zip(axes.flat, results):
        draw_panel(ax, item, ymax)
    fig.text(0.065, 0.039,
             "历史日志未直接记录 predict → 首 token 的时间戳；蓝线是日志重建的保守估计，浅蓝区间反映无法拆分的客户端开销。\n"
             "摘要完整耗时已包含在请求前准备时间中，只计一次。失败步骤未发出动作请求，因此不赋予 TTFT，也不计入均值。",
             fontsize=10, color="#536477", linespacing=1.65)
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"predict_ttft_comparison.{suffix}", dpi=175)
    plt.close(fig)
    for item in results:
        fig, ax = plt.subplots(figsize=(14, 4.8))
        fig.subplots_adjust(left=0.065, right=0.98, top=0.87, bottom=0.23)
        draw_panel(ax, item, ymax)
        fig.text(0.065, 0.035, "新口径：predict 开始至动作首 token。蓝线为重建上界，浅蓝为不确定区间；含摘要，排除桌面执行和后续生成。",
                 fontsize=10, color="#536477")
        for suffix in ("png", "svg"):
            fig.savefig(args.output_dir / f"task{item['task_id']}_predict_ttft.{suffix}", dpi=175)
        plt.close(fig)
    print(json.dumps([{k: v for k, v in item.items() if k not in {"rows", "source", "excluded_steps", "compactions"}}
                      for item in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
