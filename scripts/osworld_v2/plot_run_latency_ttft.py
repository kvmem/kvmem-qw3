#!/usr/bin/env python3
"""Plot immutable OSWorld step snapshots, including reconstructed first-token time."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

from plot_predict_ttft import configure_plot, reconstruct


BLUE, ORANGE, GRAY, SHADE = "#2563a6", "#dc6c1b", "#83909d", "#93b9de"


def analyze(snapshot, task_id):
    directory = snapshot / "snapshot" / task_id
    source = directory / "step_timing.jsonl"
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [r["step_num"] for r in records] == list(range(1, len(records) + 1))
    config = json.loads((directory / "agent_config.json").read_text())
    result_path = directory / "result.txt"
    data = []
    for record in records:
        point = reconstruct(record)
        assert record["agent_trace"]["step_num"] == record["step_num"]
        total = record["turn_wall_time_ms"] / 1000
        predict = record["predict_wall_time_ms"] / 1000
        action = record.get("action_wall_time_ms", 0) / 1000
        assert 0 <= point["summary_sec"] <= predict <= total
        assert 0 <= action <= total
        if point["upper_sec"] is not None:
            assert point["upper_sec"] <= predict
        data.append({"task_id": task_id, **point, "total_sec": total,
                     "predict_sec": predict, "desktop_action_sec": action,
                     "output_tokens": record["agent_trace"].get("api", {}).get("usage", {}).get("completion_tokens")})
    valid = [r for r in data if r["upper_sec"] is not None]
    events = [r for r in valid if r["compaction_status"] == "success"]
    ordinary = [r for r in valid if not r["compaction_status"]]
    return {"task_id": task_id, "steps": len(data),
            "finished": result_path.exists(),
            "score": float(result_path.read_text()) if result_path.exists() else None,
            "config": config, "source_snapshot": str(source.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "total_mean_sec": float(np.mean([r["total_sec"] for r in data])),
            "total_p95_sec": float(np.percentile([r["total_sec"] for r in data], 95)),
            "total_max_sec": max(r["total_sec"] for r in data),
            "ttft_mean_lower_sec": float(np.mean([r["lower_sec"] for r in valid])),
            "ttft_mean_upper_sec": float(np.mean([r["upper_sec"] for r in valid])),
            "ttft_max_upper_sec": max(r["upper_sec"] for r in valid),
            "ordinary_ttft_median_upper_sec": float(np.median([r["upper_sec"] for r in ordinary])),
            "ordinary_ttft_max_upper_sec": max(r["upper_sec"] for r in ordinary),
            "max_reconstruction_width_sec": max(r["uncertainty_sec"] for r in valid),
            "excluded_ttft_steps": [r["step"] for r in data if r["upper_sec"] is None],
            "compactions": events, "rows": data}


def axes_style(ax, item, xmax, ymax, ylabel):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, xmax + 1, 50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    state = "已结束" if item["finished"] else "进行中，仅已完成步骤"
    ax.set_title(f"{item['task_id']}  ·  {item['steps']} steps  ·  {state}"
                 f"  ·  compaction {len(item['compactions'])} 次",
                 loc="left", fontsize=11, fontweight="bold", color="#253b52", pad=10)


def make_figure(items, title, subtitle, handles):
    fig, axes = plt.subplots(len(items), 1, figsize=(14, 3.05 * len(items) + 1.7), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.835, bottom=0.105, hspace=0.62)
    fig.suptitle(title, x=0.07, ha="left", y=0.978, fontsize=20, fontweight="bold", color="#172d43")
    fig.text(0.07, 0.936, subtitle, fontsize=10.5, color="#536477")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.064, 0.914),
               ncol=3, frameon=False, fontsize=10, columnspacing=2)
    return fig, axes[:, 0]


def save(fig, directory, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def plot_total(items, directory, subtitle, xmax):
    handles = [Line2D([0], [0], color=BLUE, lw=1.3, label="每步总耗时"),
               Line2D([0], [0], color=GRAY, lw=1, label="其中：桌面动作执行"),
               Line2D([0], [0], color=ORANGE, lw=3, marker="o", markersize=4, label="其中：compaction 耗时")]
    fig, axes = make_figure(items, "OSWorld · 每个 step 的总耗时", subtitle, handles)
    for ax, item in zip(axes, items):
        rows = item["rows"]
        x = [r["step"] for r in rows]
        ax.plot(x, [r["desktop_action_sec"] for r in rows], color=GRAY, lw=0.75, alpha=0.9)
        ax.plot(x, [r["total_sec"] for r in rows], color=BLUE, lw=1.05)
        for event in item["compactions"]:
            step, total = event["step"], event["total_sec"]
            ax.axvline(step, color=ORANGE, alpha=0.25, lw=0.8, ls="--")
            ax.vlines(step, total - event["summary_sec"], total, color=ORANGE, lw=3.5)
            ax.scatter(step, total, s=25, color=ORANGE, edgecolors="white", linewidths=0.6, zorder=4)
            ax.annotate(f"#{step}", xy=(step, total), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8.5, color="#a74708")
        peak = max(rows, key=lambda r: r["total_sec"])
        ax.annotate(f"#{peak['step']}  {peak['total_sec']:.1f}s",
                    xy=(peak["step"], peak["total_sec"]), xytext=(12, 9), textcoords="offset points",
                    fontsize=9, color=BLUE, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        ymax = max(32, item["total_max_sec"] * 1.2)
        axes_style(ax, item, xmax, ymax, "总耗时（秒）")
        ax.text(0.99, 0.94, f"平均 {item['total_mean_sec']:.2f}s / step",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#536477")
    fig.text(0.07, 0.040,
             "每步总耗时：图片准备 + compaction + 模型完整生成（含 thinking）+ 桌面动作及其他步骤开销。\n"
             "不含任务初始化与最终评分。各行纵轴按样本最大值展开，保留所有峰值；橙色虚线标记 compaction 步骤。",
             fontsize=10, color="#536477", linespacing=1.6)
    save(fig, directory, "step_latency")


def plot_ttft(items, directory, subtitle, xmax):
    handles = [Line2D([0], [0], color=BLUE, lw=1.3, label="含准备 / compaction 的 TTFT（重建上界）"),
               Patch(facecolor=SHADE, alpha=0.65, label="重建上下界区间"),
               Line2D([0], [0], color=ORANGE, marker="o", lw=0, markersize=5, label="compaction 步骤")]
    fig, axes = make_figure(items, "OSWorld · 每个 step 的 TTFT（计入 compaction）", subtitle, handles)
    ymax = math.ceil((max(i["ttft_max_upper_sec"] for i in items) + 3) / 5) * 5
    for ax, item in zip(axes, items):
        rows = item["rows"]
        x = [r["step"] for r in rows]
        low = [r["lower_sec"] if r["lower_sec"] is not None else np.nan for r in rows]
        high = [r["upper_sec"] if r["upper_sec"] is not None else np.nan for r in rows]
        ax.fill_between(x, low, high, color=SHADE, alpha=0.65)
        ax.plot(x, low, color=BLUE, lw=0.55, alpha=0.45)
        ax.plot(x, high, color=BLUE, lw=1.15)
        for event in item["compactions"]:
            step, ttft = event["step"], event["upper_sec"]
            ax.axvline(step, color=ORANGE, alpha=0.3, lw=0.8, ls="--")
            ax.scatter(step, ttft, s=28, color=ORANGE, edgecolors="white", linewidths=0.7, zorder=4)
            ax.annotate(f"#{step}  {ttft:.1f}s", xy=(step, ttft), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9, color="#a74708")
        axes_style(ax, item, xmax, ymax, "TTFT 重建值（秒）")
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.text(0.99, 0.94, f"平均区间 {item['ttft_mean_lower_sec']:.2f}–{item['ttft_mean_upper_sec']:.2f}s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#536477")
    fig.text(0.07, 0.040,
             "口径：predict 开始 → 动作请求首个模型 token（可以是 thinking token）；包含之前的图片准备与完整摘要等待。\n"
             "非流式日志的区间重建，非客户端首字的直接测量；排除首 token 后生成和桌面执行。摘要耗时只计一次。各行纵轴一致。",
             fontsize=9.5, color="#536477", linespacing=1.6)
    save(fig, directory, "step_ttft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot_dir / "snapshot_manifest.json").read_text())
    items = [analyze(args.snapshot_dir, task) for task in manifest["tasks"]]
    assert items and all(item["config"] == items[0]["config"] for item in items)
    config = items[0]["config"]
    configure_plot()
    subtitle = (f"快照 {manifest['snapshot_at'][:19].replace('T', ' ')}  |  "
                f"context {config['summary_context_tokens'] // 1024}K  |  "
                f"thinking {'开启' if config['enable_thinking'] else '关闭'}  |  "
                f"输出上限 {config['max_tokens'] // 1024}K  |  整段历史摘要")
    xmax = math.ceil(max(item["steps"] for item in items) / 50) * 50
    plot_total(items, args.snapshot_dir, subtitle, xmax)
    plot_ttft(items, args.snapshot_dir, subtitle, xmax)
    all_rows = [row for item in items for row in item["rows"]]
    with (args.snapshot_dir / "step_latency_ttft.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {"snapshot": manifest,
              "ttft_endpoint": "First generated model token of the action request, which may be a thinking token; not first post-thinking action token or client-visible text.",
              "ttft_measurement": "Retrospective lower/upper interval from non-streaming logs, not a directly observed client TTFT.",
              "ttft_lower_formula": "(image_prepare + input_dump + message_build + message_dump)/1000 + server_ttft_sec",
              "ttft_upper_formula": "(predict_total - parse)/1000 - (request_total_sec - server_ttft_sec)",
              "compaction_accounting": "Complete compaction, including retries, is nested in message_build and counted only once.",
              "results": [{k: v for k, v in item.items() if k != "rows"} for item in items]}
    (args.snapshot_dir / "step_latency_ttft.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in item.items() if k not in {"rows", "config", "compactions", "source_snapshot", "source_sha256"}}
                      for item in items], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
