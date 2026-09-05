#!/usr/bin/env python3
"""Plot recorded per-step latency and whole-history compaction contributions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timing_path = args.task_dir / "step_timing.jsonl"
    event_path = args.task_dir / "history_compaction.jsonl"
    timings = read_jsonl(timing_path)
    events = [e for e in read_jsonl(event_path) if e["status"] == "success"]
    by_step = {r["step_num"]: r for r in timings}
    assert len(by_step) == len(timings), "Repeated step IDs require separate phase plots"
    assert sorted(by_step) == list(range(1, len(timings) + 1)), "Missing step timing"
    assert all(e["step_num"] in by_step for e in events)
    assert len({e["step_num"] for e in events}) == len(events)
    summaries = {e["step_num"]: e["wall_time_ms"] / 1000 for e in events}
    data = []
    for step, row in sorted(by_step.items()):
        total = row["turn_wall_time_ms"] / 1000
        predict = row["predict_wall_time_ms"] / 1000
        action = row["action_wall_time_ms"] / 1000
        summary = summaries.get(step, 0)
        assert 0 <= summary <= predict <= total
        assert 0 <= action <= total
        data.append({"step": step, "total_sec": total, "predict_sec": predict,
                     "action_sec": action, "summary_sec": summary,
                     "other_sec": total - predict - action,
                     "action_request_ttft_sec": row["agent_trace"]["api"]["server_timing"]["server_ttft_sec"],
                     "compaction": step in summaries})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"task{args.task_dir.name}_step_latency"
    with prefix.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    steps = np.array([r["step"] for r in data])
    total = np.array([r["total_sec"] for r in data])
    action = np.array([r["action_sec"] for r in data])
    ordinary = np.array([r["total_sec"] for r in data if not r["compaction"]])
    comparisons = []
    for event in events:
        step = event["step_num"]
        before = float(np.median(total[max(0, step - 6):step - 1]))
        after = float(np.median(total[step:min(len(total), step + 5)]))
        comparisons.append({"step": step, "total_sec": float(total[step - 1]),
                            "summary_sec": summaries[step], "before5_median_sec": before,
                            "after5_median_sec": after,
                            "ratio_to_before5_median": float(total[step - 1] / before),
                            "summary_output_tokens": event["api"]["usage"]["completion_tokens"],
                            "summary_request_ttft_sec": event["api"]["server_timing"]["server_ttft_sec"]})
    report = {"task_id": args.task_dir.name, "steps": len(data),
              "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in (timing_path, event_path)],
              "latency_definition": "turn_wall_time_ms / 1000; includes summary, action inference, action execution and other step overhead; excludes task setup and final evaluation",
              "all_steps_mean_sec": float(np.mean(total)),
              "ordinary_steps_mean_sec": float(np.mean(ordinary)),
              "ordinary_steps_median_sec": float(np.median(ordinary)),
              "ordinary_steps_p95_sec": float(np.percentile(ordinary, 95)),
              "summary_total_sec": sum(summaries.values()),
              "summary_amortized_sec_per_step": sum(summaries.values()) / len(data),
              "compactions": comparisons,
              "largest_steps": sorted(data, key=lambda r: r["total_sec"], reverse=True)[:10]}
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    blue, orange, gray = "#2563a6", "#df6c18", "#9aa4b0"
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.35]})
    fig.subplots_adjust(left=0.067, right=0.985, top=0.83, bottom=0.16, hspace=0.36)
    fig.suptitle(f"OSWorld 003 · 每个 step 的延迟与 compaction 峰值", x=0.067,
                 ha="left", y=0.975, fontsize=20, fontweight="bold", color="#172d43")
    fig.text(0.067, 0.933, "500 steps  |  256K context  |  thinking 关闭  |  整段历史摘要  |  MTP = 4",
             color="#536477", fontsize=11)
    handles = [Line2D([0], [0], color=blue, lw=1.7, label="每步总延迟"),
               Line2D([0], [0], color=gray, lw=1.1, label="其中：动作执行耗时"),
               Line2D([0], [0], color=orange, lw=5, marker="o", markersize=5,
                      label="其中：摘要 compaction 耗时（橙色线段）")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.061, 0.91),
               ncol=3, frameon=False, columnspacing=2.2, fontsize=11)

    for ax in axes:
        ax.plot(steps, action, color=gray, lw=0.8, alpha=0.85, zorder=1)
        ax.plot(steps, total, color=blue, lw=1.15, zorder=2)
        for item in comparisons:
            step, latency, summary = item["step"], item["total_sec"], item["summary_sec"]
            ax.axvline(step, color=orange, alpha=0.25, lw=0.9, ls="--", zorder=0)
            ax.vlines(step, latency - summary, latency, color=orange, lw=4.5, alpha=0.90, zorder=3)
            ax.scatter(step, latency, s=32, color=orange, zorder=4, edgecolors="white", linewidths=0.7)
        ax.set_xlim(0, len(data) + 1)
        ax.set_ylabel("Latency（秒）", color="#334155")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#475569")
    axes[0].set_ylim(0, max(total) * 1.1)
    axes[0].set_title("全量视图：保留所有尖峰", loc="left", fontsize=12, pad=10)
    for peak in sorted(data, key=lambda r: r["total_sec"], reverse=True)[:2]:
        step, latency = peak["step"], peak["total_sec"]
        axes[0].annotate(f"Step {step}: {latency:.1f}s\n动作执行 {peak['action_sec']:.1f}s", xy=(step, latency),
                         xytext=(step + 14, latency - 4), fontsize=10, color="#334155",
                         arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.8},
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    zoom = max(30, max(s["total_sec"] for s in comparisons) + 6)
    axes[1].set_ylim(0, zoom)
    axes[1].set_title(f"同一数据放大至 0–{zoom:.0f} 秒：观察摘要带来的单步峰值", loc="left", fontsize=12, pad=10)
    axes[1].set_xlabel("Step", labelpad=8)
    axes[1].set_xticks(np.arange(0, len(data) + 1, 25))
    for item in comparisons:
        step, latency = item["step"], item["total_sec"]
        axes[1].annotate(f"#{step}  {latency:.1f}s\n摘要 {item['summary_sec']:.1f}s", xy=(step, latency),
                         xytext=(step, latency + 1.1), ha="center", va="bottom", fontsize=10,
                         color="#a74708", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    clipped = steps[total > zoom]
    axes[1].scatter(clipped, np.repeat(zoom - 0.35, len(clipped)), marker="^", s=20, color=blue, zorder=4)
    fig.text(0.067, 0.050,
             f"口径：每步总耗时 turn_wall_time_ms，含摘要、模型生成、动作执行及其他开销；不含任务初始化和最终评分。\n"
             f"放大图顶部 ▲ 表示超出纵轴范围的峰值，完整高度见上图。5 次摘要共 {sum(summaries.values()):.1f}s，摊销 {sum(summaries.values()) / len(data):.3f}s / step。",
             fontsize=10, color="#536477", linespacing=1.6)
    fig.savefig(prefix.with_suffix(".png"), dpi=175)
    fig.savefig(prefix.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_files", "largest_steps"}}, ensure_ascii=False, indent=2))
    print(f"Plot: {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
