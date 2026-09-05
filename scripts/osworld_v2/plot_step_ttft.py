#!/usr/bin/env python3
"""Separate server TTFT from serial history-summary waiting in an OSWorld run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timing_file = args.task_dir / "step_timing.jsonl"
    summary_file = args.task_dir / "history_compaction.jsonl"
    timings = read_rows(timing_file)
    events = read_rows(summary_file)
    assert all(e["status"] == "success" for e in events), "Failed summary attempts need separate accounting"
    summaries = {e["step_num"]: e for e in events}
    assert len(summaries) == len(events)
    assert [r["step_num"] for r in timings] == list(range(1, len(timings) + 1))
    assert set(summaries).issubset(r["step_num"] for r in timings)
    data = []
    for row in timings:
        step = row["step_num"]
        api = row["agent_trace"]["api"]
        assert len(api["attempts"]) == 1, "Retries require separate timing attribution"
        server = api["server_timing"]
        event = summaries.get(step)
        summary_sec = event["wall_time_ms"] / 1000 if event else 0.0
        ttft = server["server_ttft_sec"]
        assert np.isfinite(ttft) and ttft >= 0
        data.append({"step": step, "prompt_tokens": api["usage"]["prompt_tokens"],
                     "action_server_ttft_sec": ttft,
                     "action_engine_ttft_sec": server["engine_ttft_sec"],
                     "summary_wall_sec": summary_sec,
                     "summary_server_ttft_sec": event["api"]["server_timing"]["server_ttft_sec"] if event else None,
                     "summary_plus_action_ttft_sec": summary_sec + ttft,
                     "compaction": bool(event)})

    x = np.array([r["step"] for r in data])
    y = np.array([r["action_server_ttft_sec"] for r in data])
    compact = [r for r in data if r["compaction"]]
    report = {"task_id": args.task_dir.name, "steps": len(data),
              "ttft_definition": "server_ttft_sec: server request start to first generated model token; non-streaming API, not client-observed first text",
              "combined_definition": "Recorded summary wall_time_ms / 1000 + following action request server_ttft_sec; arithmetic sum, not a directly measured end-to-end TTFT. Excludes action execution, remaining action-answer decode, client preparation, and other client overhead.",
              "mean_ttft_sec": float(np.mean(y)), "median_ttft_sec": float(np.median(y)),
              "p95_ttft_sec": float(np.percentile(y, 95)), "max_ttft_sec": float(np.max(y)),
              "max_ttft_step": int(x[np.argmax(y)]),
              "mean_summary_plus_ttft_sec": float(np.mean([r["summary_plus_action_ttft_sec"] for r in data])),
              "compactions": compact,
              "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in (timing_file, summary_file)]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"task{args.task_dir.name}_ttft"
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    blue, orange, teal = "#2563a6", "#df6c18", "#128477"
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.067, right=0.985, top=0.84, bottom=0.17, hspace=0.36)
    fig.suptitle(f"OSWorld {args.task_dir.name} · 每步 TTFT 与摘要等待", x=0.067,
                 ha="left", y=0.975, fontsize=20, fontweight="bold", color="#172d43")
    fig.text(0.067, 0.933,
             f"{len(data)} steps  |  256K context  |  thinking 关闭  |  MTP = 4  |  不计桌面动作执行耗时",
             color="#536477", fontsize=11)
    fig.text(0.067, 0.897,
             f"动作请求 TTFT：平均 {report['mean_ttft_sec']:.3f}s    中位数 {report['median_ttft_sec']:.3f}s"
             f"    P95 {report['p95_ttft_sec']:.3f}s    最大 {report['max_ttft_sec']:.3f}s（step {report['max_ttft_step']}）",
             fontsize=11, color=blue)

    for ax in axes:
        ax.plot(x, y, color=blue, lw=1.35, label="动作请求的服务端 TTFT", zorder=2)
        for row in compact:
            ax.axvline(row["step"], color=orange, alpha=0.3, ls="--", lw=0.9, zorder=0)
        ax.set_xlim(0, len(data) + 1)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#475569")
    axes[0].set_title("模型请求 TTFT：动作请求与摘要请求分别计量", loc="left", fontsize=12, pad=11)
    axes[0].set_ylabel("TTFT（秒）")
    axes[0].set_ylim(0, max(3.5, float(max(y)) * 1.25))
    axes[0].scatter([r["step"] for r in compact], [r["summary_server_ttft_sec"] for r in compact],
                    marker="^", color=teal, s=52, edgecolors="white", linewidths=0.7,
                    label="摘要请求的服务端 TTFT", zorder=4)
    for row in compact:
        axes[0].annotate(f"#{row['step']}", xy=(row["step"], row["summary_server_ttft_sec"]),
                         xytext=(0, -17), textcoords="offset points", ha="center", color=teal, fontsize=9)
    axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=10)

    axes[1].set_title("计入摘要完整生成的等待：摘要耗时 + 后续动作请求 TTFT", loc="left", fontsize=12, pad=11)
    axes[1].set_ylabel("相加耗时（秒）")
    axes[1].set_ylim(0, max(r["summary_plus_action_ttft_sec"] for r in data) + 4)
    for i, row in enumerate(compact):
        step, combined = row["step"], row["summary_plus_action_ttft_sec"]
        axes[1].vlines(step, row["action_server_ttft_sec"], combined, color=orange, lw=4,
                        label="额外的摘要完整耗时" if i == 0 else None, zorder=3)
        axes[1].scatter(step, combined, color=orange, s=32, edgecolors="white", linewidths=0.7, zorder=4)
        axes[1].annotate(f"#{step}  {combined:.2f}s", xy=(step, combined), xytext=(0, 9),
                         textcoords="offset points", ha="center", color="#a74708", fontsize=10)
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    axes[1].set_xlabel("Step", labelpad=9)
    axes[1].set_xticks(np.arange(0, len(data) + 1, 25))
    fig.text(0.067, 0.047,
             "TTFT 取 server_ttft_sec：服务端开始处理请求至生成首 token；本轮是非流式请求，不等于客户端收到首字符的时间。\n"
             "下图是两项日志耗时的相加值，排除桌面执行、动作回答后续 token 生成及其他客户端开销；橙色虚线标记 compaction 步骤。",
             color="#536477", fontsize=10, linespacing=1.65)
    fig.savefig(prefix.with_suffix(".png"), dpi=175)
    fig.savefig(prefix.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "source_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
