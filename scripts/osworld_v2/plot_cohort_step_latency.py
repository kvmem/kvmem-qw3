#!/usr/bin/env python3
"""Plot completed OSWorld pairs from frozen per-step measurements."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ARMS = ('qw3', 'kvmem')
COLORS = {'qw3': '#2563eb', 'kvmem': '#e87908'}
LABELS = {'qw3': '普通 qw3', 'kvmem': 'KVMem（guided query）'}
TTFT = 'observation_to_action_response_first_content_ms'
WAIT = 'decision_wait_ms'


def number(value):
    return float(value) if value not in ('', None) else math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zoom-seconds', type=float, default=6)
    args = parser.parse_args()
    report = args.report.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = {}

    def read(path):
        raw = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    summary = json.loads(read(report / 'summary.json'))
    tasks = summary['completed_tasks']
    events = list(csv.DictReader(read(report / 'maintenance_events.csv').splitlines()))
    by_task, exported, results = {}, [], {}
    marker_steps = {}
    for event in events:
        if event['task'] not in tasks:
            continue
        if not event['resolved_action_step']:
            raise ValueError('Maintenance event has no verified action mapping')
        key = event['task'], event['arm']
        marker_steps.setdefault(key, set()).add(int(event['resolved_action_step']))

    for task in tasks:
        candidates = [Path(root) / 'trials' / task for root in summary['cohort_summaries']
                      if (Path(root) / 'trials' / task / 'analysis/step_latency.csv').exists()]
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous source for {task}')
        trial = candidates[0]
        rows = list(csv.DictReader(read(trial / 'analysis/step_latency.csv').splitlines()))
        pair = {}
        for arm in ARMS:
            result = json.loads(read(trial / 'full' / arm / 'result.json'))
            if result['status'] != 'complete':
                raise ValueError(f'Incomplete source: {task}/{arm}')
            results[task, arm] = result
            subset = sorted((r for r in rows if r['arm'] == arm), key=lambda r: int(r['step']))
            if len({r['step'] for r in subset}) != len(subset):
                raise ValueError(f'Duplicate step: {task}/{arm}')
            if not marker_steps.get((task, arm), set()).issubset({int(r['step']) for r in subset}):
                raise ValueError(f'Missing marked step: {task}/{arm}')
            pair[arm] = subset
            for row in subset:
                for metric in (TTFT, WAIT):
                    value = number(row[metric])
                    if math.isfinite(value) and value < 0:
                        raise ValueError(f'Negative latency: {task}/{arm}/{row["step"]}')
                exported.append(dict(task=task, source_csv=str(trial / 'analysis/step_latency.csv'),
                                     maintenance_step_resolved=int(row['step']) in marker_steps.get((task, arm), set()),
                                     **row))
        by_task[task] = pair

    # A portable long table retains the original millisecond fields and gaps.
    with (out / 'step_latency_raw.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    ncols = 3
    nrows = math.ceil(len(tasks) / ncols)
    max_step = max(int(row['step']) for row in exported)
    coverage = {}

    def draw(metric, stem, zoom=None):
        fig, axes = plt.subplots(nrows, ncols, figsize=(17.5, 3.05 * nrows + 1.8), squeeze=False)
        title = '每步 TTFT' if metric == TTFT else '每步完整动作等待'
        suffix = f' · 0–{zoom:g} 秒放大' if zoom else ' · 完整原始范围'
        fig.suptitle(f'OSWorld · {len(tasks)} 个完成配对的样本 · {title}{suffix}', y=.986, fontsize=20)
        definition = ('观察就绪 → 下一动作回复首内容（包含此前的原生压缩等待）'
                      if metric == TTFT else '观察就绪 → 下一桌面动作发出（包含回复生成和维护等待）')
        fig.text(.5, .948, definition, ha='center', fontsize=12)
        handles = [Line2D([], [], color=COLORS[a], lw=1.6, label=LABELS[a]) for a in ARMS]
        handles += [Line2D([], [], linestyle='', marker='o', markerfacecolor='white',
                           markeredgecolor=COLORS['qw3'], label='compaction 所在步骤'),
                    Line2D([], [], linestyle='', marker='D', color=COLORS['kvmem'],
                           label='KVMem 重选所在步骤')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .934), ncol=4,
                   frameon=False, fontsize=10.5)
        for ax, task in zip(axes.flat, tasks):
            maxima, clipped = [], {}
            for arm in ARMS:
                rows = by_task[task][arm]
                x = np.array([int(r['step']) for r in rows])
                y = np.array([number(r[metric]) / 1000 for r in rows])
                finite = np.isfinite(y)
                maxima.extend(y[finite].tolist())
                ax.plot(x, y, color=COLORS[arm], lw=1.15, alpha=.93)
                selected = np.array([step in marker_steps.get((task, arm), set()) for step in x]) & finite
                ax.scatter(x[selected], y[selected], marker='o' if arm == 'qw3' else 'D',
                           s=26, facecolors='white' if arm == 'qw3' else COLORS[arm],
                           edgecolors=COLORS[arm], linewidths=1.1, zorder=4)
                if zoom:
                    high = finite & (y > zoom)
                    clipped[arm] = int(high.sum())
                    ax.scatter(x[high], np.full(high.sum(), zoom * .973), color=COLORS[arm],
                               marker=r'$\uparrow$', s=50, zorder=5)
                coverage[f'{task}/{arm}/{metric}'] = {
                    'rows': len(rows), 'finite': int(finite.sum()), 'missing': int((~finite).sum()),
                    'max_seconds': float(y[finite].max()) if finite.any() else None,
                    'above_zoom': int((finite & (y > args.zoom_seconds)).sum())}
            ax.set_ylim(0, zoom if zoom else max(3, max(maxima, default=0) * 1.14))
            ax.set_xlim(0, max_step + 3)
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(alpha=.45, color='#cbd5e1', linewidth=.6)
            count_text = ' / '.join(str(results[task, a]['steps']) for a in ARMS)
            ax.set_title(f'{task}   ·   步数 {count_text}（qw3 / KVMem）', fontsize=11, loc='left', pad=9)
            if zoom:
                ax.text(.025, .89, f'↑ >{zoom:g}s：qw3 {clipped["qw3"]}，KVMem {clipped["kvmem"]}',
                        transform=ax.transAxes, fontsize=8.5, color='#475569',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.5))
        for ax in list(axes.flat)[len(tasks):]:
            ax.set_axis_off()
        fig.supxlabel('各自轨迹的 step（同一步号不代表相同输入）', y=.068, fontsize=12)
        fig.supylabel('延迟（秒）', x=.012, fontsize=12)
        note = ('所有纵轴为线性；各面板独立缩放；保留原始尖峰、冷 prefill 和缺失值。'
                if zoom is None else
                f'所有纵轴线性、统一为 0–{zoom:g} 秒；↑ 标记超出显示范围的原始点，完整值见完整范围图。')
        fig.text(.045, .033, note + '\n重选标记表示该步骤包含维护；解码中的重选可能发生在首内容之后。'
                 '\n仅纳入已完成配对的 11 个样本；015 的 KVMem 仍在运行，本图未纳入。'
                 if len(tasks) == 11 and '015' not in tasks else note,
                 fontsize=9.5, va='center')
        fig.subplots_adjust(left=.052, right=.987, bottom=.125, top=.87, hspace=.40, wspace=.23)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'{stem}.{extension}', dpi=175, facecolor='white')
        plt.close(fig)

    draw(TTFT, 'step_ttft_full')
    draw(TTFT, 'step_ttft_zoom', args.zoom_seconds)
    draw(WAIT, 'step_action_wait_full')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    report_generated_at=summary['generated_at'], tasks=tasks,
                    completed_pairs=len(tasks), plotted_runs=len(tasks) * 2,
                    scope='Completed pairs only; active 015 excluded from this snapshot',
                    metric_definitions={TTFT: 'Observation ready to first content of the next action-producing response',
                                        WAIT: 'Observation ready to next desktop action dispatch'},
                    axes='Linear. Full-range panels have independent y limits. Zoom is explicitly clipped with overflow marks.',
                    smoothing=False, missing_value_policy='NaN gaps, never zero-filled',
                    coverage=coverage, maintenance_events=len(events),
                    source_sha256=evidence,
                    plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out / 'README.md').write_text(
        '# OSWorld 逐步延迟曲线\n\n'
        f'数据快照：{summary["generated_at"]}。包含 {len(tasks)} 个完整配对样本。\n\n'
        '- `step_ttft_full`：完整步骤 TTFT，观察就绪到下一动作回复首内容。\n'
        '- `step_ttft_zoom`：相同原始数据的 0–6 秒放大视图；超界点用上箭头标记。\n'
        '- `step_action_wait_full`：观察就绪到下一桌面动作发出的完整等待。\n\n'
        'PNG、PDF 和 SVG 使用同一数据。全部纵轴线性，完整范围各面板独立缩放；不做平滑、不删除冷 prefill 或其他尖峰，缺失值留断点。\n\n'
        '两组轨迹不同，同一步号不对应相同输入。压缩和重选标记来自已核对的原始事件到动作映射；重选可能发生在首内容之后，TTFT 不一定覆盖完整重选停顿。\n\n'
        '原始毫秒数据及来源见 `step_latency_raw.csv`，覆盖数和文件指纹见 `plot_manifest.json`。\n')
    print(json.dumps({'output': str(out), 'pairs': len(tasks), 'rows': len(exported),
                      'maintenance_events': len(events)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
