"""Paper-layout proposal: preserve the original KL panel and add reuse evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DOMAINS = ['code', 'tool_state', 'dialogue', 'gui']
LABELS = ['Software\nagent', 'Tool/state\nhistory', 'Dialogue', 'GUI\nagent']
BLUE, RED, GREY = '#2459e8', '#e64a4a', '#929292'


def errorbar(ax, x, statistic, color, marker='D', scale=1., filled=True):
    y = scale * statistic['mean']
    lo, hi = [scale * v for v in statistic['ci95']]
    ax.errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt=marker, color=color,
                markerfacecolor=color if filled else 'white', markeredgewidth=.7,
                markersize=3.7, capsize=2., linewidth=.8, zorder=6)


def axes_style(ax):
    ax.grid(axis='y', color='#dedede', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=2.5, width=.6, pad=2.)
    for spine in ax.spines.values():
        spine.set_linewidth(.7)
    ax.set_xlim(-.55, 3.5)
    ax.set_xticks(range(4), LABELS, fontweight='bold')


def draw_original(ax, analysis):
    axes_style(ax)
    for x, domain in enumerate(DOMAINS):
        rows = [r['settings']['32'] for r in analysis['trajectories'] if r['domain'] == domain]
        rows.sort(key=lambda r: r['matched_within_kl'])
        jitter = np.linspace(-.07, .07, len(rows))
        for offset, row in zip(jitter, rows):
            xx = [x-.15+offset, x+.15+offset]
            yy = [row['matched_within_kl'], row['matched_cross_kl']]
            ax.plot(xx, yy, color='#b9b9b9', linewidth=.45, alpha=.7, zorder=1)
            ax.scatter(xx, yy, c=[BLUE, RED], s=5, linewidths=0, alpha=.85, zorder=2)
        s = analysis['summary'][domain]['32']
        errorbar(ax, x-.29, s['matched_within_kl'], 'black', filled=False)
        errorbar(ax, x+.29, s['matched_cross_kl'], 'black', filled=False)
        ax.text(x, 1.105, f"{s['ratio_of_means']:.2f}×", ha='center', fontsize=7.5, fontweight='bold')
    ax.set(ylim=(0, 1.21), yticks=[0, .25, .5, .75, 1.],
           yticklabels=['0', '0.25', '0.5', '0.75', '1'], ylabel='KL divergence (bits)')
    ax.yaxis.label.set_fontweight('bold')
    handles = [Line2D([], [], marker='o', color=BLUE, linestyle='none', markersize=3.5, label='Within step'),
               Line2D([], [], marker='o', color=RED, linestyle='none', markersize=3.5, label='Across step'),
               Line2D([], [], marker='D', color='black', markerfacecolor='white', linestyle='none', markersize=3.5,
                      label='Mean, 95% CI')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.07), ncol=3,
              frameon=False, fontsize=6.7, handletextpad=.3, columnspacing=.8)
    pooled = analysis['pooled_kl']
    ax.text(.5, -.34, f"Overall: {pooled['matched_within_kl']['mean']:.3f} → "
            f"{pooled['matched_cross_kl']['mean']:.3f} bits ({pooled['ratio_of_means']:.2f}×)",
            transform=ax.transAxes, ha='center', fontsize=7.3, fontweight='bold')


def draw_reuse(ax, evidence):
    axes_style(ax)
    for x, domain in enumerate(DOMAINS):
        if domain == 'dialogue':
            ax.axvspan(x-.37, x+.37, color='#eeeeee', zorder=0)
            ax.text(x, 51, 'All history\nfits in 8K', ha='center', va='center', fontsize=6.7, color='#666666')
            ax.text(x, 8, 'n = 0', ha='center', fontsize=6.5, color='#666666')
            continue
        s = evidence['reuse'][domain]
        for offset, key, color, marker in [(-.22, 'uniform', GREY, 's'),
                                           (0., 'previous', BLUE, 'o'),
                                           (.22, 'oracle', RED, '^')]:
            errorbar(ax, x+offset, s[key], color, marker, scale=100.)
        ax.text(x, 8, f"n = {s['trajectories']}", ha='center', fontsize=6.5)
    ax.set(ylim=(0, 105), yticks=[0, 25, 50, 75, 100], ylabel='Historical attention\ncovered (%)')
    ax.yaxis.label.set_fontweight('bold')
    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', markersize=4, label=label)
               for marker, color, label in [('s', GREY, 'Uniform'), ('o', BLUE, 'Previous step'), ('^', RED, 'Oracle')]]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.06), ncol=3,
              frameon=False, fontsize=6.7, handletextpad=.3, columnspacing=1.)
    ax.text(.5, -.35, '8K selection; source candidate history > 8K',
            transform=ax.transAxes, ha='center', fontsize=7.)


def save(fig, target):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(target.with_suffix('.'+extension), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    args = parser.parse_args()
    original_path = args.root / 'analysis_kl/analysis.json'
    evidence_path = args.root / 'query_locality_20260923/evidence_statistics.json'
    original = json.loads(original_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    out = args.root / 'query_locality_20260923/paper_revision'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':7.5,
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.labelsize':8.})
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.9))
    fig.subplots_adjust(left=.17, right=.985, bottom=.125, top=.87, hspace=.93)
    draw_original(axes[0], original)
    draw_reuse(axes[1], evidence)
    axes[0].set_title('(a) Attention reweighting across steps', loc='left', fontsize=8., fontweight='bold', pad=34.)
    axes[1].set_title('(b) Reuse across steps within the same task', loc='left', fontsize=8., fontweight='bold', pad=32.)
    save(fig, out / 'figure4_reweighting_reuse')
    fig, ax = plt.subplots(figsize=(3.6, 2.65))
    fig.subplots_adjust(left=.17, right=.985, bottom=.27, top=.74)
    draw_reuse(ax, evidence)
    ax.set_title('Cross-step reuse of historical attention', loc='left', fontsize=8., fontweight='bold', pad=32.)
    save(fig, out / 'figure5_reuse_only')

    caption = ('Figure 4. Historical-attention reweighting and reuse across steps. '
        '(a) Across-step KL divergence is 2.47× the within-step value over 80 trajectories '
        '(20 per workload), using matched, non-overlapping 32-query windows. '
        '(b) An 8K-token historical set selected using the previous step’s final window '
        'covers 85.2–93.0% of common-history attention in the next step’s first window '
        'on eligible same-task boundaries. Source candidate histories must exceed 8K tokens; '
        '20/19/17 software/tool-state/GUI trajectories qualify. Invalid block identities are '
        'discarded without replacement. The uniform expectation and current-window oracle '
        'match the surviving block count on the same common history. Dialogue has no eligible '
        'source history exceeding 8K. Means and 95% confidence intervals bootstrap trajectories. '
        'Coverage is conditional on verified old history, excluding sink/recent and newly added content.')
    (out / 'figure4_caption.txt').write_text(caption+'\n')
    (out / 'figure4_caption.tex').write_text('\\caption{'+caption.replace('Figure 4. ', '', 1)
            .replace('2.47×', '$2.47\\times$').replace('85.2–93.0%', '85.2--93.0\\%')
            .replace('95%', '95\\%').replace('step’s', "step's")+'}\n')
    guide = '''# Figure 4 修改建议

推荐将原图作为 (a)，增加 (b) 上一步历史选集的跨步复用。已根据真实分析数据生成单栏双面板稿，也提供新增面板单独成图的版本。

## (a) 保留原有统计口径

四类各 20 条轨迹；总体 within=0.166334、across=0.410507 bits，比例为 2.467964。这里仍使用原 791 个 step 边界，包括对话里的 2 个无新 user 消息边界，因为 (a) 测的是 step，不是 user-query 边界。不要悄悄换成后续 user-query 过滤后的统计。

## (b) 补足复用证据

- 保留四类相同横轴，便于与原图对应；Dialogue 标为无 >8K 有效源历史，不画饱和的 100% 点。
- 指标是下一步窗口中，上一窗口选出的历史块覆盖的 attention mass。
- 三个对照：Uniform / Previous step / Oracle。均值及 CI，数字 n 表示各预算资格过滤后的轨迹数。
- 8K 是历史候选块预算，不是包含新内容、sink 和 recent 的总 KV 预算。
- 先在前一步完整候选历史中选块，之后核验共同前缀；失效 ID 丢弃且不补位。两个对照匹配剩余有效块数。
- oracle 使用当前窗口，是事后上界，不能写为实际在线 refresh 的收益。
- 该图测相邻边界复用，不证明整个 user query 只需选一次，更不直接测任务质量。

原图中的 “supporting step-level locality” 可以改成更明确的 “showing larger redistribution at step boundaries”。新增面板后，整图可表述为 “historical-attention reweighting and reuse across steps”。不宜称为强连续空间局部性。

## user query / 目标切换放哪里

建议将现有会话切换对照放在附录，或在主文确实需要讨论新用户轮次时单列一张图。对照标签应为 “New user turn, same session” 和 “New user turn, different session”。两组都存在新 user query，session 变化未人工标注为目标变化；不能标成 “within user query / across user query”。本次双面板主图不借它推出更强的目标切换结论。

如果篇幅允许，附录增加 4K/8K/16K 未饱和预算敏感性；跨预算的合格样本会改变，需要同时报告 n 或补共同样本对照。64K 已容纳本批全部历史，不适合作为主要选择性证据。

## 产物

- `figure4_reweighting_reuse.png/pdf/svg`：推荐的上下双面板 Figure 4。
- `figure5_reuse_only.png/pdf/svg`：保留原 Figure 4 时可单独新增的图。
- `figure4_caption.txt/tex`：完整英文图注草稿，投稿前可根据方法章节已有信息缩短。
- `provenance.json`：输入和绘图脚本 SHA256。

本方案不修改现有论文或原图文件。
'''
    (out / 'README_zh.md').write_text(guide)
    provenance = {'inputs':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [original_path, evidence_path, Path(__file__)]},
                  'upper_panel':'Original 80-trajectory, 791-boundary step analysis, unchanged.',
                  'lower_panel':'Strict 8K previous-window selection; source-unsaturated subset; same-task agent domains.',
                  'original_pooled_within':original['pooled_kl']['matched_within_kl']['mean'],
                  'original_pooled_across':original['pooled_kl']['matched_cross_kl']['mean'],
                  'new_model_calls':0}
    (out / 'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(str(out.resolve()))


if __name__ == '__main__':
    main()
