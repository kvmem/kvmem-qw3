"""Complete the four-domain reuse panel using comparable selection fractions.

The primary plot retains half of each previous call's candidate history. This
is a boundary-reuse diagnostic, not a replay with one fixed system-wide budget.
An absolute 1K-budget figure and other budget summaries are also exported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from metrics import bootstrap
from plot_paper_reweighting_reuse import (
    BLUE, RED, GREY, DOMAINS, LABELS, axes_style, draw_original, errorbar, save,
)
from plot_query_locality_evidence import trajectory_means, paired_statistics


def statistics(rows):
    previous = trajectory_means(rows, lambda r: r['strict_coverage'])
    uniform = trajectory_means(rows, lambda r:
        float(r['strict_surviving_blocks']) / int(r['common_blocks']))
    oracle = trajectory_means(rows, lambda r:
        float(r['strict_coverage']) + float(r['strict_matched_oracle_gain']))
    assert previous.keys() == uniform.keys() == oracle.keys()
    return dict(trajectories=len(previous), boundaries=len(rows),
        previous=bootstrap(list(previous.values())), uniform=bootstrap(list(uniform.values())),
        oracle=bootstrap(list(oracle.values())),
        previous_minus_uniform=paired_statistics(uniform, previous)['difference'],
        oracle_minus_previous=paired_statistics(previous, oracle)['difference'],
        common_history_fully_selected=sum(int(r['strict_surviving_blocks']) == int(r['common_blocks']) for r in rows),
        trajectory_values=[dict(trajectory=tid, previous=previous[tid], uniform=uniform[tid], oracle=oracle[tid])
                           for tid in previous])


def draw_complete_reuse(ax, summary, budget):
    axes_style(ax)
    labels = LABELS.copy()
    labels[2] = 'Dialogue*'
    ax.set_xticks(range(4), labels, fontweight='bold')
    for x, domain in enumerate(DOMAINS):
        s = summary[domain]['primary']
        for offset, key, color, marker in [(-.22, 'uniform', GREY, 's'),
                                          (0., 'previous', BLUE, 'o'),
                                          (.22, 'oracle', RED, '^')]:
            errorbar(ax, x+offset, s[key], color, marker, scale=100.)
        ax.text(x, 8, f"n = {s['trajectories']}", ha='center', fontsize=6.5)
    ax.set(ylim=(0, 105), yticks=[0, 25, 50, 75, 100], ylabel='Historical attention\ncovered (%)')
    ax.yaxis.label.set_fontweight('bold')
    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', markersize=4, label=label)
        for marker, color, label in [('s', GREY, 'Uniform'), ('o', BLUE, 'Previous-step set'),
                                    ('^', RED, 'Oracle')]]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.06), ncol=3,
              frameon=False, fontsize=6.4, handletextpad=.3, columnspacing=.7)
    if budget.endswith('%'):
        note = f'Select {budget} of previous-step candidate blocks'
    else:
        note = f'{budget} selection; source candidate history > {budget}'
    ax.text(.5, -.35, note+'\n*Dialogue: across user queries',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.7, linespacing=1.5)


def render(out, original, summary):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.labelsize': 8.})
    for budget, suffix in [('50%', ''), ('25%', '_25pct'), ('10%', '_10pct'), ('1K', '_fixed1K')]:
        fig, axes = plt.subplots(2, 1, figsize=(3.7, 5.1))
        fig.subplots_adjust(left=.17, right=.985, bottom=.155, top=.87, hspace=.93)
        draw_original(axes[0], original)
        draw_complete_reuse(axes[1], summary[budget], budget)
        axes[0].set_title('(a) Attention reweighting across steps', loc='left', fontsize=8.,
                          fontweight='bold', pad=34.)
        axes[1].set_title('(b) Historical-block reuse across steps', loc='left', fontsize=8.,
                          fontweight='bold', pad=32.)
        save(fig, out/f'figure4_all_domains{suffix}')
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    fig.subplots_adjust(left=.17, right=.985, bottom=.30, top=.74)
    draw_complete_reuse(ax, summary['50%'], '50%')
    ax.set_title('Historical-block reuse across adjacent steps', loc='left', fontsize=7.8,
                 fontweight='bold', pad=32.)
    save(fig, out/'reuse_all_domains_only')


def caption_and_report(out, summary):
    caption = ('Figure 4. Historical-attention reweighting and reuse across steps. '
        '(a) Across-step KL divergence is 2.47× the within-step value over 80 trajectories '
        '(20 per workload), using matched, non-overlapping 32-query-position windows. '
        '(b) At each boundary, the previous step’s final window selects the top '
        'ceil(0.5 N) blocks from its N historical candidates. The retained set is then evaluated '
        'on the next step’s first window. All four workloads use the same selection fraction; '
        'the absolute block count varies with history length. Source-saturated boundaries are '
        'excluded. Invalid block identities are discarded without replacement; the uniform '
        'expectation and current-window oracle match the surviving block count on the same '
        'verified common history. The reuse panel includes 179/213/187/202 boundaries from '
        '20 software/tool-state/dialogue/GUI trajectories each. Software, tool-state and GUI '
        'boundaries remain within one user task, whereas the dialogue subset crosses actual '
        'user queries. Means and 95% confidence intervals bootstrap trajectories. '
        'Coverage is conditional on common old history, excluding sink/recent and newly added '
        'content; it does not measure whole-query fixed-set coverage or pruned-model quality.')
    (out/'figure4_caption.txt').write_text(caption+'\n')
    latex = caption.replace('Figure 4. ', '', 1).replace('2.47×', '$2.47\\times$')
    latex = latex.replace('ceil(0.5 N)', '$\\lceil 0.5 N\\rceil$').replace('95%', '95\\%').replace('step’s', "step's")
    (out/'figure4_caption.tex').write_text('\\caption{'+latex+'}\n')
    lines = ['# 四类数据完整的 Figure 4', '',
        '本版解决旧版固定 8K 筛选下 Dialogue 为空的问题：四类统一采用源候选历史块数的 50% 作为选集大小，'
        '全部由已有 32-position 窗口记录统计，未运行模型。原 8K 版保留在相邻 paper_revision 目录。', '',
        '## 主图', '',
        '![完整四类数据](figure4_all_domains.png)', '',
        '- (a) 原始 KL 面板不变：80 条轨迹、791 个配对 step 边界，总体 0.166334 → 0.410507 bits。',
        '- (b) 对每个边界，以旧调用最后 32 个 decode positions 对旧调用的全部候选历史排序，'
        '取 K=ceil(0.5 N) 块；随后核验共同 token/mRoPE/图片身份，丢弃失效块且不补位，'
        '在新调用最初 32 个 decode positions 上评估。',
        '- 统一的是选择比例，不是所有轨迹/边界的绝对块数。这不是“query 首次选集后冻结”的实验，'
        '也不是对实际 query 重选/容量触发策略的在线验证。',
        '- 排除源候选历史已经能全部装下的边界：K<N。极短历史的取整会导致 K=N，'
        '因此比例预算下排除了 2 个软件边界及 6 个真实 dialogue query 边界。',
        '- 原 dialogue 的 195 个有效 step 边界中，有 193 个真实新 user query；'
        '本版复用面板仅采用后者，再做未饱和筛选后剩 187 个。它不能被称为同一 query 内的 sub-step。',
        '- source 未饱和时，共同前缀仍可能变短而被有效选集覆盖。基线与 oracle 均匹配实际存活块数；'
        'summary.json 同时给出剔除这种共同历史饱和情况的敏感性结果。',
        '- 每条轨迹先对有效边界取均值，再等权聚合；5000 次 trajectory bootstrap，seed=20260911。'
        'CI 为探索性区间，未作多重比较调整。', '',
        '## 所有预算结果', '',
        '| 预算 | 数据 | 轨迹/边界 | Uniform (%) | Previous-step set (%) | Oracle (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for budget, domains in summary.items():
        for domain in DOMAINS:
            s = domains[domain]['primary']
            cells = []
            for key in ['uniform', 'previous', 'oracle']:
                v = s[key]
                cells.append(f"{100*v['mean']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]")
            lines.append(f"| {budget} | {domain} | {s['trajectories']}/{s['boundaries']} | "+' | '.join(cells)+' |')
    lines += ['', '所有方括号为 95% CI。主展示为 50%；10%/25% 和 1K/2K/4K 全部报告，'
        '避免只保留覆盖率较高的配置。绝对预算变化会改变未饱和合格样本，不能把这些行直接当成同样本预算曲线。', '',
        '## 如何在论文中使用', '',
        '本图支持“step 边界伴随权重重分配，同时仍存在历史块复用”。其中三个 agent 领域提供同一任务内证据；'
        'Dialogue 提供跨真实 user query 的补充观察，不能将四类全部概括为同一 query 内的稳定性。'
        '跨 query 仍然有复用，不意味着目标不变或不需要重选；也不能据本图断言每个新 query 必须刷新。', '',
        '## 导出文件', '',
        '- `figure4_all_domains.png/pdf/svg`：主图，四类统一 50%。',
        '- `figure4_all_domains_25pct.png/pdf/svg`、`figure4_all_domains_10pct.png/pdf/svg`：比例预算对照。',
        '- `figure4_all_domains_fixed1K.png/pdf/svg`：四类相同绝对 1K 预算的对照，同样没有缺失领域。',
        '- `reuse_all_domains_only.png/pdf/svg`：下半面板单独导出。',
        '- `summary.json`、`plot_data.csv`、`figure4_caption.txt/tex`：统计、逐轨迹绘图数据及图注。', '',
        '```bash', '.venv/bin/python scripts/motivation_v3/plot_paper_reweighting_reuse_all_domains.py', '```', '']
    (out/'README_zh.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    args = ap.parse_args()
    original_path = args.root/'analysis_kl/analysis.json'
    budget_path = args.root/'query_locality_20260923/boundary_budget_coverage.csv'
    original = json.loads(original_path.read_text())
    with budget_path.open() as f:
        rows = list(csv.DictReader(f))
    out = args.root/'query_locality_20260923/paper_revision_all_domains'
    out.mkdir(parents=True, exist_ok=True)
    summary, plot_data = {}, []
    for budget in ['10%', '25%', '50%', '1K', '2K', '4K']:
        summary[budget] = {}
        for domain in DOMAINS:
            subset = [r for r in rows if r['domain'] == domain and r['budget'] == budget
                and r['source_saturated'] == 'False' and r['boundary_type'] ==
                ('across_user_query' if domain == 'dialogue' else 'same_user_query')]
            for r in subset:
                k, n = int(r['strict_selected_blocks']), int(r['source_candidate_blocks'])
                m, h = int(r['strict_surviving_blocks']), int(r['common_blocks'])
                assert 0 <= m <= k < n and m <= h
                assert 0 <= float(r['strict_coverage']) <= 1+1e-9
                assert float(r['strict_matched_oracle_gain']) >= -1e-9
            primary = statistics(subset)
            nonsaturated = [r for r in subset if int(r['strict_surviving_blocks']) < int(r['common_blocks'])]
            summary[budget][domain] = dict(primary=primary, common_unsaturated=statistics(nonsaturated))
            plot_data.extend(dict(budget=budget, domain=domain, **r) for r in primary['trajectory_values'])
    assert all(summary['50%'][d]['primary']['trajectories'] == 20 for d in DOMAINS)
    assert [summary['50%'][d]['primary']['boundaries'] for d in DOMAINS] == [179, 213, 187, 202]
    with (out/'plot_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_data[0]))
        writer.writeheader(); writer.writerows(plot_data)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    render(out, original, summary)
    caption_and_report(out, summary)
    sources = [original_path, budget_path, Path(__file__),
               Path(__file__).with_name('plot_paper_reweighting_reuse.py'),
               Path(__file__).with_name('plot_query_locality_evidence.py'), Path(__file__).with_name('metrics.py')]
    provenance = dict(inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        upper_panel='Unchanged original KL statistics', primary_selection_fraction=.5,
        dialogue_subset='Actual new user queries only; source history unsaturated',
        new_model_calls=0, validation=['all four workloads have 20 trajectories at 50%',
            'source selection is unsaturated', 'surviving counts fit source and common history',
            'coverage and oracle bounds', 'all reported budgets preserved'])
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({d: summary['50%'][d]['primary'] for d in DOMAINS}, indent=2))


if __name__ == '__main__':
    main()
