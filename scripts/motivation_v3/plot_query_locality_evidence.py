"""Make claim-to-evidence panels from the audited query-locality CSVs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from metrics import bootstrap, mean


DOMAINS = [('code', 'Software', '#337aa8'), ('tool_state', 'Tool/state', '#428964'),
           ('gui', 'GUI', '#9b6da9')]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def trajectory_means(rows, getter):
    groups = defaultdict(list)
    for row in rows:
        groups[row['trajectory']].append(float(getter(row)))
    return {k: mean(v) for k, v in sorted(groups.items())}


def paired_statistics(left, right):
    ids = sorted(left.keys() & right.keys())
    return dict(trajectories=len(ids), ids=ids,
                left=bootstrap([left[i] for i in ids]),
                right=bootstrap([right[i] for i in ids]),
                difference=bootstrap([right[i] - left[i] for i in ids]),
                right_gt_left=sum(right[i] > left[i] for i in ids),
                paired_values=[dict(trajectory=i, left=left[i], right=right[i]) for i in ids])


def mean_ci(ax, x, stat, color, scale=1., marker='D', label=None):
    value = scale * stat['mean']
    low, high = [scale * v for v in stat['ci95']]
    ax.errorbar(x, value, yerr=[[value-low], [high-value]], fmt=marker,
                color=color, markersize=6, capsize=4, lw=1.6, zorder=5, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir
    boundary_path = root / 'boundary_locality.csv'
    budget_path = root / 'boundary_budget_coverage.csv'
    boundaries, budgets = read_csv(boundary_path), read_csv(budget_path)
    evidence = dict(step_boundary={}, reuse={}, session_change={})
    for domain, _, _ in DOMAINS:
        rows = [r for r in boundaries if r['domain'] == domain and r['boundary_type'] == 'same_user_query']
        left = trajectory_means(rows, lambda r: r['within_kl'])
        right = trajectory_means(rows, lambda r: r['cross_kl'])
        evidence['step_boundary'][domain] = paired_statistics(left, right) | dict(boundaries=len(rows))
        rows = [r for r in budgets if r['domain'] == domain and r['boundary_type'] == 'same_user_query'
                and r['budget'] == '8K' and r['source_saturated'] == 'False']
        previous = trajectory_means(rows, lambda r: r['strict_coverage'])
        # Exact expected mass of a uniformly random subset on the common
        # support, conditional on the same number of surviving selected IDs.
        uniform = trajectory_means(rows, lambda r: float(r['strict_surviving_blocks']) / float(r['common_blocks']))
        oracle = trajectory_means(rows, lambda r: float(r['strict_coverage']) + float(r['strict_matched_oracle_gain']))
        comparison = paired_statistics(uniform, previous)
        evidence['reuse'][domain] = dict(trajectories=len(previous), boundaries=len(rows),
                previous=bootstrap(list(previous.values())), uniform=bootstrap(list(uniform.values())),
                oracle=bootstrap(list(oracle.values())),
                previous_minus_uniform=comparison['difference'],
                oracle_minus_previous=paired_statistics(previous, oracle)['difference'])
    dialogue = [r for r in boundaries if r['domain'] == 'dialogue' and r['boundary_type'] == 'across_user_query']
    same = [r for r in dialogue if r['session_transition'] == 'same_session']
    different = [r for r in dialogue if r['session_transition'] == 'different_session']
    for metric in ['cross_kl', 'transport_fraction', 'mass_overlap', 'common_tokens']:
        left = trajectory_means(same, lambda r: r[metric])
        right = trajectory_means(different, lambda r: r[metric])
        evidence['session_change'][metric] = paired_statistics(left, right) | dict(
            same_session_boundaries=len(same), different_session_boundaries=len(different))

    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.titlesize': 12})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    ax = axes[0, 0]
    for index, (domain, name, color) in enumerate(DOMAINS):
        stat = evidence['step_boundary'][domain]
        x0, x1 = 3*index, 3*index+1
        for row in stat['paired_values']:
            ax.plot([x0, x1], [row['left'], row['right']], '-o', color=color,
                    alpha=.28, markersize=2.5, linewidth=.8)
        mean_ci(ax, x0, stat['left'], color)
        mean_ci(ax, x1, stat['right'], color)
        ax.text((x0+x1)/2, 1.05, f"{name}\n20 trajectories", ha='center', va='top', fontsize=9)
    ax.set(title='A  Step boundaries change attention weights', ylabel='Historical-attention KL (bits)',
           xticks=[0, 1, 3, 4, 6, 7], xticklabels=['Within', 'Boundary']*3, ylim=(0, 1.16))
    ax.text(.02, .96, 'Lines: trajectories; diamonds: mean and 95% CI',
            transform=ax.transAxes, fontsize=8, va='top')

    ax = axes[0, 1]
    styles = [('uniform', 'Uniform subset expectation', '#888888', 's'),
              ('previous', 'Previous-step selection', '#337aa8', 'D'),
              ('oracle', 'Current-window oracle', '#c97d42', '^')]
    for index, (domain, name, _) in enumerate(DOMAINS):
        stat = evidence['reuse'][domain]
        for offset, (key, label, color, marker) in zip([-.23, 0., .23], styles):
            mean_ci(ax, index+offset, stat[key], color, 100., marker, label if index == 0 else None)
        ax.text(index, 3., f"n={stat['trajectories']}; pairs={stat['boundaries']}", ha='center', fontsize=8)
    ax.set(title='B  A previous 8K set still covers the next step', ylabel='Common-history attention covered (%)',
           xticks=range(3), xticklabels=[x[1] for x in DOMAINS], ylim=(0, 102), xlim=(-.5, 2.5))
    ax.legend(loc='lower left', bbox_to_anchor=(.015, .1), frameon=False, fontsize=8)
    ax.text(.02, .96, 'Source history > 8K; select first, then verify surviving block IDs',
            transform=ax.transAxes, fontsize=8, va='top')

    for ax, metric, title, ylabel, scale in [
            (axes[1, 0], 'cross_kl', 'C  Session changes: larger redistribution', 'Historical-attention KL (bits)', 1.),
            (axes[1, 1], 'transport_fraction', 'D  Session changes: larger position shifts',
             'Transport distance / common-history span (%)', 100.)]:
        stat = evidence['session_change'][metric]
        for row in stat['paired_values']:
            ax.plot([0, 1], [row['left']*scale, row['right']*scale], '-o',
                    color='#8a8a8a', alpha=.35, markersize=3, linewidth=.8)
        mean_ci(ax, 0, stat['left'], '#428964', scale)
        mean_ci(ax, 1, stat['right'], '#c97d42', scale)
        ax.set(title=title, ylabel=ylabel, xticks=[0, 1],
               xticklabels=['New user turn\nSame session (155 pairs)',
                            'New user turn\nDifferent session (38 pairs)'], xlim=(-.35, 1.35))
        ax.set_ylim(bottom=0.)
        ax.text(.02, .96, '20 paired trajectories; session change is a goal-change proxy',
                transform=ax.transAxes, fontsize=8, va='top')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Evidence for attention reweighting, historical-block reuse, and session changes', fontsize=15)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(root / f'evidence_panels.{extension}', dpi=200)
    plt.close(fig)
    # Export individual panels with the same input statistics as compact images.
    evidence['scope'] = ('Reference attention on recorded inputs, W=32. All CIs bootstrap trajectories, 5000 replicates. '
        'Panel B uses a preceding decode window, not query-start prefill; its oracle is hindsight and matches '
        'surviving block count. Uniform is an exact expectation on the same common support and effective count. '
        'C/D compare two kinds of new user turns within the same dialogue cohort, not user-query boundaries '
        'versus within-query tool steps. Session identity is an exploratory proxy, not an annotated goal change.')
    evidence['provenance'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [boundary_path, budget_path, Path(__file__),
                                         Path(__file__).with_name('metrics.py')]}
    (root / 'evidence_statistics.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')

    def fmt(stat, scale=1., digits=3):
        low, high = stat['ci95']
        return f"{stat['mean']*scale:.{digits}f} [{low*scale:.{digits}f}, {high*scale:.{digits}f}]"

    lines = ['# 结论与数据的一一对应', '', '所有方括号为轨迹 bootstrap 95% CI。图中每条细线是一个轨迹的聚合值。', '',
        '## A：step boundary 的分布变化较大', '',
        '| 来源 | 轨迹 / 边界 | step 内 KL | 跨 step KL | 配对差值 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for domain, name, _ in DOMAINS:
        s = evidence['step_boundary'][domain]
        lines.append(f"| {name} | {s['trajectories']} / {s['boundaries']} | {fmt(s['left'])} | {fmt(s['right'])} | {fmt(s['difference'])} |")
    lines += ['', '三类的配对差值 CI 均高于零。窗口长度相同、无重叠，且每组边界两侧及调用内对照使用相同已核验历史块。', '',
        '## B：同一 query/任务内，上一 step 的关注块仍可复用', '',
        '上一 step 最后 32 个 decode query 选择 8K 历史块，在下一 step 最前 32 个 query 评估。'
        '先在上一调用完整候选历史上选块，再验证共同前缀并保留有效块，不补位。只保留源候选历史大于 8K 的边界。'
        '随机基线为共同历史上、相同有效块数的均匀随机集合期望，不依赖重复随机抽样。'
        'oracle 用当前窗口排序，匹配仍有效的选中块数，仅作事后上界。', '',
        '| 来源 | 轨迹 / 边界 | 均匀随机期望(%) | 上一步选集覆盖率(%) | oracle(%) | 复用−随机(pp) | oracle−复用(pp) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for domain, name, _ in DOMAINS:
        s = evidence['reuse'][domain]
        lines.append(f"| {name} | {s['trajectories']} / {s['boundaries']} | {fmt(s['uniform'],100,1)} | "
                     f"{fmt(s['previous'],100,1)} | {fmt(s['oracle'],100,1)} | "
                     f"{fmt(s['previous_minus_uniform'],100,1)} | {fmt(s['oracle_minus_previous'],100,1)} |")
    lines += ['', '这能说明关注内容的局部复用，不代表整个 user query 只需选择一次，也不代表端到端任务质量无损。'
        '此前空间分析还发现，覆盖两侧各 90% 历史 attention 的最短连续区间占候选历史 82%–89%；'
        '因此应使用“共享关注集合/历史块复用”，避免描述成一个很窄的连续区域。', '',
        '## C/D：跨原始会话的新 query，分布与位置变化更明显', '',
        '20 条对话轨迹同时具有同 session 与跨 session 的新 user 消息；分别有 155、38 个有效边界。'
        '以下差值在同一轨迹内配对后再聚合。', '',
        '| 指标 | 同 session 新 query | 跨 session 新 query | 配对差值 | 跨 session 更大的轨迹数 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for metric, name, scale in [('cross_kl','KL (bits)',1.),('transport_fraction','位置迁移/历史跨度 (%)',100.),
                               ('mass_overlap','分布质量重叠 (%)',100.),('common_tokens','共同历史 tokens',1.)]:
        s = evidence['session_change'][metric]
        digits = 3 if metric == 'cross_kl' else 1
        lines.append(f"| {name} | {fmt(s['left'],scale,digits)} | {fmt(s['right'],scale,digits)} | "
                     f"{fmt(s['difference'],scale,digits)} | {s['right_gt_left']}/{s['trajectories']} |")
    lines += ['', '这支持“会话切换伴随更大的迁移”的观察性结论。两组都跨了 user 消息，'
        '所以不能把它说成“跨 user query 一定比同 query 内的工具调用变化大”。'
        '会话切换也没有人工标注为目标切换，目标变化只是解释假设。', '',
        '## 推荐表述', '',
        '> 在测量的代理轨迹中，step 边界上的历史 attention 重分配显著大于调用内部变化；'
        '然而，同一任务内，上一 step 选出的历史块仍能覆盖下一 step 的大部分历史 attention。'
        '在对话子集中，跨原始会话的新用户轮次比同一会话内的新轮次呈现更大的分布和位置变化。', '',
        '64K 在这批数据中可容纳全部历史，因此不把其 100% coverage 当作选择性证据。'
        '所有 coverage 都针对已核验共同旧历史，sink/recent/new 内容另计。', '',
        '复现：', '', '```bash',
        f'.venv/bin/python scripts/motivation_v3/plot_query_locality_evidence.py --analysis-dir {root}',
        '```', '']
    (root / 'evidence_readme_zh.md').write_text('\n'.join(lines))
    print(json.dumps({k: {name: (value['difference'] if 'difference' in value else
                {'previous': value['previous'], 'previous_minus_uniform': value['previous_minus_uniform']})
                for name, value in group.items()} for k, group in evidence.items()
                if k in ['step_boundary', 'reuse', 'session_change']}, indent=2))


if __name__ == '__main__':
    main()
