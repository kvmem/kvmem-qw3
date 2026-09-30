"""Hold one historical set fixed across a real user-query boundary."""
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

from metrics import bootstrap, mean, normalize


def trajectory_values(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['trajectory']].append(float(row[key]))
    return {k: mean(v) for k, v in sorted(groups.items())}


def errorbar(ax, x, stat, color, scale=1.):
    y = scale * stat['mean']
    lo, hi = [scale * v for v in stat['ci95']]
    ax.errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt='D', color=color,
                markersize=4, capsize=3, linewidth=1., zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    args = ap.parse_args()
    root = args.root
    out = root / 'query_locality_20260923/query_boundary_transfer'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, analysis_path = root / 'manifest.json', root / 'analysis_kl/analysis.json'
    label_path = root / 'query_locality_20260923/boundary_labels.csv'
    locality_path = root / 'query_locality_20260923/boundary_locality.csv'
    manifest = json.loads(manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    by_id = {x['id']: x for x in analysis['trajectories']}
    with label_path.open() as f:
        labels = {(x['trajectory'], int(x['to_step'])): x for x in csv.DictReader(f)}
    rows = []
    for trajectory in manifest['trajectories']:
        if trajectory['domain'] != 'dialogue':
            continue
        previous = None
        matched = {b['to_step']: b for b in by_id[trajectory['id']]['settings']['32']['matched_boundaries']}
        for i, frame in enumerate(trajectory['frames']):
            path = root / 'measurements' / trajectory['id'] / f'step_{i:03d}.json'
            record = json.loads(path.read_text())
            with np.load(record['path']) as z:
                mass, positions = z['mass'], z['positions']
            decode = mass[positions >= record['prompt_tokens']]
            current = dict(step=frame['source_step'], record=record, decode=decode)
            label = labels.get((trajectory['id'], frame['source_step']))
            if (previous is not None and frame['source_step'] in matched and
                    label['boundary_type'] == 'across_user_query'):
                assert frame['source_step'] == previous['step'] + 1
                n = matched[frame['source_step']]['common_blocks']
                end = n + 1
                # W0 selects; W1 evaluates within the old query; W2 evaluates
                # after the new query. The set is unchanged between W1 and W2.
                selection_mass = previous['decode'][-64:-32].mean(0)
                p = normalize(previous['decode'][-32:, 1:end].mean(0))
                q = normalize(decode[:32, 1:end].mean(0))
                source_end = min(len(selection_mass), previous['record']['prompt_tokens'] // 128 - 8)
                order = np.argsort(-selection_mass[1:source_end], kind='stable') + 1
                budgets = [(f'{int(f*100)}%', max(1, int(np.ceil(len(order)*f)))) for f in [.1, .25, .5]]
                budgets += [('1K', 8), ('2K', 16), ('4K', 32), ('8K', 64)]
                for name, k in budgets:
                    selected = order[:k]
                    kept = selected[selected < end]
                    m = len(kept)
                    within, across = float(p[kept-1].sum()), float(q[kept-1].sum())
                    oracle = float(np.sort(q)[-m:].sum()) if m else 0.
                    assert -1e-10 <= within <= 1+1e-10 and -1e-10 <= across <= 1+1e-10
                    assert oracle >= across-1e-10
                    rows.append(dict(trajectory=trajectory['id'], from_step=previous['step'],
                        to_step=frame['source_step'], session_transition=label['session_transition'],
                        budget=name, source_blocks=len(order), common_blocks=n, selected_blocks=len(selected),
                        surviving_blocks=m, within_coverage=within, across_coverage=across,
                        change=across-within, oracle_gain=oracle-across, saturated=m == n))
            previous = current
    assert len(rows) == 193*7
    with (out/'fixed_set_transfer.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for budget in ['10%', '25%', '50%', '1K', '2K', '4K', '8K']:
        summary[budget] = {}
        for subset in ['all', 'same_session', 'different_session']:
            subset_rows = [r for r in rows if r['budget'] == budget and
                           (subset == 'all' or r['session_transition'] == subset)]
            s = dict(trajectories=len({r['trajectory'] for r in subset_rows}), boundaries=len(subset_rows))
            for key in ['within_coverage', 'across_coverage', 'change', 'oracle_gain', 'saturated']:
                s[key] = bootstrap(list(trajectory_values(subset_rows, key).values()))
            summary[budget][subset] = s
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with locality_path.open() as f:
        raw = [r for r in csv.DictReader(f) if r['domain'] == 'dialogue' and
               r['boundary_type'] == 'across_user_query']
    within = trajectory_values(raw, 'within_kl')
    across = trajectory_values(raw, 'cross_kl')
    assert len(within) == len(across) == 20
    direct = dict(within=bootstrap(list(within.values())), across=bootstrap(list(across.values())),
                  difference=bootstrap([across[k]-within[k] for k in within]))

    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'], 'font.size':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(left=.085, right=.985, bottom=.25, top=.81, wspace=.45)
    ax = axes[0]
    for tid in within:
        ax.plot([0, 1], [within[tid], across[tid]], '-o', color='#a8a8a8', alpha=.5,
                linewidth=.65, markersize=2)
    errorbar(ax, 0, direct['within'], '#2459e8')
    errorbar(ax, 1, direct['across'], '#e64a4a')
    ax.set(xticks=[0, 1], xticklabels=['Within response', 'Across user query'],
           ylabel='KL divergence (bits)', xlim=(-.3, 1.3), ylim=(0, .56))
    ax.set_title('(a) Changes at actual user-query boundaries', loc='left', fontsize=8.5, fontweight='bold', pad=16)
    ax.text(.5, .93, f"{direct['across']['mean']/direct['within']['mean']:.2f}×; 20 trajectories, 193 boundaries",
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5)
    ax = axes[1]
    for x, (subset, name, color) in enumerate([
            ('all', 'All new\nqueries', '#888888'),
            ('same_session', 'Same\nsession', '#2459e8'),
            ('different_session', 'Different\nsession', '#e64a4a')]):
        s = summary['25%'][subset]
        errorbar(ax, x, s['change'], color, scale=100.)
        ax.text(x, -11.3, f"{s['boundaries']} pairs", ha='center', fontsize=7)
    ax.axhline(0, color='#555555', linewidth=.7, linestyle='--')
    ax.set(xticks=[0, 1, 2], xticklabels=['All new\nqueries', 'Same\nsession', 'Different\nsession'],
           ylabel='Fixed-set coverage change (pp)', xlim=(-.45, 2.45), ylim=(-12.5, 5))
    ax.set_title('(b) Does a fixed set lose coverage?', loc='left', fontsize=8.5, fontweight='bold', pad=16)
    ax.text(.5, .94, '25% source-block budget; same set before/after',
            transform=ax.transAxes, ha='center', va='top', fontsize=7)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .015, 'Means and 95% trajectory-bootstrap CIs. Session identity is a proxy, not an annotated goal change.',
             ha='center', fontsize=7)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'query_boundary_evidence.{extension}', dpi=240)
    plt.close(fig)

    lines = ['# 真实 user-query 边界：分布变化与固定选集覆盖变化', '',
        '只使用已有 20 条对话轨迹的 193 个真实新 user 消息边界，未做新生成。', '',
        '## 直接分布证据', '',
        f"同一回答内部 KL={direct['within']['mean']:.4f}，跨 user query KL={direct['across']['mean']:.4f}，"
        f"比值={direct['across']['mean']/direct['within']['mean']:.2f}。"
        f"配对差值 95% CI={direct['difference']['ci95']} bits。", '',
        '这比较的是同一次回答内的局部窗口与真实新 user query 边界；不能等同于“同一任务内工具 step”与“新任务”的对照。', '',
        '## 更直接的固定集合检验', '',
        '对每个 query 边界，旧回答末尾取两个相邻、不重叠的 32-query 窗口 W0/W1，'
        '新回答开头取 W2。W0 在旧请求全部候选历史上选块，之后按已核验共同历史保留有效 ID，不补位。'
        '同一集合分别在 W1 和 W2 上测条件历史 attention coverage。W1 未参与选集；'
        'W1/W2 使用完全相同的共同历史及同一选集。分母不包含 sink/recent、新 query 和新生成内容。', '',
        '25% 是本次展示的主预算，选择块数为 ceil(旧调用候选块数×25%)；其余预算完整报告。'
        '这不是 query 开始 prefill 选集，也不直接测任务质量。', '',
        '| 预算 | 新 query 类型 | 边界数 | 旧 query coverage | 新 query coverage | 变化(pp) [95% CI] |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for budget, strata in summary.items():
        for subset, s in strata.items():
            st = s['change']; lo, hi = st['ci95']
            lines.append(f"| {budget} | {subset} | {s['boundaries']} | {s['within_coverage']['mean']*100:.2f}% | "
                         f"{s['across_coverage']['mean']*100:.2f}% | {st['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |")
    lines += ['', '**结果区分：跨 user query 的分布变化有直接证据，但所有新 query 合并时，'
        '25% 固定集合的 coverage 没有显著下降；跨 session 子组有下降。不能把后者推广成每次新 query 必须刷新。**', '',
        '10%/25%/50% 比例预算下，全部新 query 的变化 CI 都包含零；1K 的均值下降较小（约 1.1 pp），'
        '其 CI 接近零。这些是探索性、未经多重比较调整的区间，不能只挑显著预算。'
        '8K 对这批对话的旧历史已经饱和。', '',
        '跨 session 只是目标变化的代理；同 session 的新 query 可能仍在追问相同主题。'
        '若论文要声称“新目标必然带来更大失配”，需要目标标签或受控任务对照。', '',
        '## 复现', '', '```bash', '.venv/bin/python scripts/motivation_v3/analyze_query_boundary_transfer.py', '```', '']
    (out/'README_zh.md').write_text('\n'.join(lines))
    provenance = dict(inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [manifest_path, analysis_path, label_path, locality_path, Path(__file__),
                         Path(__file__).with_name('metrics.py')]},
                      trajectories=20, boundaries=193, new_model_calls=0,
                      validation='193 boundaries × 7 budgets; coverage range and matched-oracle bounds verified.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(direct_query_divergence=direct, transfer_25_percent=summary['25%']), indent=2))


if __name__ == '__main__':
    main()
