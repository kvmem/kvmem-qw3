"""Freeze a prefill-selected set across observed calls and real query boundaries.

CPU-only analysis of existing unpruned reference attention. Selection never
uses evaluated decode rows. The primary archive H0 is fixed, including its
denominator; a separate expanding-history diagnostic counts newly arriving
content. These are attention diagnostics, not simulated pruning/quality tests.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze_query_locality import common_prefix, gzread, write_csv
from metrics import bootstrap, mean

FRACTIONS = [.25, .5, .75, 1.]
DOMAIN_NAMES = {'code': 'Software', 'tool_state': 'Tool/state', 'gui': 'GUI'}
BLUE, ORANGE, GREY = '#2763d8', '#d87320', '#68717b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(a, b):
    n = common_prefix(a['inputs']['tokens'], b['inputs']['tokens'])
    if a['inputs']['mrope'] or b['inputs']['mrope']:
        n = min(n, common_prefix(a['inputs']['mrope'], b['inputs']['mrope']))
    images = a['frame'].get('image_sha256', [])
    if images and b['frame'].get('image_sha256', [])[:len(images)] != images:
        # Conservatively exclude ambiguous image identity, even if text IDs match.
        return 0
    return n


def order(scores, end):
    return np.argsort(-scores[1:end], kind='stable') + 1


def mass_coverage(values, selected, end):
    denominator = float(values[1:end].sum())
    assert denominator > 0
    kept = selected[(selected >= 1) & (selected < end)]
    return float(values[kept].sum() / denominator)


def oracle(values, budget, end):
    return mass_coverage(values, order(values, end)[:budget], end)


def load_calls(root, trajectory, fingerprints):
    calls = []
    for i, frame in enumerate(trajectory['frames']):
        metadata_path = root / 'measurements' / trajectory['id'] / f'step_{i:03d}.json'
        rec = json.loads(metadata_path.read_text())
        assert rec['status'] == 'collected' and rec['block_tokens'] == 128
        with np.load(rec['path']) as z:
            mass = z['mass'].astype(np.float64)
            positions = z['positions']
        pre = mass[positions < rec['prompt_tokens']]
        dec = mass[positions >= rec['prompt_tokens']]
        assert len(pre) == 64 and len(dec) > 0
        assert np.max(np.abs(mass.sum(1)-1)) < 1e-5
        # All candidate blocks precede ALL 64 prefill selector query positions.
        # Exclude block zero (attention sink) and the partial/tail blocks.
        end = int(positions[0]) // 128
        assert end > 1 and positions[0] == rec['prompt_tokens']-64
        inputs = gzread(rec['input_tokens_path'])
        assert len(inputs['tokens']) == rec['prompt_tokens']
        calls.append(dict(record=rec, frame=frame, inputs=inputs, end=end,
                          prefill=pre.mean(0), decode=dec, average=dec.mean(0)))
        for path in [metadata_path, Path(rec['path']), Path(rec['input_tokens_path'])]:
            fingerprints[str(path.resolve())] = digest(path)
    return calls


def cluster_stat(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups[row['trajectory']].append(value)
    return bootstrap([mean(groups[key]) for key in sorted(groups)])


def curve(rows, xkey, field):
    return [dict(x=x, **cluster_stat([r for r in rows if r[xkey] == x], field))
            for x in sorted({r[xkey] for r in rows})]


def line(ax, stats, color, label, style='-', marker='o', band=True):
    x = np.array([r['x'] for r in stats])
    y = 100*np.array([r['mean'] for r in stats])
    ax.plot(x, y, style, marker=marker, color=color, label=label,
            markersize=3, linewidth=1.4)
    if band and all(r['ci95'][0] is not None for r in stats):
        ax.fill_between(x, [100*r['ci95'][0] for r in stats],
                        [100*r['ci95'][1] for r in stats], color=color, alpha=.12, linewidth=0)


def style_axes(ax):
    ax.set(ylim=(0, 102))
    ax.grid(axis='y', alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)


def query_panel(ax, rows, title):
    n = len({r['pair'] for r in rows})
    ax.axvspan(4.5, 8.5, color=ORANGE, alpha=.045)
    ax.axvline(4.5, color=GREY, linestyle=':', linewidth=1)
    line(ax, curve(rows, 'phase', 'frozen'), BLUE, 'Frozen at initial prefill')
    line(ax, curve([r for r in rows if r['phase'] > 4], 'phase', 'refresh'),
         ORANGE, 'Reselected at new query')
    ax.set(xticks=[1, 2, 3, 4, 5, 6, 7, 8],
           xticklabels=['0–25', '25–50', '50–75', '75–100']*2,
           xlim=(.6, 8.4), xlabel='Decode progress within each query (%)',
           ylabel='Initial-history attention covered (%)')
    ax.tick_params(axis='x', labelsize=7)
    ax.text(.23, .96, 'Old query', transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.text(.76, .96, 'New query', transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.set_title(title+f'\n{n} query pairs; {len({r["trajectory"] for r in rows})} trajectories',
                 loc='left', fontsize=9, pad=8)
    style_axes(ax)


def render(out, agents, dialogue, summary):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for fraction in [.5, .25, .75]:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
        fig.subplots_adjust(left=.09, right=.985, bottom=.145, top=.79, hspace=.75, wspace=.26)
        for ax, dom, letter in zip(axes[0], ['code', 'gui'], ['a', 'b']):
            rs = [r for r in agents if r['domain'] == dom and r['fraction'] == fraction
                  and r['all_initial_history_preserved']]
            line(ax, curve(rs, 'call', 'frozen_initial'), BLUE, 'Frozen')
            line(ax, curve(rs, 'call', 'refresh_initial'), ORANGE, 'Refresh')
            n = len({r['trajectory'] for r in rs})
            ax.set(xticks=[1, 3, 6, 9, 12], xlim=(.7, 12.3), xlabel='Observed sub-step (model call)',
                   ylabel='Initial-history attention covered (%)')
            ax.set_title(f'({letter}) {DOMAIN_NAMES[dom]}: one user query\n{n}/20 trajectories preserve all initial block identities',
                         loc='left', fontsize=9, pad=8)
            style_axes(ax)
        for ax, kind, letter, label in zip(axes[1], ['same_session', 'different_session'],
                ['c', 'd'], ['new query, same source session', 'new query, different source session']):
            rs = [r for r in dialogue if r['fraction'] == fraction and r['session_transition'] == kind]
            query_panel(ax, rs, f'({letter}) Dialogue: {label}')
        fig.suptitle('A set selected once at query start: reuse over steps and across queries',
                     y=.983, fontsize=12)
        fig.text(.5, .945, f'Fixed block count B = ceil({fraction:g} × initial candidate blocks); same H₀ denominator throughout each trace',
                 ha='center', fontsize=9)
        fig.legend(handles=[Line2D([0], [0], color=BLUE, marker='o', markersize=3,
                                  label='Frozen initial-prefill selection'),
                            Line2D([0], [0], color=ORANGE, marker='o', markersize=3,
                                  label='Same B, reselected from prefill at each call / new query')],
                   loc='upper center', bbox_to_anchor=(.5, .923), ncol=2, frameon=False, fontsize=8)
        fig.text(.5, .025, 'Means and 95% trajectory-bootstrap CIs. Rows are separate datasets, not one continuous experiment.\n'
                 'H₀ contains only initial blocks; later content is excluded here. Source-session change is not an annotated goal change.',
                 ha='center', fontsize=8)
        stem = 'query_freeze_evidence' if fraction == .5 else f'query_freeze_sensitivity_{int(fraction*100)}pct'
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'{stem}.{ext}', dpi=240)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    fig.subplots_adjust(left=.065, right=.99, bottom=.22, top=.72, wspace=.25)
    for ax, dom in zip(axes, DOMAIN_NAMES):
        rs = [r for r in agents if r['domain'] == dom and r['fraction'] == .5]
        line(ax, curve(rs, 'call', 'frozen_current'), BLUE, 'Frozen')
        line(ax, curve(rs, 'call', 'refresh_current'), ORANGE, 'Refresh')
        line(ax, curve(rs, 'call', 'oracle_current'), GREY, 'Oracle', '--', None, False)
        ax.set(xticks=[1, 3, 6, 9, 12], xlim=(.7, 12.3), xlabel='Observed model call',
               ylabel='Current-history attention covered (%)' if dom == 'code' else '')
        label = 'starts mid-task' if dom == 'tool_state' else 'starts at query beginning'
        ax.set_title(f'{DOMAIN_NAMES[dom]}: all 20 trajectories\n{label}', fontsize=9, loc='left')
        style_axes(ax)
    fig.suptitle('When newly arriving historical content counts, the initial set can become insufficient',
                 y=.985, fontsize=11)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, label='Frozen initial set'),
                        Line2D([0], [0], color=ORANGE, label='Same B, refreshed each call (prefill)'),
                        Line2D([0], [0], color=GREY, linestyle='--', label='Same-B decode oracle (upper bound)')],
               loc='upper center', bbox_to_anchor=(.5, .92), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .015, 'B is fixed to 50% of the first observed candidate history. The evaluation history grows at later calls.\n'
             'Invalidated initial IDs contribute zero; sink and prefill-tail blocks are excluded. Tool/state has 19 trajectories after call 9.',
             ha='center', fontsize=8)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'query_freeze_expanding_history.{ext}', dpi=240)
    plt.close(fig)


def build_summary(agents, dialogue, audit):
    result = {'agent': {}, 'dialogue': {}, 'audit': audit}
    for f in FRACTIONS:
        tag = str(int(100*f))+'%'
        result['agent'][tag] = {}
        result['dialogue'][tag] = {}
        for dom in DOMAIN_NAMES:
            rs = [r for r in agents if r['fraction'] == f and r['domain'] == dom]
            fixed = [r for r in rs if r['all_initial_history_preserved']]
            s = dict(all_trajectories=len({r['trajectory'] for r in rs}),
                     stable_identity_trajectories=len({r['trajectory'] for r in fixed}))
            for field in ['frozen_initial', 'refresh_initial', 'oracle_initial']:
                s[field] = curve(fixed, 'call', field)
            for field in ['frozen_current', 'refresh_current', 'oracle_current',
                          'initial_share_of_current', 'selected_survival']:
                s[field] = curve(rs, 'call', field)
            paired = []
            for tid in sorted({r['trajectory'] for r in fixed}):
                rr = sorted([r for r in fixed if r['trajectory'] == tid], key=lambda r: r['call'])
                paired.append(dict(trajectory=tid, first=rr[0]['frozen_initial'],
                    last=rr[-1]['frozen_initial'], change=rr[-1]['frozen_initial']-rr[0]['frozen_initial'],
                    minimum=min(r['frozen_initial'] for r in rr),
                    every_observed_call_ge90=float(all(r['frozen_initial'] >= .9 for r in rr)),
                    every_observed_call_ge80=float(all(r['frozen_initial'] >= .8 for r in rr))))
            s['paired_initial'] = {key: cluster_stat(paired, key) for key in
                                    ['first', 'last', 'change', 'minimum', 'every_observed_call_ge90', 'every_observed_call_ge80']}
            result['agent'][tag][dom] = s
        pair_strata = {}
        for kind in ['all', 'same_session', 'different_session']:
            rs = [r for r in dialogue if r['fraction'] == f and
                  (kind == 'all' or r['session_transition'] == kind)]
            pairs = []
            for pid in sorted({r['pair'] for r in rs}):
                rr = sorted([r for r in rs if r['pair'] == pid], key=lambda r: r['phase'])
                assert [r['phase'] for r in rr] == list(range(1, 9))
                old, new = rr[:4], rr[4:]
                pairs.append(dict(trajectory=rr[0]['trajectory'],
                    old_early=old[0]['frozen'], old_late=old[-1]['frozen'],
                    old_mean=mean([r['frozen'] for r in old]), new_mean=mean([r['frozen'] for r in new]),
                    within_change=old[-1]['frozen']-old[0]['frozen'],
                    boundary_change=new[0]['frozen']-old[-1]['frozen'],
                    mean_change=mean([r['frozen'] for r in new])-mean([r['frozen'] for r in old]),
                    refreshed_new_mean=mean([r['refresh'] for r in new]),
                    refresh_gain=mean([r['refresh']-r['frozen'] for r in new])))
            result['dialogue'][tag][kind] = dict(pairs=len(pairs), trajectories=len({r['trajectory'] for r in rs}),
                frozen=curve(rs, 'phase', 'frozen'), refresh=curve(rs, 'phase', 'refresh'),
                paired={key: cluster_stat(pairs, key) for key in pairs[0] if key != 'trajectory'})
            pair_strata[kind] = pairs
        contrast = {}
        for field in ['mean_change', 'boundary_change', 'refresh_gain']:
            by_kind = {}
            for kind in ['same_session', 'different_session']:
                grouped = defaultdict(list)
                for row in pair_strata[kind]:
                    grouped[row['trajectory']].append(row[field])
                by_kind[kind] = {tid: mean(values) for tid, values in grouped.items()}
            common = sorted(set(by_kind['same_session']) & set(by_kind['different_session']))
            contrast[field] = bootstrap([by_kind['different_session'][tid]-by_kind['same_session'][tid]
                                         for tid in common])
        result.setdefault('paired_session_contrast', {})[tag] = contrast
    return result


def report(out, summary):
    lines = ['# Query 开始时选一次集合：现有记录的严格冻结分析', '',
        '仅使用 corrected80 已保存的 attention/input-token 记录，未运行模型。', '',
        '## 方法与图的含义', '',
        '- H₀：首次调用中，在最后 64 个 prefill query positions 之前的完整 128-token 块，去掉首个 sink 块。'
        '主图始终在同一 H₀ 上归一化，后续不缩小分母。',
        '- 选择时点：首次调用最后 64 个 prefill positions 的平均 attention。取 top-B 块；'
        '之后全部 decode attention 作为未参与选择的评估数据。',
        '- 固定预算：B=ceil(f×|H₀|)，f=25%/50%/75%/100%。同一条曲线的整数 B 永远不变；'
        '不同轨迹按初始长度归一化，因此 B 可以不同。50% 在计算结果前选为主展示预算。',
        '- 蓝线：同一 S₀ 从头沿用；橙线：每个 agent 调用/新的 dialogue query 使用该次 prefill 在相同 H₀ 上重新选 B 块。'
        '橙线是因果信息可用的离线选集，并非 oracle，也不是实际剪枝后的模型运行。',
        '- 上排：Software/GUI 首次记录正好是 query 第一次调用，最多记录 12 个调用；'
        '没有记录到任务完成就不能称为覆盖整个 query。只纳入 H₀ 在所有记录中 token/mRoPE/图片身份保持一致的轨迹。',
        '- 下排：Dialogue 中每个新 user 消息定义 query，同一 query 内额外 assistant 调用合并。'
        '冻结起点是旧 query 的首次 prefill，并贯穿旧 query 和紧接着的新 query；'
        '每个 query 的 decode positions 按进度划成四份，因此横轴不是工具 sub-step。',
        '- 上下排来自不同数据，不能接成一条“多步任务→新目标”的真实实验。'
        'source session 来自原始 LongMemEval 会话标签，切换时 replay context 没有清空；它不是人工标注的目标变化。',
        '- 该 H₀ 定义不同于原 KL 图去除末尾 8 块的 support；首次软件 prompt 较短，沿用 8 块 recent 排除会让 3 条初始历史为空、其余仅 1–5 块。'
        '本图保留在 selector 之前可见的完整块，以测 query 起点的集合复用；不能直接与原 KL 绝对值比较。',
        '- 统计：先对同一 trajectory 的 query pairs 求均值，再等权聚合 trajectory；5000 次 bootstrap，seed=20260911。'
        '图中区间是逐点的探索性 95% CI，未作多重比较调整。', '',
        '## 同一 query 的初始历史覆盖率（主图，B=50%）', '',
        '| 数据 | 身份完整轨迹 | 初始覆盖 | 最后观测覆盖 | 配对变化 pp [95% CI] | 所有观测调用均 ≥90% 的比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for dom in DOMAIN_NAMES:
        s = summary['agent']['50%'][dom]; p = s['paired_initial']; v = p['change']
        lines.append(f"| {dom} | {s['stable_identity_trajectories']}/20 | {p['first']['mean']*100:.2f}% | "
                     f"{p['last']['mean']*100:.2f}% | {v['mean']*100:+.2f} {np.round(np.array(v['ci95'])*100,2).tolist()} | "
                     f"{p['every_observed_call_ge90']['mean']*100:.1f}% |")
    lines += ['', 'Tool/state 起点在任务中途，只作为补充诊断；主图上排不将它标成 query 起点。', '',
        '## 跨真实 query 的同一集合（所有预算）', '',
        '| B/初始候选数 | 边界 | pair 数 | 旧 query 平均 | 新 query 旧集合 | 均值变化 pp [95% CI] | 新 prefill 重选增益 pp [95% CI] |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for tag, strata in summary['dialogue'].items():
        for kind, s in strata.items():
            p = s['paired']; change = p['mean_change']; gain = p['refresh_gain']
            lines.append(f"| {tag} | {kind} | {s['pairs']} | {p['old_mean']['mean']*100:.2f}% | "
                         f"{p['new_mean']['mean']*100:.2f}% | {change['mean']*100:+.2f} "
                         f"{np.round(np.array(change['ci95'])*100,2).tolist()} | {gain['mean']*100:+.2f} "
                         f"{np.round(np.array(gain['ci95'])*100,2).tolist()} |")
    comparison = summary['paired_session_contrast']['50%']['mean_change']
    lines += ['', f"在同一批 20 条 trajectory 内做配对对照，跨 session 的 query 均值变化比同 session 额外下降 "
        f"{-comparison['mean']*100:.2f} pp，差值（different−same）95% CI="
        f"{np.round(np.array(comparison['ci95'])*100,2).tolist()} pp。该对照仍不是目标变化的因果实验。", '',
        '**边界处下降与两条回答均值之差必须分开：** summary 中 boundary_change 比较旧 query 最后 25% 与新 query 最初 25%；'
        'mean_change 比较两条 query 的四个进度段均值。后者包含回答内部的演化，不能标成边界瞬时下降。', '',
        '## 新增历史的对照图', '',
        '`query_freeze_expanding_history` 将分母改为各次调用在 prefill selector 之前的全部候选历史 H_t，'
        '包括新增工具结果和此前 recorded assistant 输出；所有 20 条轨迹都纳入。'
        '蓝线中失效的原始 ID 记零、不补位；橙线在 H_t 上以同样 B 重选；虚线是该 decode 的 top-B oracle 上界。'
        '另有 initial_share_of_current 和 selected_survival 列，分别记录初始历史质量占比和选集身份存活率。', '',
        '固定 H₀ 上高覆盖只表明初始内容内部的可复用性。它不意味着同一个不可变 active view 已经覆盖后续新增内容。'
        'attention coverage 也不能直接替代任务准确率。', '',
        '## 样本审计', '', '```json', json.dumps(summary['audit'], ensure_ascii=False, indent=2), '```', '',
        '## 复现', '', '```bash', '.venv/bin/python scripts/motivation_v3/analyze_query_frozen_trajectory.py', '```', '']
    (out/'README_zh.md').write_text('\n'.join(lines))
    caption = ('A fixed historical selection across observed steps and user queries. '
        'At the first prefill, we retain the top B blocks from an initial archive H0, '
        'where B = ceil(0.5 |H0|), and evaluate the same selection on later decode attention (blue). '
        'The budget and evaluation archive remain fixed along each trace. Orange curves '
        'reselect B blocks using the prefill of each model call (top) or the next user query (bottom). '
        'The top panels contain 12 observed calls of one software/GUI task, restricted to trajectories '
        'preserving every initial block identity. The bottom panels independently align consecutive '
        'dialogue queries by decode progress; a source-session change is a proxy, not a labeled goal change. '
        'Shading denotes pointwise 95% trajectory-bootstrap confidence intervals. Later-added content '
        'is excluded from H0; its contribution is evaluated in the companion expanding-history figure. '
        'These offline measurements do not evaluate pruned-model task quality.\n')
    (out/'caption.txt').write_text(caption)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    out = args.out or args.root/'query_locality_20260923/query_frozen_trajectory'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root/'manifest.json'
    label_path = args.root/'query_locality_20260923/boundary_labels.csv'
    manifest = json.loads(manifest_path.read_text())
    with label_path.open() as stream:
        labels = {(r['trajectory'], int(r['to_step'])): r for r in csv.DictReader(stream)}
    fingerprints = {str(p.resolve()): digest(p) for p in
                    [manifest_path, label_path, Path(__file__), Path(__file__).with_name('metrics.py'),
                     Path(__file__).with_name('analyze_query_locality.py')]}
    agent_rows, dialogue_rows, selection_rows, audit_rows = [], [], [], []
    dialogue_exclusions = Counter()
    call_count = 0
    for trajectory in manifest['trajectories']:
        tid, dom = trajectory['id'], trajectory['domain']
        calls = load_calls(args.root, trajectory, fingerprints)
        call_count += len(calls)
        if dom != 'dialogue':
            first = calls[0]; end = first['end']; n = end-1
            limits = [prefix(first, call)//128 for call in calls]
            preserved = all(p >= end for p in limits)
            audit_rows.append(dict(trajectory=tid, domain=dom, initial_blocks=n,
                calls=len(calls), source_start=first['frame']['source_step'],
                all_initial_history_preserved=preserved, minimum_verified_prefix_blocks=min(limits)))
            for frac in FRACTIONS:
                budget = int(np.ceil(n*frac))
                selected = order(first['prefill'], end)[:budget]
                selection_rows.append(dict(trajectory=tid, anchor_call=0, fraction=frac, budget_blocks=budget,
                                           candidate_blocks=n, block_ids=selected.tolist()))
                for i, call in enumerate(calls):
                    current_end = call['end']; v = call['average']
                    survivors = selected[selected < limits[i]]
                    valid_initial_end = min(end, limits[i], current_end)
                    refreshed = order(call['prefill'], current_end)[:budget]
                    row = dict(trajectory=tid, domain=dom, call=i+1, source_step=call['frame']['source_step'],
                        fraction=frac, budget_blocks=budget, initial_blocks=n, current_blocks=current_end-1,
                        all_initial_history_preserved=preserved, verified_prefix_blocks=limits[i],
                        selected_survival=len(survivors)/budget,
                        frozen_current=mass_coverage(v, survivors, current_end),
                        refresh_current=mass_coverage(v, refreshed, current_end),
                        oracle_current=oracle(v, budget, current_end),
                        initial_share_of_current=float(v[1:valid_initial_end].sum()/v[1:current_end].sum()),
                        frozen_initial=None, refresh_initial=None, oracle_initial=None,
                        initial_mass_total_attention=float(v[1:valid_initial_end].sum()),
                        current_mass_total_attention=float(v[1:current_end].sum()))
                    if preserved:
                        row.update(frozen_initial=mass_coverage(v, selected, end),
                            refresh_initial=mass_coverage(v, order(call['prefill'], end)[:budget], end),
                            oracle_initial=oracle(v, budget, end))
                    agent_rows.append(row)
        else:
            groups = [[0]]
            for i, call in enumerate(calls[1:], 1):
                label = labels[(tid, call['frame']['source_step'])]
                if label['boundary_type'] == 'across_user_query':
                    groups.append([i])
                else:
                    groups[-1].append(i)
            # Each actual new query yields one pair, including rare multi-call queries.
            for old, new in zip(groups, groups[1:]):
                first, following = calls[old[0]], calls[new[0]]
                end = first['end']; n = end-1
                if any(prefix(first, calls[i]) < end*128 for i in old+new):
                    dialogue_exclusions['initial_identity_not_preserved'] += 1
                    continue
                if any(calls[i]['end'] < end for i in old+new):
                    dialogue_exclusions['current_support_smaller'] += 1
                    continue
                label = labels[(tid, following['frame']['source_step'])]
                pid = f'{tid}/{old[0]}-{new[0]}'
                chunks = []
                for phase_group in [old, new]:
                    dec = np.concatenate([calls[i]['decode'][:, :end] for i in phase_group])
                    assert len(dec) >= 4
                    chunks.extend(part.mean(0) for part in np.array_split(dec, 4))
                for frac in FRACTIONS:
                    budget = int(np.ceil(n*frac))
                    selected = order(first['prefill'], end)[:budget]
                    refreshed = order(following['prefill'], end)[:budget]
                    selection_rows.append(dict(trajectory=tid, anchor_call=old[0], pair=pid, fraction=frac,
                        budget_blocks=budget, candidate_blocks=n, block_ids=selected.tolist(),
                        refreshed_block_ids=refreshed.tolist()))
                    for phase, v in enumerate(chunks, 1):
                        dialogue_rows.append(dict(trajectory=tid, pair=pid,
                            old_first_source_step=first['frame']['source_step'],
                            new_first_source_step=following['frame']['source_step'],
                            old_calls=len(old), new_calls=len(new),
                            session_transition=label['session_transition'], fraction=frac,
                            budget_blocks=budget, initial_blocks=n, phase=phase,
                            frozen=mass_coverage(v, selected, end),
                            refresh=mass_coverage(v, selected if phase <= 4 else refreshed, end),
                            oracle=oracle(v, budget, end), initial_mass_total_attention=float(v[1:end].sum())))
        print(tid, 'done', flush=True)

    assert call_count == 957
    # Metric invariants exercise all calls/budgets, rather than just displayed means.
    for row in agent_rows:
        for suffix in ['initial', 'current']:
            if row[f'frozen_{suffix}'] is None:
                continue
            assert 0 <= row[f'frozen_{suffix}'] <= row[f'oracle_{suffix}']+1e-10 <= 1+1e-9
            assert 0 <= row[f'refresh_{suffix}'] <= row[f'oracle_{suffix}']+1e-10
        if row['fraction'] == 1 and row['all_initial_history_preserved']:
            assert abs(row['frozen_initial']-1) < 1e-10
        if row['call'] == 1:
            assert abs(row['frozen_current']-row['refresh_current']) < 1e-10
    for row in dialogue_rows:
        assert 0 <= row['frozen'] <= row['oracle']+1e-10 <= 1+1e-9
        assert 0 <= row['refresh'] <= row['oracle']+1e-10
        if row['fraction'] == 1:
            assert abs(row['frozen']-1) < 1e-10 and abs(row['refresh']-1) < 1e-10
    audit = dict(calls=call_count, new_model_calls=0, agent_trajectories=60,
        primary_fraction=.5, fractions=FRACTIONS, dialogue_exclusions=dict(dialogue_exclusions),
        dialogue_pairs=len({r['pair'] for r in dialogue_rows}),
        agent_identity_exclusions=[r for r in audit_rows if not r['all_initial_history_preserved']],
        main_figure_agent_domains=['code', 'gui'], tool_state_starts_mid_task=True)
    summary = build_summary(agent_rows, dialogue_rows, audit)
    write_csv(out/'agent_call_coverage.csv', agent_rows)
    write_csv(out/'dialogue_query_progress.csv', dialogue_rows)
    write_csv(out/'trajectory_audit.csv', audit_rows)
    (out/'frozen_selections.json').write_text(json.dumps(selection_rows, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    render(out, agent_rows, dialogue_rows, summary)
    report(out, summary)
    (out/'provenance.json').write_text(json.dumps(dict(input_sha256=fingerprints,
        new_model_calls=0, selector='initial prefill only; no decode selection',
        validation=['all probability/oracle bounds', '100% H0 coverage sanity control',
                    'prefill/decode disjoint', 'token/mRoPE/image prefix identity',
                    'initial frozen equals refreshed', '957 saved calls read']), indent=2)+'\n')
    compact = dict(agent={d: summary['agent']['50%'][d]['paired_initial'] for d in DOMAIN_NAMES},
                   dialogue={k: v['paired'] for k, v in summary['dialogue']['50%'].items()})
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
