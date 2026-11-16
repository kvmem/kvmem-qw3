"""Measure top-block identity overlap on the audited frozen attention cohort.

Common-support boundary windows, whole-call means, and each-call's own history
are separate estimands. Source measurements are read-only. No model execution.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from metrics import bootstrap, kl, mean

FRACTIONS = (.1, .2, .3)
WINDOWS = (16, 32, 64)
DOMAINS = ('code', 'tool_state', 'dialogue', 'gui')
NAMES = {'all': '全部', 'agents': 'Agent 合计', 'code': '软件 Agent',
         'tool_state': '工具/状态', 'dialogue': '对话', 'gui': 'GUI Agent',
         'same_user_query': '同一任务/query', 'across_user_query': '新 user query',
         'same_session': '对话：同 session', 'different_session': '对话：跨 session'}
SEED = 20260911


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive(scores):
    return len(scores) > 0 and float(np.sum(scores, dtype=np.float64)) > 0


def selection(scores, ids, fraction):
    if not positive(scores):
        raise ValueError('Undefined selection: empty or zero-mass history')
    # Source order is increasing position. Stable sort matches archive tie policy.
    k = math.ceil(fraction * len(ids))
    ranked = np.argsort(-scores, kind='stable')
    cutoff_ties = (int(np.count_nonzero(scores == scores[ranked[k-1]]))
                   if k < len(ids) and scores[ranked[k-1]] == scores[ranked[k]] else 0)
    return set(map(int, ids[ranked[:k]])), k, cutoff_ties


def compare(p, q, ids, fraction):
    a, k, ties_a = selection(p, ids, fraction)
    b, kb, ties_b = selection(q, ids, fraction)
    assert k == kb and len(a) == len(b) == k
    intersection = len(a & b)
    overlap = intersection / k
    baseline = k / len(ids)
    p64, q64 = p.astype(np.float64), q.astype(np.float64)
    p64, q64 = p64 / p64.sum(), q64 / q64.sum()
    old_mask, new_mask = np.isin(ids, list(a)), np.isin(ids, list(b))
    coverage, oracle = float(q64[old_mask].sum()), float(q64[new_mask].sum())
    assert oracle >= coverage - 1e-7
    return dict(intersection_blocks=intersection, selected_blocks=k,
                overlap=overlap, replacement=1-overlap,
                jaccard=intersection/(2*k-intersection), random_overlap=baseline,
                chance_adjusted_overlap=(overlap-baseline)/(1-baseline) if baseline < 1 else None,
                saturated=k == len(ids),
                earlier_topk_coverage=float(p64[old_mask].sum()),
                frozen_next_coverage=coverage, next_oracle_coverage=oracle,
                next_oracle_gap=oracle-coverage,
                mass_overlap=float(np.minimum(p64, q64).sum()), cross_kl=kl(p, q),
                earlier_cutoff_ties=ties_a, later_cutoff_ties=ties_b), a, b


def own_history(old, new, old_scores, new_scores, fraction):
    old_columns, new_columns = np.flatnonzero(old['history']), np.flatnonzero(new['history'])
    p, q = old_scores[old_columns], new_scores[new_columns]
    if not positive(p) or not positive(q):
        return None
    old_ids, new_ids = old['ids'][old_columns], new['ids'][new_columns]
    a, ka, ta = selection(p, old_ids, fraction)
    b, kb, tb = selection(q, new_ids, fraction)
    old_h, new_h = set(map(int, old_ids)), set(map(int, new_ids))
    old_visible, new_visible = set(map(int, old['ids'])), set(map(int, new['ids']))
    overlap = len(a & b)
    lost_identity = len(a - new_visible)
    lost_eligibility = len((a & new_visible) - new_h)
    demoted = len((a & new_h) - b)
    newly_matched_identity = len(b - old_visible)
    newly_eligible = len((b & old_visible) - old_h)
    promoted = len((b & old_h) - a)
    assert overlap + lost_identity + lost_eligibility + demoted == ka
    assert overlap + newly_matched_identity + newly_eligible + promoted == kb
    q64 = q.astype(np.float64)
    q64 /= q64.sum()
    surviving = a & new_h
    kept = np.isin(new_ids, list(surviving))
    current = np.isin(new_ids, list(b))
    return dict(earlier_history_blocks=len(old_h), later_history_blocks=len(new_h),
                shared_eligible_blocks=len(old_h & new_h), earlier_selected_blocks=ka,
                later_selected_blocks=kb, intersection_blocks=overlap,
                earlier_retention=overlap/ka, next_set_reuse=overlap/kb,
                jaccard=overlap/(ka+kb-overlap), selected_identity_survival=len(a & new_visible)/ka,
                selected_eligibility_survival=len(surviving)/ka,
                earlier_unmatched_identity_blocks=lost_identity,
                earlier_ineligible_blocks=lost_eligibility, earlier_demoted_blocks=demoted,
                later_unmatched_identity_blocks=newly_matched_identity,
                later_newly_eligible_blocks=newly_eligible, later_promoted_blocks=promoted,
                earlier_unmatched_identity_fraction=lost_identity/ka,
                earlier_ineligible_fraction=lost_eligibility/ka,
                earlier_demoted_fraction=demoted/ka,
                later_unmatched_identity_fraction=newly_matched_identity/kb,
                later_newly_eligible_fraction=newly_eligible/kb,
                later_promoted_fraction=promoted/kb,
                frozen_next_coverage=float(q64[kept].sum()),
                next_oracle_coverage=float(q64[current].sum()),
                earlier_cutoff_ties=ta, later_cutoff_ties=tb,
                earlier_topk_ids=json.dumps(sorted(a)), later_topk_ids=json.dumps(sorted(b)))


def subset(rows, group):
    if group == 'all':
        return rows
    if group == 'agents':
        return [r for r in rows if r['domain'] != 'dialogue']
    if group in DOMAINS:
        return [r for r in rows if r['domain'] == group]
    if group in ('same_user_query', 'across_user_query'):
        return [r for r in rows if r['boundary_type'] == group]
    return [r for r in rows if r['domain'] == 'dialogue' and r['session_transition'] == group]


def cluster_stats(rows, fields):
    result = dict(trajectories=len({r['trajectory'] for r in rows}), boundaries=len(rows))
    for field in fields:
        grouped = defaultdict(list)
        for r in rows:
            if r.get(field) is not None:
                grouped[r['trajectory']].append(r[field])
        result[field] = bootstrap([mean(v) for _, v in sorted(grouped.items())], seed=SEED)
    return result


def summarize(rows, fields, groups):
    summary = {}
    keys = sorted({(r['window'], r['fraction']) for r in rows})
    for window, fraction in keys:
        matched = [r for r in rows if r['window'] == window and r['fraction'] == fraction]
        key = f'w{window}_r{int(round(fraction*100)):02d}'
        summary[key] = {g: cluster_stats(subset(matched, g), fields) for g in groups}
        # Preserve the full primary cohort, and report separate small-history,
        # natural-end, and common-window-cohort controls.
        for name, selected in [('history_ge20', [r for r in matched if r['common_blocks'] >= 20]),
                               ('natural_end_only', [r for r in matched if not r['either_truncated']]),
                               ('matched_w64_cohort', [r for r in matched if r['eligible_w64']])]:
            summary[key][name] = cluster_stats(selected, fields)
    return summary


def self_check():
    ids = np.arange(100)
    p = np.r_[np.arange(20, 0, -1), np.zeros(80)]
    q = np.zeros(100)
    q[5:25] = np.arange(20, 0, -1)
    s, _, _ = compare(p, q, ids, .2)
    assert s['intersection_blocks'] == 15 and s['overlap'] == .75 and s['jaccard'] == .6
    identical, _, _ = compare(p, p, ids, .2)
    assert identical['overlap'] == 1 and identical['next_oracle_gap'] == 0
    disjoint, _, _ = compare(p, np.roll(p, 20), ids, .2)
    assert disjoint['overlap'] == 0
    tie, _, _ = compare(np.ones(100), np.ones(100), ids, .2)
    assert tie['earlier_cutoff_ties'] == 100
    assert not positive(np.zeros(100))
    singleton, _, _ = compare(np.ones(1), np.ones(1), np.array([1]), .2)
    assert singleton['saturated'] and singleton['chance_adjusted_overlap'] is None
    old = dict(ids=np.arange(10), history=np.ones(10, dtype=bool))
    new = dict(ids=np.arange(5, 25), history=np.ones(20, dtype=bool))
    own = own_history(old, new, np.arange(10.), np.arange(20., 0, -1), .2)
    assert own['intersection_blocks'] == 1 and own['earlier_retention'] == .5
    assert own['next_set_reuse'] == .25


def validate_archive(root, rows, whole_rows, own_rows, fingerprints):
    reference = root/'query_locality_20260923/boundary_budget_coverage.csv'
    locality = root/'query_locality_20260923/boundary_locality.csv'
    fingerprints[str(reference)], fingerprints[str(locality)] = digest(reference), digest(locality)
    with reference.open() as stream:
        saved = {(r['trajectory'],int(r['from_step']),int(r['to_step'])):r
                 for r in csv.DictReader(stream) if r['budget']=='10%'}
    with locality.open() as stream:
        kl_saved = {(r['trajectory'],int(r['from_step']),int(r['to_step'])):r
                    for r in csv.DictReader(stream)}
    max_kl_error, max_coverage_error = 0., 0.
    archived_count = 0
    for r in rows:
        for name in ('overlap','replacement','jaccard','random_overlap','frozen_next_coverage',
                     'next_oracle_coverage','within_overlap'):
            assert -1e-7 <= r[name] <= 1+1e-7
        if r['window']!=32 or r['fraction']!=.1:
            continue
        key = r['trajectory'],r['from_step'],r['to_step']
        ref, refkl = saved[key], kl_saved[key]
        assert r['common_blocks'] == int(ref['common_blocks'])
        assert r['overlap'] == float(ref['topk_overlap'])
        max_coverage_error = max(max_coverage_error,abs(r['frozen_next_coverage']-float(ref['frozen_coverage'])))
        max_kl_error = max(max_kl_error,abs(r['cross_kl']-float(refkl['cross_kl'])),
                           abs(r['within_kl']-float(refkl['within_kl'])))
        archived_count += 1
    assert archived_count == 791 and max_kl_error < 1e-6 and max_coverage_error < 1e-6

    # Independent reconstruction from the original query matrices, using Python
    # sorting and set intersections instead of the exported selection helper.
    source = root/'block_attention_export_20260913'
    cache = {}
    reconstructed = 0
    for r in rows:
        if r['window'] != 32 or r['fraction'] != .2:
            continue
        ids = json.loads(r['earlier_topk_ids'])
        calls = []
        for step in (r['from_step_id'],r['to_step_id']):
            key = r['trajectory'],step
            if key not in cache:
                meta = json.loads((source/'trajectories'/key[0]/f'step_{step:03d}.json').read_text())
                raw_path = Path(meta['source_record']['path'])
                actual = digest(raw_path)
                assert actual == meta['source_npz_sha256']
                fingerprints[str(raw_path)] = actual
                with np.load(raw_path,allow_pickle=False) as z:
                    raw_mass, pos = z['mass'],z['positions']
                cache[key] = raw_mass[pos >= meta['prompt_tokens']]
            calls.append(cache[key])
        # The archive's verified common support is the contiguous position range
        # [1, 1+B); independently check against the exported identity arrays.
        n,k = r['common_blocks'],r['selected_blocks']
        p = calls[0][-32:,1:1+n].mean(0)
        q = calls[1][:32,1:1+n].mean(0)
        a = set(sorted(range(n),key=lambda i:(-float(p[i]),i))[:k])
        b = set(sorted(range(n),key=lambda i:(-float(q[i]),i))[:k])
        assert len(a & b) == r['intersection_blocks']
        assert len(ids) == k
        reconstructed += 1
    assert reconstructed == 791
    for r in own_rows:
        assert abs(r['earlier_retention'] + r['earlier_unmatched_identity_fraction'] +
                   r['earlier_ineligible_fraction'] + r['earlier_demoted_fraction'] - 1) < 1e-12
        assert abs(r['next_set_reuse'] + r['later_unmatched_identity_fraction'] +
                   r['later_newly_eligible_fraction'] + r['later_promoted_fraction'] - 1) < 1e-12
    return dict(archived_w32_r10_boundaries_reproduced=archived_count,
                max_archived_kl_absolute_error=max_kl_error,
                max_archived_coverage_absolute_error=max_coverage_error,
                independent_original_matrix_w32_r20_reconstructions=reconstructed,
                paired_rows=len(rows),whole_step_rows=len(whole_rows),own_history_rows=len(own_rows),
                own_history_partition_checks=len(own_rows)*2)


def render(out, summary, rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/qw3-topk-overlap-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':10,
                         'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    blue,orange,grey = '#2563eb','#ea580c','#64748b'
    def save(fig,name):
        for suffix in ('png','pdf','svg'):
            fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def bar(ax,x,stat,color,label=None,width=.32):
        lo,hi = stat['ci95']
        v = stat['mean']*100
        err = [[v-100*lo],[100*hi-v]] if lo is not None else None
        ax.bar(x,v,width=width,color=color,label=label,yerr=err,capsize=3,alpha=.9)
        ax.text(x,v+2.4,f'{v:.1f}%',ha='center',fontsize=8,color=color)
    def decorate(ax):
        ax.set_ylim(0,108)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    groups = ['code','tool_state','gui','dialogue','all']
    primary = summary['paired']['w32_r20']
    fig,axes = plt.subplots(1,2,figsize=(12.2,4.4))
    for i,g in enumerate(groups):
        s=primary[g]
        bar(axes[0],i-.18,s['within_overlap'],blue,'步内' if i==0 else None)
        bar(axes[0],i+.18,s['overlap'],orange,'跨 step' if i==0 else None)
        bar(axes[1],i-.18,s['frozen_next_coverage'],blue,'上一选集在下一窗口的覆盖率' if i==0 else None)
        bar(axes[1],i+.18,s['next_oracle_coverage'],orange,'下一窗口重新选 top-20%' if i==0 else None)
    for ax in axes:
        decorate(ax)
        ax.set_xticks(range(len(groups)),[NAMES[g]+f"\n{primary[g]['boundaries']} 边界" for g in groups],fontsize=9)
        ax.legend(loc='upper left',fontsize=8)
    axes[0].set_title('(a) top-20% 集合重叠率',loc='left',fontweight='bold')
    axes[0].set_ylabel('交集 / top-k 数量 (%)')
    axes[1].set_title('(b) 同一选集在下一窗口仍覆盖多少 attention',loc='left',fontweight='bold')
    axes[1].set_ylabel('共同历史 attention mass 覆盖率 (%)')
    fig.suptitle('相邻 step 的高 attention KV block 重叠：20% 预算，32-query 窗口',fontsize=13,y=1.03)
    fig.text(.5,-.06,'同一已核验共同历史；128-token blocks；轨迹等权均值与 95% bootstrap 区间。\n步内：边界两侧各一对相邻窗口；覆盖率分母仅为共同历史 attention。',ha='center',fontsize=9,color=grey)
    fig.tight_layout()
    save(fig,'top20_overlap')

    fig,axes = plt.subplots(1,2,figsize=(11.2,4.4))
    colors = dict(zip(DOMAINS,['#2563eb','#059669','#9333ea','#ea580c']))
    for ax in axes:
        decorate(ax)
        ax.set_ylabel('跨 step 集合重叠率 (%)')
    for g in DOMAINS:
        for ax,settings,xs,linestyle,label in [
            (axes[0],[f'w32_r{int(f*100):02d}' for f in FRACTIONS],[10,20,30],'-',NAMES[g]),
            (axes[1],[f'w{w}_r20' for w in WINDOWS],list(WINDOWS),'-',NAMES[g]),
            (axes[1],[f'w{w}_r20' for w in WINDOWS],list(WINDOWS),'--',None)]:
            values = [summary['paired'][s][g if linestyle=='-' else 'matched_w64_cohort'] for s in settings]
            if linestyle=='--':
                # Same trajectories/boundaries across windows, within domain.
                values = [cluster_stats([r for r in rows if r['domain']==g and r['eligible_w64'] and
                    r['window']==w and r['fraction']==.2],['overlap']) for w in WINDOWS]
            ys=[100*v['overlap']['mean'] for v in values]
            ax.plot(xs,ys,linestyle,marker='o' if linestyle=='-' else '.',color=colors[g],label=label,linewidth=1.5)
            if linestyle=='-':
                ax.fill_between(xs,[100*v['overlap']['ci95'][0] for v in values],
                                [100*v['overlap']['ci95'][1] for v in values],color=colors[g],alpha=.1)
    axes[0].set(xticks=[10,20,30],xlabel='选集占候选 block 的比例 (%)',title='(a) 预算敏感性：W=32，同一批边界')
    axes[1].set(xticks=WINDOWS,xlabel='decode query 窗口长度',title='(b) 窗口敏感性：20% 预算')
    axes[0].legend(fontsize=9)
    axes[1].text(.02,.04,'实线：各窗口所有合格边界\n虚线：固定 W=64 合格边界',transform=axes[1].transAxes,fontsize=8,color=grey)
    fig.text(.5,-.03,'轨迹等权均值；阴影为 95% trajectory-bootstrap 区间。窗口长度变化可能改变样本资格，虚线用于控制这一影响。',ha='center',fontsize=9)
    fig.tight_layout()
    save(fig,'sensitivity')

    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for i,g in enumerate(groups):
        bar(axes[0],i-.18,summary['paired']['w32_r20'][g]['overlap'],blue,'边界窗口' if i==0 else None)
        bar(axes[0],i+.18,summary['whole_step_matched_w32'][g]['overlap'],orange,'整步均值' if i==0 else None)
        s=summary['own_history']['w32_r20'][g]
        bar(axes[1],i-.18,s['earlier_retention'],blue,'交集 / 上一步选集' if i==0 else None)
        bar(axes[1],i+.18,s['next_set_reuse'],orange,'交集 / 下一步选集' if i==0 else None)
    for ax in axes:
        decorate(ax)
        ax.set_xticks(range(len(groups)),[NAMES[g] for g in groups],fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylabel('集合重叠比例 (%)')
    axes[0].set_title('(a) 共同历史：边界窗口 vs 整步平均',loc='left',fontweight='bold')
    axes[1].set_title('(b) 两侧各自在完整候选历史选 top-20%',loc='left',fontweight='bold')
    fig.text(.5,-.05,'(a) 两种窗口定义使用相同的 791 个边界；共同历史与 top-k 预算一致。\n(b) 候选历史和预算可能变化，因此两个分母分别报告；新增/无法匹配身份的块计入变化。',ha='center',fontsize=9,color=grey)
    fig.tight_layout()
    save(fig,'whole_step_and_own_history')


def write_report(out, summary, validation):
    def fmt(stat):
        if stat['mean'] is None:
            return '无有效样本'
        low,high=stat['ci95']
        return f"{100*stat['mean']:.2f}%"+(f" [95% CI {100*low:.2f}, {100*high:.2f}]" if low is not None else '')
    lines=['Top-20% KV block 跨 step 重叠统计', '',
        '数据：corrected80；80 条独立源轨迹，957 次调用；Qwen3.8-27B-Q8_0，FP16 KV。',
        '这是记录输入 replay 的完整可见 attention 离线统计；不包含真实 KVMem 选择或任务质量实验。',
        '仅包含 16 个标准 attention 层的聚合分数，不包含 hybrid linear-state 层。', '',
        '主口径：相邻真实源 step 的已核验共同历史；完整 128-token blocks，排除首块与两侧最近 8 个 prompt blocks。',
        '分数：窗口原 attention mass 的 query 平均。每侧先在相同共同历史选 k=ceil(0.2*B)，然后求集合交集。',
        '跨步：上一调用最后 32 个 query vs 下一调用最前 32 个 query。',
        '步内：上一调用最后两个不重叠窗口、下一调用最前两个不重叠窗口的重叠率取平均。',
        '两侧都至少 64 个 decode query；使用已记录 query 行数，避免输出 token 数与 forward 次数混淆。',
        '主重叠率=交集/k；Jaccard=交集/并集；替换率=1-主重叠率。',
        '先在轨迹内平均边界，再跨轨迹等权；95% CI 为轨迹 bootstrap 5000 次，seed=20260911。', '',
        'W=32，20% 预算：']
    for group in ['agents',*DOMAINS,'all']:
        s=summary['paired']['w32_r20'][group]
        lines += [f"{NAMES[group]}：{s['trajectories']} 轨迹 / {s['boundaries']} 边界",
                  '  跨步重叠 '+fmt(s['overlap'])+'；步内重叠 '+fmt(s['within_overlap']),
                  '  步内减跨步 '+fmt(s['within_minus_cross'])+'（此项单位为百分点）',
                  '  替换率 '+fmt(s['replacement'])+'；Jaccard '+fmt(s['jaccard']),
                  '  上一选集在下一窗口 coverage '+fmt(s['frozen_next_coverage'])+
                  '；下一窗口 oracle '+fmt(s['next_oracle_coverage']),
                  '  平均随机集合基线 '+fmt(s['random_overlap'])]
    lines += ['', '整步平均口径：每侧在全部 decode query 求平均；相邻两步仍使用同一已核验共同历史。']
    for group in ['agents',*DOMAINS,'all']:
        s=summary['whole_step']['w0_r20'][group]
        paired=summary['whole_step_matched_w32'][group]
        lines += [f"{NAMES[group]}：{s['boundaries']} 边界；整步重叠 {fmt(s['overlap'])}；"
                  f"固定 W=32 边界：{fmt(paired['overlap'])}"]
    lines += ['', '各自完整候选历史口径（边界 W=32）：先选后求交，保留两侧不同预算分母。']
    for group in ['agents',*DOMAINS,'all']:
        s=summary['own_history']['w32_r20'][group]
        lines += [f"{NAMES[group]}：上一步选集保留 {fmt(s['earlier_retention'])}；下一步选集复用 {fmt(s['next_set_reuse'])}",
                  '  上一选集：无法匹配身份 '+fmt(s['earlier_unmatched_identity_fraction'])+
                  '；不再合格 '+fmt(s['earlier_ineligible_fraction'])+'；仍合格但跌出排名 '+fmt(s['earlier_demoted_fraction']),
                  '  下一选集：无法匹配上一身份 '+fmt(s['later_unmatched_identity_fraction'])+
                  '；新进入历史资格 '+fmt(s['later_newly_eligible_fraction'])+'；旧候选排名上升 '+fmt(s['later_promoted_fraction'])]
    lines += ['', '敏感性：10%/20%/30% 预算、16/32/64-query 窗口。',
              '另外保存 history_ge20、natural_end_only、固定 W=64 合格边界的控制统计。',
              '短历史中向上取整可能使实际预算比例高于 20%，小至 k=B 时重叠必为 100%；完整样本保留并单列控制。',
              '整步资格与边界窗口资格不同，whole_step_matched_w32 保留与主统计完全相同的边界用于比较。',
              '汇总中的 p10/p90 是轨迹均值的分位数，不是逐 token 或逐边界的分位数。',
              '身份只能证明未改写完整共同前缀；身份不匹配不能直接等同实际内容被删除。',
              '软件/工具/GUI 边界主要属于同一任务，dialogue 包含真实新 user query；跨来源差异不作因果解释。', '',
              '验证：791 个 W=32/10% 边界重现已有 overlap、coverage 和 KL；791 个 20% 边界从原始 query 矩阵独立重算。',
              f"最大已有 KL 误差：{validation['max_archived_kl_absolute_error']:.3g}；最大 coverage 误差：{validation['max_archived_coverage_absolute_error']:.3g}。", '',
              '产物：top20_overlap.png/pdf/svg（主图），sensitivity.png/pdf/svg（敏感性），whole_step_and_own_history.png/pdf/svg（补充口径）。',
              'boundary_overlap.csv：逐边界、窗口、预算的重叠和覆盖率，包含两侧选中 IDs。',
              'whole_step_overlap.csv：整步均值；own_history_overlap.csv：各自历史与变化分解。',
              'trajectory_summary.csv：每轨迹聚合；summary.csv/json：汇总与区间。',
              'selected_sets.jsonl：四个窗口的选集和共同历史 IDs；validation.json：验证结果。',
              'provenance.json：分析口径与来源 SHA256；checksums.json：产物 SHA256。', '',
              '复现：在仓库根目录运行 .venv/bin/python scripts/motivation_v3/analyze_topk_overlap.py']
    (out/'统计说明.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2] /
                        'benchmark/results/motivation_v3_20260910/corrected80')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'block_attention_export_20260913'
    out = args.out.resolve() if args.out else root / 'topk_overlap_20261013'
    out.mkdir(parents=True, exist_ok=True)
    self_check()
    checksums = json.loads((source/'checksums.json').read_text())
    fingerprints = {}
    def verify(path):
        actual = digest(path)
        relative = str(path.relative_to(source))
        if relative in checksums and actual != checksums[relative]:
            raise ValueError(f'Source checksum mismatch: {relative}')
        fingerprints[str(path)] = actual
    for filename in ('index.csv', 'manifest.json', 'boundaries.json', 'audit.json', 'checksums.json'):
        verify(source/filename)
    manifest = json.loads((source/'manifest.json').read_text())
    boundaries = json.loads((source/'boundaries.json').read_text())
    labels_path = root/'query_locality_20260923/boundary_labels.csv'
    fingerprints[str(labels_path)] = digest(labels_path)
    with labels_path.open() as stream:
        labels = {(r['trajectory'], int(r['to_step'])): r for r in csv.DictReader(stream)}
    with (source/'index.csv').open(encoding='utf-8-sig') as stream:
        index = list(csv.DictReader(stream))
    indexed = {(r['trajectory_id'], int(r['step_id'])): r for r in index}
    by_trajectory = defaultdict(list)
    for b in boundaries:
        by_trajectory[b['trajectory_id']].append(b)
    independent = [t['independent_unit'] for t in manifest['trajectories']]
    assert len(independent) == len(set(independent)) == 80
    matched_rows, whole_rows, own_rows, selected_sets = [], [], [], []
    excluded = Counter()
    loaded_count, max_row_error = 0, 0.
    for trajectory in manifest['trajectories']:
        tid = trajectory['id']
        calls = {}
        for i in range(len(trajectory['frames'])):
            row = indexed[tid, i]
            metadata_path, npz_path = source/row['metadata_file'], source/row['npz_file']
            verify(metadata_path)
            verify(npz_path)
            meta = json.loads(metadata_path.read_text())
            verify(metadata_path.parent/meta['input_identity_file'])
            with np.load(npz_path, allow_pickle=False) as z:
                ids, mass = z['block_ids'], z['decode_block_mass']
                history, full = z['prompt_history_mask'], z['is_full_prompt_block']
                positions = z['decode_query_positions']
                common_mask = z['common_history_with_previous_mask']
            assert meta['trajectory_id'] == tid and meta['step_id'] == i
            assert meta['source_step_id'] == trajectory['frames'][i]['source_step']
            assert mass.dtype == np.float32 and np.isfinite(mass).all() and (mass >= 0).all()
            assert len(ids) == len(set(map(int, ids))) and history.shape == ids.shape
            assert (full[history]).all() and not history[0]
            assert len(positions) == len(mass) == meta['decode_queries']
            assert np.array_equal(positions, np.arange(meta['prompt_tokens'], meta['prompt_tokens']+len(mass)))
            error = float(np.abs(mass.sum(1)-1).max()) if len(mass) else 0.
            max_row_error = max(max_row_error, error)
            assert error < 1e-6
            calls[i] = dict(ids=ids, mass=mass, history=history, full=full, meta=meta,
                            common_mask=common_mask, mapping={int(b): j for j, b in enumerate(ids)})
            loaded_count += 1
        for boundary in by_trajectory[tid]:
            old, new = calls[boundary['from_step_id']], calls[boundary['to_step_id']]
            om, nm = old['meta'], new['meta']
            assert nm['source_step_id'] == om['source_step_id']+1
            assert nm['prefix_identity_verified']
            label = labels[tid, nm['source_step_id']]
            ids = np.asarray(boundary['common_block_ids'], dtype=np.int64)
            old_columns = np.asarray([old['mapping'][int(b)] for b in ids], dtype=np.int64)
            new_columns = np.asarray([new['mapping'][int(b)] for b in ids], dtype=np.int64)
            assert np.array_equal(new['ids'][new['common_mask']], ids)
            assert len(ids) == boundary['common_history_blocks']
            assert old['history'][old_columns].all() and new['history'][new_columns].all()
            common = dict(trajectory=tid, domain=trajectory['domain'], from_step=om['source_step_id'],
                          to_step=nm['source_step_id'], from_step_id=om['step_id'], to_step_id=nm['step_id'],
                          boundary_type=label['boundary_type'], session_transition=label['session_transition'],
                          common_blocks=len(ids), earlier_decode_queries=len(old['mass']),
                          later_decode_queries=len(new['mass']),
                          earlier_truncated=om['truncated'], later_truncated=nm['truncated'],
                          either_truncated=om['truncated'] or nm['truncated'],
                          eligible_w64=64 in boundary['eligible_windows'])
            for window in (0, *WINDOWS):
                if window:
                    if window not in boundary['eligible_windows']:
                        excluded[f'w{window}_not_matched_length_or_empty_history'] += 1
                        continue
                    p_all = old['mass'][-window:].mean(0)
                    q_all = new['mass'][:window].mean(0)
                else:
                    if not len(old['mass']) or not len(new['mass']):
                        excluded['whole_step_empty_decode'] += 1
                        continue
                    p_all, q_all = old['mass'].mean(0), new['mass'].mean(0)
                p, q = p_all[old_columns], q_all[new_columns]
                valid = positive(p) and positive(q)
                if window and valid:
                    before = old['mass'][-2*window:-window].mean(0)[old_columns]
                    after = new['mass'][window:2*window].mean(0)[new_columns]
                    valid = positive(before) and positive(after)
                if not valid:
                    excluded[f'w{window}_empty_or_zero_common_mass'] += 1
                for fraction in FRACTIONS:
                    if valid:
                        result, a, b = compare(p, q, ids, fraction)
                        entry = dict(**common, window=window, fraction=fraction,
                                     earlier_historical_mass=float(p.sum()), later_historical_mass=float(q.sum()),
                                     **result, earlier_topk_ids=json.dumps(sorted(a)), later_topk_ids=json.dumps(sorted(b)))
                        if window:
                            left, ba, _ = compare(before, p, ids, fraction)
                            right, _, ab = compare(q, after, ids, fraction)
                            within = (left['overlap']+right['overlap'])/2
                            entry.update(within_earlier_overlap=left['overlap'], within_later_overlap=right['overlap'],
                                         within_overlap=within, within_minus_cross=within-result['overlap'],
                                         cross_less_than_within=result['overlap'] < within,
                                         within_frozen_next_coverage=(left['frozen_next_coverage']+right['frozen_next_coverage'])/2,
                                         within_oracle_gap=(left['next_oracle_gap']+right['next_oracle_gap'])/2,
                                         within_kl=(left['cross_kl']+right['cross_kl'])/2,
                                         within_earlier_cutoff_ties=max(left['earlier_cutoff_ties'],left['later_cutoff_ties']),
                                         within_later_cutoff_ties=max(right['earlier_cutoff_ties'],right['later_cutoff_ties']))
                            matched_rows.append(entry)
                            selected_sets.append(dict(trajectory=tid, from_step=om['source_step_id'],
                                to_step=nm['source_step_id'], window=window, fraction=fraction,
                                common_block_ids=list(map(int,ids)), before=sorted(ba), tail=sorted(a),
                                head=sorted(b), after=sorted(ab)))
                        else:
                            whole_rows.append(entry)
                    own = own_history(old, new, p_all, q_all, fraction)
                    if own is not None:
                        own_rows.append(dict(**common, window=window, fraction=fraction, **own))
        print(f'{len(calls)} calls analyzed: {tid}; total calls {loaded_count}', flush=True)

    assert loaded_count == 957
    for w, expected in [(16, 833), (32, 791), (64, 597)]:
        assert sum(r['window']==w and r['fraction']==.2 for r in matched_rows) == expected
    write_csv(out/'boundary_overlap.csv', matched_rows)
    write_csv(out/'whole_step_overlap.csv', whole_rows)
    write_csv(out/'own_history_overlap.csv', own_rows)
    with (out/'selected_sets.jsonl').open('w') as stream:
        for row in selected_sets:
            stream.write(json.dumps(row)+'\n')
    fields = ['overlap','replacement','jaccard','random_overlap','chance_adjusted_overlap',
              'intersection_blocks','selected_blocks','frozen_next_coverage','next_oracle_coverage',
              'next_oracle_gap','earlier_historical_mass','later_historical_mass','cross_kl',
              'saturated','earlier_cutoff_ties','later_cutoff_ties']
    paired_fields = fields+['within_overlap','within_minus_cross','cross_less_than_within',
                            'within_frozen_next_coverage','within_oracle_gap','within_kl']
    own_fields = ['earlier_retention','next_set_reuse','jaccard','selected_identity_survival',
                  'selected_eligibility_survival','earlier_unmatched_identity_fraction',
                  'earlier_ineligible_fraction','earlier_demoted_fraction','later_unmatched_identity_fraction',
                  'later_newly_eligible_fraction','later_promoted_fraction','frozen_next_coverage','next_oracle_coverage']
    groups = ['all','agents',*DOMAINS,'same_user_query','across_user_query','same_session','different_session']
    summary = dict(primary_setting='w32_r20', aggregation='Boundary mean within trajectory, then equal-weight trajectory mean',
                   seed=SEED, bootstrap_replicates=5000,
                   paired=summarize(matched_rows,paired_fields,groups),
                   whole_step=summarize(whole_rows,fields,groups),
                   own_history=summarize(own_rows,own_fields,groups))
    primary_keys = {(r['trajectory'],r['from_step'],r['to_step']) for r in matched_rows
                    if r['window']==32 and r['fraction']==.2}
    whole_matched = [r for r in whole_rows if r['fraction']==.2 and
                     (r['trajectory'],r['from_step'],r['to_step']) in primary_keys]
    assert len(whole_matched) == len(primary_keys)
    summary['whole_step_matched_w32'] = {g:cluster_stats(subset(whole_matched,g),fields) for g in groups}
    write_json(out/'summary.json', summary)
    flattened = []
    for kind in ('paired','whole_step','own_history'):
        for setting, sections in summary[kind].items():
            for group, section in sections.items():
                for metric, stat in section.items():
                    if not isinstance(stat,dict):
                        continue
                    flattened.append(dict(analysis=kind,setting=setting,group=group,metric=metric,
                        trajectories=section['trajectories'],boundaries=section['boundaries'],
                        mean=stat['mean'],median=stat.get('median'),p10=stat.get('p10'),p90=stat.get('p90'),
                        ci95_low=stat['ci95'][0],ci95_high=stat['ci95'][1]))
    for group,section in summary['whole_step_matched_w32'].items():
        for metric,stat in section.items():
            if isinstance(stat,dict):
                flattened.append(dict(analysis='whole_step_matched_w32',setting='w0_r20',group=group,
                    metric=metric,trajectories=section['trajectories'],boundaries=section['boundaries'],
                    mean=stat['mean'],median=stat.get('median'),p10=stat.get('p10'),p90=stat.get('p90'),
                    ci95_low=stat['ci95'][0],ci95_high=stat['ci95'][1]))
    write_csv(out/'summary.csv',flattened)
    trajectory_rows = []
    for setting in sorted({(r['window'],r['fraction']) for r in matched_rows}):
        for trajectory in manifest['trajectories']:
            rows = [r for r in matched_rows if (r['window'],r['fraction'])==setting and r['trajectory']==trajectory['id']]
            if rows:
                trajectory_rows.append(dict(trajectory=trajectory['id'],domain=trajectory['domain'],
                    window=setting[0],fraction=setting[1],boundaries=len(rows),
                    **{f:mean([r[f] for r in rows]) for f in paired_fields}))
    write_csv(out/'trajectory_summary.csv',trajectory_rows)
    validation = validate_archive(root, matched_rows, whole_rows, own_rows, fingerprints)
    validation.update(status='passed',calls=loaded_count,trajectories=len(manifest['trajectories']),
                      verified_boundaries=len(boundaries),source_files_hashed=len(fingerprints),
                      max_mass_row_sum_error=max_row_error,excluded=dict(excluded),
                      synthetic_cases='75%/60% example, identical/disjoint sets, stable ties, zero mass, singleton, unequal budgets')
    write_json(out/'validation.json',validation)
    fingerprints[str(Path(__file__).resolve())] = digest(Path(__file__))
    metrics_path = Path(__file__).resolve().parent/'metrics.py'
    fingerprints[str(metrics_path)] = digest(metrics_path)
    write_json(out/'provenance.json',dict(source_files=fingerprints,
        protocol=dict(block_tokens=128,exclude_sink_blocks=1,exclude_recent_prompt_blocks=8,
            fractions=list(FRACTIONS),windows=list(WINDOWS),strict_pair_minimum_queries='2W on both sides',
            score='Mean raw float32 layer-aggregated block mass over decode window, then common-history conditioning',
            identity='Trajectory-scoped inherited IDs on independently verified full token/mRoPE/image prefix blocks',
            own_history='Select independently over prompt_history_mask; unequal budgets retain both denominators',
            ties='Stable descending score sort in position order',
            uncertainty='5000 bootstrap replicates on 80 distinct source trajectories; fixed seed 20260911',
            scope='Recorded-input replay, dense full-visible attention; standard-attention layers only; no actual KVMem selections')))
    render(out,summary,matched_rows)
    write_report(out,summary,validation)
    write_json(out/'checksums.json',{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='checksums.json'})
    print(json.dumps(summary['paired']['w32_r20']['all'],ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
