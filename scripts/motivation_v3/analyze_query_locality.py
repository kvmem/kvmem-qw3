"""Offline boundary locality audit on the frozen motivation-v3 cohort.

No generation or changes to source measurements. Fixed sets use the preceding
window only; shared intervals/sets and current top-k are explicitly hindsight
diagnostics. User-turn changes are distinguished from environment feedback.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from metrics import bootstrap, js, kl, mean, normalize

DOMAINS = ['code', 'tool_state', 'dialogue', 'gui']
BUDGETS = [('1K', 8), ('2K', 16), ('4K', 32), ('8K', 64),
           ('16K', 128), ('32K', 256), ('64K', 512)]
FRACTIONS = [('10%', .1), ('25%', .25), ('50%', .5)]


def gzread(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def message_text(message):
    content = message.get('content', '')
    if isinstance(content, list):
        return '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text').strip()
    return str(content or '').strip()


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def shared_interval(p, q, threshold):
    """Exact shortest contiguous interval containing threshold mass in BOTH."""
    cp, cq = np.r_[0., np.cumsum(p)], np.r_[0., np.cumsum(q)]
    best = None
    for start in range(len(p)):
        if cp[-1] - cp[start] < threshold - 1e-12 or cq[-1] - cq[start] < threshold - 1e-12:
            break
        end = max(int(np.searchsorted(cp, cp[start] + threshold - 1e-12)),
                  int(np.searchsorted(cq, cq[start] + threshold - 1e-12)))
        if end <= len(p) and (best is None or end - start < best[1] - best[0]):
            best = (start, end)
    assert best is not None
    start, end = best
    return dict(start_block=start + 1, stop_block=end + 1, blocks=end - start,
                tokens=(end - start) * 128, fraction=(end - start) / len(p),
                earlier_mass=float(p[start:end].sum()), later_mass=float(q[start:end].sum()))


def budget_metrics(p, q, k):
    k = min(k, len(p))
    before = np.argsort(-p, kind='stable')[:k]
    after = np.argsort(-q, kind='stable')[:k]
    cp, cq = np.r_[0., np.cumsum(p)], np.r_[0., np.cumsum(q)]
    ip, iq = cp[k:] - cp[:-k], cq[k:] - cq[:-k]
    start = int(np.argmax(ip))
    frozen, oracle = float(q[before].sum()), float(q[after].sum())
    return dict(blocks=k, tokens=k * 128, saturated=k == len(p),
                earlier_coverage=float(p[before].sum()), frozen_coverage=frozen,
                oracle_coverage=oracle, oracle_gain=oracle - frozen,
                coverage_change=frozen - float(p[before].sum()),
                topk_overlap=len(set(before) & set(after)) / k,
                earlier_best_interval_coverage=float(ip[start]),
                frozen_interval_coverage=float(iq[start]),
                shared_interval_oracle_coverage=float(np.minimum(ip, iq).max()))


def locality(p, q):
    p, q = normalize(p), normalize(q)
    n = len(p)
    both = shared_interval(p, q, .9)
    ip, iq = shared_interval(p, p, .9), shared_interval(q, q, .9)
    order = np.argsort(-(p + q), kind='stable')
    hit = np.flatnonzero((np.cumsum(p[order]) >= .9 - 1e-12) &
                        (np.cumsum(q[order]) >= .9 - 1e-12))
    joint_count = int(hit[0]) + 1
    overlap = float(np.minimum(p, q).sum())
    budgets = {name: budget_metrics(p, q, k) for name, k in BUDGETS}
    budgets.update({name: budget_metrics(p, q, max(1, int(np.ceil(n * f))))
                    for name, f in FRACTIONS})
    w1 = float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]).sum())
    concentration = {}
    for threshold in [.5, .8]:
        tag = str(int(threshold * 100))
        region = shared_interval(p, q, threshold)
        count = int(np.flatnonzero((np.cumsum(p[order]) >= threshold - 1e-12) &
                                  (np.cumsum(q[order]) >= threshold - 1e-12))[0]) + 1
        concentration.update({f'shared_interval{tag}_fraction': region['fraction'],
                              f'pooled_ranked_set{tag}_fraction': count / n})
    return dict(mass_overlap=overlap, total_variation=1 - overlap,
                transport_tokens=w1 * 128, transport_fraction=w1 / max(1, n - 1),
                shared_interval90_tokens=both['tokens'], shared_interval90_fraction=both['fraction'],
                individual_interval90_fraction=(ip['fraction'] + iq['fraction']) / 2,
                shared_interval90_expansion=(both['tokens'] - max(ip['tokens'], iq['tokens'])),
                pooled_ranked_set90_tokens=joint_count * 128,
                pooled_ranked_set90_fraction=joint_count / n,
                shared_interval90=both, budgets=budgets, **concentration)


def classify(domain, previous, current):
    old_users = [m for m in previous if m['role'] == 'user']
    users = [m for m in current if m['role'] == 'user']
    if domain == 'dialogue':
        assert current[:len(previous)] == previous
        added = current[len(previous):]
        return ('across_user_query' if any(m['role'] == 'user' for m in added)
                else 'same_user_query'), 'dialogue_user_message'
    if domain == 'code':
        assert len(users) == len(old_users) == 1 and users == old_users
        return 'same_user_query', 'single_software_task'
    if domain == 'tool_state':
        assert current[:len(previous)] == previous
        added_users = [m for m in current[len(previous):] if m['role'] == 'user']
        assert all(message_text(m).startswith('Round ') for m in added_users)
        return 'same_user_query', 'environment_round' if added_users else 'tool_result'
    # GUI snapshots/feedback have role=user but do not introduce a new goal.
    # The harness can update its action summary inside the first user message.
    # The actual instruction remains before this explicitly labeled section.
    assert (message_text(users[0]).split('Previous actions:')[0] ==
            message_text(old_users[0]).split('Previous actions:')[0])
    assert all(message_text(m).startswith('<tool_response>') for m in users[1:])
    return 'same_user_query', 'gui_observation'


def summarize(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[r['trajectory']].append(r)
    result = dict(boundaries=len(rows), trajectories=len(groups), **{
        key: bootstrap([mean([r[key] for r in group]) for group in groups.values()])
        for key in fields})
    if 'frozen_coverage' in fields:
        result['coverage_p10_within_trajectory'] = bootstrap([
            float(np.quantile([r['frozen_coverage'] for r in group], .1)) for group in groups.values()])
        result['fraction_boundaries_ge90'] = bootstrap([
            mean([r['frozen_coverage'] >= .9 for r in group]) for group in groups.values()])
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout='constrained')
    names = ['Software: same query', 'Tool/state: same task',
             'Dialogue: new query', 'GUI: same query']
    for col, domain in enumerate(DOMAINS):
        group = domain + '/' + ('across_user_query' if domain == 'dialogue' else 'same_user_query')
        s = result['summary'][group]
        ax = axes[0, col]
        for i, key in enumerate(['within_kl', 'cross_kl']):
            stat = s['locality'][key]
            ax.errorbar(i, stat['mean'], yerr=[[stat['mean'] - stat['ci95'][0]],
                        [stat['ci95'][1] - stat['mean']]], fmt='o', capsize=4,
                        color=['#3679aa', '#cf7945'][i])
        ax.set(xticks=[0, 1], xticklabels=['Within call', 'Call boundary'],
               title=names[col], ylabel='KL (bits)' if col == 0 else '', ylim=(0, .95), xlim=(-.4, 1.4))
        ax.text(.03, .95, f"{s['locality']['trajectories']} trajectories; {s['locality']['boundaries']} boundaries",
                transform=ax.transAxes, va='top', fontsize=8)
        ax = axes[1, col]
        for key, label, style, color in [('frozen_coverage', 'Previous-window set', '-', '#3679aa'),
                ('oracle_coverage', 'Current-window oracle', '--', '#777777')]:
            x, y = [], []
            for budget, _ in BUDGETS:
                stat = s['budgets'][budget]['all'][key]
                x.append(int(budget[:-1])); y.append(100 * stat['mean'])
            ax.plot(x, y, style + 'o', label=label, color=color, markersize=3)
        counts = [s['budgets'][b]['unsaturated']['boundaries'] for b, _ in BUDGETS]
        ax.text(.02, .03, 'Unsaturated pairs (1K to 64K):\n' + ' / '.join(map(str, counts)),
                transform=ax.transAxes, fontsize=7, va='bottom')
        ax.set(xscale='log', xticks=[1, 2, 4, 8, 16, 32, 64],
               xticklabels=['1', '2', '4', '8', '16', '32', '64'], ylim=(0, 103),
               xlabel='Historical selection budget (Ki tokens)',
               ylabel='Common-history attention coverage (%)' if col == 0 else '')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Attention reweighting and reuse of historical blocks\n'
                 '32-query boundary windows; exact common prefix; trajectory means and 95% CIs above', fontsize=13)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'query_locality.{extension}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
    for col, domain in enumerate(DOMAINS):
        group = domain + '/' + ('across_user_query' if domain == 'dialogue' else 'same_user_query')
        s = result['summary'][group]['locality']
        ax = axes[col]
        for prefix, label, style in [('pooled_ranked_set', 'Distributed blocks (hindsight)', '-o'),
                                     ('shared_interval', 'Contiguous interval (hindsight)', '--s')]:
            ax.plot([50, 80, 90], [100 * s[f'{prefix}{t}_fraction']['mean'] for t in [50, 80, 90]],
                    style, label=label, markersize=4)
        ax.set(title=names[col], xticks=[50, 80, 90], ylim=(0, 100),
               xlabel='Mass required in EACH window (%)',
               ylabel='Common history retained (%)' if col == 0 else '')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        if col == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Shared support can be distributed: block reuse is not necessarily a short contiguous region', fontsize=11)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'spatial_concentration.{extension}', dpi=180)
    plt.close(fig)


def report(result, out):
    summary = result['summary']
    names = {'code': '软件', 'tool_state': '工具/状态', 'gui': 'GUI', 'dialogue': '对话'}
    selected = [(dom, summary[dom + '/' + ('across_user_query' if dom == 'dialogue' else 'same_user_query')])
                for dom in DOMAINS]
    lines = ['# 现有轨迹：step 边界变化、历史块复用与 user query 边界', '',
        '本次仅分析已保存的 957 次模型调用，没有新生成。复用了原 W=32 的 791 个严格配对边界，'
        '逐项重算 KL 与原值完全一致；共同 token/mRoPE 前缀及图片序列重新核验。', '',
        '**结论：step 边界的分布变化较大，且扩大选集预算能提高相邻 step 的历史块复用。'
        '但当前数据不支持把主要历史 attention 描述为始终集中在一个很小的连续区域，'
        '也没有足够的同分布样本证明真实新 user query 比同 query 内工具边界迁移更大。**', '',
        '## 1. 边界类型与分布差异', '',
        '| 来源 | 边界语义 | 轨迹 / 有效边界 | 调用内 KL | 跨调用 KL | 倍数 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for dom, s in selected:
        l = s['locality']; within = l['within_kl']['mean']; cross = l['cross_kl']['mean']
        lines.append(f"| {names[dom]} | {'真实新 user 消息' if dom == 'dialogue' else '同一任务/query'} | "
                     f"{l['trajectories']} / {l['boundaries']} | {within:.4f} | {cross:.4f} | {cross / within:.2f} |")
    lines += ['', '软件全部 220 个源相邻边界保持同一条用户任务；工具/状态的 116 次新增 user-role 消息都是 Round 环境反馈；'
        'GUI 的 user-role 消息是截图/工具反馈，首条消息中的 Previous actions 是运行时摘要。'
        '对话有 218 次真实新 user 消息、2 次没有新 user 的连续 assistant；有效配对分别为 193 和 2。'
        '“新 user 消息”不等于“目标必然改变”，同一话题的追问也属于新消息。', '',
        '## 2. 上一 step 的关注集合能否复用', '',
        '以下使用前一步最后 32 个 decode query 对共同历史选 top-k，固定这组块，评估下一步前 32 个 query。'
        '共同历史范围在事后由前缀身份确定，主表是共同支持集上的排名复用控制，不能直接当作部署中的实际选集。'
        'Current-window oracle 使用当前窗口选相同数量的块，是事后上界；“oracle 增益”不是在线刷新策略的实测收益。'
        '这个口径检验相邻边界的局部复用，不能替代 query 开始时只选一次并跨整段任务使用的实验。', '',
        '| 同一任务/query | 预算 | 全样本复用 coverage | 全样本 oracle 增益(pp) | 未装下全部历史：轨迹/边界 | 未饱和 coverage | 未饱和 oracle 增益(pp) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for dom, s in selected:
        if dom == 'dialogue':
            continue
        for budget in ['2K', '4K', '8K', '16K', '32K', '64K']:
            a, u = s['budgets'][budget]['all'], s['budgets'][budget]['unsaturated']
            uc = '—' if not u['boundaries'] else f"{100 * u['frozen_coverage']['mean']:.1f}%"
            ug = '—' if not u['boundaries'] else f"{100 * u['oracle_gain']['mean']:.2f}"
            lines.append(f"| {names[dom]} | {budget} | {100 * a['frozen_coverage']['mean']:.1f}% | "
                         f"{100 * a['oracle_gain']['mean']:.2f} | {u['trajectories']}/{u['boundaries']} | {uc} | {ug} |")
    lines += ['', '再做严格的“先选后求交”检查：先在上一调用完整候选历史上选择 8K，'
        '之后只保留身份仍在共同前缀中的块，不补位。oracle 与剩余块数匹配。'
        '以下只统计上一调用候选历史超过 8K 的边界；低覆盖可能同时涉及选集迁移与可核验前缀收缩。', '',
        '| 来源 | 轨迹/边界 | 严格复用 coverage | 选中块身份保留率 | 同有效预算 oracle 增益(pp) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for dom, s in selected:
        u = s['budgets']['8K']['source_unsaturated']
        if not u['boundaries']:
            continue
        lines.append(f"| {names[dom]} | {u['trajectories']}/{u['boundaries']} | {100 * u['strict_coverage']['mean']:.1f}% | "
                     f"{100 * u['strict_surviving_fraction']['mean']:.1f}% | {100 * u['strict_matched_oracle_gain']['mean']:.2f} |")
    lines += ['', '8K 预算下，仅看每条轨迹跨步 KL 最高四分之一、并排除预算已装下全部共同历史的边界：', '',
        '| 来源 | 轨迹/边界 | 复用 coverage | oracle 增益(pp) |', '| --- | ---: | ---: | ---: |']
    for dom, s in selected:
        if dom == 'dialogue':
            continue
        h = s['budgets']['8K']['high_drift_unsaturated']
        lines.append(f"| {names[dom]} | {h['trajectories']}/{h['boundaries']} | "
                     f"{100 * h['frozen_coverage']['mean']:.1f}% | {100 * h['oracle_gain']['mean']:.2f} |")
    lines += ['', '完整 1K–64K、10%/25%/50% 比例预算，以及 coverage 的轨迹内低十分位数、'
        'coverage≥90% 的边界比例、bootstrap CI 都在 analysis.json。'
        '预算增大本就会使 top-k 覆盖率单调增加；更有判别力的是未饱和样本的覆盖率及其与 oracle 的差距。', '',
        '## 3. 关注范围是否集中', '',
        '分别测量能够同时覆盖两窗口各 50%、80%、90% 历史 attention 的共同块集合和共同连续区间。'
        '集合按两分布平均质量排序后扩展，是事后构造，不是保证最优的最小集合；连续区间使用穷举等价算法求最短区间。'
        '两者都使用未来窗口信息，因此只能描述共享关注范围，不能当作在线选集结果。', '',
        '| 来源 | 80% 共同分散块集 / 历史 | 80% 最短连续区间 / 历史 | 90% 共同分散块集 / 历史 | 90% 最短连续区间 / 历史 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for dom, s in selected:
        l = s['locality']
        vals = [l[k]['mean'] for k in ['pooled_ranked_set80_fraction', 'shared_interval80_fraction',
                                     'pooled_ranked_set90_fraction', 'shared_interval90_fraction']]
        lines.append('| ' + names[dom] + ' | ' + ' | '.join(f'{100 * v:.1f}%' for v in vals) + ' |')
    lines += ['', '因此更适合说“历史块具有可复用性/存在共享关注集合”，而非“高度集中于一个小连续空间区域”。'
        '这里的空间是输入 token 顺序，不等价于文件、网页或语义对象的空间。', '',
        '## 4. 新 query 与目标切换', '',
        '不同来源的历史长度、任务和输出模式不同，不能把软件/GUI 的同 query 边界与对话的新 query 边界直接比较成因果结论。'
        '对话内部只有 2 个同 query 跨调用边界，无法提供充分的同来源对照。', '',
        '补充使用 LongMemEval 源会话标识：以下两组都是新 user query，区别是是否跨原始 session。'
        '会话切换只是目标/话题切换的代理指标，没有人工标注目标变化。', '',
        '| 对话中新 user query 类型 | 轨迹/边界 | 跨步 KL | 质量重叠 | 位置迁移 / 历史跨度 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for tag, title in [('same_session', '同 session 追问'), ('different_session', '跨 session')]:
        s = result['dialogue_session_control'][tag]
        lines.append(f"| {title} | {s['trajectories']}/{s['boundaries']} | {s['cross_kl']['mean']:.4f} | "
                     f"{100 * s['mass_overlap']['mean']:.1f}% | {100 * s['transport_fraction']['mean']:.1f}% |")
    lines += ['', '位置迁移使用一维 Wasserstein 距离（块序累计概率差的和），除以候选块位置跨度。'
        '跨 session 的变化更大是探索性关联，不能替代同任务中新 query 与工具返回的配对实验。', '',
        '## 5. 64K 与解释边界', '',
        f"本批最大 prompt={result['validation']['max_prompt_tokens']} tokens，最大 prompt+completion="
        f"{result['validation']['max_prompt_plus_completion_tokens']} tokens。64K 预算对全部样本均已饱和，"
        '其 coverage=100%、oracle 增益=0 是完整容纳历史的结果。较小预算的未饱和对照才衡量选择与复用能力。', '',
        'coverage 仅以两步共同、身份核验后的旧历史为分母；排除首块、近期 8 个 prompt blocks、'
        '新输入和新增生成内容。新增内容没有被本实验假设为可丢弃。各预算是这部分旧历史的选择预算，'
        '不等于包含 sink/recent/new tokens 的总 KV 预算。原始共同历史 attention 总质量已单列在 CSV/JSON。', '',
        '统计先在轨迹内平均边界，再跨轨迹等权平均；bootstrap 5000 次，seed=20260911。'
        '各预算未饱和组的样本可能变化，不能把差异全部归因于预算。少于两个轨迹时不报告总体 CI。'
        '这是 full-visible attention 离线诊断，不能直接推出生产检索器质量或端到端任务成功率。', '',
        '## 复现与文件', '', '```bash',
        '.venv/bin/python scripts/motivation_v3/analyze_query_locality.py --out benchmark/results/motivation_v3_20260910/corrected80/query_locality_20260923',
        '```', '',
        '- `analysis.json`：统计、CI、输入与脚本 SHA256、核验结果。',
        '- `boundary_labels.csv`：全部 877 个源相邻调用边界的标签。',
        '- `boundary_locality.csv`：791 个有效边界的分布与空间指标。',
        '- `boundary_budget_coverage.csv`：各预算的边界级 coverage、oracle 差距及预算饱和标记。',
        '- `request_context_lengths.csv`：957 次调用长度。',
        '- `query_locality.png/pdf`：分布差异及选集复用曲线。',
        '- `spatial_concentration.png/pdf`：共同块集合与连续区间大小。', '']
    (out / 'README_zh.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('benchmark/results/motivation_v3_20260910/corrected80'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path, analysis_path = args.root / 'manifest.json', args.root / 'analysis_kl/analysis.json'
    manifest = json.loads(manifest_path.read_text())
    original = json.loads(analysis_path.read_text())
    original_by_id = {r['id']: r for r in original['trajectories']}
    dialogue_rows = [r for r in manifest['trajectories'] if r['domain'] == 'dialogue']
    dialogue_source = Path(dialogue_rows[0]['source'])
    with dialogue_source.open() as f:
        dialogue_data = json.load(f)
    dialogue_sessions = {}
    for row in dialogue_rows:
        source = dialogue_data[row['source_row']]
        position, latest_user_session = 1, None
        by_cut = {}
        for session_id, session in enumerate(source['haystack_sessions']):
            for message in session:
                if message['role'] == 'user':
                    latest_user_session = session_id
                if message['role'] == 'assistant':
                    by_cut[position] = latest_user_session
                position += 1
        dialogue_sessions[row['id']] = {f['source_step']: by_cut[f['cut']] for f in row['frames']}
    del dialogue_data
    labels, boundaries, budget_rows, context = [], [], [], []
    max_error = 0.
    for row in manifest['trajectories']:
        previous = None
        matched = {r['to_step']: r for r in original_by_id[row['id']]['settings']['32']['matched_boundaries']}
        for i, frame in enumerate(row['frames']):
            folder = args.root / 'measurements' / row['id']
            record = json.loads((folder / f'step_{i:03d}.json').read_text())
            assert record['status'] == 'collected' and record['block_tokens'] == 128
            messages = gzread(folder / f'request_{i:03d}.json.gz')['messages']
            inputs = gzread(record['input_tokens_path'])
            with np.load(record['path']) as z:
                mass, positions = z['mass'], z['positions']
            decode = mass[positions >= record['prompt_tokens']]
            windows = [decode[:32].mean(0), decode[32:64].mean(0) if len(decode) >= 64 else None,
                       decode[-64:-32].mean(0) if len(decode) >= 64 else None, decode[-32:].mean(0)]
            context.append(dict(trajectory=row['id'], domain=row['domain'], step=i,
                                prompt_tokens=record['prompt_tokens'], completion_tokens=record['completion_tokens']))
            current = dict(record=record, messages=messages, inputs=inputs, frame=frame, windows=windows)
            if previous is not None:
                assert frame['source_step'] == previous['frame']['source_step'] + 1
                kind, reason = classify(row['domain'], previous['messages'], messages)
                identity = dict(trajectory=row['id'], domain=row['domain'],
                                from_step=previous['frame']['source_step'], to_step=frame['source_step'],
                                boundary_type=kind, label_reason=reason)
                if row['domain'] == 'dialogue':
                    source_sessions = dialogue_sessions[row['id']]
                    identity['session_transition'] = ('different_session' if
                        source_sessions[frame['source_step']] != source_sessions[previous['frame']['source_step']]
                        else 'same_session')
                else:
                    identity['session_transition'] = 'not_applicable'
                labels.append(identity)
                if frame['source_step'] in matched:
                    saved = matched[frame['source_step']]
                    n = saved['common_blocks']; end = n + 1
                    prefix = common_prefix(previous['inputs']['tokens'], inputs['tokens'])
                    if inputs['mrope'] and previous['inputs']['mrope']:
                        prefix = min(prefix, common_prefix(previous['inputs']['mrope'], inputs['mrope']))
                    old_images = previous['frame'].get('image_sha256', [])
                    assert not old_images or old_images == frame.get('image_sha256', [])[:len(old_images)]
                    assert record['prefix_identity_verified'] and end * 128 <= prefix
                    before = previous['windows'][2][1:end]
                    p = previous['windows'][3][1:end]
                    q = windows[0][1:end]
                    after = windows[1][1:end]
                    cross, within = kl(p, q), mean([kl(before, p), kl(q, after)])
                    error = max(abs(cross - saved['cross_kl']), abs(within - saved['within_kl']))
                    max_error = max(max_error, error)
                    assert error < 1e-6, (identity, error)
                    loc = locality(p, q)
                    controls = [locality(before, p), locality(q, after)]
                    flat = {k: v for k, v in loc.items() if k not in ['budgets', 'shared_interval90']}
                    boundary = identity | flat | dict(common_blocks=n, common_tokens=n * 128,
                        earlier_historical_mass=float(p.sum()), later_historical_mass=float(q.sum()),
                        within_kl=within, cross_kl=cross, kl_difference=cross - within,
                        within_js=saved['within_js'], cross_js=saved['cross_js'],
                        within_mass_overlap=mean([c['mass_overlap'] for c in controls]),
                        within_transport_fraction=mean([c['transport_fraction'] for c in controls]),
                        within_shared_interval90_fraction=mean([c['shared_interval90_fraction'] for c in controls]))
                    boundaries.append(boundary)
                    source_end = min(len(previous['windows'][3]), previous['record']['prompt_tokens'] // 128 - 8)
                    source_order = np.argsort(-previous['windows'][3][1:source_end], kind='stable') + 1
                    normalized_q = normalize(q)
                    # Additional strict-order control: select on ALL source
                    # candidates first, then intersect with the audited common
                    # history. Never backfill discarded selected block IDs.
                    for budget, values in loc['budgets'].items():
                        source_k = (max(1, int(np.ceil(len(source_order) * dict(FRACTIONS)[budget])))
                                    if budget.endswith('%') else dict(BUDGETS)[budget])
                        selected = source_order[:source_k]
                        kept = selected[selected < end]
                        strict_coverage = float(normalized_q[kept - 1].sum())
                        strict_oracle = (float(np.sort(normalized_q)[-len(kept):].sum()) if len(kept) else 0.)
                        budget_rows.append(identity | dict(budget=budget, common_blocks=n) | values |
                            dict(within_frozen_coverage=mean([c['budgets'][budget]['frozen_coverage'] for c in controls]),
                                 within_oracle_gain=mean([c['budgets'][budget]['oracle_gain'] for c in controls]),
                                 source_candidate_blocks=len(source_order), source_saturated=source_k >= len(source_order),
                                 strict_selected_blocks=len(selected), strict_surviving_blocks=len(kept),
                                 strict_surviving_fraction=len(kept) / len(selected),
                                 strict_coverage=strict_coverage, strict_matched_oracle_gain=strict_oracle - strict_coverage))
            previous = current
        print(row['id'], 'done', flush=True)
    summary = {}
    fields = ['within_kl', 'cross_kl', 'kl_difference', 'within_js', 'cross_js', 'mass_overlap',
              'within_mass_overlap', 'transport_fraction', 'within_transport_fraction',
              'shared_interval90_fraction', 'within_shared_interval90_fraction', 'shared_interval90_tokens',
              'individual_interval90_fraction', 'shared_interval90_expansion',
              'shared_interval50_fraction', 'shared_interval80_fraction',
              'pooled_ranked_set50_fraction', 'pooled_ranked_set80_fraction',
              'pooled_ranked_set90_fraction', 'pooled_ranked_set90_tokens', 'common_tokens',
              'earlier_historical_mass', 'later_historical_mass']
    budget_fields = ['earlier_coverage', 'frozen_coverage', 'oracle_coverage', 'oracle_gain',
                     'topk_overlap', 'saturated', 'frozen_interval_coverage', 'shared_interval_oracle_coverage',
                     'within_frozen_coverage', 'within_oracle_gain', 'strict_coverage',
                     'strict_surviving_fraction', 'strict_matched_oracle_gain', 'source_saturated']
    for domain in DOMAINS:
        for kind in ['same_user_query', 'across_user_query']:
            rows = [r for r in boundaries if r['domain'] == domain and r['boundary_type'] == kind]
            if not rows:
                continue
            key = domain + '/' + kind
            s = dict(locality=summarize(rows, fields), budgets={})
            grouped = defaultdict(list)
            for r in rows:
                grouped[r['trajectory']].append(r)
            high_drift = set()
            for tid, rr in grouped.items():
                cutoff = np.quantile([r['cross_kl'] for r in rr], .75)
                high_drift.update((tid, r['to_step']) for r in rr if r['cross_kl'] >= cutoff)
            for budget in [x[0] for x in BUDGETS + FRACTIONS]:
                br = [r for r in budget_rows if r['domain'] == domain and
                      r['boundary_type'] == kind and r['budget'] == budget]
                s['budgets'][budget] = {name: summarize(rs, budget_fields) for name, rs in
                    [('all', br), ('unsaturated', [r for r in br if not r['saturated']]),
                     ('source_unsaturated', [r for r in br if not r['source_saturated']]),
                     ('high_drift', [r for r in br if (r['trajectory'], r['to_step']) in high_drift]),
                     ('high_drift_unsaturated', [r for r in br if not r['saturated'] and
                                                (r['trajectory'], r['to_step']) in high_drift])]}
            summary[key] = s
    session_control = {}
    for transition in ['same_session', 'different_session']:
        rs = [r for r in boundaries if r['domain'] == 'dialogue' and
              r['boundary_type'] == 'across_user_query' and r['session_transition'] == transition]
        session_control[transition] = summarize(rs, fields)
    result = dict(scope='Exploratory offline analysis of existing reference attention; no new model calls. '
                  'Primary top-k curves rank the preceding window on the retrospectively verified common support. '
                  'The strict_* control instead selects on all previous-call candidates before intersecting identities, '
                  'without backfilling and with oracle matched to the surviving count. This is a local boundary-reuse '
                  'diagnostic, NOT a set selected once at user-query start and NOT a task-quality test. '
                  'Spatial locality refers to token order in the verified common prefix. '
                  'Different domains are not matched within/across-user-query populations.',
                  aggregation='Boundary means within trajectory, then equal trajectory weights; 5000 bootstrap replicates, seed=20260911. '
                  'Intervals containing 90% must cover 90% in each distribution, not merely their average. '
                  'Pooled-ranked sets and shared intervals use both sides and are hindsight diagnostics.',
                  provenance={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [manifest_path, analysis_path, dialogue_source,
                                        Path(__file__), Path(__file__).with_name('metrics.py')]},
                  validation=dict(calls=len(context), matched_boundaries=len(boundaries),
                                  maximum_kl_reconstruction_error=max_error,
                                  label_counts=dict(Counter(r['domain'] + '/' + r['boundary_type'] for r in labels)),
                                  max_prompt_tokens=max(r['prompt_tokens'] for r in context),
                                  max_prompt_plus_completion_tokens=max(r['prompt_tokens'] + r['completion_tokens'] for r in context)),
                  summary=summary, dialogue_session_control=session_control)
    assert len(context) == 957 and len(boundaries) == 791
    assert all(abs(r['frozen_coverage'] - 1) < 1e-9 for r in budget_rows if r['saturated'])
    assert all(r['oracle_gain'] >= -1e-10 for r in budget_rows)
    assert all(r['saturated'] for r in budget_rows if r['budget'] == '64K')
    (args.out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_csv(args.out / 'boundary_labels.csv', labels)
    write_csv(args.out / 'boundary_locality.csv', boundaries)
    write_csv(args.out / 'boundary_budget_coverage.csv', budget_rows)
    write_csv(args.out / 'request_context_lengths.csv', context)
    render(result, args.out)
    report(result, args.out)
    print(json.dumps(result['validation'], indent=2))


if __name__ == '__main__':
    main()
