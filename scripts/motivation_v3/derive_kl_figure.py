"""Aggregate archived paired-boundary KL values and render the main figure."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from metrics import bootstrap, mean
from plot_main_figure import DOMAINS, LABELS, render


def eligible_boundaries(settings):
    rows = settings['matched_boundaries']
    for row in rows:
        for key in ('within_kl', 'cross_kl'):
            value = row[key]
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid saved paired KL: {row}')
    return rows


def summarize(rows):
    paired = [r for r in rows if r['matched_within_kl'] is not None
              and r['matched_cross_kl'] is not None]
    result = {key: bootstrap([r[key] for r in paired])
              for key in ('matched_within_kl', 'matched_cross_kl')}
    result.update(paired_difference=bootstrap([
        r['matched_cross_kl']-r['matched_within_kl'] for r in paired]),
        cross_gt_within=sum(r['matched_cross_kl'] > r['matched_within_kl'] for r in paired),
        paired_rollouts=len(paired))
    if paired:
        within = result['matched_within_kl']['mean']
        result['ratio_of_means'] = result['matched_cross_kl']['mean']/within if within > 0 else None
    else:
        result['ratio_of_means'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.analysis.read_text())
    manifest = json.loads(args.manifest.read_text())
    report = copy.deepcopy(original)
    rows = report['trajectories']
    assert len(rows) == len(manifest['trajectories']) == 80
    assert {r['id'] for r in rows} == {r['id'] for r in manifest['trajectories']}
    assert all(r['collected_steps'] == r['planned_steps'] for r in rows)
    assert sum(r['collected_steps'] for r in rows) == 957
    for row in rows:
        for group in ('settings', 'natural_end_settings'):
            for settings in row[group].values():
                boundaries = eligible_boundaries(settings)
                for metric in ('within', 'cross'):
                    settings[f'matched_{metric}_kl'] = mean([r[f'{metric}_kl'] for r in boundaries])
                    # Validate that these are the same matched boundaries as the JS analysis.
                    js_mean = mean([r[f'{metric}_js'] for r in boundaries])
                    stored = settings[f'matched_{metric}_js']
                    assert (js_mean is None and stored is None) or np.isclose(js_mean, stored, rtol=1e-12)
    for domain in DOMAINS:
        domain_rows = [r for r in rows if r['domain'] == domain]
        assert len(domain_rows) == 20
        for window in ('16', '32', '64', '128'):
            for group, target in [('settings', report['summary'][domain]),
                                  ('natural_end_settings', report['summary'][domain]['natural_end_only'])]:
                old = copy.deepcopy(target[window])
                target[window] = summarize([r[group][window] for r in domain_rows])
                target[window]['js_reference'] = old
        assert report['summary'][domain]['coverage'] == original['summary'][domain]['coverage']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report['divergence_metric'] = 'kl'
    report['kl_derivation'] = dict(
        source_analysis=str(args.analysis.resolve()), source_analysis_sha256=sha(args.analysis),
        manifest_sha256=sha(args.manifest),
        direction='D_KL(earlier window || later window)', log_base=2,
        smoothing='Normalize each window distribution, add 1e-9 to each probability, renormalize.',
        aggregation='Average matched boundaries within each trajectory, then weight trajectories equally.',
        bootstrap_seed=20260911, bootstrap_replicates=5000,
        source_metric_sha256=original['analysis_source_sha256']['metrics.py'],
        derived_source_sha256={name: sha(Path(__file__).with_name(name))
                               for name in ('derive_kl_figure.py', 'plot_main_figure.py', 'metrics.py')})
    report['pooled_kl'] = summarize([r['settings']['32'] for r in rows])
    boundaries = sum(len(r['settings']['32']['matched_boundaries']) for r in rows)
    assert boundaries == 791
    (args.out/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    render(report, manifest, args.out, metric='kl')
    lines = ['# KL divergence 重绘结果', '',
             '80 条完整主轨迹，四类各 20 条；957 次采集调用，W=32 包含 791 个配对 step 边界。', '',
             '从已归档逐边界 KL 重算轨迹均值；使用相同已核验历史、相同非重叠窗口与同一套样本。',
             '方向为 KL(前一窗口 || 后一窗口)，以 2 为底，单位 bit。先归一化，每块概率加 1e-9，再归一化。',
             '跨步：前一步最后 32 query 与后一步最前 32 query。步内：该边界两侧各一对相邻 32-query 窗口的 KL 均值。',
             '均值和置信区间以轨迹为单位；5,000 次 bootstrap，seed=20260911。图中使用统一线性纵轴，没有对 divergence 额外取 log。', '',
             '| 类型 | 步内 KL 均值 | 跨步 KL 均值 | 跨步 / 步内 | 跨步较高轨迹 | 配对差值 [95% CI] |',
             '|---|---:|---:|---:|---:|---:|']
    for domain in DOMAINS + ['all']:
        s = report['pooled_kl'] if domain == 'all' else report['summary'][domain]['32']
        diff = s['paired_difference']
        lines.append(f"| {LABELS.get(domain, 'Overall')} | {s['matched_within_kl']['mean']:.4f} | "
                     f"{s['matched_cross_kl']['mean']:.4f} | {s['ratio_of_means']:.3f} | "
                     f"{s['cross_gt_within']}/{s['paired_rollouts']} | "
                     f"{diff['mean']:.4f} [{diff['ci95'][0]:.4f}, {diff['ci95'][1]:.4f}] |")
    lines += ['', '窗口敏感性（跨步 / 步内；括号内为符合条件的轨迹数）：', '',
              '| 类型 | W=16 | W=32 | W=64 | W=128 |', '|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        lines.append('| '+LABELS[domain]+' | '+' | '.join(
            f"{report['summary'][domain][w]['ratio_of_means']:.3f} "
            f"(n={report['summary'][domain][w]['paired_rollouts']})"
            for w in ('16','32','64','128'))+' |')
    exceptions = [r['id'] for r in rows if r['settings']['32']['matched_cross_kl'] <= r['settings']['32']['matched_within_kl']]
    lines += ['', 'W=32 跨步不高于步内的轨迹：'+', '.join(exceptions), '',
              'KL 有方向性且对接近零的概率及平滑参数敏感；本次没有改变平滑方式，也没有把 JS 倍数当成 KL 倍数。'
              '下排保留原有 attention coverage 统计。KL 描述分布变化，不直接证明重选的质量或性能收益。', '',
              f'输入分析 SHA256：`{sha(args.analysis)}`', '',
              '复现命令（仓库根目录）：', '```bash',
              '.venv/bin/python scripts/motivation_v3/derive_kl_figure.py '
              f'--analysis {args.analysis} --manifest {args.manifest} --out {args.out}', '```']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    for name in ('derive_kl_figure.py', 'plot_main_figure.py', 'metrics.py'):
        (args.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
