"""Export the reviewed KL figure's full plotting data as portable JSON and CSV."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmark/results/motivation_v3_20260910/corrected80'
OUT = SOURCE / 'kl_plot_data_export_20260913'
OUT.mkdir(parents=True, exist_ok=True)
a = json.loads((SOURCE / 'analysis_kl/analysis.json').read_text())
m = json.loads((SOURCE / 'manifest.json').read_text())
counts = {}


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    counts[name] = len(rows)


trajectories, boundaries, steps, curves, coverage_stats = [], [], [], [], []
for r in a['trajectories']:
    identity = dict(trajectory_id=r['id'], domain=r['domain'])
    for group in ('settings', 'natural_end_settings'):
        for window, s in r[group].items():
            base = identity | dict(subset='all' if group == 'settings' else 'natural_end_only', window_queries=int(window))
            v = base | dict(collected_steps=r['collected_steps'], planned_steps=r['planned_steps'],
                            truncated_steps=r['truncated_steps'], matched_boundary_count=len(s['matched_boundaries']))
            for name in ('matched_within_kl', 'matched_cross_kl', 'matched_within_js', 'matched_cross_js'):
                v[name] = s[name]
            within, cross = v['matched_within_kl'], v['matched_cross_kl']
            v['kl_cross_minus_within'] = cross-within if within is not None and cross is not None else None
            v['kl_cross_over_within'] = cross/within if within and cross is not None else None
            trajectories.append(v)
            boundaries.extend(base | b for b in s['matched_boundaries'])
            steps.extend(base | {k: v for k, v in d.items() if not isinstance(v, (dict, list))} for d in s['details'])
    for budget, c in r['coverage'].items():
        coverage_stats.append(identity | dict(budget=budget) | {k:v for k,v in c.items() if not isinstance(v, (dict,list))})
        for i in range(10):
            curves.append(identity | dict(budget=budget, bin_index=i, decode_progress_percent=5+10*i) |
                          {key: c[key][i] for key in ('frozen', 'oracle', 'gap')})

write_csv('trajectory_divergence_all_windows.csv', trajectories)
main = [r for r in trajectories if r['subset'] == 'all' and r['window_queries'] == 32]
write_csv('trajectory_divergence_w32.csv', main)
write_csv('boundary_divergence_all_windows.csv', boundaries)
write_csv('boundary_divergence_w32.csv', [r for r in boundaries if r['subset'] == 'all' and r['window_queries'] == 32])
write_csv('step_metrics_all_windows.csv', steps)
write_csv('coverage_trajectory_curves.csv', curves)
write_csv('coverage_trajectory_summary.csv', coverage_stats)


def flatten_summary(domain, subset, window, s):
    result = dict(domain=domain, subset=subset, window_queries=window,
                  paired_trajectories=s['paired_rollouts'], cross_gt_within=s['cross_gt_within'],
                  ratio_of_means=s['ratio_of_means'])
    for metric in ('matched_within_kl', 'matched_cross_kl', 'paired_difference'):
        for key, value in s[metric].items():
            if key == 'ci95':
                result[f'{metric}_ci95_low'], result[f'{metric}_ci95_high'] = value
            else:
                result[f'{metric}_{key}'] = value
    return result


summary, coverage_summary = [], []
for domain, s in a['summary'].items():
    for subset, settings in [('all', s), ('natural_end_only', s['natural_end_only'])]:
        for window in (16,32,64,128):
            summary.append(flatten_summary(domain, subset, window, settings[str(window)]))
    for budget, c in s['coverage'].items():
        for i in range(10):
            v = dict(domain=domain, budget=budget, bin_index=i, decode_progress_percent=5+10*i)
            for metric in ('frozen', 'oracle', 'gap'):
                for key, value in c[metric][i].items():
                    if key == 'ci95':
                        v[f'{metric}_ci95_low'], v[f'{metric}_ci95_high'] = value
                    else:
                        v[f'{metric}_{key}'] = value
            coverage_summary.append(v)
summary.append(flatten_summary('all', 'all', 32, a['pooled_kl']))
write_csv('divergence_summary_all_windows.csv', summary)
write_csv('divergence_summary_w32.csv', [r for r in summary if r['subset']=='all' and r['window_queries']==32])
write_csv('coverage_summary_curves.csv', coverage_summary)
write_csv('trajectory_metadata.csv', [
    {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in r.items() if k!='frames'}
    for r in m['trajectories']])

for name in ('analysis.json', 'figure_caption.txt', 'report.md', 'multi_domain_scheduling.pdf',
             'multi_domain_scheduling.png', 'plot_main_figure.py', 'metrics.py', 'derive_kl_figure.py', 'validation.json'):
    shutil.copy2(SOURCE / 'analysis_kl' / name, OUT / name)
shutil.copy2(SOURCE / 'manifest.json', OUT / 'manifest.json')
shutil.copy2(Path(__file__), OUT / 'export_kl_plot_data.py')
(OUT / 'requirements.txt').write_text('numpy\nmatplotlib\n')

readme = '''# KVMem KL 图完整数据导出包

本包用于在另一个任务或机器上重新绘图。包含完整分析 JSON、原始精度 CSV、绘图脚本、当前 PNG/PDF、图注及方法说明。无需 GPU，也不依赖原采集服务器。

## 直接交给另一任务的说明

请根据本目录数据重画 KVMem step-level memory scheduling 图。上排使用 trajectory_divergence_w32.csv 的 matched_within_kl / matched_cross_kl，按 domain 分成四组，每条轨迹用连线连接 Within 与 Across。均值及 95% CI 使用 divergence_summary_w32.csv；domain=all 是总体，不是第五种数据类型。下排使用 coverage_summary_curves.csv，筛选 budget=r10，用 decode_progress_percent 为横轴、frozen_mean / oracle_mean 为纵轴，并画对应 ci95_low/high。所有覆盖率字段在 CSV 中为 0–1 比例，画百分数时乘 100。保持四组上排共享线性纵轴，显示所有样本。可重新设计样式，但不要改动统计定义、样本或单位。

## 文件与字段

- trajectory_divergence_w32.csv：主图全部 80 条轨迹，每行一条。KL 和 JS 的配对均值、KL 差值和轨迹内比值。
- divergence_summary_w32.csv：四类及总体的均值、95% CI、配对差值、均值比及跨步较高数量。
- boundary_divergence_w32.csv：791 个配对 step 边界。from_step/to_step 为采集器记录的源 step 标识；common_blocks 为该边界的共同候选历史块数。
- trajectory_divergence_all_windows.csv / boundary_divergence_all_windows.csv：W=16/32/64/128 的全部轨迹级与边界级数据，同时包括 all 与 natural_end_only 子集。子集和窗口存在重复观测，不能混合计作独立样本。
- divergence_summary_all_windows.csv：上述窗口/子集的汇总与 CI。
- step_metrics_all_windows.csv：各窗口/子集的逐 step 标量诊断。这里的 within_kl / cross_kl 是未做边界配对聚合的诊断，不能替代主图 matched_* 列。
- coverage_trajectory_curves.csv：各轨迹、各预算、十个 decode 进度区间的 frozen/oracle/gap，用于重新 bootstrap 或修改呈现。gap=oracle-frozen。
- coverage_trajectory_summary.csv：各轨迹的覆盖率均值、有效调用数、平均候选块/选中块数及预算饱和比例。
- coverage_summary_curves.csv：各类、各预算、十个区间的均值、CI 及样本数。
- trajectory_metadata.csv / manifest.json：轨迹来源、独立样本单位、模型及采集配置。source 路径只是来源记录，绘图不需要访问它们。
- analysis.json：完整 KL 分析，无精度截断，包含逐 step 的嵌套覆盖率诊断、自然结束子集及原 JS 对照。重绘当前图以此为唯一输入统计文件即可。
- report.md / figure_caption.txt：统计结果、窗口敏感性与图注。
- checksums.json：包内文件 SHA256 与 CSV 行数。

CSV 使用 UTF-8 BOM，空字段表示缺失/不符合条件，不是零。数值按 Python 浮点可往返精度导出。

## 必须保留的统计口径

80 条主轨迹（code/tool_state/dialogue/gui 各 20），957 次模型调用；W=32 有 791 个配对边界。不混入旧 49 条或新增的 1 条辅助重复任务轨迹。
模型为 Qwen3.8-27B-Q8_0。历史块大小 128 token。主图为全可见历史参考 attention 的离线诊断，输入来自记录的轨迹；新生成动作没有被执行，不是任务成功率测试。
KL 方向为 D_KL(前一窗口 || 后一窗口)，以 2 为底，单位 bit。归一化后每个块概率加 1e-9，再归一化；未对结果额外取 log。
跨步比较前一步最后 W 个 query 与后一步最前 W 个 query。步内取该边界两侧各一对相邻、不重叠 W-query 窗口的 KL 平均。两者使用相同已核验共同历史，要求相邻两步都至少有 2W 个 decode query。
先对各轨迹的有效配对边界平均，再对轨迹等权平均；CI 为轨迹 bootstrap 5,000 次、seed=20260911。总体 CI 使用普通轨迹 bootstrap；各类别 CI 在各自类别内 bootstrap。
ratio_of_means 是跨步总体均值 / 步内总体均值，不是逐轨迹比值的平均。
W=32 总体步内 KL=0.16633406708895256，跨步 KL=0.410506564391839；跨步/步内约 2.468 倍。78/80 条轨迹跨步较高；例外是 tool_kf-c_21、dialogue_6456829e_abs，必须保留。
coverage 是候选历史 attention 内的条件覆盖率；排除第一个块与最近 8 个 prompt 块。r05/r10/r20 为候选块数的 5%/10%/20%，向上取整；8/16/32/64 为绝对块数预算，可能饱和。frozen 为 prefill 尾部信号选集后整步固定；oracle 为同预算逐 query 事后上界。
逐 step 的 prefill_diagnostic 不一定都符合主图覆盖率资格；主图请使用已筛选汇总的 coverage_* CSV 或 analysis.json 中的 summary/trajectory coverage。
KL 更大并不直接证明重选的任务质量/延迟收益。

## 复现当前图

在本目录运行（Python 需安装 numpy 与 matplotlib；原环境为 Matplotlib 3.11.0）：

```bash
python plot_main_figure.py --analysis analysis.json --manifest manifest.json --out redrawn --metric kl
```

本包包含全部已有绘图分析数据，不包含 GPU 采集的逐 query Q/K、attention 矩阵、模型权重或截图。更换 KL 方向、平滑方式或任意新窗口，需要原逐 query 数据，不能从本包的聚合值反推。
'''
# Insert full-precision current means from the authoritative JSON.
pooled = a['pooled_kl']
readme = readme.replace('0.16633406708895256', repr(pooled['matched_within_kl']['mean'])).replace('0.410506564391839', repr(pooled['matched_cross_kl']['mean']))
(OUT / 'README_zh.md').write_text(readme)
assert len(main)==80 and len({r['trajectory_id'] for r in main})==80
assert counts['boundary_divergence_w32.csv']==791
assert all(sum(r['domain']==d for r in main)==20 for d in a['summary'])
assert sum(r['matched_cross_kl']>r['matched_within_kl'] for r in main)==78
for row in main:
    b = [v for v in boundaries if v['trajectory_id']==row['trajectory_id'] and v['subset']=='all' and v['window_queries']==32]
    for kind in ('within','cross'):
        assert abs(sum(v[kind+'_kl'] for v in b)/len(b)-row['matched_'+kind+'_kl'])<1e-12
    # Verify CSV serialization preserves every main-figure float exactly.
with (OUT/'trajectory_divergence_w32.csv').open(encoding='utf-8-sig',newline='') as stream:
    for stored, expected in zip(csv.DictReader(stream),main):
        for key in ('matched_within_kl','matched_cross_kl','matched_within_js','matched_cross_js'):
            assert float(stored[key])==expected[key]
checks = dict(csv_row_counts=counts, validation='80 unique trajectories, four groups of 20, 791 paired boundaries; boundary-to-trajectory means and CSV float round trips verified.',
              sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='checksums.json'})
(OUT/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
archive = OUT.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,arcname=OUT.name+'/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps(dict(directory=str(OUT),zip=str(archive),zip_bytes=archive.stat().st_size,csv_rows=counts),indent=2))
