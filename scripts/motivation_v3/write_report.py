"""Build the paper-facing report from measured aggregates and integrity evidence."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from metrics import bootstrap,mean

LABELS={'code':'软件代理','tool_state':'工具/状态历史','dialogue':'多会话对话','gui':'GUI 代理'}


def estimate(d,scale=1,digits=3):
    if d.get('mean') is None:return '—'
    if d.get('n',0)<2 or d['ci95'][0] is None:
        return f"{d['mean']*scale:.{digits}f}（n={d.get('n',0)}，CI 不可估计）"
    return f"{d['mean']*scale:.{digits}f} [{d['ci95'][0]*scale:.{digits}f}, {d['ci95'][1]*scale:.{digits}f}]"


def paired_control(rows,fn):
    groups=defaultdict(list)
    for r in rows:groups[r['trajectory']].append(fn(r))
    return bootstrap([mean(v) for v in groups.values()])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--legacy',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    args=ap.parse_args();out=args.campaign/'analysis'
    data=json.loads((out/'analysis.json').read_text());diag=json.loads((out/'schedule_diagnostics.json').read_text())
    audit=json.loads((args.campaign/'integrity_audit.json').read_text())
    manifest=json.loads(args.manifest.read_text());legacy=json.loads(args.legacy.read_text())
    complete=sum(v['complete_trajectories'] for v in data['summary'].values())
    ready=audit['complete'] and complete==len(manifest['trajectories'])
    lines=['# Step-level memory scheduling：多样本实验报告','',
        f'状态：{"全量采集及逐请求完整性检查已完成；仍需人工审阅图表和结论" if ready else "采集/核验尚未完成，下列结果仅为中间快照"}。',
        f'新增完整源轨迹 {complete}/{len(manifest["trajectories"])}；已采集调用 {data["status_counts"].get("collected",0)}/{audit["planned_requests"]}；独立核验通过 {audit["verified_requests"]} 次。',
        '', '## 1. 49 条旧数据是什么','',
        '49 条是历史 OpenHands/SWE-bench Lite rollout，共 343 次唯一匹配的模型请求。它们全部在六次 agent iteration 上限停止，未进行官方任务评分，因此不是 49 个解题成功样本。原始 trace 的完整关联和分析已经恢复。',
        '旧队列使用 Qwen3.6-27B-Q8_0，与新增 Qwen3.8 队列分开报告。旧 trace 没有保存实际 token 前缀，位置交集不能证明相同的历史内容；其结果用于诊断窗口重叠与样本选择的影响。',
        '', '## 2. 新增数据和方法','',
        manifest.get('sampling_note_zh','四类各预选 20 条源轨迹，合计 957 个连续调用前缀；样本在正式测量前冻结并保存来源校验值。') + ' 每个记录输入重新生成 continuation，动作不执行，也不被写入下一次记录输入。因此这是 recorded-input replay，不能用来计算新的闭环任务成功率。',
        '', '| 数据类与来源 | 主样本已完成 / 计划 | 测量单位和边界 |',
        '| --- | ---: | --- |',
        f'| 软件代理：本地 mini-swe-agent / Pier 任务记录 | {data["summary"]["code"]["complete_trajectories"]} / {sum(r["domain"]=="code" for r in manifest["trajectories"])} | 按真实任务名去重；同任务不同 attempt 不重复计数 |',
        f'| 工具/状态：AgentLongBench 的 32K 配置记录 | {data["summary"]["tool_state"]["complete_trajectories"]} / {sum(r["domain"]=="tool_state" for r in manifest["trajectories"])} | 跨任务变体按基础样本编号去重；实际输入长度另报 |',
        f'| 对话：LongMemEval-S 历史会话 | {data["summary"]["dialogue"]["complete_trajectories"]} / {sum(r["domain"]=="dialogue" for r in manifest["trajectories"])} | 重放记录中的助手回复前缀；不运行最后的 LongMemEval 问答评测 |',
        f'| GUI：OSWorld 截图交互记录 | {data["summary"]["gui"]["complete_trajectories"]} / {sum(r["domain"]=="gui" for r in manifest["trajectories"])} | 按任务编号去重，保留连续可重放的消息和截图；不把 API 调用完成视为任务成功 |', '',
        f'模型：{manifest["model"]}，FP16 KV，64K context cap；生成上限 2048，thinking budget 512，temperature 0。使用全可见 dense attention；MTP 和 KVMem 关闭。',
        'prefill 信号由原批量 prefill 路径中的 Q/K 旁路读取，采样末尾 64 个查询；逐 token scalar 化的试采因改变生成结果而被排除。正式旁路版本通过文本与图片请求的普通 prefill 输出对照。',
        '跨步与步内比较采用相同两步、相同已核验 token/图片/mRoPE 前缀、相同窗口宽度，且窗口不重叠。每条边界需要两侧 decode 都至少包含两个窗口。先在轨迹内平均，再按轨迹 bootstrap 5000 次；以下方括号均为 95% CI。',
        '', '## 3. 跨步与步内 attention 变化','',
        '| 数据类 | 窗口 | 配对轨迹数 | 跨步大于步内的轨迹数 | 步内 JS (bits) | 跨步 JS (bits) | 配对差值：跨步−步内 |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for domain,s in data['summary'].items():
        for w in ['16','32','64','128']:
            v=s[w];lines.append(f"| {LABELS[domain]} | {w} | {v['paired_rollouts']} | {v['cross_gt_within']} | {estimate(v['matched_within_js'])} | {estimate(v['matched_cross_js'])} | {estimate(v['paired_difference'])} |")
    oldlines=['旧数据的严格配对控制（历史身份只能按位置近似）：','',
        '| 窗口 | 配对轨迹数 | 步内 JS | 跨步 JS | 配对差值 |',
        '| --- | ---: | --- | --- | --- |']
    for w in [16,32,64,128]:
        s=legacy['summary'][f'{w}/{w}']['strict_matched']
        oldlines.append(f"| {w} | {s['difference']['n']} | {estimate(s['matched_within_js'])} | {estimate(s['matched_cross_js'])} | {estimate(s['difference'])} |")
    oldlines+=['','窗口宽度变化时，可配对样本数也会变化；重叠窗口的差异倍数不能与不重叠控制直接等同。','']
    at=lines.index('## 2. 新增数据和方法');lines[at:at]=oldlines
    lines+=['','自然结束敏感性检查（W=32）：排除触及上限的请求，并保持源 step 相邻，避免把删去请求两侧重新连成一个边界。主表仍保留全部请求。','',
        '| 数据类 | 自然结束配对轨迹数 | 跨步−步内 JS |',
        '| --- | ---: | --- |']
    for domain,s in data['summary'].items():
        d=s['natural_end_only']['32']['paired_difference']
        lines.append(f"| {LABELS[domain]} | {d['n']} | {estimate(d)} |")
    lines+=['','## 4. 固定集合与更频繁更新','',
        '候选历史排除首个 block 和近期八个 prompt blocks，并限制在已核验的历史前缀。主预算为候选 block 数的 10%，向上取整；当候选很少时，实际比例会高于 10%，下表单独列出。coverage 是候选历史内部的条件 attention 比例。',
        'periodic 对照从此前 32 个查询读取因果信号，在每 32/128/512 个 decode token 更新；oracle 使用当前 query 的完整参考 attention 事后选择同预算最佳集合，仅作离线覆盖率上界，不是实际检索策略。以下收益在相同请求和相同轨迹内配对，不能解释为任务正确率收益。',
        '', '| 数据类 | 可用轨迹 / 步数 | 候选历史占总 attention (%) | 实际保留 block 比例 (%) | 固定集合 coverage (%) | 每 128 token 更新的 coverage 收益 (pp) | 固定集合末四分之一−首四分之一 (pp) |',
        '| --- | ---: | --- | --- | --- | --- | --- |']
    paired={}
    for domain,s in diag['summary']['domain'].items():
        rows=[r for r in diag['steps'] if r['domain']==domain];paired[domain]={}
        for interval in [32,128,512]:
            paired[domain][f'periodic_{interval}_gain']=paired_control(rows,
                lambda r,i=interval:r['schedules'][f'periodic_{i}']['coverage']-r['schedules']['prefill_fixed']['coverage'])
        drift=paired_control(rows,lambda r:r['schedules']['prefill_fixed']['last_quarter']-r['schedules']['prefill_fixed']['first_quarter'])
        paired[domain]['fixed_quarter_change']=drift
        lines.append(f"| {LABELS[domain]} | {s['trajectories']} / {s['steps']} | {estimate(s['historical_mass'],100,1)} | {estimate(s['actual_selected_fraction'],100,1)} | {estimate(s['prefill_fixed']['coverage'],100,1)} | {estimate(paired[domain]['periodic_128_gain'],100,1)} | {estimate(drift,100,1)} |")
    lines+=['','完整 5%/10%/20% 与绝对 8/16/32/64-block 预算见 `analysis.json`；输入长度、decode 长度、候选历史大小与截断分组见 `schedule_diagnostics.json`。',
        '', '分组描述统计（各组的数据类型构成可能不同，不作跨组因果比较）：','',
        '| 分组 | 轨迹 / 调用数 | 固定集合 coverage (%) | 每 128 token 更新收益 (pp) | 末四分之一−首四分之一 (pp) |',
        '| --- | ---: | --- | --- | --- |']
    for group,title in [('context_group','输入长度'),('decode_group','decode-query 数'),('truncated','上限/截断')]:
        for value,s in diag['summary'][group].items():
            rows=[r for r in diag['steps'] if str(r[group])==value]
            gain=paired_control(rows,lambda r:r['schedules']['periodic_128']['coverage']-r['schedules']['prefill_fixed']['coverage'])
            drift=paired_control(rows,lambda r:r['schedules']['prefill_fixed']['last_quarter']-r['schedules']['prefill_fixed']['first_quarter'])
            lines.append(f"| {title}: {value} | {s['trajectories']} / {s['steps']} | {estimate(s['prefill_fixed']['coverage'],100,1)} | {estimate(gain,100,1)} | {estimate(drift,100,1)} |")
    extra_path=out/'refresh_distance_controls.json'
    if extra_path.exists():
        extra=json.loads(extra_path.read_text())
        lines+=['','沿用上一步集合的配对对照：旧集合只保留当前已核验历史中的有效块，不补位；本步 prefill 重新选集与 oracle 均使用剩余块数，因此该表预算与主表的本步 10% 预算不同。只比较源 step 真正相邻且两次选择均可定义的请求。','',
            '| 数据类 | 轨迹 / 配对调用 | 实际保留比例 (%) | 平均失效旧块数 | 本步刷新覆盖率收益 (pp) |',
            '| --- | ---: | --- | --- | --- |']
        for domain,item in extra['summary'].items():
            s=item['stale']
            lines.append(f"| {LABELS[domain]} | {s['trajectories']} / {s['requests']} | {estimate(s['effective_fraction'],100,1)} | {estimate(s['lost_selected'],1,1)} | {estimate(s['gain'],100,1)} |")
        lines+=['','绝对 decode-query 距离的覆盖率见 `refresh_distance_controls.pdf` 上排；仅纳入达到各区间末端的请求，每个区间的轨迹数/调用数见同名 `.md`。末区间止于 2047，匹配 2048 输出上限的实际 forward-query 数。越靠后的区间样本构成会变化，不能将曲线直接解释为固定队列的因果长度效应。']
    lines+=[
        '', '## 5. 样本覆盖与截断','',
        '| 数据类 | 实测调用数 | 输入长度 p10 / p50 / p90 | decode-query 数 p10 / p50 / p90 | 触及生成上限的调用数 |',
        '| --- | ---: | --- | --- | ---: |']
    calls=defaultdict(list)
    byid={r['id']:r['domain'] for r in manifest['trajectories']}
    for r in audit['requests']:calls[byid[r['trajectory']]].append(r)
    for domain,rows in calls.items():
        q=lambda key:' / '.join(f'{v:.0f}' for v in np.quantile([r[key] for r in rows],[.1,.5,.9]))
        lines.append(f"| {LABELS[domain]} | {len(rows)} | {q('prompt_tokens')} | {q('decode_tokens')} | {sum(r['budget_reached'] for r in rows)} |")
    lines+=['','首个输出由 prefill logits 产生；触及输出上限时，最后一个输出不会再次 forward，因此 decode-query trace 比输出 token 数少一条。自然 EOS 结束的请求会 forward 最后一个可见输出，其 query 数与可见输出数相同。工具调用标签可能覆盖 `length`，故上限统计结合实际输出 token 数判断。']
    lines+=['','## 6. 图的安排和结论边界','',
        '- 正文主图：`multi_domain_scheduling.pdf`，上排为四类数据的轨迹级配对分布，下排为 prefill 固定集合与同预算 token oracle 的 decode coverage。',
        '- 补充控制图：`scheduling_controls.pdf`，展示输入规模、可比较历史量、历史总 attention 和更新次数—coverage 关系。',
        '- 预算敏感性图：`budget_sensitivity.pdf`，展示比例预算与绝对 block 预算；`budget_sensitivity.md` 列出预算足以覆盖全部候选历史的比例，避免将平凡的满覆盖当作选择性证据。',
        '- 分层稳健性图：`layer_robustness.pdf`，从原始 trace 按 16 个标准 attention 层分别计算边界差异；`layer_reconstruction_check.json` 核对重新合并层后与主分析一致。原始 trace 已对 heads 求和，因此不是 per-head 检查。',
        '- 绝对距离与刷新对照：`refresh_distance_controls.pdf`，上排展示绝对 decode-query 距离，下排比较有效预算相同的旧集合、本步重新选集与 oracle；预算和入组范围区别于正文主图。',
        '- 原始单轨迹图只作机制例图；旧 49 条的 `legacy49_controls.pdf` 用于说明窗口重叠、窗口宽度和早期 decode 代理信号的影响。',
        '- coverage 使用精确 attention 诊断选集，不是系统 compact-index selector；更新次数不是检索耗时或端到端延迟。',
        '- 需要按分组结果收窄论文措辞。即使跨步 JS 更大，也不能直接推导整步固定集合无损；需要一起检查末尾 coverage、长 decode 和 periodic 收益。',
        '- 此处没有验证新闭环任务质量、实际检索/搬运开销或百万 token workspace 的扩展性；这些主张需要独立实验。',
        '', '## 7. 复核入口','',
        '`integrity_audit.json` 保存逐请求独立核验和采集清单来源；`model_provenance.json` 保存模型及视觉权重文件 SHA256；`protocol_snapshot/` 保存原始启动脚本，`final_analysis_snapshot/` 保存最终分析脚本；源采集目录的各 `server_*/` 保存二进制哈希、启动参数、原始 trace 和两秒一次的显存记录。']
    auxiliary = out/'auxiliary_comparison.md'
    if auxiliary.exists():
        lines += ['', '## 8. 保留的同任务辅助尝试', '',
                  '另一条 wasmi 尝试已单独核验，保留为描述性对照，不计入主队列样本数或置信区间。'
                  '两次尝试的记录输入不同，不能作为相同输入下的重复性实验。详见 `auxiliary_comparison.md`。']
    if (out/'resource_summary.md').exists():
        lines += ['', '采集时间和显存监控见 `resource_summary.md`。整卡历史峰值与后期分进程峰值'
                  '的统计范围不同，不能混用为采集最低显存要求；记录耗时不作为系统性能结果。']
    if (out/'boundary_support_controls.md').exists():
        lines += ['', '## 探索性补充：共同历史长度', '',
                  '采集中观察到 GUI 历史裁剪后只剩短共同前缀，因此另外按共同历史块数 '
                  '<16、16–63、64+ 分组检查 W=32 的边界差值，详见 `boundary_support_controls.md`。'
                  '这是观察数据后增加的探索性分析，不改变主队列和主估计。64+ 组要求至少 '
                  '8192 个已核验共同历史 token；各组样本构成不同，不作跨组因果比较，'
                  '也不等同于显式排除 system/tool-schema token 的消融。']
    (out/'paper_report_zh.md').write_text('\n'.join(lines)+'\n')
    (out/'paired_schedule_controls.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(f'Report generated; full audit ready={ready}, complete trajectories={complete}')


if __name__=='__main__':main()
