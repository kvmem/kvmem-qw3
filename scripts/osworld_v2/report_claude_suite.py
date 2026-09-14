#!/usr/bin/env python3
"""Render completed-pair results and auditable maintenance events without rerunning trials."""
import argparse
import csv
import datetime
import io
import json
from pathlib import Path

from summarize_claude_suite import summarize


EVENT_FIELDS = (
    'task', 'arm', 'kind', 'api_id', 'action_step', 'event_count', 'input_tokens',
    'output_tokens', 'api_ttft_ms', 'api_total_ms', 'native_compaction_ms',
    'action_wait_ms', 'action_first_content_ms', 'prefill_ms', 'decode_ms',
    'reused_tokens', 'prefilled_tokens', 'postprefill_semantic_ms',
    'postprefill_query_replay_ms', 'postprefill_guided_query_ms',
    'middecode_refreshes', 'middecode_query_ms', 'middecode_reselect_ms',
)


def event_row(task, arm, kind, api, action, count, duration=None):
    api = api or {}
    action = action or {}
    phase = api.get('server', {})
    return dict(zip(EVENT_FIELDS, (
        task, arm, kind, api.get('id'), action.get('step'), count,
        api.get('input_usage', {}).get('input_tokens'),
        api.get('output_usage', {}).get('output_tokens'),
        api.get('ttft_ms'), api.get('total_ms'), duration,
        action.get('decision_wait_ms'),
        action.get('observation_to_action_response_first_content_ms'),
        phase.get('native_prefill_ms'), phase.get('native_decode_ms'),
        phase.get('reused_tokens', phase.get('kvmem_reuse')), phase.get('prefilled'),
        phase.get('native_post_semantic_ms'), phase.get('native_post_query_replay_ms'),
        phase.get('native_post_guided_query_ms'), phase.get('kvmem_middecode_refreshes'),
        phase.get('kvmem_middecode_query_ms'), phase.get('kvmem_middecode_reselect_ms'),
    )))


def maintenance_events(root, tasks):
    events = []
    for task in tasks:
        folder = root / 'trials' / task / 'analysis'
        metrics = json.loads((folder / 'metrics.json').read_text())
        actions = {'qw3': {}, 'kvmem': {}}
        with (folder / 'step_latency.csv').open() as source:
            for row in csv.DictReader(source):
                for api_id in json.loads(row.get('kvmem_selection_api_ids') or '[]'):
                    actions[row['arm']].setdefault(api_id, row)
        for arm, metric in metrics.items():
            details = list(metric.get('compaction_details', []))
            for boundary in metric.get('compact_boundaries', []):
                metadata = boundary['compact_metadata']
                # Preserve a completed compaction even if no subsequent action
                # was produced. Unobserved request/action fields remain empty.
                index = next((i for i, d in enumerate(details) if d['metadata'] == metadata), None)
                detail = {} if index is None else details.pop(index)
                events.append(event_row(task, arm, 'native_compaction', detail.get('summary_api'),
                    detail.get('next_action'), 1, metadata.get('duration_ms')))
            for api in metric.get('selection_apis', []):
                phase = api.get('server', {})
                prefill = phase.get('native_post_semantic_ms', 0) > 1
                middecode = int(phase.get('kvmem_middecode_refreshes', 0))
                kind = '+'.join(name for name, present in (
                    ('prefill_selection', prefill), ('middecode_refresh', middecode)) if present)
                events.append(event_row(task, arm, kind, api, actions[arm].get(api['id']),
                                        int(prefill) + middecode))
    return events


def seconds(value):
    return '未观测' if value is None else f'{float(value) / 1000:.3f}'


def compaction_profile(events):
    compact = [e for e in events if e['kind'] == 'native_compaction']
    result = {'events': len(compact),
        'scope': 'Completed pairs only; event-weighted phase means, not equal-task means. Decode share uses only events with both native duration and decode timing.'}
    for key in ('native_compaction_ms', 'prefill_ms', 'decode_ms',
                'input_tokens', 'reused_tokens', 'prefilled_tokens'):
        observed = [float(e[key]) for e in compact if e[key] is not None]
        result[key] = {'n': len(observed),
            'mean': sum(observed) / len(observed) if observed else None,
            'max': max(observed) if observed else None}
    decoded = [e for e in compact if e['native_compaction_ms'] and e['decode_ms'] is not None]
    result['decode_share_events'] = len(decoded)
    result['decode_share_of_native_duration'] = (sum(e['decode_ms'] for e in decoded) /
        sum(e['native_compaction_ms'] for e in decoded)) if decoded else None
    reused = [e for e in compact if e['reused_tokens'] is not None]
    result['zero_reuse_summary_requests'] = sum(e['reused_tokens'] == 0 for e in reused)
    result['reuse_observed_events'] = len(reused)
    return result


def write_atomic(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def write_report(root):
    root = Path(root).resolve()
    summary, rows = summarize(root)
    events = maintenance_events(root, summary['completed_tasks'])
    compact = compaction_profile(events)
    write_atomic(root / 'compaction_phase_profile.json', json.dumps(compact, indent=2) + '\n')
    outcomes = {key: [] for key in ('both_success', 'qw3_only', 'kvmem_only', 'neither_success')}
    tool_errors = dict.fromkeys(('qw3', 'kvmem'), 0)
    for task in summary['completed_tasks']:
        pair = {r['arm']: r for r in rows if r['task'] == task}
        q, k = pair['qw3']['full_success'], pair['kvmem']['full_success']
        key = 'both_success' if q and k else 'qw3_only' if q else 'kvmem_only' if k else 'neither_success'
        outcomes[key].append(task)
        metrics = json.loads((root / 'trials' / task / 'analysis/metrics.json').read_text())
        for arm in tool_errors:
            tool_errors[arm] += len(metrics[arm]['tool_errors'])
    qualification_path = root / 'cohort_qualification.json'
    qualification = json.loads(qualification_path.read_text()) if qualification_path.exists() else {}
    write_atomic(root / 'paired_outcomes.json', json.dumps({
        'completed_pairs': summary['completed_pairs'],
        'definition': 'Success means official score exactly 1; partial scores remain in the per-task table.',
        'outcomes': {key: {'count': len(tasks), 'tasks': tasks} for key, tasks in outcomes.items()},
        'tool_errors': tool_errors,
        'cohort_qualification': qualification,
    }, indent=2) + '\n')
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EVENT_FIELDS)
    writer.writeheader()
    writer.writerows(events)
    write_atomic(root / 'maintenance_events.csv', output.getvalue())

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    text = [f'# OSWorld：普通 qw3 与 KVMem', '',
        f'生成时间：{now}。已完成 **{summary["completed_pairs"]}/{summary["planned_pairs"]} 对**。', '',
        '只统计双方均完成的配对，保留官方失败和部分得分。未完成样本不进入通过率分母。',
        '通过指官方分数恰为 1；进程正常退出不代表任务成功。单 seed 结果不能证明模型能力等效。', '',
        '| 指标 | 普通 qw3 | KVMem |', '|---|---:|---:|']
    if qualification.get('final_capability_comparison_eligible') is False:
        text[2:2] = [f'> 本批次已归档，不纳入最终能力比较。原因及替代批次见 [cohort_qualification.json]({qualification_path})。', '']

    def row(label, values):
        text.append(f'| {label} | {values[0]} | {values[1]} |')

    arms = [summary['arms'][arm] for arm in ('qw3', 'kvmem')]
    row('完整成功 / 已完成样本', [f'{a["full_successes"]}/{a["trials"]}' for a in arms])
    row('官方平均分', ['未观测' if a['mean_official_score'] is None else f'{a["mean_official_score"]:.3f}' for a in arms])
    for label, key in [('常规步骤 TTFT（秒）', 'regular_step_ttft_ms'),
                       ('全部步骤 TTFT（秒）', 'step_ttft_ms'),
                       ('常规动作等待（秒）', 'regular_decision_wait_ms'),
                       ('维护时动作等待（秒）', 'maintenance_action_wait_ms')]:
        row(label, [seconds(a['latency'][key]['equal_trial_mean_ms']) for a in arms])
    row('原生 compaction 次数', [a['native_compactions'] for a in arms])
    row('发生 selection 的请求数', [a['kvmem_selections'] for a in arms])
    row('其中输出过程中的刷新次数', ['未观测' if a['kvmem_middecode_refreshes'] is None else a['kvmem_middecode_refreshes'] for a in arms])
    row('执行期 API 错误数', [a['execution_api_errors'] for a in arms])
    row('客户端收到的工具错误数', [tool_errors[arm] for arm in ('qw3', 'kvmem')])
    for label, key in [('最高采样 GPU 总占用（GiB）', 'peak_sampled_gpu_gib'),
                       ('最高观测服务器 RSS（GiB）', 'peak_server_rss_gib'),
                       ('最高观测服务器 swap（GiB）', 'peak_server_swap_gib')]:
        values = []
        for arm in ('qw3', 'kvmem'):
            observed = [r[key] for r in rows if r['arm'] == arm and r[key] is not None]
            values.append(f'{max(observed):.2f}' if observed else '未观测')
        row(label, values)
    text += ['', '延迟主表先计算每个样本的均值，再对有观测的样本等权平均。维护事件缺失时不按零耗时计算。',
        'TTFT 从上一动作的观察就绪计时，到产生下一动作的响应开始输出；动作等待持续到实际发出下一动作。',
        '同一模型响应包含多个动作时，后续动作没有新的模型请求，TTFT 留空；实际动作等待仍计入统计。',
        '输出过程中发生的 reselect 可能不抬高 TTFT，因此必须同时看动作等待。不同方法的操作轨迹和生成量可能不同。', '',
        '## 配对成功情况', '', '| 结果 | 样本数 | 样本 |', '|---|---:|---|']
    for key, label in [('both_success', '双方完整成功'), ('qw3_only', '仅普通 qw3 完整成功'),
                       ('kvmem_only', '仅 KVMem 完整成功'), ('neither_success', '双方均未完整成功')]:
        text.append(f'| {label} | {len(outcomes[key])} | {", ".join(outcomes[key]) or "无"} |')
    text += ['', '未完整成功包含部分得分，详细分数保留在下表中。工具错误可能来自模型参数错误，不等于服务故障。', '',
        '## 已完成样本', '',
        '| 样本 | 官方分数：qw3 / KVMem | 动作数：qw3 / KVMem | 常规 TTFT 秒：qw3 / KVMem | 维护动作等待秒：qw3 / KVMem |',
        '|---|---:|---:|---:|---:|']
    for task in summary['completed_tasks']:
        pair = [next(r for r in rows if r['task'] == task and r['arm'] == arm) for arm in ('qw3', 'kvmem')]
        chart = root / 'trials' / task / 'analysis/latency_comparison.png'
        label = f'[{task}]({chart})' if chart.exists() else task
        text.append(f'| {label} | ' + ' | '.join((
            ' / '.join(f'{r["official_score"]:g}' for r in pair),
            ' / '.join(str(r['steps']) for r in pair),
            ' / '.join(seconds(r['regular_step_ttft_ms_mean']) for r in pair),
            ' / '.join(seconds(r['maintenance_action_wait_ms_mean']) for r in pair))) + ' |')
    pending = ', '.join(p['task'] for p in summary['pending_pairs'])
    text += ['', f'待完成配对：{pending or "无"}。', '']
    if summary.get('matched_build_groups'):
        text += ['## 程序版本与显存观测', '',
            '下表使用实际 server manifest 的程序指纹；同一样本的两组程序一致。',
            '不同批次版本分别保留，不把整个集合描述为同一程序版本。', '',
            '| 程序 SHA256 | 已完成的匹配样本 |', '|---|---|']
        for version, tasks in summary['matched_build_groups'].items():
            text.append(f'| `{version}` | {", ".join(tasks)} |')
        text += ['', 'GPU 数值是整张卡的采样占用；未录制显存的早期样本显示为未观测，不能据此推断其峰值。',
            '采样不能排除采样间隔内的短暂峰值，显存优化另用完整轨迹的 CUDA 分配观测验证。', '']
    if summary.get('pairs_with_different_binaries'):
        text += ['程序版本不匹配、需要单独审查的样本：' + ', '.join(summary['pairs_with_different_binaries']) + '。', '']
    profile_path = root / 'regular_request_profile.json'
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        covered = [task for task in summary['completed_tasks'] if task in profile['tasks']]
        text += ['## 常规请求的阶段观察', '',
            '范围为双方首次维护前的共同动作区间；只统计产生新动作响应的请求。',
            '轨迹、截图和请求体大小仍然不同，因此这些分项是观测对比，不能视为同输入速度测试。',
            '图片验证阶段包含 GPU 视觉编码，不能全部记为 CPU 管理开销。', '',
            '| 样本 | 共同动作区间 | 新图编码请求 / 观测请求：qw3；KVMem | 模型内部 TTFT 秒：qw3 / KVMem | 图片验证阶段秒：qw3 / KVMem |',
            '|---|---:|---|---:|---:|']
        for task in covered:
            trial = profile['tasks'][task]
            pair = [trial['arms'][arm]['common_steps_before_first_maintenance'] for arm in ('qw3', 'kvmem')]
            text.append(f'| {task} | 1–{trial["common_steps_before_first_maintenance"]} | ' + ' | '.join((
                '；'.join(f'{r["image_encoding_calls"]}/{r["action_responses"]}' for r in pair),
                ' / '.join(seconds(r['engine_ttft_ms']['mean'] if r['engine_ttft_ms'] else None) for r in pair),
                ' / '.join(seconds(r['pre_validate_ms']['mean'] if r['pre_validate_ms'] else None) for r in pair))) + ' |')
        text += ['', f'本分项分析当前覆盖 {", ".join(covered) or "无"}；详见 [regular_request_profile.json]({profile_path})。', '']
    text += ['## 维护事件与复核', '',
        f'逐事件数据见 [maintenance_events.csv]({root / "maintenance_events.csv"})，包含完整 compaction 时长、',
        'prefill、摘要生成、KV 复用、query/reselect 分项以及下一动作等待。未观测字段留空。',
        'query 总计已经包含它的生成和状态处理分项，不应重复相加；API TTFT 不等于完整维护耗时。', '',
        f'运行配置见 [plan.json]({root / "plan.json"})。每个样本的详细指标和曲线保存在对应 analysis 目录。',
        '固定请求回放用于验证效率及输出一致性，真实桌面任务使用原始判分器；二者不能互相替代。', '']
    text += ['## 原生 compaction 的耗时分解', '',
        f'已完成配对共记录 {compact["events"]} 次原生 compaction。下表按有观测的事件平均，计数分别列出。', '',
        '| 阶段 | 平均秒数 | 有观测事件数 |', '|---|---:|---:|']
    for key, label in [('native_compaction_ms', 'Claude Code 完整 compaction'),
                       ('prefill_ms', '摘要请求的模型 prefill'),
                       ('decode_ms', '摘要请求的模型生成')]:
        stat = compact[key]
        text.append(f'| {label} | {seconds(stat["mean"])} | {stat["n"]} |')
    share = compact['decode_share_of_native_duration']
    if share is not None:
        text += ['', f'在同时观测到完整压缩和生成耗时的 {compact["decode_share_events"]} 次事件中，生成累计耗时占完整压缩累计耗时的 {share:.1%}。']
    text += [f'有缓存统计的 {compact["reuse_observed_events"]} 次摘要请求中，零 token 复用为 {compact["zero_reuse_summary_requests"]} 次。',
        '这些 prefill 时间属于生成摘要的请求；压缩完成后，短摘要替换原历史所需的新请求另行计入动作等待。',
        f'具体输入、复用量和覆盖计数见 [compaction_phase_profile.json]({root / "compaction_phase_profile.json"})。', '']
    write_atomic(root / 'REPORT.md', '\n'.join(text))
    return {'completed_pairs': summary['completed_pairs'], 'planned_pairs': summary['planned_pairs'],
            'maintenance_event_rows': len(events), 'report': str(root / 'REPORT.md')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(write_report(args.root), indent=2))
