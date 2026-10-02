"""Read-only calibration report and continuous-video comparison page."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import statistics

from osworld_windows.capabilities import DEFINITIONS

DISPLAY_NAMES = dict(zip((f'cap{i:02d}' for i in range(1, 11)), [
    '创建并保存笔记', '修改笔记并保留原文件', '浏览器提取信息后保存', '网页填表并下载结果',
    '表格公式及求和', '文档标题与文字替换', '修改与删除指定幻灯片',
    '读取新旧规则并计算', '文件改名与归档', '表格整行排序及求和']))


def summarize(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    terminal = result['native_result']['task']
    started = terminal['started_unix_ns'] / 1e9
    database = directory / 'guest/app/desktop/desktop.sqlite'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only = ON')
        rows = db.execute('SELECT request,status,result,created FROM actions ORDER BY created').fetchall()
    first_rejection = None
    for request, status, response, created in rows:
        if status == 'rejected':
            response = json.loads(response or '{}')
            first_rejection = dict(task_seconds=created - started, action=json.loads(request).get('action'),
                                   error=response.get('error'))
            break
    metrics = [json.loads(p.read_text()) for p in (directory / 'guest/app/inference').glob('*/metrics.json')]
    request_seconds = [m['elapsed_seconds'] for m in metrics if isinstance(m.get('elapsed_seconds'), (int, float))]
    tokens = [m.get('usage', {}).get('input_tokens') for m in metrics]
    tokens = [n for n in tokens if isinstance(n, int)]
    grade = result['independent_grade']
    status = result['status']
    correction = directory / 'independent-grade-v2.json'
    if correction.is_file():
        corrected = json.loads(correction.read_text())
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = directory / 'guest/artifacts/cap07.pptx'
        if (result['task'] != 'cap07'
                or corrected['original_result_sha256'] != digest(directory / 'result.json')
                or corrected['grader_source_sha256'] != digest(Path(__file__).with_name('capability_regrade.py'))
                or corrected.get('output_sha256') != digest(artifact)):
            raise ValueError('Derived presentation grade provenance mismatch')
        grade = corrected
        if status in {'passed', 'task_failed'} and terminal['outcome'] == 'agent_completed':
            status = 'passed' if grade['strict_success'] else 'task_failed'
    tool_items = {}
    for line in (directory / 'guest/app/codex-events.jsonl').read_text().splitlines():
        event = json.loads(line); item = event.get('params', {}).get('item', {})
        if event.get('method') == 'item/completed' and item.get('type') == 'mcpToolCall':
            tool_items[item['id']] = item
    contract_errors = []
    for item in tool_items.values():
        result_text = ' '.join(c.get('text', '') for c in (item.get('result') or {}).get('content', []) if c.get('type') == 'text')
        if result_text.startswith(('action_id must ', 'desktop_id must ', 'observation_id must ')):
            contract_errors.append(dict(call_id=item['id'], error=result_text))
    resources = [json.loads(line) for line in (directory / 'engine-resources.jsonl').read_text().splitlines()]
    timing_rows = []
    for line in (directory / 'engine/engine.log').read_text().splitlines():
        if line.startswith('[qw3-server-ttft]'):
            timing_rows.append(dict(re.findall(r'(\w+)=([0-9.]+)', line)))
    first_tokens = [float(r['server_ttft_ms']) for r in timing_rows if r.get('model_token_seen') == '1']
    visible_output = [float(r['response_ttft_ms']) for r in timing_rows if r.get('visible_output_seen') == '1']
    review = directory / 'trajectory-review.json'
    return dict(task=result['task'], name=DEFINITIONS[result['task']][0], kvmem=result['kvmem'],
        run=result['run'], formal_run=False, pipeline_passed=result['pipeline_passed'],
        status=status, original_status=result['status'],
        grader_revision=grade.get('grader_revision'), strict_success=grade['strict_success'],
        valid_trial=result['pipeline_passed'] and result['status'] in {'passed', 'task_failed', 'timed_out', 'tool_budget_exhausted'},
        passed=status == 'passed', checks=grade['checks'],
        partial_check_fraction=grade.get('partial_check_fraction', 0),
        task_seconds=terminal['elapsed_seconds'], inference_requests=len(metrics),
        inference_seconds_sum=sum(request_seconds),
        median_request_seconds=statistics.median(request_seconds) if request_seconds else None,
        max_request_seconds=max(request_seconds) if request_seconds else None,
        median_server_first_token_ms=statistics.median(first_tokens) if first_tokens else None,
        median_visible_response_ms=statistics.median(visible_output) if visible_output else None,
        largest_input_tokens=max(tokens) if tokens else None,
        sampled_gpu_device_peak_bytes=max((r.get('gpu_memory_bytes', 0) for r in resources), default=0),
        sampled_engine_tree_peak_rss_bytes=max((r.get('peak_rss_bytes', 0) for r in resources), default=0),
        actions=len(rows), action_statuses=dict(Counter(row[1] for row in rows)),
        completed_tool_calls=len(tool_items), tool_contract_errors=contract_errors,
        action_kinds=dict(Counter(json.loads(row[0]).get('action', {}).get('kind') for row in rows)),
        first_action_rejection=first_rejection,
        semantic_first_error='Requires trajectory review; absence of rejected actions does not prove correctness',
        trajectory_review=json.loads(review.read_text()) if review.is_file() else None,
        prompt_sha256=terminal.get('prompt_sha256'),
        snapshot=result.get('snapshot', {}).get('name'),
        timeout_seconds=result.get('timeout_seconds'),
        tool_call_limit=result.get('tool_call_limit'),
        visual_history=result.get('visual_history'),
        directory=str(directory.resolve()), video_seconds=result.get('recording', {}).get('video_duration_seconds'))


def cohort_key(row):
    return tuple(row.get(key) for key in ('task', 'prompt_sha256', 'snapshot',
                                         'timeout_seconds', 'tool_call_limit', 'visual_history'))


def matched_pairs(rows):
    groups = {}
    for row in rows:
        if row['valid_trial']:
            groups.setdefault(cohort_key(row), {}).setdefault(row['kvmem'], []).append(row)
    pairs = []
    for key, arms in groups.items():
        # Repeated trials remain separate; the newest attempt is not substituted
        # for an earlier failure when computing a pair.
        for on, off in zip(arms.get('on', []), arms.get('off', [])):
            pairs.append(dict(task=key[0], on=on['run'], off=off['run'],
                on_passed=on['passed'], off_passed=off['passed'],
                on_artifact_correct=on['strict_success'], off_artifact_correct=off['strict_success'],
                on_seconds=on['task_seconds'], off_seconds=off['task_seconds'],
                speedup=off['task_seconds'] / on['task_seconds']
                    if on['passed'] and off['passed'] else None,
                snapshot=on['snapshot'], timeout_seconds=on['timeout_seconds']))
    return pairs


def write_markdown(result, output):
    rows = result['trials']
    lines = ['# Windows 能力边界实测', '',
        '自定义能力校准，真实 QW3 + DFlash2 + Codex；不是官方 OSWorld 分数。', '',
        f"已结束 {result['completed_attempts']} 条试次：{result['valid_attempts']} 条有效，"
        f"{result['invalid_attempts']} 条环境受阻或链路无效；"
        f"{result['passed']} 条按时完整通过，{result['valid_artifact_correct']} 条有效试次产物正确。", '',
        '## 同配置开关结果', '',
        '| 场景 | 预算 | KVMem 开 | KVMem 关 |', '|---|---:|---|---|']
    groups = {}
    for row in rows:
        if row['valid_trial']:
            groups.setdefault(cohort_key(row), {}).setdefault(row['kvmem'], []).append(row)

    def outcome(row):
        if row is None:
            return '尚未完成'
        state = '通过' if row['passed'] else ('产物正确，未按时结束' if row['strict_success'] else '未通过')
        return f"{state}，{row['task_seconds']:.1f} 秒"

    for key, arms in groups.items():
        for i in range(max(len(v) for v in arms.values())):
            pair = {arm: trials[i] if i < len(trials) else None for arm, trials in arms.items()}
            lines.append(f"| {key[0]} · {DISPLAY_NAMES[key[0]]} | {key[3]} 秒 | "
                f"{outcome(pair.get('on'))} | {outcome(pair.get('off'))} |")
    lines += ['', '## 有效试次耗时与等待', '',
        '| 试次 | 推理次数 | 推理等待合计 | 首 token 中位数 | 动作完成/拒绝 |',
        '|---|---:|---:|---:|---|']
    for row in rows:
        if row['valid_trial']:
            ttft = row['median_server_first_token_ms']
            first_token = f'{ttft / 1000:.2f} 秒' if ttft is not None else '未测得'
            statuses = row['action_statuses']
            lines.append(f"| {row['run']} | {row['inference_requests']} | "
                f"{row['inference_seconds_sum']:.1f} 秒 | {first_token} | "
                f"{statuses.get('completed', 0)}/{statuses.get('rejected', 0)} |")
    lines += ['', '## 解释范围', '',
        '- 保存文件独立验收正确与任务按时正常结束分别统计；超时不改判为完整通过。',
        '- 每配置单次摸底只能证明已观察到的可行性，不能证明稳定可靠或统计显著收益。',
        '- 开关使用当前产品配置，包含前缀缓存差异；轨迹不同也会改变耗时。',
        '- 任务时间不包含快照恢复和模型加载；录像包含启动与收尾，未按动作对齐。',
        '- 不同快照、提示或预算分组展示，保留全部失败和受阻录像。', '',
        f"[完整统计与轨迹说明]({(output / 'summary.json').resolve()}) · "
        '[连续录像回放](http://127.0.0.1:18089/capability/index.html)', '']
    (output / 'report.md').write_text('\n'.join(lines))


def build(queues, output, preview):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for queue in queues:
        manifest = json.loads((queue / 'result.json').read_text())
        for trial in manifest['trials']:
            path = Path(trial['result_path']).parent
            if (path / 'result.json').exists() and (path / 'independent-grade.json').exists():
                rows.append(summarize(path))
    result = dict(suite='custom_capability_calibration', formal_run=False,
        osworld_official_tasks=False, completed_attempts=len(rows),
        valid_attempts=sum(row['valid_trial'] for row in rows),
        invalid_attempts=sum(not row['valid_trial'] for row in rows),
        passed=sum(row['passed'] for row in rows), trials=rows, matched_pairs=matched_pairs(rows),
        valid_artifact_correct=sum(row['valid_trial'] and row['strict_success'] for row in rows),
        limitations=['Single attempts establish observed feasibility, not stable reliability.',
                     'Short contexts do not measure long-history KVMem retention benefits.',
                     'On/off use the current product configurations, including prefix cache differences.',
                     'Videos start at capture start and include startup/cleanup; actions are not aligned.'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    write_markdown(result, output)
    fields = ['task', 'name', 'kvmem', 'run', 'status', 'passed', 'partial_check_fraction', 'task_seconds',
              'inference_requests', 'median_request_seconds', 'median_server_first_token_ms',
              'median_visible_response_ms', 'largest_input_tokens', 'actions']
    with (output / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        sections = []
        for task in dict.fromkeys(row['task'] for row in rows):
            task_rows = [r for r in rows if r['task'] == task]
            valid_rows = [r for r in task_rows if r['valid_trial']]
            # Put one common-configuration cohort in the comparison view. Old
            # snapshots, budgets and invalid attempts remain available below.
            primary_key = cohort_key(valid_rows[-1]) if valid_rows else None
            primary = [r for r in valid_rows if cohort_key(r) == primary_key]
            archived = [r for r in task_rows if r not in primary]
            videos = []
            for row in primary + archived:
                source = Path(row['directory']) / 'video/continuous.mp4'
                if not source.exists():
                    # Qualified recorder versions may name the final file console.mp4.
                    candidates = list((Path(row['directory']) / 'video').glob('*.mp4'))
                    source = candidates[0] if len(candidates) == 1 else None
                label = html.escape(row['run'])
                media = '<p>暂无可播放 MP4</p>'
                if source and source.is_file():
                    link = preview / (row['run'] + '.mp4')
                    if not link.exists(): link.symlink_to(source.resolve())
                    media = f'<video controls preload="metadata" src="{label}.mp4"></video>'
                statuses = row['action_statuses']
                review = row.get('trajectory_review') or {}
                note = '<p>最终画面确认编辑正确，但未保存目标文件。</p>' if review.get('visible_edit_milestones_verified') else ''
                outcome = '受阻／无效试次' if not row['valid_trial'] else ('通过' if row['passed'] else '未通过')
                if review.get('failure_class') in {'environment_default_app_choice_dialog',
                        'environment_default_text_app_choice_dialog'}:
                    note += '<p>默认打开触发系统选择弹窗，受执行范围限制而暂停；环境修正后重新测试。</p>'
                if row['strict_success'] and not row['valid_trial']:
                    note += '<p>保存产物独立验收正确；收尾链路失败，此条不计入模型成功率。</p>'
                if row['valid_trial'] and row['strict_success'] and not row['passed']:
                    note += '<p>目标文件全部通过验收；达到时间／工具上限，未正常结束。</p>'
                if row['grader_revision']:
                    note += '<p>按实际标题和正文复核；LibreOffice 占位符编号变化造成的首次错误判分已保留。</p>'
                policy = '开' if row['kvmem'] == 'on' else '关'
                videos.append(f'<article><h3>KVMem {policy} · {outcome}</h3>'
                    f'<p>试次：{label}</p>'
                    f'{media}<p>任务 {row["task_seconds"]:.1f} 秒 · {row["inference_requests"]} 次推理 · '
                    f'{row["actions"]} 次动作 · {statuses.get("rejected", 0)} 次动作拒绝 · '
                    f'{len(row["tool_contract_errors"])} 次 ID 参数错误</p>'
                    f'{note}'
                    f'<details><summary>独立验收</summary><pre>{html.escape(json.dumps(row["checks"], ensure_ascii=False, indent=2))}</pre></details></article>')
            common = primary[0] if primary else None
            configuration = (f'<p>共同快照：{html.escape(common["snapshot"] or "未记录")} · '
                f'预算 {common["timeout_seconds"]} 秒 · 工具次数上限 {common["tool_call_limit"]}</p>') if common else ''
            main_videos = ''.join(videos[:len(primary)])
            history = ('<details class="history"><summary>历史环境受阻／其他配置试次（保留）</summary>'
                f'<div class="pair">{"".join(videos[len(primary):])}</div></details>') if archived else ''
            pending = '<p>同配置的另一侧尚未完成。</p>' if len({r['kvmem'] for r in primary}) < 2 else ''
            sections.append(f'<section><h2>{task} · {html.escape(DISPLAY_NAMES[task])}</h2>{configuration}{pending}'
                '<button class="play">从头同时播放</button> <button class="pause">同时暂停</button>'
                f'<div class="pair current">{main_videos}</div>{history}</section>')
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Windows 能力边界回放</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#172b40;margin:0}main{max-width:1400px;margin:auto;padding:28px}section{background:white;padding:20px;border-radius:12px;margin:22px 0}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%;background:#161b22}button{padding:9px 16px}pre{white-space:pre-wrap}.note{padding:16px;background:#e8edf7}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<main><h1>Windows 能力边界回放</h1><p class="note">自定义短任务 · 真实 QW3+DFlash2 与原生 Codex · 相同快照、提示与预算的开关配对。连续录像包含启动和收尾，从录像开头播放，未按操作步骤对齐。结果不是官方 OSWorld 分数；单次结果不能证明稳定性或 KVMem 长历史收益。</p>'''
        page += f'<p>已完成 {len(rows)} 次：{result["valid_attempts"]} 次有效、{result["invalid_attempts"]} 次受阻或无效；{result["passed"]} 次完整通过，{result["valid_artifact_correct"]} 次有效试次产物正确。受阻或无效测试单独保留。运行中请刷新查看已完成结果。</p>' + ''.join(sections)
        page += '''</main><script>for(const s of document.querySelectorAll('section')){const vs=[...s.querySelectorAll('.current video')];s.querySelector('.play').onclick=()=>{for(const v of document.querySelectorAll('video'))v.pause();for(const v of vs)v.currentTime=0;Promise.allSettled(vs.map(v=>v.play()));};s.querySelector('.pause').onclick=()=>s.querySelectorAll('video').forEach(v=>v.pause());}</script></html>'''
        (preview / 'index.html').write_text(page)
    print(json.dumps({k: result[k] for k in ['completed_attempts', 'passed', 'formal_run']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    build(args.queue, args.output, args.preview)
