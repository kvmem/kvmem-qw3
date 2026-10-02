"""Read-only calibration report and continuous-video comparison page."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import re
import sqlite3
import statistics

from osworld_windows.capabilities import DEFINITIONS


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
        status=result['status'], strict_success=grade['strict_success'],
        valid_trial=result['pipeline_passed'] and result['status'] in {'passed', 'task_failed', 'timed_out', 'tool_budget_exhausted'},
        passed=result['status'] == 'passed', checks=grade['checks'],
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
        directory=str(directory.resolve()), video_seconds=result.get('recording', {}).get('video_duration_seconds'))


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
        passed=sum(row['passed'] for row in rows), trials=rows,
        limitations=['Single attempts establish observed feasibility, not stable reliability.',
                     'Short contexts do not measure long-history KVMem retention benefits.',
                     'On/off use the current product configurations, including prefix cache differences.',
                     'Videos start at capture start and include startup/cleanup; actions are not aligned.'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    fields = ['task', 'name', 'kvmem', 'run', 'status', 'passed', 'partial_check_fraction', 'task_seconds',
              'inference_requests', 'median_request_seconds', 'median_server_first_token_ms',
              'median_visible_response_ms', 'largest_input_tokens', 'actions']
    with (output / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        sections = []
        for task in dict.fromkeys(row['task'] for row in rows):
            videos = []
            for row in [r for r in rows if r['task'] == task]:
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
                if review.get('failure_class') == 'environment_default_app_choice_dialog':
                    note += '<p>默认打开触发系统选择弹窗，受执行范围限制而暂停；环境修正后重新测试。</p>'
                if row['strict_success'] and not row['valid_trial']:
                    note += '<p>保存产物独立验收正确；收尾链路失败，此条不计入模型成功率。</p>'
                videos.append(f'<article><h3>KVMem {row["kvmem"]} · {outcome}</h3>'
                    f'<p>试次：{label}</p>'
                    f'{media}<p>任务 {row["task_seconds"]:.1f} 秒 · {row["inference_requests"]} 次推理 · '
                    f'{row["actions"]} 次动作 · {statuses.get("rejected", 0)} 次动作拒绝 · '
                    f'{len(row["tool_contract_errors"])} 次参数错误</p>'
                    f'{note}'
                    f'<details><summary>独立验收</summary><pre>{html.escape(json.dumps(row["checks"], ensure_ascii=False, indent=2))}</pre></details></article>')
            sections.append(f'<section><h2>{task} · {html.escape(DEFINITIONS[task][0])}</h2>'
                '<button class="play">从头同时播放</button> <button class="pause">同时暂停</button>'
                f'<div class="pair">{"".join(videos)}</div></section>')
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Windows 能力边界回放</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#172b40;margin:0}main{max-width:1400px;margin:auto;padding:28px}section{background:white;padding:20px;border-radius:12px;margin:22px 0}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%;background:#161b22}button{padding:9px 16px}pre{white-space:pre-wrap}.note{padding:16px;background:#e8edf7}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<main><h1>Windows 能力边界回放</h1><p class="note">自定义短任务 · 真实 QW3+DFlash2 与原生 Codex · 相同快照、提示与预算的开关配对。连续录像包含启动和收尾，从录像开头播放，未按操作步骤对齐。结果不是官方 OSWorld 分数；单次结果不能证明稳定性或 KVMem 长历史收益。</p>'''
        page += f'<p>已完成 {len(rows)} 次：{result["valid_attempts"]} 次有效、{result["invalid_attempts"]} 次链路无效；{result["passed"]} 次完整通过。无效测试不计入模型成功率。运行中请刷新查看已完成结果。</p>' + ''.join(sections)
        page += '''</main><script>for(const s of document.querySelectorAll('section')){const vs=[...s.querySelectorAll('video')];s.querySelector('.play').onclick=()=>{for(const v of document.querySelectorAll('video'))v.pause();for(const v of vs)v.currentTime=0;Promise.allSettled(vs.map(v=>v.play()));};s.querySelector('.pause').onclick=()=>vs.forEach(v=>v.pause());}</script></html>'''
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
