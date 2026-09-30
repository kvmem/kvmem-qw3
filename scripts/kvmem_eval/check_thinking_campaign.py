#!/usr/bin/env python3
"""Thirty-minute experiment heartbeat; notify this Codex task only on findings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/data/chaidi/qw3-experiments/thinking_summary_first16_pass4_promptfix_20260923')
THREAD = '01a0c8ad-2510-76e1-a033-b2fb0d9bbb7d'
TIMER = 'qw3-thinking-first16-monitor.timer'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def alive(pid, start=None):
    try:
        values = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return values[0] != 'Z' and (start is None or int(values[19]) == int(start))
    except (OSError, ValueError, TypeError, IndexError):
        return False


def inspect(root):
    now = time.time()
    state, summary, live, owner = [read(root / name) for name in
        ('state.json', 'summary.json', 'live.json', 'supervisor_owner.json')]
    phase = state.get('phase')
    completed = summary.get('completed_trials', 0)
    finished = phase == 'finished' and completed == 64
    issues = []
    compactions = {}
    cache = {'after_compaction': 0, 'hits': 0, 'query_snapshot_reuse': 0,
             'query_boundary_misses': 0}
    if not state:
        issues.append('experiment state missing or unreadable')
    elif phase == 'needs_attention':
        issues.append('runner needs attention: ' + str(state.get('reason')))
    elif phase not in ('finished', 'stopped', 'stopped_by_user'):
        if not alive(owner.get('pid'), owner.get('start_ticks')):
            issues.append('supervisor is no longer running')
        if phase == 'running_trial':
            heartbeat_age = now - (root / 'live.json').stat().st_mtime if (root / 'live.json').exists() else None
            if heartbeat_age is None or heartbeat_age > 180:
                issues.append('supervisor heartbeat is missing or older than 180 seconds')
            if not alive(state.get('worker_pid')):
                issues.append('current trial worker is no longer running')
            artifact = Path(state.get('artifact', '/nonexistent'))
            attempt = artifact.parents[2] if len(artifact.parents) >= 3 else root
            trace = attempt / 'requests.jsonl'
            if trace.exists():
                previous_compacted = False
                for line in trace.open():
                    try:
                        request = json.loads(line)
                    except ValueError:  # Writer may be in the middle of a line.
                        continue
                    if previous_compacted:
                        prefix = request.get('prefix_cache', {})
                        cache['after_compaction'] += 1
                        cache['hits'] += bool(prefix.get('hit'))
                        cache['query_snapshot_reuse'] += bool(prefix.get('hit')) and prefix.get('query_snapshot') == 1
                        cache['query_boundary_misses'] += prefix.get('reason') == 'checkpoint_after_query_boundary'
                    previous_compacted = any(e.get('status') == 'compacted'
                        for e in request.get('compactions', []))
                    for event in request.get('compactions', []):
                        status = event.get('status', 'unknown')
                        compactions[status] = compactions.get(status, 0) + 1
                attempts = sum(compactions.get(k, 0) for k in ('compacted', 'invalid-summary', 'not-shorter'))
                if attempts >= 3 and compactions.get('compacted', 0) == 0:
                    issues.append('at least three summary attempts with no successful compaction; inspect summary output')
                if cache['query_boundary_misses'] >= 2:
                    issues.append('repeated query-boundary cache misses after compaction; inspect query snapshots and prompt changes')
            services = list((attempt / 'services').glob('*/service_owner.json'))
            for service_file in services:
                service = read(service_file)
                if not alive(service.get('server_pgid'), service.get('server_start_ticks')):
                    issues.append('current inference service is no longer running')
            paths = [artifact / 'claude.stream.jsonl', artifact / 'agent.console.log',
                     artifact / 'verifier.console.log', *list((attempt / 'services').glob('*/qw3_server.log'))]
            mtimes = [p.stat().st_mtime for p in paths if p.exists()]
            if mtimes and now - max(mtimes) > 1800:
                issues.append('no agent/verifier/server log progress for 30 minutes; inspect before restarting')
            for log in (attempt / 'services').glob('*/qw3_server.log'):
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 262144))
                    tail = stream.read().decode(errors='replace')
                tail = '\n'.join(line for line in tail.splitlines() if '[thinking-summary-output]' not in line)
                if re.search(r'terminal_status=error|POST .* -> 5\d\d|CUDA error|illegal memory access', tail):
                    issues.append('inference error found in the current server log')
    free = shutil.disk_usage(root).free
    if free < 16 * 1024**3:
        issues.append('experiment data disk has less than 16 GiB free')
    last = live.get('last_request') or {}
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
            'completed_trials': completed, 'total_trials': 64,
            'round': state.get('round'), 'sample_index': state.get('sample_index'),
            'task': state.get('task'), 'attempt': state.get('attempt'),
            'requests': live.get('requests'), 'elapsed_sec': live.get('elapsed_sec'),
            'latest_context_tokens': last.get('output_context_tokens'), 'latest_pool': last.get('pool'),
            'current_trial_compaction_counts': compactions,
            'post_compaction_cache': cache,
            'data_disk_free_bytes': free, 'issues': issues, 'finished': finished,
            'completed_rounds': summary.get('completed_rounds', [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--no-notify', action='store_true')
    args = parser.parse_args()
    directory = args.root / 'monitor'
    directory.mkdir(exist_ok=True)
    previous = read(directory / 'status.json')
    check = inspect(args.root)
    # Keep start notification pending across failed queue deliveries and manual
    # --no-notify inspections. A phase transition alone would lose that wakeup.
    check['start_notified'] = previous.get('start_notified', False)
    started = check['phase'] == 'running_trial' and not check['start_notified']
    kind = 'completed' if check['finished'] else 'issue' if check['issues'] else 'started' if started else None
    signature = hashlib.sha256(json.dumps([kind, check['task'] if kind == 'issue' else None,
        check['attempt'] if kind == 'issue' else None, check['issues']], sort_keys=True).encode()).hexdigest()
    if kind and signature != previous.get('notified_signature') and not args.no_notify:
        prompt = ('[DeepSWE 每30分钟定时检查] 用户已授权监控和及时修复。'
            '本任务的修复目标以新版测试启动为完成条件；若目标尚未完成（active或因GPU排队blocked）且现在已实际启动，'
            '核实真实请求和定时巡检后可标记该目标完成。此后继续监控后台64次摘要实验，'
            '不要重新创建或恢复已完成目标。每轮完整16个样本，4轮，禁止重新跑baseline。'
            f'实验目录：{args.root}。检查结果：{json.dumps(check, ensure_ascii=False)}。'
            '请读取当前state/live/summary和相关日志核实；只处理本实验的推理或测试环境问题，'
            '保留失败attempt及冻结版本审计，修复验证后恢复，不能改任务答案或按通过率筛选重试。'
            '不要对正在计时的GPU服务发送额外模型探测请求。只在异常、重要进展、需用户输入或全部完成时通知用户；'
            '状态正常或问题已报告且无新变化时保持安静，不发例行总结。'
            '全部64次有效trial完成后检查并汇总报告，关闭此定时巡检。')
        try:
            result = subprocess.run(['/home/chaidi/.local/bin/codex', 'queue', '--thread', THREAD,
                '--message', prompt], capture_output=True, text=True, timeout=60)
            check['notification_returncode'] = result.returncode
            check['notification_output'] = (result.stdout + result.stderr)[-2000:]
            if result.returncode == 0:
                check['notified_signature'] = signature
                if kind == 'started':
                    check['start_notified'] = True
        except (OSError, subprocess.TimeoutExpired) as error:
            check['notification_error'] = str(error)
    if 'notified_signature' not in check and kind:
        check['notified_signature'] = previous.get('notified_signature')
    write(directory / 'status.json', check)
    with (directory / 'checks.jsonl').open('a') as log:
        log.write(json.dumps(check, ensure_ascii=False) + '\n')
    if check['finished'] and check.get('notified_signature') == signature and not args.no_notify:
        subprocess.run(['systemctl', '--user', 'disable', '--now', TIMER], check=True)
    print(json.dumps({'checked_at': check['checked_at'], 'phase': check['phase'],
                      'completed_trials': check['completed_trials'], 'issues': check['issues']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
