#!/usr/bin/env python3
"""Combine disjoint OSWorld cohorts, retaining failed tasks and version strata.

Reads completed artifacts only; never launches trials or changes their inputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze_claude_pair import server_records
from summarize_claude_suite import ARMS, LATENCIES, official_score, summarize


def read_json(path, evidence):
    raw = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def audit_manifests(plan, task, arm, server, client):
    """Check recorded launch configuration without copying credentials to output."""
    expected = plan.get('pair_overrides', {}).get(task, plan)
    revision = plan.get('episode_end_revision', {})
    driver_sources = (revision['old_driver_sources']
                      if f'{task}/{arm}' in revision.get('retained_result_sha256', {})
                      else plan['driver_sources'])
    checks = {
        'binary_sha256': server.get('binary_sha256') == expected['binary_sha256'],
        'server_command': server.get('command') == expected['arms'][arm]['command'],
        'server_environment': server.get('environment') == expected['arms'][arm]['environment'],
        'server_identity': (server.get('task'), server.get('arm'), server.get('seed')) == (task, arm, plan['seed']),
        'client_version': client.get('version') == plan['client']['version'],
        'native_history': client.get('native_history') is True,
        'no_custom_summary': client.get('custom_summary') is False,
        'subagents': client.get('subagents') == plan['client']['subagents'],
        'client_seed': client.get('seed') == plan['seed'],
        'max_actions': client.get('max_actions') == plan['execution']['max_desktop_actions'],
        'driver_sources': client.get('driver_sources') == driver_sources,
    }
    env_revision = plan.get('environment_revision')
    if env_revision and task not in env_revision['retained_pairs']:
        checks['environment_monitor'] = client.get('environment_monitor') == {
            'interval_seconds': env_revision['root_mount_check_interval_seconds'],
            'read_only_guest_metadata': True,
            'retain_environment_on_error': env_revision['retain_environment_on_error'],
            'guest_volume_size_gb': env_revision['guest_volume_size_gb'],
        }
    environment = client.get('environment', {})
    # The manifest intentionally records a subset of client environment fields.
    # Updater settings and transport credentials are not part of this comparison.
    expected_env = {k: plan['client']['common_env'][k] for k in
                    ('CLAUDE_CODE_MAX_OUTPUT_TOKENS', 'MAX_THINKING_TOKENS')}
    expected_env.update(plan['client'][arm + '_env'])
    checks['client_budget_and_compaction_environment'] = all(environment.get(k) == v for k, v in expected_env.items())
    if arm == 'qw3':
        checks['no_compaction_disable_override'] = not any(
            environment.get(k) not in (None, '', '0') for k in ('DISABLE_COMPACT', 'DISABLE_AUTO_COMPACT'))
    failed = [k for k, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'Launch manifest mismatch for {task}/{arm}: ' + ', '.join(failed))
    return sorted(checks)


def guided_query_events(path, cohort, task, arm, evidence):
    raw = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(raw).hexdigest()
    events = []
    prefixes = {
        '[qw3] native kvmem guided query: ': ('prefill', 'mode'),
        '[qw3] native kvmem middecode guided reselect: refresh=': ('middecode', 'query_mode'),
    }
    for lineno, line in enumerate(raw.decode().splitlines(), 1):
        matched = next((value for prefix, value in prefixes.items() if line.startswith(prefix)), None)
        if matched is None:
            continue
        phase, mode_key = matched
        mode = re.search(r'\b' + mode_key + r'=(\S+)', line)
        if mode is None:
            raise ValueError(f'Missing guided query mode at {path}:{lineno}')
        reason = re.search(r'\breason=([a-z][a-z-]*)\s*$', line)
        event = dict(cohort=cohort, task=task, arm=arm, phase=phase,
                     mode=mode.group(1), fallback_original=mode.group(1) == 'fallback-original',
                     fallback_reason=reason.group(1) if reason else None,
                     server_log=str(path), server_log_line=lineno)
        # Do not export query text, which can contain complete user instructions.
        header = line.split(' query=', 1)[0]
        for key in ('query_tokens', 'private_prompt_ms', 'generated_query_ms', 'state_ms'):
            value = re.search(r'\b' + key + r'=([0-9.]+)', header)
            event[key] = float(value.group(1)) if value else None
        events.append(event)
    return events


def model_work(run, metric, evidence):
    path = run / 'api.jsonl'
    raw = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(raw).hexdigest()
    requests = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    summary_ids = {d['summary_api']['id'] for d in metric.get('compaction_details', []) if d.get('summary_api')}
    if len(summary_ids) != metric['native_compactions']:
        raise ValueError('Unmatched compaction API records: ' + str(run))
    server_path = run / 'qw3_server.log'
    evidence[str(server_path)] = hashlib.sha256(server_path.read_bytes()).hexdigest()
    server = server_records(server_path)
    decoded = [server[r['id']]['native_decode_ms'] for r in requests
               if r['id'] in server and 'native_decode_ms' in server[r['id']]]
    step_path = run / 'steps.jsonl'
    raw_steps = step_path.read_bytes()
    evidence[str(step_path)] = hashlib.sha256(raw_steps).hexdigest()
    terminal = next((s for s in (json.loads(line) for line in raw_steps.decode().splitlines())
                     if s.get('done')), None)
    cutoff = terminal['observation_ready_ns'] if terminal else None
    after = [r for r in requests if cutoff is not None and r['started_ns'] >= cutoff]
    errors = [r for r in requests if r.get('status', 200) >= 400 or r.get('stream_error') or r.get('relay_error')]
    return dict(
        api_requests=len(requests),
        output_usage_observed_requests=sum('output_tokens' in r.get('output_usage', {}) for r in requests),
        output_usage_missing_requests=sum('output_tokens' not in r.get('output_usage', {}) for r in requests),
        output_tokens=sum(r.get('output_usage', {}).get('output_tokens', 0) for r in requests),
        non_summary_output_tokens=sum(r.get('output_usage', {}).get('output_tokens', 0)
                                      for r in requests if r['id'] not in summary_ids),
        summary_api_requests=len(summary_ids),
        api_total_seconds=sum(r['total_ms'] for r in requests) / 1000,
        native_decode_seconds=sum(decoded) / 1000,
        native_decode_observed_requests=len(decoded),
        desktop_end_boundary_observed=cutoff is not None,
        post_desktop_end_api_requests=len(after) if cutoff is not None else None,
        post_desktop_end_output_usage_observed_requests=sum('output_tokens' in r.get('output_usage', {}) for r in after) if cutoff is not None else None,
        post_desktop_end_output_tokens=sum(r.get('output_usage', {}).get('output_tokens', 0) for r in after) if cutoff is not None else None,
        post_desktop_end_api_total_seconds=sum(r['total_ms'] for r in after) / 1000 if cutoff is not None else None,
        post_desktop_end_api_errors=sum(r in after for r in errors) if cutoff is not None else None,
        api_total_seconds_before_desktop_end=sum(r['total_ms'] for r in requests if r not in after) / 1000 if cutoff is not None else None)


def resolve_event_actions(root, events, evidence):
    """Keep exported fields and add action mapping from the full decision interval.

    A non-action response can follow a summary before the actual tool call.
    Match the event request to raw desktop timestamps, then use the existing
    per-step latency row instead of treating that intermediate response as an
    action. Multiple events can belong to one action; their waits are not additive.
    """
    cached = {}
    resolved = []
    for event in events:
        key = event['task'], event['arm']
        if key not in cached:
            trial = root / 'trials' / key[0]
            run = trial / 'full' / key[1]
            data = []
            for path in (run / 'api.jsonl', run / 'steps.jsonl', trial / 'analysis' / 'step_latency.csv'):
                raw = path.read_bytes()
                evidence[str(path)] = hashlib.sha256(raw).hexdigest()
                data.append(raw.decode())
            apis = {r['id']: r for r in map(json.loads, data[0].splitlines())}
            steps = list(map(json.loads, data[1].splitlines()))
            latencies = {int(r['step']): r for r in csv.DictReader(data[2].splitlines()) if r['arm'] == key[1]}
            cached[key] = apis, steps, latencies
        apis, steps, latencies = cached[key]
        api = apis[int(event['api_id'])]
        matches = [step for previous, step in zip(steps, steps[1:])
                   if previous['observation_ready_ns'] <= api['started_ns'] < step['started_ns']]
        if len(matches) > 1:
            raise ValueError(f'Overlapping decision intervals for {key}, API {api["id"]}')
        mapping = dict(action_mapping='no_enclosing_decision_interval', resolved_action_step=None,
                       resolved_action_api_id=None, resolved_action_wait_ms=None,
                       resolved_action_first_content_ms=None)
        if matches:
            step = matches[0]
            latency = latencies[step['step']]
            if not math.isclose(float(latency['decision_wait_ms']), step['decision_wait_ms'], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f'Stale step latency for {key}, step {step["step"]}')
            for field, value in (('action_step', step['step']), ('action_wait_ms', float(latency['decision_wait_ms'])),
                                 ('action_first_content_ms', latency['observation_to_action_response_first_content_ms'])):
                if event.get(field) and (value == '' or not math.isclose(float(event[field]), float(value), rel_tol=1e-9, abs_tol=1e-6)):
                    raise ValueError(f'Conflicting {field} for {key}, API {api["id"]}')
            mapping.update(action_mapping='recorded' if event.get('action_step') else 'resolved_from_decision_interval',
                           resolved_action_step=step['step'], resolved_action_api_id=latency['api_id'] or None,
                           resolved_action_wait_ms=float(latency['decision_wait_ms']),
                           resolved_action_first_content_ms=(float(latency['observation_to_action_response_first_content_ms'])
                                                            if latency['observation_to_action_response_first_content_ms'] else None))
        elif event.get('action_step'):
            raise ValueError(f'Recorded action lacks a decision interval for {key}, API {api["id"]}')
        resolved.append(dict(event, **mapping))
    return resolved


def aggregate(rows):
    out = {}
    for arm in ARMS:
        values = [row for row in rows if row['arm'] == arm]
        n = len(values)
        passed = sum(row['full_success'] for row in values)
        item = dict(trials=n, full_successes=passed,
                    full_success_rate=passed / n if n else None,
                    mean_official_score=statistics.mean(r['official_score'] for r in values) if n else None,
                    native_compactions=sum(r['native_compactions'] for r in values),
                    kvmem_selection_requests=sum(r['kvmem_selections'] for r in values),
                    execution_api_errors=sum(r['execution_api_errors'] for r in values),
                    mean_trial_elapsed_minutes=statistics.mean(r['elapsed_minutes'] for r in values) if n else None,
                    latency={})
        peaks = [r['peak_sampled_gpu_gib'] for r in values if r['peak_sampled_gpu_gib'] is not None]
        item['gpu'] = dict(trials_with_samples=len(peaks), peak_sampled_gib=max(peaks) if peaks else None)
        item['model_work'] = {key: sum(r[key] for r in values) for key in
                              ('api_requests', 'output_tokens', 'non_summary_output_tokens',
                               'output_usage_observed_requests', 'output_usage_missing_requests',
                               'summary_api_requests', 'api_total_seconds', 'native_decode_seconds',
                               'native_decode_observed_requests')}
        item['model_work']['equal_trial_mean_api_total_seconds'] = (
            statistics.mean(r['api_total_seconds'] for r in values) if values else None)
        bounded = [r for r in values if r['desktop_end_boundary_observed']]
        item['model_work']['trials_with_explicit_desktop_end'] = len(bounded)
        for key in ('post_desktop_end_api_requests', 'post_desktop_end_output_tokens',
                    'post_desktop_end_output_usage_observed_requests',
                    'post_desktop_end_api_total_seconds', 'post_desktop_end_api_errors',
                    'api_total_seconds_before_desktop_end'):
            item['model_work'][key] = sum(r[key] for r in bounded) if bounded else None
        for key in LATENCIES:
            observed = [r for r in values if r[key + '_n']]
            count = sum(r[key + '_n'] for r in observed)
            item['latency'][key] = dict(
                trials_with_observations=len(observed), observations=count,
                equal_trial_mean_ms=statistics.mean(r[key + '_mean'] for r in observed) if count else None,
                pooled_observation_mean_ms=sum(r[key + '_mean'] * r[key + '_n'] for r in observed) / count if count else None,
                max_ms=max(r[key + '_max'] for r in observed) if count else None)
        out[arm] = item
    return out


def combine(roots):
    evidence, plans, rows, events, pending, tasks, run_status = {}, [], [], [], [], [], []
    manifest_checks = []
    query_events = []
    cohort_summaries = {}
    for root in roots:
        plan = read_json(root / 'plan.json', evidence)
        if len(set(plan['tasks'])) != len(plan['tasks']) or set(plan['tasks']) & set(tasks):
            raise ValueError('Tasks must be unique within and across cohorts: ' + str(root))
        if plans:
            for key in ('benchmark_release', 'arms', 'client', 'seed', 'runs_per_method_per_task'):
                if plan.get(key) != plans[0].get(key):
                    raise ValueError('Cohort configuration mismatch: ' + key)
        plans.append(plan)
        tasks.extend(plan['tasks'])
        summary, cohort_rows = summarize(root)
        cohort_summaries[str(root)] = summary
        pending.extend(dict(cohort=str(root), **item) for item in summary['pending_pairs'])
        complete = set(summary['completed_tasks'])
        if summary['pairs_with_different_binaries']:
            raise ValueError('Different binaries within a pair: ' + str(root))
        for task in plan['tasks']:
            metric_path = root / 'trials' / task / 'analysis' / 'metrics.json'
            metrics = read_json(metric_path, evidence) if task in complete else {}
            pair_builds = set()
            for arm in ARMS:
                run = root / 'trials' / task / 'full' / arm
                result_path = run / 'result.json'
                result = read_json(result_path, evidence) if result_path.exists() else {}
                run_status.append(dict(cohort=str(root), task=task, arm=arm,
                                       status=result.get('status', 'not_started')))
                if task not in complete:
                    continue
                recorded = metrics[arm]['result']
                # The pair analyzer normalizes structured evaluation payloads to
                # their official numeric score; auxiliary data stays in result.json.
                for key in ('status', 'task', 'arm', 'steps', 'started_at', 'finished_at'):
                    if recorded.get(key) != result.get(key):
                        raise ValueError('Stale analysis: ' + str(result_path) + ' / ' + key)
                if official_score(recorded['score']) != official_score(result['score']):
                    raise ValueError('Stale official score: ' + str(result_path))
                manifest = read_json(run / 'server_manifest.json', evidence)
                client = read_json(run / 'client_manifest.json', evidence)
                checks = audit_manifests(plan, task, arm, manifest, client)
                manifest_checks.append(dict(cohort=str(root), task=task, arm=arm, checked=checks))
                if arm == 'kvmem':
                    query_events.extend(guided_query_events(run / 'qw3_server.log', str(root), task, arm, evidence))
                build = manifest.get('binary_sha256')
                if not isinstance(build, str) or len(build) != 64:
                    raise ValueError('Missing binary identity: ' + str(run))
                pair_builds.add(build)
                row = next(r for r in cohort_rows if r['task'] == task and r['arm'] == arm)
                if row['official_score'] != official_score(result['score']):
                    raise ValueError('Score mismatch: ' + str(run))
                elapsed = (datetime.fromisoformat(result['finished_at']) - datetime.fromisoformat(result['started_at'])).total_seconds()
                if elapsed < 0:
                    raise ValueError('Negative trial duration: ' + str(run))
                rows.append(dict(cohort=str(root), elapsed_minutes=elapsed / 60,
                                 driver_sha256=client['driver_sources']['run_claude_pair.py'],
                                 post_episode_client_error=result.get('post_episode_client_error', False),
                                 **model_work(run, metrics[arm], evidence), **row))
            if len(pair_builds) > 1:
                raise ValueError('Different binaries within task ' + task)
        event_path = root / 'maintenance_events.csv'
        if complete:
            raw = event_path.read_bytes()
            evidence[str(event_path)] = hashlib.sha256(raw).hexdigest()
            completed_events = [event for event in csv.DictReader(raw.decode().splitlines()) if event['task'] in complete]
            for event in resolve_event_actions(root, completed_events, evidence):
                events.append(dict(cohort=str(root), **event))
            observed = sum(e['kind'] == 'native_compaction' and e['cohort'] == str(root) for e in events)
            expected = sum(r['native_compactions'] for r in cohort_rows)
            if observed != expected:
                raise ValueError('Incomplete compaction event export: ' + str(root))
    builds = sorted({r['binary_sha256'] for r in rows})
    maintenance_tasks = {r['task'] for r in rows if r['native_compactions'] or r['kvmem_selections']}
    completed_tasks = [t for t in tasks if any(r['task'] == t for r in rows)]
    summary = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), planned_pairs=len(tasks),
        planned_runs=2 * len(tasks), completed_pairs=len(completed_tasks),
        completed_runs=sum(r['status'] == 'complete' for r in run_status),
        tasks=tasks, completed_tasks=completed_tasks, pending_pairs=pending, run_status=run_status,
        arms=aggregate(rows),
        by_binary={b: dict(tasks=[t for t in completed_tasks if any(r['task'] == t and r['binary_sha256'] == b for r in rows)],
                           arms=aggregate([r for r in rows if r['binary_sha256'] == b])) for b in builds},
        by_observed_maintenance={
            label: dict(tasks=[t for t in completed_tasks if (t in maintenance_tasks) == observed],
                        arms=aggregate([r for r in rows if (r['task'] in maintenance_tasks) == observed]))
            for label, observed in [('at_least_one_arm', True), ('neither_arm', False)]},
        cohort_summaries=cohort_summaries, evidence_sha256=evidence,
        launch_manifest_checks=manifest_checks,
        episode_end_revisions={str(root): plan['episode_end_revision'] for root, plan in zip(roots, plans)
                               if 'episode_end_revision' in plan},
        environment_revisions={str(root): plan['environment_revision'] for root, plan in zip(roots, plans)
                               if 'environment_revision' in plan},
        maintenance_action_mapping={label: sum(e['action_mapping'] == label for e in events)
                                    for label in ('recorded', 'resolved_from_decision_interval', 'no_enclosing_decision_interval')},
        guided_query_modes={
            'scope': 'Completed pairs only. Preserve raw log mode labels; fallback still pays for query generation.',
            'events': len(query_events),
            'fallback_original_events': sum(e['fallback_original'] for e in query_events),
            'by_mode': {mode: sum(e['mode'] == mode for e in query_events)
                        for mode in sorted({e['mode'] for e in query_events})},
            'by_task': {task: dict(events=sum(e['task'] == task for e in query_events),
                                  fallback_original_events=sum(e['task'] == task and e['fallback_original'] for e in query_events))
                        for task in completed_tasks},
            'details': query_events},
        notes=[
            'Only complete pairs enter comparative statistics; official failures remain included.',
            'Full success is official score exactly 1. Partial scores are not successes.',
            'Latency means are equal-weighted by trial; pooled means are separately labelled.',
            'Missing maintenance events are unobserved, not zero latency.',
            'Trajectories differ. Total runtime and same-step comparisons are not matched-input speedups.',
            'Maintenance strata describe observed trajectories, not randomized causal subgroups.',
            'Single-seed task scores do not establish capability equivalence.',
            'Elapsed trial time includes setup/evaluation; it is not online model API time.',
            'Output usage excludes private guided-query tokens. Native decode time covers observed main-response decode records, with coverage counts retained.',
            'Middecode maintenance can occur after first content. Action wait includes response generation and maintenance; it is not an isolated maintenance timer.',
            'Native main-response decode timing already includes middecode maintenance. Do not add the recorded query/reselect subphase timers to that total again.',
            'GPU samples cannot exclude transient peaks; memory acceptance has separate allocation evidence.',
            'A guided-query attempt may fall back to original query Q after format validation. Do not label all attempts as adopted generated queries.',
            'Snapshot hashes record sources used; this report does not replace the final experiment audit.'])
    return summary, rows, events


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(summary, rows):
    def number(value, scale=1):
        return '未观测' if value is None else f'{value / scale:.3f}'

    text = [
        '# OSWorld 两批实验合并统计', '',
        f"生成时间：{summary['generated_at']}。完成 {summary['completed_pairs']}/{summary['planned_pairs']} 对，"
        f"{summary['completed_runs']}/{summary['planned_runs']} 次运行。", '',
        '比较统计仅包含双方均完成的样本，保留失败和部分得分；官方分数等于 1 才算完整成功。',
        '每步 TTFT 为观察就绪到下一动作响应首内容，包含发生在该区间的原生压缩等待。',
        '各组先按样本计算均值，再对有观测的样本等权平均；无维护事件不按零延迟处理。', '',
        '| 指标 | 普通 qw3 | KVMem |', '|---|---:|---:|']
    arms = summary['arms']
    text.append('| 完整成功 / 样本 | ' + ' | '.join(f"{arms[a]['full_successes']}/{arms[a]['trials']}" for a in ARMS) + ' |')
    text.append('| 官方平均分 | ' + ' | '.join(number(arms[a]['mean_official_score']) for a in ARMS) + ' |')
    for label, key in [('常规步骤 TTFT（秒）', 'regular_step_ttft_ms'),
                       ('全部步骤 TTFT（秒）', 'step_ttft_ms'),
                       ('维护时动作等待（秒）', 'maintenance_action_wait_ms')]:
        text.append('| ' + label + ' | ' + ' | '.join(number(arms[a]['latency'][key]['equal_trial_mean_ms'], 1000) for a in ARMS) + ' |')
    for label, key in [('原生压缩次数', 'native_compactions'), ('发生重选的请求数', 'kvmem_selection_requests')]:
        text.append('| ' + label + ' | ' + ' | '.join(str(arms[a][key]) for a in ARMS) + ' |')
    for label, key, scale in [('已报告的非摘要输出 tokens 总量', 'non_summary_output_tokens', 1),
                              ('在线 API 累计耗时（分钟）', 'api_total_seconds', 60),
                              ('主响应 decode 累计耗时（分钟）', 'native_decode_seconds', 60)]:
        text.append('| ' + label + ' | ' + ' | '.join(number(arms[a]['model_work'][key], scale) for a in ARMS) + ' |')
    text += ['', '不同方法的轨迹、生成量和结束时间不同，不能把总耗时差当作相同输入的加速比。',
             '输出用量不包含私有 guided query；decode 累计耗时的记录覆盖数保留在 summary.json。',
             '取消请求可能缺少最终 API 输出用量；token 总量只加总已报告值，未报告不代表实际生成零 tokens，覆盖数单列。',
             '本结果每方法只有一个 seed，无法据此证明能力等效。', '',
             '| 样本 | 官方分数 qw3 / KVMem | 步数 qw3 / KVMem | 分钟 qw3 / KVMem（含准备与判分） |',
             '|---|---:|---:|---:|']
    for task in summary['completed_tasks']:
        pair = {r['arm']: r for r in rows if r['task'] == task}
        fields = [' / '.join(number(pair[a][key]) if key != 'steps' else str(pair[a][key]) for a in ARMS)
                  for key in ('official_score', 'steps', 'elapsed_minutes')]
        text.append('| ' + task + ' | ' + ' | '.join(fields) + ' |')
    text += ['', '程序版本分组（每个样本的两组版本一致）：', '']
    for build, group in summary['by_binary'].items():
        text.append(f"- `{build}`：{', '.join(group['tasks'])}")
    if summary['episode_end_revisions']:
        text += ['', '驱动另有一次任务结束后的收尾修复：桌面 DONE/FAIL 响应送达后等待 5 秒，'
                 '仍未退出则结束客户端并执行官方判分；不改变任务结束前的模型计算或操作。',
                 '已完成运行保留原驱动指纹；逐次指纹与收尾错误标记见 paired_trials.csv。',
                 '044 普通组原运行在 DONE 后重复 terminate，人工中断后旧驱动未判分且清理了临时系统盘；'
                 '该次轨迹完整归档、分数未知，不进入有效配对统计，按相同 seed 重跑普通组。']
        text += ['原始 API 总量全部保留；同时按首个明确 DONE/FAIL 的观察完成时刻，单列此后发起的请求数、'
                 '输出量、API 耗时和错误。没有明确结束边界的轨迹留空，覆盖数见 summary.json。',
                 '步骤等待只统计实际桌面动作之间的区间；收尾取消产生的连接错误须结合上述边界核对，'
                 '不能直接视作任务执行期故障。']
    if summary['environment_revisions']:
        text += ['', '015 普通组首次尝试在轨迹中出现虚拟机根文件系统只读，最终官方判分文件上传失败；'
                 '该次 447 步轨迹完整归档、无有效分数，不进入配对统计，按相同 seed 重跑。',
                 '后续两组统一使用官方 volume_size=50 扩展根分区，并在后台每 30 秒读取挂载状态；'
                 '异常时保留虚拟机。任务、判分器、模型程序及推理参数保持原计划。',
                 '只读故障的底层触发原因尚未复现，不能断言扩容已根治该问题。'
                 '此前已完成的 11 对保留原环境配置；环境修订与无效尝试证据路径见 summary.json。']
    modes = summary['guided_query_modes']
    text += ['', f"已完成配对记录 {modes['events']} 次 guided-query 事件，"
             f"其中 {modes['fallback_original_events']} 次回退原 query。",
             '回退仍支付 query 生成耗时；它不代表该次选择采用了生成 query。',
             '逐次模式、回退原因和日志位置见 guided_query_events.csv；未记录的原因留空。']
    text += ['', 'summary.json 同时保留各版本分组、双方均未触发维护与至少一方触发维护的描述性统计。',
             'paired_trials.csv 保留逐样本指标；maintenance_events.csv 保留逐事件原始耗时分项。',
             '事件表的原始动作字段保留；resolved_action_* 字段按原始请求与桌面动作时间戳核对并补齐，覆盖压缩后先返回非动作回复的情况。',
             '多个维护事件可能对应同一次动作等待，该等待不能重复相加。',
             'API 首内容时间、完整摘要耗时、下一动作等待是不同计时区间，不应混用。',
             '解码中的重选可能发生在首内容之后，TTFT 不能完整反映这次停顿；维护时动作等待还包含正常回复生成，不能全部归为重选成本。',
             '主响应 decode 原始计时已包含解码中的维护，不能再叠加该事件的 query/reselect 分项；分项计时也不覆盖未单独计时的最后 MTP 状态恢复。',
             '显存为采样观测，完整显存验收另见 prefix_checkpoint_memory_20260910/VALIDATION_REPORT.md。',
             (f"已逐次核对 {len(summary['launch_manifest_checks'])} 次运行的服务与客户端 manifest，"
              '包括程序、启动参数、预算、历史管理、seed 和驱动指纹。'),
             ('全部计划配对结果已齐全；整体目标仍需完成最终验收及环境收尾核对。'
              if summary['completed_pairs'] == summary['planned_pairs']
              else '本文件是进行中实验的快照，不代表全部实验已经完成。'), '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohorts', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [p.resolve() for p in args.cohorts]
    output = args.output.resolve()
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise ValueError('Output must be separate from input cohorts')
    summary, rows, events = combine(roots)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'paired_trials.csv', rows)
    write_csv(output / 'maintenance_events.csv', events)
    write_csv(output / 'guided_query_events.csv', summary['guided_query_modes']['details'])
    (output / 'REPORT.md').write_text(report(summary, rows))
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(output / 'summary.json')
    print(json.dumps({k: summary[k] for k in ('planned_pairs', 'completed_pairs', 'completed_runs')}))


if __name__ == '__main__':
    main()
