#!/usr/bin/env python3
"""Summarize completed native Claude Code pairs without discarding failures."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ARMS = ('qw3', 'kvmem')
LATENCIES = ('regular_step_ttft_ms', 'step_ttft_ms', 'decision_wait_ms',
             'regular_decision_wait_ms', 'maintenance_action_wait_ms', 'maintenance_step_ttft_ms')
SELECTION_COUNTS = ('kvmem_prefill_selection_requests', 'kvmem_middecode_selection_requests',
                    'kvmem_middecode_refreshes', 'kvmem_selection_events')


def gpu_observation(path):
    if not path.exists():
        return None, 0
    peak, count = None, 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            value = next((v for k, v in row.items() if k and k.strip().startswith('memory.used')), None)
            if value is None:
                continue
            parts = value.strip().split()
            if len(parts) != 2 or parts[1] != 'MiB':
                continue
            try:
                gib = float(parts[0]) / 1024
            except ValueError:
                continue
            if not math.isfinite(gib) or gib < 0:
                continue
            peak = gib if peak is None else max(peak, gib)
            count += 1
    return peak, count


def official_score(evaluation):
    """Read either return form documented by OSWorld's BaseTask.evaluate.

    The official scorer owns its auxiliary checks and any penalties. Read its
    score directly; keep the complete original payload in the trial artifacts.
    """
    value = evaluation.get('score') if isinstance(evaluation, dict) else evaluation
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Official evaluation must be numeric or contain a numeric score field')
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Expected a finite official score in [0, 1]')
    return score


def summarize(root):
    plan = json.loads((root / 'plan.json').read_text())
    rows, pending, completed = [], [], []
    per_arm = {arm: [] for arm in ARMS}
    for task in plan['tasks']:
        path = root / 'trials' / task / 'analysis' / 'metrics.json'
        metrics = json.loads(path.read_text()) if path.exists() else {}
        missing = [arm for arm in ARMS if
                   metrics.get(arm, {}).get('result', {}).get('status') != 'complete']
        if missing:
            pending.append({'task': task, 'missing_complete_arms': missing})
            continue
        completed.append(task)
        for arm in ARMS:
            metric = metrics[arm]
            result = metric['result']
            if result.get('task') != task or result.get('arm') != arm:
                raise ValueError('Mismatching task or method: ' + str(path))
            score = official_score(result['score'])
            row = {'task': task, 'arm': arm, 'official_score': score,
                   'full_success': score == 1, 'steps': result['steps'],
                   'termination': result.get('termination', 'client_finished'),
                   'native_compactions': metric['native_compactions'],
                   'kvmem_selections': metric['kvmem_selections'],
                   'execution_api_errors': len(metric['execution_api_errors']),
                   'max_input_tokens': metric['input_tokens_max']}
            run = root / 'trials' / task / 'full' / arm
            manifest_path = run / 'server_manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            row['binary_sha256'] = manifest.get('binary_sha256')
            row['peak_sampled_gpu_gib'], row['gpu_samples'] = gpu_observation(run / 'gpu.csv')
            for key in SELECTION_COUNTS:
                row[key] = metric.get(key)
            for key in LATENCIES:
                stat = metric.get(key)
                for field in ('n', 'mean', 'p95', 'max'):
                    row[key + '_' + field] = stat[field] if stat else None
            resources = metric.get('resource_samples') or {}
            for key in ('peak_server_rss_gib', 'peak_server_swap_gib', 'min_host_available_gib'):
                row[key] = resources.get(key)
            rows.append(row)
            per_arm[arm].append(row)
    summary = {'planned_pairs': len(plan['tasks']), 'completed_pairs': len(completed),
               'completed_tasks': completed, 'pending_pairs': pending,
               'scope': 'Only complete pairs enter comparative statistics; official failures remain included.',
               'score_definition': 'Use the official numeric return or structured return.score directly; retain auxiliary evaluation data in the original trial artifacts.',
               'success_definition': 'Official score exactly 1; partial credit is reported separately as mean score.',
               'arms': {}}
    builds = {}
    unmatched = []
    for task in completed:
        pair = [r for r in rows if r['task'] == task]
        versions = {r['binary_sha256'] for r in pair}
        if len(versions) == 1 and None not in versions:
            builds.setdefault(next(iter(versions)), []).append(task)
        elif None not in versions:
            unmatched.append(task)
    summary['matched_build_groups'] = builds
    summary['pairs_with_different_binaries'] = unmatched
    summary['version_scope'] = 'Build groups use recorded server manifests. Different completed pairs can retain different validated versions; every within-task comparison should use matching builds.'
    for arm, values in per_arm.items():
        full_successes = sum(row['full_success'] for row in values)
        item = {'trials': len(values), 'full_successes': full_successes,
                'full_success_rate': full_successes / len(values) if values else None,
                'mean_official_score': statistics.mean(row['official_score'] for row in values) if values else None,
                'timeouts': sum(row['termination'] == 'timeout' for row in values),
                'action_limits': sum(row['termination'] == 'action_limit' for row in values),
                'native_compactions': sum(row['native_compactions'] for row in values),
                'kvmem_selections': sum(row['kvmem_selections'] for row in values),
                'execution_api_errors': sum(row['execution_api_errors'] for row in values),
                'latency': {}}
        for key in SELECTION_COUNTS:
            item[key] = sum(row[key] for row in values) if values and all(row[key] is not None for row in values) else None
        for key in LATENCIES:
            available = [row for row in values if row[key + '_n']]
            count = sum(row[key + '_n'] for row in available)
            item['latency'][key] = {
                'trials_with_observations': len(available), 'observations': count,
                'equal_trial_mean_ms': statistics.mean(row[key + '_mean'] for row in available) if count else None,
                'pooled_observation_mean_ms': sum(row[key + '_mean'] * row[key + '_n'] for row in available) / count if count else None,
                'max_ms': max(row[key + '_max'] for row in available) if count else None}
        summary['arms'][arm] = item
    return summary, rows


def write_summary(root):
    summary, rows = summarize(root)
    path = root / 'summary.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(path)
    with (root / 'paired_trials.csv').open('w', newline='') as output:
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(write_summary(args.root), indent=2))
