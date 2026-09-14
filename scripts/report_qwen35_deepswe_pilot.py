#!/usr/bin/env python3
"""Report the fixed pilot without mixing partial, unpaired or invalid trials."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark/claude_deepswe_ab'))
import batch


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(values) if values else None


def arm_row(job, root):
    arm = root / 'pilot' / job['model'] / job['policy'] / 'runs' / job['task_id'] / job['condition']
    metrics_path = arm / 'metrics.json'
    if not metrics_path.exists() or not (arm / 'cell_summary.json').exists():
        return None, None
    d = json.loads(metrics_path.read_text())
    server, claude = d.get('server') or {}, d.get('claude') or {}
    reward = d.get('reward') or {}
    manifest = json.loads((root / 'pilot' / job['model'] / job['policy'] / 'manifest.json').read_text())
    row = dict(model=job['model'], policy=job['policy'], task_id=job['task_id'],
        seed=manifest['seed'], reward=reward.get('reward'),
        full_pass=reward.get('reward') == 1,
        f2p_passed=reward.get('f2p_passed'), f2p_total=reward.get('f2p_total'),
        p2p_passed=reward.get('p2p_passed'), p2p_total=reward.get('p2p_total'),
        f2p_rate=reward.get('f2p'), p2p_rate=reward.get('p2p'),
        completed_naturally=batch._natural_completion(d), protocol_clean=batch._protocol_clean(d),
        agent_elapsed_sec=d.get('agent_elapsed_sec'),
        request_wall_sec=server.get('request_wall_sec'), prefill_sec=server.get('prefill_sec'),
        decode_sec=server.get('decode_sec'), decoded_tokens=server.get('decoded_tokens'),
        decode_tps_completed=server.get('decode_tps_completed'),
        request_count=server.get('request_count'), max_prompt_tokens=server.get('max_prompt_tokens_server'),
        compactions=claude.get('compactions', 0), compaction_total_sec=claude.get('compaction_total_sec'),
        semantic_reselections=server.get('kvmem_semantic_reselections', 0),
        selection_pass_sec=server.get('kvmem_selection_pass_sec'),
        guided_query_count=server.get('guided_query_count', 0), guided_query_sec=server.get('guided_query_sec'),
        unexpected_no_reusable_checkpoints=server.get('kvmem_unexpected_no_reusable_checkpoints', 0),
        metrics_path=str(metrics_path), binary_sha256=manifest['locked_artifacts']['qw3_binary_sha256'],
        model_sha256=manifest['locked_artifacts']['model_sha256'])
    return row, d


def mtp_row(arm):
    """Aggregate counters, not averages of rounded per-request accept rates.

    Phase values are the existing engine's host timers; they are not a CUDA
    profile and should not be assumed to partition decode wall time exactly.
    """
    log = Path(arm['metrics_path']).parent / 'server/qw3_server.log'
    summaries = []
    for line in log.read_text(errors='replace').splitlines():
        if '[qw3] native mtp_spec_summary:' not in line:
            continue
        fields = dict(re.findall(r'([a-z0-9_]+)=([^ ]+)', line))
        if fields.get('enabled') == 'true':
            summaries.append(fields)
    total = lambda key: sum(float(s.get(key, '0').rstrip('s')) for s in summaries)
    drafted, accepted = int(total('drafted')), int(total('accepted'))
    return dict(model=arm['model'], policy=arm['policy'], task_id=arm['task_id'],
        summaries=len(summaries), model_requests=arm['request_count'],
        drafted=drafted, accepted=accepted,
        acceptance=accepted / drafted if drafted else None,
        batches=int(total('batches')), rollbacks=int(total('rollbacks')),
        draft_sec=total('draft_s'), verify_sec=total('verify_s'),
        restore_sec=total('restore_s'), snapshot_sec=total('snapshot_s'),
        replay_sec=total('replay_s'), prefix_sec=total('prefix_s'),
        decode_sec=arm['decode_sec'], decoded_tokens=arm['decoded_tokens'],
        weighted_decode_tokens_per_s=arm['decode_tps_completed'], log_path=str(log))


def report(root):
    plan = json.loads((root / 'pilot_plan.json').read_text())
    arms, raw = [], {}
    for job in plan['jobs']:
        row, d = arm_row(job, root)
        if row:
            arms.append(row)
            raw[(job['model'], job['task_id'], job['policy'])] = d
    pairs = []
    seen = set()
    for job in plan['jobs']:
        key = job['model'], job['task_id']
        if key in seen:
            continue
        seen.add(key)
        q, k = raw.get((*key, 'qw3'), {}), raw.get((*key, 'kvmem'), {})
        assessment = batch.paired_assessment(q, k, selection_budget=131072,
            expected_ordinary_context=262144, expected_kvmem_context=3145728,
            expected_effective_window=242144)
        compactions = (q.get('claude') or {}).get('compactions', 0)
        reselections = (k.get('server') or {}).get('kvmem_semantic_reselections', 0)
        pairs.append(dict(model=key[0], task_id=key[1], pair_complete=bool(q and k),
            ordinary_compactions=compactions, kvmem_semantic_reselections=reselections,
            maintenance_events_observed=bool(compactions or reselections),
            # Matching reward signatures does not match the generated work.
            # Preserve the legacy assessment, but never interpret its
            # "claimable" label as a causal engine/maintenance speedup.
            maintenance_speedup_isolated=False, **assessment))
    # Retain quality-divergent pairs in the main comparison. Quality-equivalence
    # is a separate diagnostic; selecting on equal scores would bias accuracy.
    valid = {(p['model'], p['task_id']) for p in pairs if p['pair_complete']
             and p['both_completed_naturally'] and p['protocol_clean']}
    summary = []
    for model in ('4B', '9B'):
        for policy in ('qw3', 'kvmem'):
            rows = [r for r in arms if r['model'] == model and r['policy'] == policy
                    and (model, r['task_id']) in valid]
            count = len(rows)
            tokens = sum(r['decoded_tokens'] or 0 for r in rows)
            decode = sum(r['decode_sec'] or 0 for r in rows)
            summary.append(dict(model=model, policy=policy, paired_tasks=count,
                full_passes=sum(r['full_pass'] for r in rows),
                full_pass_rate=sum(r['full_pass'] for r in rows) / count if count else None,
                mean_reward=mean(rows, 'reward'), mean_f2p_rate=mean(rows, 'f2p_rate'),
                mean_p2p_rate=mean(rows, 'p2p_rate'), mean_agent_sec=mean(rows, 'agent_elapsed_sec'),
                mean_request_sec=mean(rows, 'request_wall_sec'), mean_prefill_sec=mean(rows, 'prefill_sec'),
                mean_decoded_tokens=mean(rows, 'decoded_tokens'),
                weighted_decode_tokens_per_s=tokens / decode if decode else None,
                compactions=sum(r['compactions'] for r in rows),
                semantic_reselections=sum(r['semantic_reselections'] for r in rows)))
    out = root / 'report'
    out.mkdir(exist_ok=True)
    mtp = [mtp_row(arm) for arm in arms]
    for name, rows in (('arms', arms), ('pairs', pairs), ('paired_summary', summary),
                       ('mtp_counters', mtp)):
        (out / f'{name}.json').write_text(json.dumps(rows, indent=2) + '\n')
        if rows:
            with (out / f'{name}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    result = dict(updated_at=batch.utc_now(), completed_arms=len(arms), total_arms=plan['trials'],
        completed_pairs=sum(p['pair_complete'] for p in pairs), valid_pairs=len(valid),
        summary=summary, notes=[
            'Only completed, naturally terminated, protocol-valid pairs enter paired averages.',
            'Quality-divergent pairs remain included; they are separately flagged.',
            'Agent/request wall-time ratios compare complete trajectories, not matched engine inputs.',
            'The legacy long_context_pressure_observed field can be true below the actual refresh threshold; maintenance_events_observed requires a logged compaction or semantic refresh.',
            'Equal reward signatures, including equal failures, do not isolate maintenance-induced speedup.',
            'Pure decode throughput excludes prefill, retrieval, compaction and tools.',
            'MTP acceptance is total accepted / total drafted; recorded host phase timers are not an exclusive GPU-time breakdown.',
            'All completed arm results, including invalid ones, remain in arms.csv for audit.'])
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('completed_arms', 'total_arms', 'completed_pairs', 'valid_pairs')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=ROOT / 'benchmark/qwen35_small_20260914')
    args = parser.parse_args()
    report(args.experiment.resolve())
