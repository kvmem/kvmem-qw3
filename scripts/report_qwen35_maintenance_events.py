#!/usr/bin/env python3
"""Export maintenance events from completed, serial, block=32 pilot arms.

Before-decode accounting and cumulative mid-decode timers are kept separate.
This is a log export, not a causal speedup or task-quality assessment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def fields(line):
    # Generated query text is data and may itself contain key=value strings.
    return dict(re.findall(r'([a-z0-9_]+)=([^ ]+)', line.split(' query=', 1)[0]))


def export_arm(arm):
    log = Path(arm['metrics_path']).parent / 'server/qw3_server.log'
    raw = log.read_bytes()
    rows, pending = [], []
    accounting, generation = {}, {}
    for number, line in enumerate(raw.decode(errors='replace').splitlines(), 1):
        if line.startswith(('[qw3] native kvmem guided query: mode=',
                            '[qw3] native kvmem middecode guided reselect: refresh=',
                            '[kvmem-reselect-perf] kind=semantic',
                            '[qw3] native kvmem query replay:')):
            pending.append((number, line.split(' query=', 1)[0], fields(line)))
        elif line.startswith('[qw3] [qw3-native-accounting] '):
            accounting = fields(line)
        elif line.startswith('[qw3] native generate:'):
            generation = fields(line)
        elif line.startswith('[qw3-server-ttft] '):
            if pending:
                timing = fields(line)
                assert accounting and generation, (log, number, 'missing accounting')
                selections = [f for _, s, f in pending if s.startswith('[kvmem-reselect-perf]')]
                replay = [f for _, s, f in pending if 'native kvmem query replay:' in s]
                queries = [f for _, s, f in pending if 'guided' in s]
                assert selections or replay, (log, number, 'query without refresh')
                milliseconds = lambda key: float(accounting[key])
                observed = lambda key: float(timing[key]) if float(timing[key]) >= 0 else None
                middecode = int(generation.get('kvmem_middecode_refreshes', 0))
                rows.append(dict(
                    model=arm['model'], policy=arm['policy'], task_id=arm['task_id'],
                    request_id=int(timing['rid']), prompt_tokens=int(generation['prompt_tokens']),
                    decoded_tokens=int(generation['decoded']), semantic_refreshes=len(selections),
                    query_modes=';'.join(f.get('query_mode', f.get('mode', '')) for f in queries),
                    query_tokens=sum(int(f.get('query_tokens', 0)) for f in queries),
                    before_decode_guided_ms=milliseconds('post_guided_query_ms'),
                    before_decode_query_generation_ms=milliseconds('post_guided_decode_ms'),
                    before_decode_selection_accounted_ms=milliseconds('post_semantic_ms'),
                    before_decode_query_replay_ms=milliseconds('post_query_replay_ms'),
                    ordinary_prefill_ms=milliseconds('prefill_ms'),
                    replay_tokens=sum(int(f['suffix_tokens']) for f in replay),
                    replay_origins=';'.join(f.get('warm_checkpoint', '') for f in replay),
                    replay_boundaries=';'.join(f['boundary'] for f in replay),
                    replay_warm=all(f.get('warm_reuse') == '1' for f in replay) if replay else None,
                    replay_block_aligned=all(int(f['boundary']) % 32 == 0 for f in replay) if replay else None,
                    selection_event_wall_ms=sum(float(f['total_ms']) for f in selections),
                    middecode_refreshes=middecode,
                    middecode_query_ms=float(generation.get('kvmem_middecode_query_ms', 0)),
                    middecode_reselect_ms=float(generation.get('kvmem_middecode_reselect_ms', 0)),
                    engine_decode_ms=milliseconds('decode_ms'),
                    server_visible_ttft_ms=observed('response_ttft_ms'),
                    request_total_ms=observed('request_total_ms'),
                    source_event_lines=';'.join(str(n) for n, _, _ in pending),
                    source_ttft_line=number, log_path=str(log)))
            # Pilot services serialize requests. Reset at every completed
            # request, including ordinary requests with no maintenance.
            pending, accounting, generation = [], {}, {}
    assert not pending, (log, 'incomplete maintenance request in completed arm')
    assert sum(r['semantic_refreshes'] for r in rows) == arm['semantic_reselections'], log
    assert sum(r['middecode_refreshes'] for r in rows) == arm['middecode_refreshes'], log
    return rows, {str(log): hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'benchmark/qwen35_small_20260914')
    args = parser.parse_args()
    report = args.root / 'report'
    arms_path = report / 'arms.json'
    arms = json.loads(arms_path.read_text())
    rows, sources = [], {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (arms_path, Path(__file__).resolve())}
    for arm in arms:
        if arm['policy'] == 'kvmem':
            events, hashes = export_arm(arm)
            rows.extend(events)
            sources.update(hashes)
    (report / 'maintenance_events.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (report / 'maintenance_events.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = dict(completed_arms=len(arms), maintenance_requests=len(rows), source_sha256=sources,
        notes=[
            'Only completed KVMem arms are exported; task failures remain included.',
            'Server visible TTFT is not client observation-to-action latency.',
            'Before-decode guided, selection and replay timers are separate from ordinary prefill.',
            'Mid-decode query/reselection timers are cumulative per request and exported once. They are already included in engine_decode_ms.',
            'selection_event_wall_ms overlaps the accounting fields; do not add it to those fields.',
            'Events do not establish task success, causal speedup, or an exclusive GPU-time breakdown.'])
    (report / 'maintenance_events_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('completed_arms', 'maintenance_requests')}))


if __name__ == '__main__':
    main()
