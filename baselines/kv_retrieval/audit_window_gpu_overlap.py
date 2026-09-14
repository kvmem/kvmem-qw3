"""Annotate raw request intervals with sampled GPU co-residency and activity.

CPU-only, read-only on experiment artifacts. Samples are observations, not proof
of exclusivity, occupancy, contention duration, or a latency correction factor.
No request or trial is dropped from the primary comparison.
"""
import argparse
from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
import time

from compare_agent_windows import csv_rows, jsonl, write_json


class Samples:
    def __init__(self, rows, clock):
        self.rows = sorted((r for r in rows if isinstance(r.get(clock), (int, float))
                            and math.isfinite(r[clock])), key=lambda r: r[clock])
        self.times = [r[clock] for r in self.rows]

    def between(self, start, end):
        return self.rows[bisect_left(self.times, start):bisect_right(self.times, end)]


def annotate(request, server_pid, memory, activity):
    start = request['received_at']
    duration = request['request_wall_s']
    if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
        raise ValueError('invalid request wall interval')
    end = start + duration
    mem = memory.between(start, end)
    sm = activity.between(start, end)
    valid_sm = [r for r in sm if 'processes' in r and not r.get('sampling_error')]
    other_pids = set()
    active_pids = set()
    shared_count = active_count = 0
    for row in mem:
        other = {int(pid) for pid in row['gpu_processes_gib'] if int(pid) != server_pid}
        other_pids.update(other)
        shared_count += bool(other)
    for row in valid_sm:
        active = {p['pid'] for p in row['processes']
                  if p.get('pid') is not None and p['pid'] != server_pid
                  and 'C' in str(p.get('type', '')) and (p.get('sm') or 0) > 0}
        active_pids.update(active)
        active_count += bool(active)
    points = [r['at'] for r in mem]
    boundaries = [start, *points, end]
    return {
        'request_id': request['request_id'], 'received_at': start, 'finished_at_estimate': end,
        'request_wall_s': duration, 'prompt_tokens': request.get('prompt_tokens'),
        'output_tokens': request.get('output_tokens'), 'decode_s': request.get('decode_s'),
        'api_first_content_s': request.get('api_first_content_s'),
        'memory_samples_inside_request': len(mem),
        'max_gap_including_request_boundaries_s': max((b-a for a,b in zip(boundaries,boundaries[1:])), default=duration),
        'memory_samples_with_other_compute_process': shared_count,
        'other_compute_pids_observed': sorted(other_pids),
        'other_compute_process_observed': bool(other_pids) if mem else None,
        'activity_samples_inside_request': len(valid_sm),
        'activity_sampling_errors': len(sm)-len(valid_sm),
        'activity_samples_with_other_compute_sm': active_count,
        'other_active_compute_pids_observed': sorted(active_pids),
        'other_compute_sm_activity_observed': bool(active_pids) if valid_sm else None,
        'gpu_exclusivity_verified': False, 'latency_adjusted': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    activity = Samples(jsonl(campaign/'gpu_process_utilization.jsonl'), 'observed_at')
    rows = []
    for result_file in sorted(campaign.glob('*/small/attempt_*/result.json')):
        run = result_file.parent
        result = json.loads(result_file.read_text())
        if not result.get('server_pid'):
            continue
        memory = Samples(jsonl(run/'resource_samples.jsonl'), 'at')
        for request in jsonl(run/'server/requests.jsonl'):
            if request.get('status') != 'completed':
                continue
            rows.append({'task_id': result['task']['task_id'], 'arm': str(run),
                         **annotate(request, result['server_pid'], memory, activity)})
    out = campaign/'gpu_overlap'
    out.mkdir(exist_ok=True)
    report = {'created_at': time.time(), 'rows': rows, 'request_count': len(rows),
        'other_compute_process_observed_requests': sum(r['other_compute_process_observed'] is True for r in rows),
        'other_compute_sm_activity_observed_requests': sum(r['other_compute_sm_activity_observed'] is True for r in rows),
        'no_memory_samples_requests': sum(r['other_compute_process_observed'] is None for r in rows),
        'note': 'Sampled presence/activity during whole API requests only. Missing samples are unknown. '
                'No observed competing process does not prove exclusivity. Do not subtract sampled '
                'activity from latency or filter these requests out of the primary task comparison.'}
    write_json(out/'requests.json', report)
    csv_rows(out/'requests.csv', rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','note')}))


if __name__ == '__main__':
    main()
