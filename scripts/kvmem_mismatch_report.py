#!/usr/bin/env python3
"""Summarize shadow CPU mismatch JSONL. Does not modify inference or results."""
import argparse
import collections
import json
import math


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    def percentile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return dict(n=len(values), mean=sum(values) / len(values), min=values[0],
                p50=percentile(.5), p90=percentile(.9), p95=percentile(.95), max=values[-1])


def summarize(paths):
    phases = collections.defaultdict(list)
    statuses = collections.Counter()
    incomplete = 0
    for path in paths:
        with open(path) as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    incomplete += 1
                    continue
                if item.get('kind') != 'kvmem_mismatch':
                    continue
                statuses[item['status']] += 1
                if item['status'] == 'ok':
                    phases[item['phase']].append(item)
    result = {'status_counts': dict(statuses), 'unparsed_lines': incomplete, 'phases': {}}
    for phase, rows in phases.items():
        metrics = {}
        for key in ('missing', 'debt', 'cpu_ms', 'capture_ms', 'snapshot_ms',
                    'indexed_tokens', 'unindexed_tokens'):
            metrics[key] = distribution([r[key] for r in rows if key in r and math.isfinite(r[key])])
        metrics['missing_histogram_0_to_1_step_0_1'] = [sum(min(9, int(r['missing'] * 10)) == i for r in rows) for i in range(10)]
        result['phases'][phase] = metrics
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()
    print(json.dumps(summarize(args.logs), ensure_ascii=False, indent=2))
