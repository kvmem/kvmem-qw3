"""Export completed pairs, with explicit paired exclusions and untrimmed retained tails."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


def number(value):
    if value in ('', None):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Invalid latency: {value}')
    return result


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-task', action='append', default=[])
    parser.add_argument('--exclusion-reason')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    def table(path):
        return list(csv.DictReader(io.StringIO(read(path).decode())))

    summary = json.loads(read(args.report / 'summary.json'))
    excluded = set(args.exclude_task)
    if excluded and not args.exclusion_reason:
        raise ValueError('Record the reason for explicit paired exclusions')
    if not excluded.issubset(summary['completed_tasks']):
        raise ValueError('An excluded task is absent from this snapshot')
    tasks = [t for t in summary['completed_tasks'] if t not in excluded]
    if not tasks:
        raise ValueError('No paired tasks remain')
    pairs = table(args.report / 'paired_trials.csv')
    pairs = [r for r in pairs if r['task'] in tasks]
    assert len(pairs) == len(tasks) * 2
    events = table(args.report / 'maintenance_events.csv')
    events = [r for r in events if r['task'] in tasks]
    markers = defaultdict(list)
    for event in events:
        if not event['resolved_action_step']:
            raise ValueError('Unresolved maintenance event')
        markers[event['task'], event['arm'], int(event['resolved_action_step'])].append(event)

    raw_rows, rows = [], []
    for pair in pairs:
        task, arm = pair['task'], pair['arm']
        trial = Path(pair['cohort']) / 'trials' / task
        source = trial / 'analysis/step_latency.csv'
        result = json.loads(read(trial / 'full' / arm / 'result.json'))
        assert result['status'] == 'complete'
        read(trial / 'full' / arm / 'server_manifest.json')
        read(trial / 'full' / arm / 'client_manifest.json')
        subset = sorted((r for r in table(source) if r['arm'] == arm), key=lambda r: int(r['step']))
        assert len({r['step'] for r in subset}) == len(subset)
        for r in subset:
            key = task, arm, int(r['step'])
            related = markers.get(key, [])
            kinds = sorted({e['kind'] for e in related})
            compaction = any(e['kind'] == 'native_compaction' for e in related)
            reselection = any(e['kind'] != 'native_compaction' for e in related)
            raw_rows.append(dict(task=task, source_csv=str(source), **r))
            values = {}
            for dest, src in [('step_ttft_s', 'observation_to_action_response_first_content_ms'),
                              ('action_wait_s', 'decision_wait_ms'), ('api_ttft_s', 'api_ttft_ms'),
                              ('engine_ttft_s', 'engine_ttft_ms')]:
                value = number(r[src])
                values[dest] = value / 1000 if value is not None else None
            rows.append(dict(task=task, arm=arm, step=int(r['step']), api_id=r['api_id'],
                **values, compaction=int(compaction), reselection=int(reselection),
                regular=int(not related), maintenance_kinds=';'.join(kinds),
                maintenance_api_ids=';'.join(e['api_id'] for e in related),
                input_tokens=r['input_tokens'], binary_sha256=pair['binary_sha256'],
                driver_sha256=pair['driver_sha256'], source_csv=str(source),
                known_issue='screenshot_timeout_in_preceding_action' if key == ('053', 'kvmem', 100) else ''))
        values = [r['step_ttft_s'] for r in rows if r['task'] == task and r['arm'] == arm and r['step_ttft_s'] is not None]
        assert len(values) == int(pair['step_ttft_ms_n'])
        assert math.isclose(statistics.mean(values) * 1000, float(pair['step_ttft_ms_mean']), abs_tol=1e-6)

    row_keys = {(r['task'], r['arm'], r['step']) for r in rows}
    assert set(markers).issubset(row_keys)
    write_csv(out / 'step_latency.csv', rows)
    write_csv(out / 'step_latency_raw_ms.csv', raw_rows)
    (out / 'traces').mkdir()
    for task in tasks:
        write_csv(out / 'traces' / f'task_{task}.csv', [r for r in rows if r['task'] == task])

    cdf, stats, coverage = [], [], []
    for arm in ('qw3', 'kvmem'):
        all_rows = [r for r in rows if r['arm'] == arm]
        valid = [r for r in all_rows if r['step_ttft_s'] is not None]
        counts = {task: sum(r['task'] == task for r in valid) for task in tasks}
        assert all(counts.values())
        # One ECDF jump per distinct observation; ties carry their full probability mass.
        mass = defaultdict(lambda: [0, 0.0])
        for r in valid:
            mass[r['step_ttft_s']][0] += 1
            mass[r['step_ttft_s']][1] += 1 / (len(tasks) * counts[r['task']])
        cumulative_n, weighted = 0, 0.0
        for latency, (n, weight) in sorted(mass.items()):
            cumulative_n += n
            weighted += weight
            cdf.append(dict(arm=arm, ttft_s=latency, count_at_value=n,
                            cumulative_steps=cumulative_n, total_valid_steps=len(valid),
                            cdf_step_pooled_pct=100 * cumulative_n / len(valid),
                            cdf_equal_task_pct=100.0 if cumulative_n == len(valid) else 100 * weighted))
        assert cumulative_n == len(valid) and math.isclose(weighted, 1, abs_tol=1e-10)
        for scope in ['all', *tasks]:
            selected = [r for r in all_rows if scope == 'all' or r['task'] == scope]
            values = [r['step_ttft_s'] for r in selected if r['step_ttft_s'] is not None]
            coverage.append(dict(task=scope, arm=arm, action_rows=len(selected), valid_ttft=len(values),
                                 missing_ttft=len(selected) - len(values)))
            for category in ('all', 'regular', 'maintenance'):
                part = [r['step_ttft_s'] for r in selected if r['step_ttft_s'] is not None and
                        (category == 'all' or bool(r['regular']) == (category == 'regular'))]
                if not part:
                    continue
                stats.append(dict(task=scope, arm=arm, category=category, n=len(part),
                    mean_s=statistics.mean(part), p50_s=quantile(part, .5), p95_s=quantile(part, .95),
                    p99_s=quantile(part, .99), p999_s=quantile(part, .999), max_s=max(part),
                    above_30s=sum(x > 30 for x in part), above_30s_pct=100 * sum(x > 30 for x in part) / len(part)))
    write_csv(out / 'ttft_cdf.csv', cdf)
    write_csv(out / 'ttft_statistics.csv', stats)
    write_csv(out / 'coverage.csv', coverage)
    write_csv(out / 'maintenance_events.csv', events)
    snapshot_dir = out / 'source_snapshot' if excluded else out
    snapshot_dir.mkdir(exist_ok=True)
    for name in ('paired_trials.csv', 'summary.json', 'REPORT.md', 'guided_query_events.csv'):
        (snapshot_dir / name).write_bytes(read(args.report / name))
    if excluded:
        write_csv(out / 'paired_trials.csv', pairs)
        guided = [r for r in table(args.report / 'guided_query_events.csv') if r['task'] in tasks]
        if guided:
            write_csv(out / 'guided_query_events.csv', guided)
        view = dict(completed_tasks=tasks, completed_pairs=len(tasks), completed_runs=2 * len(tasks),
                    source_summary_time=summary['generated_at'], excluded_tasks=sorted(excluded),
                    exclusion_reason=args.exclusion_reason, exclusion_scope='both arms, entire task',
                    arms={})
        for arm in ('qw3', 'kvmem'):
            selected = [r for r in pairs if r['arm'] == arm]
            view['arms'][arm] = dict(
                task_equal_mean_step_ttft_s=statistics.mean(float(r['step_ttft_ms_mean']) / 1000 for r in selected),
                official_mean_score=statistics.mean(float(r['official_score']) for r in selected),
                full_successes=sum(float(r['official_score']) == 1 for r in selected))
        (out / 'summary.json').write_text(json.dumps(view, indent=2) + '\n')
        (out / 'REPORT.md').write_text(
            '# Paired exclusion analysis\n\n'
            + f'Included pairs: {len(tasks)}. Excluded entire tasks from both arms: {", ".join(sorted(excluded))}.\n\n'
            + args.exclusion_reason + '\n\n'
            + 'All retained observations, including cold prefill and other outliers, remain unchanged. '
            + 'The unfiltered report in source_snapshot is provenance, not this analysis. '
            + 'See ttft_statistics.csv and summary.json for recomputed results.\n')
    read(Path(__file__))
    manifest = dict(exported_at=datetime.now(timezone.utc).isoformat(),
                    source_summary_time=summary['generated_at'], tasks=tasks, paired_tasks=len(tasks),
                    excluded_tasks=sorted(excluded), exclusion_reason=args.exclusion_reason,
                    raw_step_rows=len(rows), maintenance_event_rows=len(events),
                    source_sha256=evidence, coverage=coverage[:1] + [r for r in coverage[1:] if r['task'] == 'all'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'paired_tasks': len(tasks), 'step_rows': len(rows),
                      'stats': [r for r in stats if r['task'] == 'all' and r['category'] == 'all']}, indent=2))


if __name__ == '__main__':
    main()
