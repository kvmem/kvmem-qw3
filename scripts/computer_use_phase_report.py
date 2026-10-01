#!/usr/bin/env python3
"""Audit serialized QW3 request phases from retained computer-use engine logs.

The outer server and native phase timers are additive within their own scope.
Query-replay timers are nested inside those phases; never add them again.
Server validation includes vision preparation but is not a pure vision timer.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


FIELD = re.compile(r'\b([A-Za-z_]\w*)=("(?:\\.|[^"\\])*"|[^\s]+)')


def fields(line):
    result = {}
    for key, value in FIELD.findall(line):
        try:
            result[key] = json.loads(value)
        except ValueError:
            result[key] = value
    return result


def check_sum(values, total, parts):
    for key in (total, *parts):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid duration {key}: {value}")
    actual = sum(values[key] for key in parts)
    if abs(actual - values[total]) > 0.0001:
        raise ValueError(f"Non-additive {total}: {values[total]} != {actual}")


def parse_log(path):
    requests, active, vision = [], None, None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if '[qw3-serve] multimodal prompt images=' in line:
            if active is not None or vision is not None:
                raise ValueError(f"Overlapping image/request records at line {number}")
            vision = fields(line)
        elif '[qw3-server-dispatch]' in line:
            if active is not None:
                raise ValueError(f"Concurrent or unfinished request at line {number}")
            active = {'dispatch': fields(line), 'vision': vision, 'query_replay': []}
            vision = None
            if active['dispatch']['dispatch'] != 'plain':
                raise ValueError('Only serialized plain native requests are supported')
            if any(r['dispatch']['rid'] == active['dispatch']['rid'] for r in requests):
                raise ValueError('Duplicate server request ID')
            requests.append(active)
        elif '[qw3-native-accounting]' in line:
            if active is None or 'native' in active:
                raise ValueError(f"Unmatched native accounting at line {number}")
            active['native'] = values = fields(line)
            check_sum(values, 'total_ms', ('setup_ms', 'prefill_ms', 'postprefill_ms', 'decode_ms', 'finalize_ms'))
        elif 'native kvmem query-replay' in line and 'select_ms=' in line:
            if active is None:
                raise ValueError(f"Unmatched query replay at line {number}")
            active['query_replay'].append(fields(line))
        elif 'native generate: prompt_tokens=' in line:
            if active is None or 'generation' in active:
                raise ValueError(f"Unmatched generation summary at line {number}")
            active['generation'] = fields(line)
        elif '[qw3-server-ttft]' in line or '[qw3-server-accounting]' in line:
            values = fields(line)
            if active is None or active['dispatch']['rid'] != values['rid']:
                raise ValueError(f"Unmatched server request at line {number}")
            if '[qw3-server-ttft]' in line:
                active['ttft'] = values
            else:
                check_sum(values, 'total_ms', ('preprocess_ms', 'queue_ms', 'engine_ms', 'response_ms'))
                check_sum(values, 'preprocess_ms', ('pre_json_ms', 'pre_validate_ms', 'pre_render_ms', 'pre_tokenize_ms', 'pre_span_setup_ms'))
                active['server'] = values
                if values['terminal_status'] == 'success' and ('native' not in active or 'generation' not in active):
                    raise ValueError('Successful request lacks native timing or token evidence')
                if active.get('native', {}).get('total_ms', 0) > values['engine_ms'] + 0.0001:
                    raise ValueError('Native duration exceeds its enclosing server engine phase')
                active = None
    if vision is not None:
        raise ValueError('Unmatched trailing multimodal preparation')
    if not requests:
        raise ValueError('No native requests found')
    return requests


def correlate(requests, directory):
    metrics = [(p, json.loads(p.read_text())) for p in directory.glob('*/metrics.json')]
    metrics.sort(key=lambda item: item[1]['started_at'])
    if len(metrics) != len(requests):
        raise ValueError('Bridge/server request counts differ')
    for record, (path, metric) in zip(requests, metrics):
        record['bridge_request_id'] = path.parent.name
        record['bridge_status'] = metric['status']
        tag = metric.get('backend_trace_tag')
        native_tag = record.get('native', {}).get('trace_tag')
        # This server only forwards kvmem_trace_tag when KVMem is enabled.
        # Dense requests therefore correlate by serial order and token counts,
        # with this weaker attribution recorded explicitly.
        record['trace_tag_matched'] = bool(tag and native_tag == tag)
        if tag and native_tag not in (None, '-') and native_tag != tag:
            raise ValueError('Bridge/native trace tags differ')
        if metric['status'] == 'completed':
            usage = metric['usage']
            if record.get('server', {}).get('terminal_status') != 'success':
                raise ValueError('Completed bridge request has no successful server terminal')
            if usage['input_tokens'] != record['generation']['prompt_tokens'] or usage['output_tokens'] != record['generation']['decoded']:
                raise ValueError('Bridge/native token counts differ')
        elif 'server' not in record and metric['status'] not in ('cancelled', 'failed'):
            raise ValueError('Unfinished request has no cancellation/failure evidence')


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'kind': 'serialized_native_request_phase_audit', 'passed': False,
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'engine_log_sha256': hashlib.sha256(args.engine_log.read_bytes()).hexdigest(),
              'scopes': ['Server phases include HTTP preprocessing and response emission.',
                         'Native phases are nested within server engine time.',
                         'Query-replay events are nested timings, not additional total time.',
                         'Validation includes vision preparation; pure vision time is unavailable in this log.']}
    try:
        requests = parse_log(args.engine_log)
        if args.inference_dir:
            correlate(requests, args.inference_dir)
        rows = []
        for record in requests:
            row = {'rid': record['dispatch']['rid'], 'trace_tag': record.get('native', {}).get('trace_tag'),
                   'terminal_status': record.get('server', {}).get('terminal_status', 'log_incomplete'),
                   'bridge_request_id': record.get('bridge_request_id'), 'bridge_status': record.get('bridge_status'),
                   'trace_tag_matched': record.get('trace_tag_matched')}
            for scope in ('server', 'native', 'ttft', 'vision'):
                row.update({scope + '_' + key: value for key, value in (record.get(scope) or {}).items()
                            if isinstance(value, (int, float))})
            row.update({key: record.get('generation', {}).get(key) for key in ('prompt_tokens', 'decoded', 'kvmem_reuse', 'prefilled')})
            row['query_replay_events'] = len(record['query_replay'])
            for key in ('checkpoint_ms', 'provisional_ms', 'restore_ms', 'select_ms', 'replay_ms'):
                row['query_replay_' + key] = sum(event[key] for event in record['query_replay']) if record['query_replay'] and all(key in event for event in record['query_replay']) else None
            rows.append(row)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output / 'requests.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        (args.output / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
        result.update(requests=len(rows), incomplete_requests=sum(r['terminal_status'] == 'log_incomplete' for r in rows),
                      bridge_correlated=bool(args.inference_dir),
                      trace_tag_matched_requests=sum(bool(r['trace_tag_matched']) for r in rows),
                      terminal_statuses={s: sum(r['terminal_status'] == s for r in rows) for s in sorted({r['terminal_status'] for r in rows})})
        complete = [r for r in rows if r['terminal_status'] == 'success']
        result['successful_request_medians_ms'] = {key: statistics.median(r[key] for r in complete if r.get(key) is not None)
            for key in columns if key.endswith('_ms') and any(r.get(key) is not None for r in complete)}
        result['complete_phase_coverage'] = result['incomplete_requests'] == 0
        result['passed'] = all(r['terminal_status'] != 'log_incomplete' or r['bridge_status'] in ('cancelled', 'failed') for r in rows)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result['error'] = repr(exc)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-log', type=Path, required=True)
    parser.add_argument('--inference-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
