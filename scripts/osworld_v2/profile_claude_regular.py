#!/usr/bin/env python3
"""Describe regular request costs on completed pairs without causal matching."""
import argparse
import csv
import json
from pathlib import Path
import re

from analyze_claude_pair import jsonl, server_records, stats


def vision_records(path):
    records, pending = {}, None
    with path.open() as source:
        for line in source:
            if '[qw3-vision-cache]' in line:
                pending = {k: int(v) for k, v in re.findall(r'\b(images|encoded_images)=(\d+)', line)}
            elif '[qw3-server-ttft]' in line:
                rid = int(re.search(r'\brid=(\d+)', line).group(1))
                records[rid], pending = pending, None
    return records


def profile(root):
    plan = json.loads((root / 'plan.json').read_text())
    result = {'binary_sha256': plan['binary_sha256'], 'tasks': {},
        'scope': 'Completed pairs only. Common steps still contain different images, actions and outputs; these are observational strata, not matched-input causal estimates.',
        'timing': 'pre_validate_ms includes GPU vision encoding and is not pure CPU time. Wrapper total overlaps handler preprocessing; phase timers must not be added indiscriminately. Output tokens belong to the complete action-producing API response, which may contain multiple actions.'}
    fields = ('input_tokens', 'output_tokens', 'request_bytes', 'api_ttft_ms',
        'engine_ttft_ms', 'pre_validate_ms', 'preprocess_ms',
        'pre_render_ms', 'pre_tokenize_ms', 'pre_span_setup_ms',
        'prefilled', 'native_prefill_ms', 'encoded_images',
        'native_wrapper_parse_ms', 'native_wrapper_convert_ms',
        'native_wrapper_total_ms')

    def describe(rows):
        return {'action_responses': len(rows),
            'image_encoding_calls': sum((r.get('encoded_images') or 0) > 0 for r in rows),
            'unknown_vision_calls': sum(r.get('encoded_images') is None for r in rows),
            **{field: stats(r.get(field) for r in rows) for field in fields}}

    for task in plan['tasks']:
        folder = root / 'trials' / task
        path = folder / 'analysis/metrics.json'
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        if any(metrics.get(a, {}).get('result', {}).get('status') != 'complete' for a in ('qw3', 'kvmem')):
            continue
        with (folder / 'analysis/step_latency.csv').open() as source:
            actions = list(csv.DictReader(source))
        common = min(metrics[a]['result']['steps'] for a in ('qw3', 'kvmem'))
        maintenance = [int(r['step']) for r in actions if
            r['native_compaction_before_action'] == 'True' or r['kvmem_selection_before_action'] == 'True']
        cutoff = min(common, min(maintenance, default=common + 1) - 1)
        trial = {'common_steps_before_first_maintenance': cutoff, 'arms': {}}
        for arm in ('qw3', 'kvmem'):
            p = folder / 'full' / arm
            phases, vision = server_records(p / 'qw3_server.log'), vision_records(p / 'qw3_server.log')
            apis = sorted((a for a in jsonl(p / 'api.jsonl') if a['path'].split('?')[0] == '/v1/messages'), key=lambda a: a['id'])
            requests = {}
            for rid, api in enumerate(apis, 1):
                native = phases.get(rid, {})
                tokens = api.get('input_usage', {}).get('input_tokens')
                if tokens is not None and 'prompt_tokens' in native and native['prompt_tokens'] != tokens:
                    raise ValueError(f'{task}/{arm}: request/phase mismatch at {rid}')
                requests[api['id']] = {**native, **(vision.get(rid) or {}),
                    'input_tokens': tokens, 'output_tokens': api.get('output_usage', {}).get('output_tokens'),
                    'request_bytes': api['request_bytes'], 'api_ttft_ms': api.get('ttft_ms')}
            regular = [{**requests[int(r['api_id'])], 'step': int(r['step'])} for r in actions if
                r['arm'] == arm and r['api_id'] and r['native_compaction_before_action'] == 'False'
                and r['kvmem_selection_before_action'] == 'False']
            regular.sort(key=lambda r: r['step'])
            early = [r for r in regular if r['step'] <= cutoff]
            trial['arms'][arm] = {
                'all_regular_action_responses': describe(regular),
                'common_steps_before_first_maintenance': describe(early),
                'common_steps_image_encoded': describe([r for r in early if (r.get('encoded_images') or 0) > 0]),
                'common_steps_image_cached': describe([r for r in early if r.get('encoded_images') == 0]),
                'early_late_regular_windows': {
                    'scope': 'First and last 50 regular action-producing responses, requiring at least 100 so windows do not overlap. Inputs and image encoding differ; compaction can reset ordinary context length. Descriptive, not causal.',
                    'first50': describe(regular[:50]) if len(regular) >= 100 else None,
                    'last50': describe(regular[-50:]) if len(regular) >= 100 else None}}
        result['tasks'][task] = trial
    output = root / 'regular_request_profile.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    return {'tasks': list(result['tasks']), 'report': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(json.dumps(profile(parser.parse_args().root.resolve()), indent=2))
