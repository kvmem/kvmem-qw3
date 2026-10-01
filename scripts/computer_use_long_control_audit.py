#!/usr/bin/env python3
"""Audit a completed six-profile, single-repetition long visual control run.

This is an offline evidence reader. It dispatches no inference or desktop tools.
The answer checks deliberately apply only to the frozen historical-image corpus.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, binary_sha256):
    summary = read(run / 'summary.json')
    checks = read(run / 'visual-checks.json')
    aggregate = read(run / 'aggregate.json')
    assert summary['prefix_reuse'] is False
    assert checks['passed'] and len(checks['checks']) == 12
    labels = {f'{method}-kvmem-{policy}' for method, policy in
              itertools.product(('none', 'mtp', 'dflash2'), ('on', 'off'))}
    expected = set(itertools.product(labels, ('present_image', 'absent_image'), (0,)))
    assert {(c['profile'], c['case'], c['repetition']) for c in checks['checks']} == expected
    assert all(c['passed'] for c in checks['checks'])
    corpus = Path(summary['corpus'])
    cases = {case['name']: case for case in summary['cases']}
    assert set(cases) == {'present_image', 'absent_image'} and len(summary['cases']) == 2
    bodies = {}
    for name, case in cases.items():
        assert sha(corpus / case['path']) == case['sha256']
        bodies[name] = read(corpus / case['path'])
        assert case['prompt_tokens'] == 158911
    profiles, observed, common = [], set(), None
    for profile in summary['profiles']:
        method, policy = profile['method'], profile['kvmem']
        label = f'{method}-kvmem-{policy}'
        directory = run / label
        manifest = read(directory / 'engine/manifest.json')
        command = manifest['command']
        def value(flag):
            assert command.count(flag) == 1
            return command[command.index(flag) + 1]
        assert profile['status'] == 'completed' and len(profile['runs']) == 2
        assert manifest['binary_sha256'] == binary_sha256
        assert manifest['prefix_reuse'] is False and '--kvmem-prefix-cache' not in command
        assert manifest['guided_refresh'] == 'off' and manifest['visual_history'] == 'retrievable'
        assert manifest['kvmem'] == policy and ('--kvmem' in command) == (policy == 'on')
        assert value('--speculative-method') == method
        assert ('--mtp-chain' in command) == (method == 'mtp')
        assert ('--dflash-model' in command) == (method == 'dflash2')
        controls = {flag: value(flag) for flag in
                    ('--model', '--vision-model', '--vision-device', '--ctx', '--kv-dtype', '--prefill-chunk')}
        if common is None:
            common = controls
        assert controls == common
        metadata = []
        if policy == 'on':
            metadata = [json.loads(line) for line in
                        (directory / 'engine/retrieval.jsonl').read_text().splitlines()]
            metadata = [m for m in metadata if m.get('type') == 'meta']
            assert len(metadata) == 2
        evidence = []
        for trial in profile['runs']:
            name = trial['case']
            key = (label, name, trial['repetition'])
            assert key in expected and key not in observed
            observed.add(key)
            retained_path = Path(trial['result_path'])
            retained = read(retained_path)
            for field in ('status', 'http_status', 'usage', 'text', 'tool_calls', 'elapsed_seconds'):
                assert retained[field] == trial[field]
            assert trial['status'] == 'completed' and trial['http_status'] == 200
            assert trial['usage']['prompt_tokens'] == cases[name]['prompt_tokens']
            expected_answer = {'reference': 'QW3-WINDOWS-2026-ALPHA', 'name': 'Alice', 'priority': 'High'}
            if name == 'absent_image':
                expected_answer = dict.fromkeys(expected_answer)
            assert json.loads(trial['text']) == expected_answer and not trial['tool_calls']
            record = dict(case=name, result_sha256=sha(retained_path), answer=expected_answer,
                          elapsed_seconds=trial['elapsed_seconds'], first_output_seconds=trial['first_output_seconds'])
            if policy == 'on':
                candidates = [m for m in metadata if m['trace_tag'] == bodies[name]['kvmem_trace_tag']]
                assert len(candidates) == 1
                meta = candidates[0]
                assert meta['prompt_tokens'] == cases[name]['prompt_tokens']
                assert 0 < meta['selected'] < meta['block_count']
                assert meta['query_ready'] and meta['index_ready']
                assert meta['scorer_requested'] == meta['scorer_used'] == 'sub-block-mean-k-max'
                assert meta['scorer_fallback_reason'] == 'none'
                assert meta['query_expected'] == meta['query_captured']
                assert meta['qc_total_blocks'] == meta['qc_captured_blocks']
                record['retrieval'] = meta
            evidence.append(record)
        log_path = directory / 'engine/engine.log'
        spec = re.findall(r'native (mtp|dflash)_spec_summary: enabled=true .*?drafted=(\d+) accepted=(\d+).*?fallback=(\w+)', log_path.read_text())
        if method == 'none':
            assert not spec
        else:
            assert len(spec) == 2
            assert all(row[0] == ('mtp' if method == 'mtp' else 'dflash') and row[3] == 'false' for row in spec)
            assert sum(int(row[2]) for row in spec) > 0
        rows = [row for row in aggregate if row['profile'] == label]
        assert len(rows) == 2 and all(row['completed'] == row['trials'] == 1 for row in rows)
        profiles.append(dict(profile=label, runs=2, checks=evidence, binary_sha256=binary_sha256,
                             source_commit=manifest['source_commit'], engine_manifest_sha256=sha(directory / 'engine/manifest.json'),
                             engine_log_sha256=sha(log_path), drafted=sum(int(row[1]) for row in spec),
                             accepted=sum(int(row[2]) for row in spec), fallback_requests=0,
                             sampled_gpu_peak_bytes=max(row['sampled_profile_gpu_peak_bytes'] for row in rows),
                             engine_rss_high_water_bytes=max(row['profile_rss_high_water_bytes'] for row in rows)))
    assert len(profiles) == 6 and observed == expected
    csv_path = run / 'requests.csv'
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert {(row['profile'], row['case'], int(row['repetition'])) for row in rows} == expected
    return dict(kind='qualified_long_fixed_input_controls_current_engine', reviewed_at=time.time(), passed=True,
                requests=12, profiles=profiles, cases=summary['cases'], matched_model_controls=common,
                prefix_reuse=False, guided_refresh='off', vision_cache=summary['vision_cache'], case_metrics=aggregate,
                scope='Two immutable 158911-token historical-image bodies, one positive and one changed-image negative, once per profile. No desktop actions. Prefix reuse is disabled on both policies. Images are cached within each profile, so the second case is not a fully cold vision run. Sampled whole-device GPU peaks are not allocator bounds. These fixed-input checks do not qualify same-turn continuation with guided refresh disabled.',
                artifacts={str(path): sha(path) for path in
                           (run / 'summary.json', run / 'visual-checks.json', run / 'aggregate.json', csv_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), args.binary_sha256)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'requests': result['requests'], 'output': str(args.output)}))
