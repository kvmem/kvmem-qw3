#!/usr/bin/env python3
"""Serial fixed-transcript off/shadow/delete check within one shared KV capacity.

This is a management microbenchmark, not a task-quality or full agent benchmark.
Token-ready intervals are measured at the native backend callback, before HTTP.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    return {f'p{p}': ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
            for p in (50, 95, 99)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=80)
    parser.add_argument('--output-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('refusing to overwrite an existing measurement')
    if args.rounds > 86 or args.rounds < 8:
        raise ValueError('this shared-capacity control supports 8..86 rounds')
    args.out.mkdir(parents=True)
    binary = str(args.binary.resolve())
    model = str(args.model.resolve())
    manifest = {'binary': binary, 'binary_sha256': hashlib.sha256(
        Path(binary).read_bytes()).hexdigest(), 'model': model,
        'rounds': args.rounds, 'output_tokens': args.output_tokens,
        'repeats': args.repeats, 'capacity_tokens': 8192,
        'selection_tokens': 1024, 'generation_tokens': 1024,
        'block_tokens': 32, 'kv': 'FP8', 'mtp': 4,
        'order': 'rotate off/shadow/delete each repeat', 'warmup_rounds': 8,
        'scope': 'native backend full fixed transcript, CPU plain Mean-K, no NVMe',
        'caveat': 'All inputs fit one repository. Delete may discard cold blocks near '
                  'the high watermark. Active-window and input/output limits match. '
                  'Not an unlimited-history no-GC baseline or HTTP latency.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    modes = ['off', 'shadow', 'delete']
    results = []
    for rep in range(args.repeats):
        order = modes[rep % 3:] + modes[:rep % 3]
        for mode in order:
            occupied = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if occupied:
                raise RuntimeError(f'GPU is occupied; do not overlap timing: PIDs {occupied}')
            name = f'{mode}_r{rep + 1}'
            env = os.environ.copy()
            env.update(QW3_TEST_REPOSITORY_GC=mode,
                       QW3_TEST_OUTPUT_TOKENS=str(args.output_tokens),
                       QW3_TEST_TIMING_EVENTS=str((args.out / (name + '.tokens.csv')).resolve()))
            command = [binary, model, str(args.rounds), 'guided-text']
            with (args.out / (name + '.csv')).open('w') as out, (
                    args.out / (name + '.log')).open('w') as err:
                subprocess.run(command, env=env, stdout=out, stderr=err,
                               timeout=1200, check=True)
            rows = list(csv.DictReader((args.out / (name + '.csv')).open()))
            events = list(csv.DictReader((args.out / (name + '.tokens.csv')).open()))
            log = (args.out / (name + '.log')).read_text()
            if len(rows) != args.rounds or 'prefix-cache MISS' in log:
                raise RuntimeError(f'{name}: incomplete trajectory or unexpected prefix miss')
            if any(int(r['callbacks']) != args.output_tokens for r in rows):
                raise RuntimeError(f'{name}: callbacks are not one per requested output token')
            if int(rows[-1]['history']) + args.output_tokens > 8192:
                raise RuntimeError('off/shadow control exceeded the physical capacity')
            valid = rows[8:]
            intervals = []
            last_round, last_time = -1, 0.0
            for event in events:
                rid, elapsed = int(event['round']), float(event['elapsed_ms'])
                if rid >= 8 and rid == last_round:
                    intervals.append(elapsed - last_time)
                last_round, last_time = rid, elapsed
            captures = [line for line in log.splitlines() if 'kvmem bounded history=' in line]
            gc = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', captures[-1]))
            result = {'mode': mode, 'repeat': rep + 1, 'requests': len(rows),
                'prefix_misses': log.count('prefix-cache MISS'),
                'ttft_ms': quantiles([float(r['ttft_ms']) for r in valid]),
                'tbt_ms': quantiles(intervals),
                'decode_tokens_per_second': sum(int(r['callbacks']) - 1 for r in valid) /
                    (sum(float(r['callback_span_ms']) for r in valid) / 1000),
                'request_tokens_per_second': sum(int(r['callbacks']) for r in valid) /
                    (sum(float(r['elapsed_ms']) for r in valid) / 1000),
                'gc_and_resources_final': gc}
            results.append(result)
            (args.out / 'results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
