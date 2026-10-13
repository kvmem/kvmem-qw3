#!/usr/bin/env python3
"""Deterministic retention microbenchmark, not an official dataset/agent score.

Same observation events and questions, Q8/FP8/MTP, and active/generation budgets.
Full-retention off/shadow use a larger repository; delete has an 8K repository.
Generated answers are scored but not inserted into subsequent fixed inputs.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def trajectory(rounds):
    records = {}
    for i in range(rounds):
        service = 0 if i == 64 else i
        # Fixed public synthetic data, not credentials or actual customer codes.
        code = f'NOTE-{(i * 7919 + 4817) % 10000:04d}'
        records[service] = code
        event = (f'Configuration record {i}. Service service-{service:03d}. '
                 f'The current test marker is {code}. Revision {2 if i == 64 else 1}. '
                 'This observation comes from the configuration file after reading it. '
                 'This revision supersedes the earlier value for the same service. '
                 f'Log sequence {i}: the connection opened, the status check completed, '
                 'the retry counter stayed at zero, and the connection closed normally. '
                 'The record concerns a test environment and contains no production secrets.')
        question = None
        if i % 8 == 7:
            kind = ['frequent', 'recent', 'cold', 'frequent'][i // 8 % 4]
            target = 0 if kind == 'frequent' else service if kind == 'recent' else max(1, i - 40)
            question = {'kind': ('revised-frequent' if target == 0 and i >= 64 else kind),
                        'service': target, 'expected': records[target],
                        'text': f'What is the current test marker for service-{target:03d}? '
                                'Use the latest revision in the observations. Reply with only '
                                'the marker, or UNKNOWN if it is unavailable.'}
        yield {'step': i, 'event': event, 'question': question}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=128)
    p.add_argument('--port', type=int, default=18497)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = list(trajectory(args.rounds))
    (args.out / 'trajectory.json').write_text(json.dumps(data, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f'http://127.0.0.1:{args.port}'

    def request(path, body=None):
        value = None if body is None else json.dumps(body).encode()
        with opener.open(urllib.request.Request(url + path, data=value,
                headers={'Content-Type': 'application/json',
                         'anthropic-version': '2023-06-01'}), timeout=300) as r:
            return json.load(r)

    results = []
    for mode, capacity in [('off', 32768), ('shadow', 32768), ('delete', 8192)]:
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU is occupied: {occupied}')
        run = args.out / mode
        run.mkdir()
        cmd = [str(args.binary.resolve()), 'serve', '--model', str(args.model.resolve()),
               '--host', '127.0.0.1', '--port', str(args.port), '--ctx', str(capacity),
               '--prefill-chunk', '128', '--paged-kv', '--kv-page-size', '16',
               '--kv-dtype', 'fp8', '--mtp-chain', '4', '--kvmem',
               '--kvmem-repository-blocks', str(capacity // 32),
               '--kvmem-repository-gc', mode, '--kvmem-block-tokens', '32',
               '--kvmem-budget', '1024', '--kvmem-gen-budget', '1024',
               '--kvmem-sink-blocks', '1', '--kvmem-recent-blocks', '4',
               '--kvmem-index-placement', 'cpu', '--kvmem-cpu-gb', '1',
               '--kvmem-query-conditioned', '--kvmem-guided-reselect', 'both',
               '--kvmem-middecode-trigger-tokens', '512', '--kvmem-guided-query-tokens', '64']
        (run / 'config.json').write_text(json.dumps({'argv': cmd,
            'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            'capacity_tokens': capacity, 'sampling': 'greedy, thinking disabled',
            'protocol': 'retention-qa-v3: text observations; transient lookup questions; no tools',
            'max_tokens': {'ingest': 1, 'answer': 64},
            'scope': 'synthetic retention microbenchmark; not latency baseline'}, indent=2))
        env = os.environ.copy()
        env.update(QW3_KVMEM_PREFIX_CACHE_TRACE='1', QW3_KVMEM_GUIDED_MTP='1')
        with (run / 'server.log').open('w') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{mode} server exited during startup')
                    try:
                        model = request('/v1/models')['data'][0]['id']
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'{mode} server startup timed out')
                        time.sleep(0.25)
                messages = []
                body = {'model': model,
                        'system': 'Track configuration observations. Keep the current marker for '
                                  'each service, preferring the latest revision. For an explicit '
                                  'lookup question, output only the exact marker or UNKNOWN. '
                                  'Otherwise acknowledge the observation.',
                        'thinking': {'type': 'disabled'}, 'temperature': 0,
                        'messages': messages}
                answers = []
                with (run / 'responses.jsonl').open('w') as responses:
                    for event in data:
                        i, q = event['step'], event['question']
                        messages.append({'role': 'user', 'content': event['event']})
                        if q:
                            messages.append({'role': 'user', 'content': q['text']})
                        body['max_tokens'] = 64 if q else 1
                        response = request('/v1/messages', body)
                        if response.get('type') != 'message':
                            raise RuntimeError(f'{mode} step {i}: invalid message response')
                        responses.write(json.dumps({'step': i, 'response': response}) + '\n')
                        responses.flush()
                        if mode != 'delete' and response['usage']['input_tokens'] + 64 > capacity:
                            raise RuntimeError('full retention reference exceeded physical capacity')
                        if q:
                            answer = ''.join(x.get('text', '') for x in response['content']
                                             if x['type'] == 'text')
                            found = sorted(set(re.findall(r'NOTE-\d{4}', answer)))
                            item = dict(q, step=i, answer=answer, markers=found,
                                        correct=found == [q['expected']],
                                        input_tokens=response['usage']['input_tokens'])
                            answers.append(item)
                            (run / 'answers.json').write_text(json.dumps(answers, indent=2))
                        # Lookup questions are read-only probes of a fixed
                        # observation history. Never teach the next request a
                        # ground-truth answer or an invalid tool action.
                        if q:
                            messages.pop()
                        messages.append({'role': 'assistant', 'content': 'Observation recorded.'})
                result = {'mode': mode, 'capacity_tokens': capacity,
                          'questions': len(answers), 'correct': sum(x['correct'] for x in answers),
                          'by_kind': {kind: {'n': sum(x['kind'] == kind for x in answers),
                              'correct': sum(x['kind'] == kind and x['correct'] for x in answers)}
                              for kind in sorted(set(x['kind'] for x in answers))}}
                results.append(result)
                (args.out / 'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    main()
