#!/usr/bin/env python3
"""Manual GPU regression for isolated summaries and cooperative HTTP serving."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dflash', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--continuous', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith('QW3_')}
    env.update(QW3_KVMEM_PREFIX_CACHE_TRACE='1', QW3_CONTINUOUS_BATCHING_TRACE='1',
               QW3_KVMEM_PROBE_RESERVE_TOKENS='4096', QW3_CONTINUOUS_BATCHING_ADMISSION_WAIT_US='50000')
    if args.continuous:
        env.update(QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS='1', QW3_CONTINUOUS_KVMEM_CONTROLLER='1',
                   QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL='1', QW3_CONTINUOUS_KVMEM_SESSION_REUSE='1',
                   QW3_CONTINUOUS_DFLASH='1')
    command = [str(args.binary.resolve()), 'serve', '--model', str(args.model.resolve()),
        '--host', '127.0.0.1', '--port', str(port), '--ctx', '8192', '--prefill-chunk', '128',
        '--paged-kv', '--kv-page-size', '16', '--kv-dtype', 'fp8', '--max-active', '2',
        '--continuous-batching' if args.continuous else '--no-continuous-batching',
        '--enable-thinking', '--thinking-budget', '256', '--thinking-summary',
        '--thinking-summary-min-tokens', '64', '--kvmem', '--kvmem-prefix-cache',
        '--kvmem-query-conditioned', '--kvmem-update-mode', 'step',
        '--kvmem-repository-blocks', '256', '--kvmem-repository-gc', 'delete',
        '--kvmem-block-tokens', '32', '--kvmem-budget', '1024', '--kvmem-prefill-budget', '1024',
        '--kvmem-gen-budget', '2048', '--kvmem-sink-blocks', '1', '--kvmem-recent-blocks', '4',
        '--kvmem-index-placement', 'cpu', '--kvmem-cpu-gb', '3', '--kvmem-gpu-memory-ratio', '1']
    if args.dflash:
        command += ['--speculative-method', 'dflash2', '--dflash-model', str(args.dflash.resolve())]
    else:
        command += ['--speculative-method', 'mtp', '--mtp-chain', '4']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{port}'
    log_path = args.output / 'server.log'
    records = []
    prompt = ('Review a memoization decorator for recursive functions. Compare three designs for mutable arguments, '
              'exceptions, recursion cycles and concurrent calls; reason through races and counterexamples. '
              'The next action is to inspect src/cache.py. The public API must remain unchanged; no tests have run. '
              'Give only a short recommended next step in the answer.')
    def request(label, overrides=None, messages=None, barrier=None):
        body = {'model': 'test', 'messages': messages or [{'role': 'user', 'content': prompt}],
                'temperature': 0, 'max_tokens': 768, 'enable_thinking': True,
                'stream': True, **(overrides or {})}
        if barrier: barrier.wait(timeout=30)
        call = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'x-qw3-harness': 'generic-tool-client'})
        result = {'reasoning': '', 'content': '', 'done': False}
        with opener.open(call, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b'data: '): continue
                data = raw[6:].strip()
                if data == b'[DONE]': result['done'] = True; break
                event = json.loads(data)
                if event.get('error'): raise RuntimeError(event['error'])
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    result['reasoning'] += delta.get('reasoning_content', '')
                    result['content'] += delta.get('content', '')
                    if choice.get('finish_reason'): result['finish_reason'] = choice['finish_reason']
        assert result['done'], (label, result)
        assert 'PRIVATE REASONING COMPACTION' not in json.dumps(result), result
        (args.output / (label + '.json')).write_text(json.dumps({'request': body, 'response': result}, indent=2))
        records.append(label)
        return result
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError(f'server exited: {log_path}')
                try:
                    with opener.open(base + '/health', timeout=2): break
                except OSError: time.sleep(.5)
            else: raise RuntimeError('server readiness timeout')
            if args.continuous:
                barrier = threading.Barrier(2)
                with concurrent.futures.ThreadPoolExecutor(2) as workers:
                    a = workers.submit(request, 'summary-with-peer', {}, None, barrier)
                    b = workers.submit(request, 'decoding-peer', {'thinking_summary': False, 'max_tokens': 1024,
                        'ignore_eos': True, 'thinking_budget': 0}, None, barrier)
                    first = a.result(); b.result()
                text = log_path.read_text()
                phases = list(re.finditer(r'continuous_controller_summary_yield: request=(\d+) phase=(\w+)', text))
                assert phases and any(m[2] == 'decode' for m in phases), 'summary never yielded'
                during = text[phases[0].start():phases[-1].end()]
                assert re.search(r'continuous_(?:controller|dflash)_verify:', during), 'peer made no verification progress during summary'
                barrier = threading.Barrier(2)
                start = log_path.stat().st_size
                with concurrent.futures.ThreadPoolExecutor(2) as workers:
                    futures = [workers.submit(request, f'simultaneous-summary-{i}', {}, None, barrier) for i in range(2)]
                    for future in futures: future.result()
                text = log_path.read_text()[start:]
                ids = set(re.findall(r'continuous_controller_summary_yield: request=(\d+)', text))
                assert len(ids) == 2, 'two occupied parent slots did not both run summaries'
            else:
                first = request('serial-summary')
            assert 'status=compacted' in log_path.read_text(), 'fixture never produced an accepted summary'
            request('invalid-summary', {'thinking_summary_tokens': 1})
            request('short-thinking', {'thinking_summary_min_tokens': 4096, 'thinking_budget': 32, 'max_tokens': 128})
            messages = [{'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'reasoning_content': first['reasoning'], 'content': first['content']},
                        {'role': 'user', 'content': 'Have tests been run? Answer briefly.'}]
            request('followup', {'thinking_budget': 32, 'max_tokens': 128}, messages)
            cache_options = {'kvmem_prefix_cache_key': 'summary-cache'} if args.continuous else {}
            warm = request('cache-summary', cache_options)
            warm_messages = [{'role': 'user', 'content': prompt},
                {'role': 'assistant', 'reasoning_content': warm['reasoning'], 'content': warm['content']},
                {'role': 'user', 'content': 'Have tests been run? Answer briefly.'}]
            start = log_path.stat().st_size
            request('cache-followup', {**cache_options, 'thinking_budget': 32, 'max_tokens': 128}, warm_messages)
            cache_log = log_path.read_text()[start:]
            assert 'prefix-cache HIT' in cache_log, 'summary lost usable prefix state'
            if args.continuous:
                assert 'event=hit' in cache_log and 'key=summary-cache' in cache_log
            if args.continuous:
                # Disconnect precisely while the child owns its page lease,
                # then prove that the parent slot can serve another request.
                start = log_path.stat().st_size
                body = json.dumps({'model': 'test', 'messages': [{'role': 'user', 'content': prompt}],
                    'stream': True, 'temperature': 0, 'max_tokens': 768, 'enable_thinking': True}).encode()
                with socket.create_connection(('127.0.0.1', port), timeout=10) as client:
                    headers = (f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                               f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                               'Connection: close\r\n\r\n').encode()
                    client.sendall(headers + body)
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        if 'continuous_controller_summary_yield:' in log_path.read_text()[start:]: break
                        time.sleep(.01)
                    else: raise AssertionError('cancel fixture did not enter summary')
                    client.shutdown(socket.SHUT_RDWR)
                request('after-summary-cancel')
            text = log_path.read_text()
            assert 'summary_context=isolated-current-thinking' in text
            assert 'status=short-thinking' in text
            assert re.search(r'status=invalid-summary.*rollback_ms=0(?:\.0+)?\b.*replay_ms=0', text), 'invalid summary touched parent'
            report = {'passed': True, 'continuous': args.continuous, 'route': 'dflash2' if args.dflash else 'mtp',
                      'requests': records, 'compactions': text.count('[thinking-compaction] status=compacted')}
            (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()

if __name__ == '__main__': main()
