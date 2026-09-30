#!/usr/bin/env python3
"""CUDA/HTTP regression: Claude tool continuation immediately after compaction."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=REPO / 'models/Qwen3.5-4B-Q8_0-with-MTP.gguf')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guided', action='store_true', help='Exercise private direct guided Q and forced boundary selection')
    parser.add_argument('--min-thinking-tokens', type=int, default=1024,
                        help='Production threshold by default; smaller values isolate cache mechanics on 4B')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), 'serve', '--model', str(args.model.resolve()),
        '--host', '127.0.0.1', '--port', str(port), '--ctx', '16384',
        '--prefill-chunk', '128', '--paged-kv', '--kv-page-size', '16', '--kv-dtype', 'fp8',
        '--mtp-chain', '4', '--native-mtp-speculate', '--no-continuous-batching',
        '--enable-thinking', '--thinking-budget', '1536', '--kvmem', '--kvmem-prefix-cache',
        '--kvmem-query-conditioned', '--kvmem-update-mode', 'step', '--kvmem-repository-blocks', '512',
        '--kvmem-repository-gc', 'delete', '--kvmem-block-tokens', '32',
        '--kvmem-budget', '4096', '--kvmem-prefill-budget', '4096',
        '--kvmem-gen-budget', '8192', '--kvmem-sink-blocks', '1', '--kvmem-recent-blocks', '4',
        '--kvmem-index-placement', 'cpu', '--kvmem-cpu-gb', '4', '--kvmem-gpu-memory-ratio', '1.0',
        '--thinking-summary', '--thinking-summary-min-tokens', str(args.min_thinking_tokens)]
    if args.guided:
        command += ['--kvmem-guided-reselect', 'both', '--kvmem-guided-query-tokens', '512',
                    '--kvmem-guided-thinking-tokens', '0', '--kvmem-middecode-trigger-tokens', '6144']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    log_path = args.output / 'server.log'
    records = []
    outcome = {'passed': False, 'guided': args.guided, 'command': command, 'records': records}
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=dict(os.environ, QW3_KVMEM_PREFIX_CACHE_TRACE='1'),
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError('service exited before ready')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise RuntimeError('readiness timeout')

            def request(body, label, expect_hit=False):
                before = log_path.stat().st_size
                started = time.monotonic()
                route = '/v1/messages'
                payload = body
                if args.guided:
                    # Explicit guided-query controls are OpenAI extensions;
                    # Anthropic intentionally forwards only its own fields.
                    route = '/v1/chat/completions'
                    converted = [{'role': 'system', 'content': body['system']}]
                    for message in body['messages']:
                        if isinstance(message['content'], str):
                            converted.append(message)
                        elif message['role'] == 'assistant':
                            blocks = message['content']
                            converted.append({'role': 'assistant',
                                'content': ''.join(b.get('text', '') for b in blocks if b['type'] == 'text'),
                                'reasoning_content': ''.join(b.get('thinking', '') for b in blocks if b['type'] == 'thinking'),
                                'tool_calls': [{'id': b['id'], 'type': 'function', 'function': {
                                    'name': b['name'], 'arguments': json.dumps(b['input'])}}
                                    for b in blocks if b['type'] == 'tool_use']})
                        else:
                            converted += [{'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': b['content']}
                                          for b in message['content']]
                    payload = {'model': body['model'], 'messages': converted, 'max_tokens': body['max_tokens'],
                        'temperature': 0, 'enable_thinking': True, 'thinking_budget': body['thinking']['budget_tokens'],
                        'tool_choice': 'required',
                        'kvmem_query_message_range': {'message_begin': 3, 'message_end': 4},
                        'tools': [{'type': 'function', 'function': {'name': t['name'],
                            'description': t['description'], 'parameters': t['input_schema']}} for t in body['tools']]}
                    if not expect_hit:
                        payload.update(kvmem_reselect='force',
                            kvmem_query_guided_direct=True,
                            kvmem_query_guided_thinking_max_tokens=0, kvmem_query_guided_query_max_tokens=2048)
                req = urllib.request.Request(f'http://127.0.0.1:{port}{route}',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01', 'User-Agent': 'claude-cli/2.1.148 (cache regression)'})
                with opener.open(req, timeout=900) as response:
                    result = json.load(response)
                if args.guided:
                    msg = result['choices'][0]['message']
                    result = {'content': [{'type': 'thinking', 'thinking': msg.get('reasoning_content', '')},
                        {'type': 'text', 'text': msg.get('content', '') or ''},
                        *[{'type': 'tool_use', 'id': t['id'], 'name': t['function']['name'],
                           'input': json.loads(t['function']['arguments'])} for t in msg.get('tool_calls', [])]],
                        'openai_response': result}
                events = log_path.read_bytes()[before:].decode(errors='replace')
                hit = re.search(r'kvmem prefix-cache HIT \(mtp\): reuse=(\d+)', events)
                accounting = re.search(r'kvmem_reuse=(\d+) prefilled=(\d+)', events)
                row = {'label': label, 'elapsed_sec': time.monotonic() - started,
                    'request': payload, 'response': result, 'hit': bool(hit),
                    'reuse': int(hit[1]) if hit else 0,
                    'prefilled': int(accounting[2]) if accounting else None,
                    'compacted': 'status=compacted' in events,
                    'short': 'status=short-thinking' in events,
                    'invalid_summary': 'status=invalid-summary' in events}
                records.append(row)
                (args.output / f'{label}.json').write_text(json.dumps(row, indent=2) + '\n')
                (args.output / f'{label}.log').write_text(events)
                print(json.dumps({k: v for k, v in row.items() if k not in ('request', 'response')}), flush=True)
                if expect_hit:
                    assert hit and accounting, f'{label}: continuation lost prefix reuse'
                    assert re.search(r'prefix-cache HIT.*query_snapshot=1', events), 'query snapshot was not reused'
                    assert int(accounting[1]) == int(hit[1]) > 0
                    assert int(accounting[2]) < int(hit[1]), 'replayed most of the retained prompt'
                return result, row

            instruction = ('We must design a lock-free concurrent hash map using hazard pointers. '
                'Before opening the source, reason carefully about all interleavings of insertion, deletion, '
                'resizing, pointer reuse, ABA, stalled readers and memory reclamation; compare at least three '
                'designs and work through correctness counterexamples and memory ordering. '
                'Keep SENTINEL_R7: do not change the public API. No tests have been run. '
                'Do the entire detailed comparison and correctness analysis privately in your reasoning. '
                'Your visible response must contain ONLY a single Read call on src/cache.py. '
                'After a tool result, inspect the next requested file using Read. Do not implement anything yet.')
            if args.guided:
                instruction = ('We must design a lock-free concurrent hash map using hazard pointers. '
                    'Before opening the source, reason carefully about all interleavings of insertion, deletion, '
                    'resizing, pointer reuse, ABA, stalled readers and memory reclamation; compare at least three '
                    'designs and work through correctness counterexamples and memory ordering. '
                    'Keep the exact constraint SENTINEL_R7: do not change the public API. No tests have been run. '
                    'Do the entire detailed comparison, counterexamples, and correctness analysis privately in your reasoning. '
                    'Your visible response must contain ONLY a single Read call on src/cache.py, with no explanation or analysis. '
                    'Reading that file is the pending next action after the private analysis. Do not implement anything yet.')
            messages = [{'role': 'user', 'content': instruction}]
            if args.guided:
                # Provide retrievable history outside the mandatory root/current
                # query pins so selection can fit while the prompt is over budget.
                messages = [{'role': 'user', 'content': 'Review our cache implementation and preserve its public API.'},
                    {'role': 'assistant', 'content': '\n'.join(
                        f'Historical note {i}: compare allocation strategies and reader lifetime before changing internal storage.'
                        for i in range(300))}, *messages]
            common = {'model': 'test', 'max_tokens': 3072, 'temperature': 0,
                'system': 'You are Claude Code, a coding assistant. Use tools to inspect source. Never invent test results.',
                'tools': [{'name': 'Read', 'description': 'Read a source file.', 'input_schema': {
                    'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}}]}
            first, row = request({**common, 'messages': messages,
                'thinking': {'type': 'enabled', 'budget_tokens': 1536}}, '01-long-thinking')
            assert row['compacted'], 'fixture did not exercise successful long-thinking compaction'
            result = first
            for i in range(2, 5):
                calls = [b for b in result['content'] if b['type'] == 'tool_use']
                assert len(calls) == 1 and calls[0]['name'] == 'Read', 'expected one real Read tool call'
                messages = [*messages, {'role': 'assistant', 'content': result['content']},
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0]['id'],
                    'content': f'// Source stub, no tests run. Next inspect src/cache_{i}.py using Read only.'}]}]
                result, _ = request({**common, 'messages': messages,
                    'thinking': {'type': 'enabled', 'budget_tokens': 64}}, f'0{i}-tool-continuation', True)
            text = log_path.read_text()
            assert not re.search(r'POST .* -> 500|terminal_status=error', text)
            if args.guided:
                assert re.search(r'native kvmem guided query: mode=direct .*query_tokens=[1-9]', text), 'guided Q not exercised'
            outcome['passed'] = True
        except Exception as error:
            outcome['error'] = str(error)
            raise
        finally:
            (args.output / 'result.json').write_text(json.dumps(outcome, indent=2) + '\n')
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
