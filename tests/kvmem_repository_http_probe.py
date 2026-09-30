#!/usr/bin/env python3
"""Manual local HTTP continuation test; synthetic tool trajectory, not agent quality."""
import argparse
import base64
import struct
import zlib
import copy
import csv
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18497')
    parser.add_argument('--rounds', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-max-output', type=int, default=1024,
                        help='model-list output allowance including configured refresh epochs')
    parser.add_argument('--images', action='store_true', help='include a small deterministic PNG test fixture every 16 tool results')
    parser.add_argument('--fault-checks', action='store_true',
                        help='after normal growth, test tail/old-prefix rewrite and stream cancellation')
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('output directory already exists; do not overwrite evidence')
    args.out.mkdir(parents=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(args.url + path, data=data, headers={
            'Content-Type': 'application/json', 'anthropic-version': '2023-06-01',
            'User-Agent': 'claude-cli/2.1.148 (bounded repository protocol probe)',
        })
        try:
            with opener.open(req, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from error

    models = request('/v1/models')
    (args.out / 'models.json').write_text(json.dumps(models, indent=2))
    model = models['data'][0]
    capacity = model['kvmem_repository_tokens']
    assert model['context_window'] > capacity
    assert model['max_output_tokens'] == args.expected_max_output
    def image_fixture(index):
        # Minimal test pixels exercise the real visual frontend without network
        # inputs, screenshot tools, or any task-quality interpretation.
        def chunk(kind, payload):
            return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind+payload))
        rgb = bytes([(index*17)%256, 80, 160])
        raw = b''.join(b'\0' + rgb*32 for _ in range(32))
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B',32,32,8,2,0,0,0)) + chunk(b'IDAT',zlib.compress(raw)) + chunk(b'IEND',b'')
        return {'type':'image','source':{'type':'base64','media_type':'image/png','data':base64.b64encode(png).decode()}}

    messages = [{'role': 'user', 'content':
        'Inspect the network retry issue in src/client.cpp. Read the tool results, '
        'retain evidence, and determine the function, retry limit and observed error.'}]
    body = {
        'model': model['id'],
        'system': "You are Claude Code, a coding assistant. Inspect the logs and answer briefly.",
        'tools': [{'name': 'Read', 'description': 'Read a source file.',
                   'input_schema': {'type': 'object', 'properties': {
                       'file_path': {'type': 'string'}}, 'required': ['file_path']}}],
        'messages': messages, 'max_tokens': 1, 'temperature': 0,
    }
    (args.out / 'config.json').write_text(json.dumps({
        'rounds': args.rounds, 'capacity_tokens': capacity, 'images': args.images,
        'kind': 'fixed synthetic tool trajectory over native Anthropic protocol',
        'quality_measurement': False, 'controlled_latency_measurement': False,
    }, indent=2))
    with (args.out / 'requests.csv').open('w') as f, gzip.open(
            args.out / 'responses.jsonl.gz', 'wt') as responses:
        out = csv.DictWriter(f, fieldnames=['round', 'input_tokens', 'output_tokens', 'wall_ms'])
        out.writeheader()
        last_input = 0
        for round_id in range(args.rounds):
            if round_id and round_id % 16 == 0:
                # A real new user query must retain the normal private-query
                # generation/selection/replay path, not only pressure eviction.
                messages.append({'role': 'user', 'content':
                    f'Investigation update {round_id}: find the retry limit and '
                    'network error in the prior logs, and check cancellation behavior.'})
            call_id = f'read_{round_id}'
            messages.extend([
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': call_id,
                    'name': 'Read', 'input': {'file_path': 'src/client.cpp'}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call_id,
                    'content': f'Tool log iteration {round_id}. Function fetch_record retries '
                        'network requests. The retry limit is three and timeout is 500 ms. '
                        'Error: connection reset by peer. The previous request used a stale '
                        'connection. Regression tests cover timeout and cancellation. '
                        'Line 10 opens the connection; line 20 checks its state; line 30 '
                        'handles the error; line 40 retries; line 50 closes the resource.'}]},
            ])
            if args.images and round_id % 16 == 0:
                result_block = messages[-1]['content'][0]
                result_block['content'] = [{'type':'text', 'text': result_block['content']}, image_fixture(round_id//16)]
            body['max_tokens'] = 16 if round_id % 16 == 0 else 1
            start = time.monotonic()
            result = request('/v1/messages', body)
            elapsed = (time.monotonic() - start) * 1000
            responses.write(json.dumps({'round': round_id, 'response': result}) + '\n')
            if result.get('type') != 'message':
                raise RuntimeError(f'invalid message response at round {round_id}: {result}')
            usage = result['usage']
            last_input = usage['input_tokens']
            out.writerow({'round': round_id, 'input_tokens': last_input,
                          'output_tokens': usage['output_tokens'], 'wall_ms': elapsed})
            f.flush()
        with gzip.open(args.out / 'last_request.json.gz', 'wt') as payload:
            json.dump(body, payload)
    summary = {'requests': args.rounds, 'last_input_tokens': last_input,
               'capacity_tokens': capacity, 'capacity_multiple': last_input / capacity,
               'complete': True}
    if args.fault_checks:
        checks = []

        def check(name, value, note):
            value['kvmem_trace_tag'] = name
            start = time.monotonic()
            result = request('/v1/messages', value)
            assert result.get('type') == 'message', result
            checks.append({'name': name, 'usage': result['usage'], 'note': note,
                           'wall_ms': (time.monotonic() - start) * 1000})

        # Keep the edit inside the final block covered by the branch-safe P.
        tail = copy.deepcopy(body)
        tail['max_tokens'] = 1
        content = tail['messages'][-1]['content'][0]['content']
        if isinstance(content, list):
            content[0]['text'] = content[0]['text'][:-9] + 'handle.'
        else:
            tail['messages'][-1]['content'][0]['content'] = content[:-9] + 'handle.'
        check('bounded-tail-rewrite', tail, 'expect retained-tail checkpoint reuse')
        ancient = copy.deepcopy(tail)
        ancient['messages'][0]['content'] += ' Also check retry cancellation.'
        check('bounded-old-prefix-rewrite', ancient,
              'changed before retained tail; safe cold rebuild expected')
        ancient['messages'].extend([
            {'role': 'assistant', 'content': 'The tool result is recorded.'},
            {'role': 'user', 'content': 'Summarize the current findings in detail.'}])
        ancient['max_tokens'] = 1024
        ancient['stream'] = True
        ancient['kvmem_trace_tag'] = 'bounded-client-cancel'
        req = urllib.request.Request(args.url + '/v1/messages',
            data=json.dumps(ancient).encode(), headers={'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'})
        deltas = 0
        with opener.open(req, timeout=300) as response:
            for line in response:
                if line.startswith(b'data: '):
                    event = json.loads(line[6:])
                    if event.get('type') == 'content_block_delta':
                        deltas += 1
                        if deltas >= 2:
                            break  # close a live stream, not a completed answer
        assert deltas >= 2, 'stream completed before cancellation could be tested'
        checks.append({'name': 'bounded-client-cancel', 'received_deltas': deltas})
        # Allow disconnect detection; this sleep is outside normal-run timing.
        time.sleep(1)
        ancient.pop('stream')
        ancient['max_tokens'] = 1
        check('bounded-after-cancel', ancient,
              'request must succeed; conservative warm-cache invalidation is allowed')
        summary['fault_checks'] = checks
        (args.out / 'fault_checks.json').write_text(json.dumps(checks, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
