#!/usr/bin/env python3
"""Separate, unscored multimodal KVMem capacity and replay check."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import urllib.error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--profile', choices=('stepwise', 'bulk-stress', 'image-trajectory'), default='stepwise')
    ap.add_argument('--image-dir', type=Path)
    args = ap.parse_args()
    settings = json.loads(args.settings.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(settings)
    env['QW3_KVMEM_PREFIX_CACHE_TRACE'] = '1'
    port = 18081
    command = [settings['QW3_OSWORLD_BINARY'], 'serve', '--model', 'models/Qwen3.8-27B-Q8_0.gguf',
               '--vision-model', 'models/Qwen3.8-27B-BF16', '--vision-device', 'cuda',
               '--host', '127.0.0.1', '--port', str(port), '--ctx', settings['QW3_OSWORLD_CONTEXT_TOKENS'],
               '--kv-dtype', settings.get('QW3_OSWORLD_KV_DTYPE', 'fp16'), '--prefill-chunk', '2048', '--mtp-chain', '4',
               '--no-continuous-batching', '--kvmem', '--kvmem-prefix-cache',
               '--kvmem-block-tokens', settings['QW3_OSWORLD_KVMEM_BLOCK_TOKENS'],
               '--kvmem-budget', settings['QW3_OSWORLD_KVMEM_BUDGET'],
               '--kvmem-prefill-budget', settings['QW3_OSWORLD_KVMEM_BUDGET'],
               '--kvmem-gen-budget', settings['QW3_OSWORLD_KVMEM_GEN_BUDGET'],
               '--kvmem-method', 'retrieval', '--kvmem-retrieval-method', 'mean-k',
               '--kvmem-query-conditioned', '--kvmem-index-placement', 'cpu',
               '--kvmem-opt-stage-out', settings.get('QW3_OSWORLD_KVMEM_OPT_STAGE_OUT', 'on'),
               '--kvmem-update-mode', 'step', '--kvmem-guided-reselect', 'both',
               '--kvmem-guided-thinking-tokens', '0', '--kvmem-guided-query-tokens', '256',
               '--kvmem-middecode-trigger-tokens', settings['QW3_OSWORLD_KVMEM_REFRESH_TOKENS'],
               '--kvmem-middecode-max-refreshes', '2', '--kvmem-sink-tokens', '2048',
               '--kvmem-recent-tokens', '16384', '--kvmem-gpu-memory-ratio', '0.90',
               '--kvmem-cpu-gb', settings['QW3_OSWORLD_KVMEM_CPU_GB'],
               '--kvmem-nvme-gb', '0', '--no-kvmem-raw-k-nvme',
               '--temp', '0', '--thinking-budget', '0', '-n', '32768']
    (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None, timeout=600):
        req = urllib.request.Request(f'http://127.0.0.1:{port}'+path,
              data=None if body is None else json.dumps(body).encode(),
              headers={'Content-Type': 'application/json', 'x-qw3-harness': 'osworld'})
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            raise RuntimeError(f'HTTP {exc.code} from {path}: {detail}') from exc
    log_path = out/'server.log'
    rows = []
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        (out/'server.pid').write_text(str(proc.pid)+'\n')
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f'Preflight server exited with code {proc.returncode}; see server.log')
                try:
                    request('/health', timeout=1)
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('Preflight server startup timed out')
            unit = 'Recorded observation: the application state is unchanged.\n'
            unit_count = request('/tokenize', {'content': unit*1000})['count']
            image = 'data:image/png;base64,'+base64.b64encode(args.image.read_bytes()).decode()
            image_pool = [image]
            used_images = {image}
            if args.image_dir:
                image_pool = ['data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()
                              for f in sorted(args.image_dir.glob('step_*.png'))]
                assert image_pool, 'No recorded screenshots found'
            reply = ('OK, observation recorded.' if args.profile == 'image-trajectory' else 'OK')
            messages = [{'role': 'system', 'content': 'Check the observation history. Reply with exactly '+reply+' Do not execute actions.'},
                        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}},
                                                   {'type': 'text', 'text': 'Task: verify the screenshot remains available. Reply '+reply}]}]
            # The first large append prepares history below selection capacity.
            # Later stepwise appends stay near the GUI's 32K output ceiling.
            # Preserve the separate bulk stress case: consecutive 50K/80K
            # additions can require a replay suffix exceeding the 128K budget.
            additions = ([0, 100000, 45000, 50000, 80000, 2048]
                         if args.profile == 'bulk-stress'
                         else [0, 100000] + [30000] * 6 + [2048])
            if args.profile == 'image-trajectory':
                additions = [0] * 140
            for turn, extra_tokens in enumerate(additions):
                if turn:
                    used_images.add(image_pool[turn % len(image_pool)])
                    filler = unit * max(1, extra_tokens * 1000 // unit_count)
                    messages.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': '<tool_response>\n'+filler},
                        {'type': 'image_url', 'image_url': {'url': image_pool[turn % len(image_pool)]}},
                        {'type': 'text', 'text': '\n</tool_response>'}]})
                payload = {'model': 'gpt-qw3', 'messages': messages, 'temperature': 0,
                           'max_tokens': 16, 'seed': 0, 'enable_thinking': False, 'preserve_thinking': False}
                started = time.monotonic()
                response = request('/v1/chat/completions', payload)
                answer = response['choices'][0]['message']
                assert answer.get('content') or answer.get('tool_calls'), response
                row = {'turn': turn+1, 'prompt_tokens': response['usage']['prompt_tokens'],
                       'wall_sec': time.monotonic()-started, 'timing': response.get('timing'),
                       'message': answer}
                rows.append(row)
                (out/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                messages.append({'role': 'assistant', 'content': answer.get('content') or 'OK'})
            text = log_path.read_text()
            assert rows[-1]['prompt_tokens'] > 262144
            assert text.count('kvmem prefix-cache HIT') >= len(rows) - 1
            assert 'kvmem_index_placement=cpu' in text
            assert 'kvmem_nvme_tier=0 bytes' in text
            assert 'native kvmem query replay:' in text, 'No completed semantic replay recorded'
            if args.profile == 'image-trajectory':
                assert 'images=140 ' in text
                assert re.search(r'retrievable_visual_spans=[1-9]', text)
                assert re.search(r'native mtp_spec_summary: enabled=true batches=[1-9]', text)
            (out/'report.json').write_text(json.dumps({'ok': True, 'scored_trial': False,
                          'profile': args.profile,
                          'unique_image_inputs': len(used_images),
                          'available_unique_image_inputs': len(set(image_pool)),
                          'semantic_replays': text.count('native kvmem query replay:'),
                          'rows': rows, 'prefix_hits': text.count('kvmem prefix-cache HIT'),
                          'prefix_misses': text.count('kvmem prefix-cache MISS')}, indent=2)+'\n')
        except BaseException as exc:
            (out/'failure.json').write_text(json.dumps({'error': str(exc), 'server_exit_code': proc.poll(),
                                                       'rows_completed': len(rows)}, indent=2)+'\n')
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    main()
