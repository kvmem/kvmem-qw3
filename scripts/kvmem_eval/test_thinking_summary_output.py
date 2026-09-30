#!/usr/bin/env python3
"""27B summary diagnostics: historical reasoning replay and actual tool transaction.

The old historical replay includes task context; the new replay supplies only
the current reasoning to an isolated summary prompt. Neither reuses original KV.
The live transaction below separately exercises the production compaction path.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

REPO = Path('/home/chaidi/qw3')
OLD_PROMPT = ('\n<|im_end|>\n<|im_start|>user\nPRIVATE REASONING COMPACTION\n'
    'Summarize the immediately preceding reasoning in the shortest description '
    'that retains the key conclusion and information needed to continue the task. '
    'Keep only essential decisions, constraints, and unresolved next steps; '
    'omit exploratory details and repetition. Output only a concise plain-text '
    'summary. Do not redo the task, call tools, or invent completed results.\n'
    '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-artifact', type=Path, required=True)
    p.add_argument('--reuse-history', type=Path,
                   help='Reuse completed old/new replay responses; rerun only the live transaction')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    header = (REPO / 'src/thinking_compaction.hpp').read_text()
    def prompt_fragment(name):
        body = header.split('inline std::string ' + name + '()', 1)[1].split('return ', 1)[1].split(';\n}', 1)[0]
        return ''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"', body))
    new_prefix = prompt_fragment('thinking_summary_prefix')
    new_suffix = prompt_fragment('thinking_summary_suffix')
    notes = []
    for line in (args.historical_artifact / 'claude.stream.jsonl').open():
        r = json.loads(line)
        if r.get('type') != 'assistant':
            continue
        for b in r['message'].get('content', []):
            text = b.get('thinking', '') if b.get('type') == 'thinking' else ''
            if len(text) > 6000 and text not in notes:
                notes.append(text)
    assert len(notes) >= 3, 'three failed reasoning spans missing'
    tools = [{'type': 'function', 'function': {'name': 'Read', 'description': 'Read a source file.',
        'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}}},
        {'type': 'function', 'function': {'name': 'Bash', 'description': 'Run a shell command.',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
    prefix = ('<|im_start|>system\nYou are Claude Code, a coding assistant.\n# Tools\n<tools>\n'
        + '\n'.join(json.dumps(t) for t in tools) + '\n</tools>\n'
        'Function calls use <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>. '
        'Use tools to inspect and modify the repository and run tests.\n<|im_end|>\n<|im_start|>user\n'
        + (args.historical_artifact / 'prompt.md').read_text() + '\n<|im_end|>\n<|im_start|>assistant\n<think>\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), 'serve', '--model', str(REPO/'models/Qwen3.8-27B-Q8_0.gguf'),
        '--host', '127.0.0.1', '--port', str(port), '--ctx', '262144', '--prefill-chunk', '512',
        '--paged-kv', '--kv-page-size', '16', '--kv-dtype', 'fp8', '--mtp-chain', '4',
        '--native-mtp-speculate', '--no-continuous-batching', '--enable-thinking', '--thinking-budget', '2048',
        '--kvmem', '--kvmem-prefix-cache', '--kvmem-query-conditioned', '--kvmem-update-mode', 'step',
        '--kvmem-repository-blocks', '8192', '--kvmem-repository-gc', 'delete', '--kvmem-block-tokens', '32',
        '--kvmem-budget', '131072', '--kvmem-prefill-budget', '131072', '--kvmem-gen-budget', '32768',
        '--kvmem-index-placement', 'cpu', '--kvmem-cpu-gb', '48', '--thinking-summary',
        '--thinking-summary-min-tokens', '1024']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    def request(route, body, label):
        (args.output / (label + '.request.json')).write_text(json.dumps(body, ensure_ascii=False))
        started = time.monotonic()
        req = urllib.request.Request(f'http://127.0.0.1:{port}{route}', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'x-qw3-harness': 'generic-tool-client'})
        with opener.open(req, timeout=900) as response:
            result = json.load(response)
        record = {'label': label, 'elapsed_sec': time.monotonic()-started, 'response': result}
        records.append(record)
        (args.output / (label + '.response.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(label, round(record['elapsed_sec'], 2), flush=True)
        return result
    log_path = args.output / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, QW3_KVMEM_PREFIX_CACHE_TRACE='1'))
        try:
            for _ in range(600):
                if process.poll() is not None: raise RuntimeError('27B service exited')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health', timeout=2): break
                except OSError: time.sleep(.5)
            else: raise RuntimeError('27B readiness timed out')
            for i, note in enumerate(notes[:3], 1):
                for variant in ('old', 'new'):
                    summary_prompt = (prefix + note + '\n</think>' + OLD_PROMPT if variant == 'old'
                                      else new_prefix + note + new_suffix)
                    if args.reuse_history:
                        saved = json.loads((args.reuse_history / f'history-{i}-{variant}.response.json').read_text())
                        records.append(saved)
                        result = saved['response']
                    else:
                        result = request('/v1/completions', {'model': 'test', 'prompt': summary_prompt,
                            'temperature': 0, 'enable_thinking': False, 'thinking_summary': False, 'max_tokens': 2048},
                            f'history-{i}-{variant}')
                    if variant == 'new':
                        choice = result['choices'][0]
                        assert choice['finish_reason'] == 'stop', 'new replay did not end naturally'
                        assert choice['text'].strip(), 'empty replay summary'
                        assert not re.search(r'<(?:\||/?(?:tool|think|function|parameter))|```', choice['text']), 'invalid replay summary'

            messages = [{'role': 'system', 'content': 'You are a coding assistant. Use Read to inspect actual source before making edits. Never invent file contents or test results.'},
                {'role': 'user', 'content': 'We must design a lock-free concurrent hash map using hazard pointers. '
                    'Before opening the source, reason carefully about all interleavings of insertion, deletion, '
                    'resizing, pointer reuse, ABA, stalled readers and memory reclamation; compare at least three '
                    'designs and work through correctness counterexamples and memory ordering. '
                    'Keep the exact constraint SENTINEL_R7: do not change the public API. No tests have been run. '
                    'Do the entire detailed comparison, counterexamples, and correctness analysis privately in your reasoning. '
                    'Your visible response must contain ONLY a single Read call on src/cache.py, with no explanation or analysis. '
                    'Reading that file is the pending next action after the private analysis. Do not implement anything yet.'}]
            body = {'model': 'test', 'messages': messages, 'tools': tools[:1], 'tool_choice': 'required',
                    'enable_thinking': True, 'max_tokens': 4096, 'temperature': 0, 'thinking_budget': 2048}
            before = log_path.stat().st_size
            result = request('/v1/chat/completions', body, 'transaction-tool')
            msg = result['choices'][0]['message']
            events = log_path.read_text()[before:]
            assert 'status=compacted' in events, '27B transaction did not exercise long compaction'
            assert 'SENTINEL_R7' in msg.get('reasoning_content', ''), 'summary lost the required constraint'
            calls = msg.get('tool_calls', [])
            assert len(calls) == 1 and calls[0]['function']['name'] == 'Read', 'main tool invocation broken'
            assert json.loads(calls[0]['function']['arguments'])['file_path'] == 'src/cache.py', 'wrong tool target'
            messages += [msg, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                'content': 'The file contains only: // TODO concurrent map. No test command was run. '
                    'Next inspect src/cache_followup.py using Read only.'}]
            before = log_path.stat().st_size
            result = request('/v1/chat/completions', {'model': 'test', 'messages': messages, 'tools': tools[:1],
                'tool_choice': 'required', 'max_tokens': 384, 'thinking_budget': 64, 'temperature': 0}, 'transaction-followup')
            followup_events = log_path.read_bytes()[before:].decode(errors='replace')
            assert re.search(r'prefix-cache HIT.*query_snapshot=1', followup_events), 'post-summary query snapshot was not reused'
            accounting = re.search(r'kvmem_reuse=(\d+) prefilled=(\d+)', followup_events)
            assert accounting and int(accounting[1]) > int(accounting[2]), 'post-summary continuation replayed the retained prefix'
            followup_message = result['choices'][0]['message']
            followup_calls = followup_message.get('tool_calls', [])
            assert len(followup_calls) == 1 and followup_calls[0]['function']['name'] == 'Read'
            assert json.loads(followup_calls[0]['function']['arguments'])['file_path'] == 'src/cache_followup.py'
            messages += [followup_message, {'role': 'tool', 'tool_call_id': followup_calls[0]['id'],
                'content': 'This file is empty. No test command was run.'},
                {'role': 'user', 'content': 'Do not call more tools. State the unchanged API constraint and whether tests have been run, in one sentence.'}]
            result = request('/v1/chat/completions', {'model': 'test', 'messages': messages, 'tools': tools[:1],
                'tool_choice': 'none', 'max_tokens': 384, 'thinking_budget': 64, 'temperature': 0}, 'transaction-state-check')
            content = result['choices'][0]['message'].get('content', '')
            assert content and not result['choices'][0]['message'].get('tool_calls'), 'follow-up failed'
            assert 'api' in content.lower() and re.search(r'not|no tests', content, re.I), 'follow-up lost task state'
            text = log_path.read_text()
            assert not re.search(r'POST .* -> 500|terminal_status=error', text), 'inference error'
            (args.output/'result.json').write_text(json.dumps({'passed': True, 'records': records,
                'historical_replay_limit': __doc__}, ensure_ascii=False, indent=2)+'\n')
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == '__main__':
    main()
