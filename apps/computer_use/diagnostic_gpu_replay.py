"""Replay three historical inference bodies without dispatching desktop tools.

Use the existing, confirmed idle computer-use engine only. No other GPU process
is stopped. Historical/current pairs are diagnostic observations, not a causal
estimate: engine cache state and generated output may differ.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time

import aiohttp

from diagnostic_priority_checks import ROOT, SOURCE, digest, save
from kvmem_osworld_queue import process_identity, same_process
from osworld_linux.archive import read_request

TRACES = ('5c36fe4705894897afc5e8f7c1a6a212', '0a02b59c51c64f41b5b8bc4ee127d480',
          '7f781cfa630e4ac6b9b99fe0b5497458')


def gpu_sample():
    rows = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=10).stdout
    return dict(at=time.time(), processes=[dict(pid=int(line.split(',')[0]), memory=line.split(',')[1].strip())
        for line in rows.splitlines() if line.strip()])


async def run(args):
    from qw3_computer.protocol import read_sse
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'apps/computer_use/.runtime'):
        raise ValueError('Use a goal-owned diagnostic runtime')
    engine_dir = args.engine.resolve()
    engine = json.loads((engine_dir / 'manifest.json').read_text())
    identity = process_identity(engine['pid'])
    if not identity or identity['command'] != engine['command'] or identity['cwd'] != str(ROOT):
        raise ValueError('Actual engine identity changed')
    old_dir = SOURCE / 'attempts/004-full-retrievable-r1'
    old_engine = json.loads((old_dir / 'engine-manifest.json').read_text())
    for k in ('command', 'qw3_env', 'binary_sha256', 'kvmem', 'prefix_reuse', 'context_tokens'):
        if old_engine.get(k) != engine.get(k):
            raise ValueError('Historical/current profile differs: ' + k)
    if digest(identity['command'][0]) != engine['binary_sha256']:
        raise ValueError('Registered engine binary changed')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = dict(kind='isolated_exact_historical_body_inference_diagnostic', started_at=time.time(),
        passed=False, actual_engine_identity=identity, old_engine_manifest_sha256=digest(old_dir / 'engine-manifest.json'),
        current_engine_manifest_sha256=digest(engine_dir / 'manifest.json'), model_profile_equal=True,
        real_model_requests=0, device_inputs=0, historical_other_compute_pids=old_engine['existing_compute_pids'],
        samples=[], trials=[], failures=[], causal_gpu_slowdown_proven=False,
        source_sha256=digest(__file__), limitation='Single replay per body; old requests include bridge overhead. '
        'Generated output and engine cache/session state may differ. Same prompt does not isolate all causal variables.')
    save(output / 'result.json', result)
    inflight = False

    async def sample_loop():
        while True:
            sample = await asyncio.to_thread(gpu_sample)
            sample['inflight'] = inflight
            sample['other_compute_pids'] = [p['pid'] for p in sample['processes'] if p['pid'] != identity['pid']]
            result['samples'].append(sample)
            save(output / 'result.json', result)
            await asyncio.sleep(1)

    sampler = asyncio.create_task(sample_loop())
    try:
        await asyncio.sleep(2)
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=600, sock_read=180)) as session:
            for origin in ('http://127.0.0.1:18215', 'http://127.0.0.1:18505'):
                async with session.get(origin + '/api/state') as reply:
                    state = await reply.json()
                if state['status'] not in {'idle', 'paused'} or not state['desktop_paused'] or state['uncertain_actions']:
                    raise RuntimeError('Old desktop task is not fully paused/idle')
            for trace in TRACES:
                if any(s['other_compute_pids'] for s in result['samples'][-2:]):
                    raise RuntimeError('Another GPU process appeared before inference; no request sent')
                if not same_process(identity, process_identity(identity['pid'])):
                    raise RuntimeError('Registered inference generation changed')
                source = old_dir / 'app/inference' / trace
                data = read_request(source)
                historical = json.loads((source / 'metrics.json').read_text())
                row = dict(trace=trace, request_sha256=hashlib.sha256(data).hexdigest(),
                    original_metrics=historical, started_at=time.time(), status='requesting',
                    exact_backend_body_reused=True, model_tools_executed=0)
                result['trials'].append(row); result['real_model_requests'] += 1
                save(output / 'result.json', result)
                start = time.monotonic(); content = reasoning = 0; calls = []
                inflight = True
                async with session.post('http://127.0.0.1:18080/v1/chat/completions', data=data,
                                        headers={'Content-Type': 'application/json'}) as reply:
                    reply.raise_for_status()
                    with (output / (trace + '-backend.jsonl')).open('x') as log:
                        async for event in read_sse(reply.content):
                            if event == '[DONE]':
                                break
                            chunk = json.loads(event)
                            if chunk.get('error'):
                                raise RuntimeError('Backend returned an inference error')
                            log.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                            if chunk.get('usage'): row['usage'] = chunk['usage']
                            for choice in chunk.get('choices', []):
                                delta = choice.get('delta', {})
                                if any(delta.get(k) for k in ('content', 'reasoning_content', 'tool_calls')):
                                    row.setdefault('first_model_delta_seconds', time.monotonic() - start)
                                content += len(delta.get('content', ''))
                                reasoning += len(delta.get('reasoning_content', ''))
                                calls.extend(delta.get('tool_calls', []))
                                if choice.get('finish_reason'): row['finish_reason'] = choice['finish_reason']
                inflight = False
                row.update(elapsed_seconds=time.monotonic()-start, status='finished', content_chars=content,
                    reasoning_chars=reasoning, tool_call_chunks_retained_not_executed=len(calls), finished_at=time.time())
                row['isolated_gpu_residency'] = not any(s['other_compute_pids'] for s in result['samples']
                    if row['started_at'] <= s['at'] <= row['finished_at'])
                row['same_input_tokens'] = row.get('usage', {}).get('prompt_tokens') == historical['usage']['input_tokens']
                row['same_output_tokens'] = row.get('usage', {}).get('completion_tokens') == historical['usage']['output_tokens']
                if not row['isolated_gpu_residency']:
                    result['failures'].append('Other GPU residency detected; subsequent probes stopped')
                    break
                save(output / 'result.json', result)
    except Exception as exc:
        result['failures'].append(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        sampler.cancel(); await asyncio.gather(sampler, return_exceptions=True)
        result.update(finished_at=time.time(), passed=len(result['trials']) == len(TRACES) and not result['failures']
            and all(r.get('isolated_gpu_residency') and r.get('same_input_tokens') and r.get('finish_reason')
                    for r in result['trials']))
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ('passed', 'real_model_requests', 'failures')}))
