"""Two inference-only arms for the actual missing-asterisk field failure.

Keep the historical user task, images, messages, inference settings and output
budget. Change only the tool condition declarations; retain proposed tools as
data and never dispatch them. This is a decision probe, not desktop acceptance.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import aiohttp

from diagnostic_gpu_replay import gpu_sample
from diagnostic_priority_checks import ROOT, SOURCE, digest, save
from kvmem_osworld_queue import process_identity, same_process
from osworld_linux.archive import read_request

TRACE = 'a54abd2bbadb46c988e9d17c353e9912'


async def run(args):
    candidate = args.candidate.resolve()
    build = json.loads((candidate / 'build.json').read_text())
    for name, sha in build['candidate_files_sha256'].items():
        if digest(candidate / 'framework' / name) != sha:
            raise ValueError('Declared private wire candidate changed')
    sys.path.insert(0, str(candidate / 'framework'))
    from qw3_computer.condition_schema_wire import restore_inline_conditions
    from qw3_computer.tool_schema import WORKFLOW_TOOLS
    from qw3_computer.execution import validate_checks
    from qw3_computer.protocol import read_sse
    source = SOURCE / 'attempts/013-recent-observations-r1/app/inference' / TRACE
    original = json.loads(read_request(source))
    repaired = copy.deepcopy(original)
    declaration = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    for tool in repaired['tools']:
        function = tool['function']
        function['parameters'] = restore_inline_conditions(function['name'], function['parameters'], declaration)
    expected_field = None
    for message in reversed(original['messages']):
        if message.get('role') != 'tool': continue
        for part in message.get('content', []):
            if not isinstance(part, dict) or part.get('type') != 'text': continue
            try: body = json.loads(part['text'])
            except (ValueError, KeyError): continue
            if not isinstance(body, dict): continue
            for field in body.get('observation', body).get('accessibility', {}).get('controls', []):
                if field.get('kind') == 'entry' and field.get('name') == 'First Prize Total Amount *':
                    expected_field = field; break
        if expected_field: break
    if not expected_field: raise ValueError('Actual last visible target not found')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'apps/computer_use/.runtime'):
        raise ValueError('Use a private diagnostic runtime')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    engine = json.loads((args.engine / 'manifest.json').read_text())
    identity = process_identity(engine['pid'])
    if not identity or identity['command'] != engine['command']:
        raise ValueError('Engine identity changed')
    result = dict(kind='actual_qw3_field_failure_condition_contract_inference_probe', started_at=time.time(),
        source_trace=str(source), source_request_sha256=hashlib.sha256(read_request(source)).hexdigest(),
        expected_visible_field=expected_field, candidate_build_sha256=digest(candidate / 'build.json'),
        model_settings_and_task_messages_unchanged=True, real_model_requests=0, device_inputs=0,
        model_profile=engine, actual_engine_identity=identity, cases=[], samples=[], passed=False,
        business_success_improvement_proven=False, limitation='One decision per arm; no tool execution or task success evaluation.')
    save(output / 'result.json', result)
    inflight = False
    async def sample_loop():
        while True:
            sample = await asyncio.to_thread(gpu_sample)
            sample['inflight'] = inflight
            sample['other_compute_pids'] = [p['pid'] for p in sample['processes'] if p['pid'] != identity['pid']]
            result['samples'].append(sample);save(output / 'result.json', result)
            await asyncio.sleep(1)
    sampler = asyncio.create_task(sample_loop())
    try:
        await asyncio.sleep(2)
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=600, sock_read=180)) as session:
            for arm, body in (('original-opaque', original), ('complete-conditions', repaired)):
                if any(s['other_compute_pids'] for s in result['samples'][-2:]) or not same_process(identity, process_identity(identity['pid'])):
                    raise RuntimeError('Engine/GPU isolation changed; no next arm dispatched')
                data = json.dumps(body, ensure_ascii=False).encode()
                save(output / (arm + '-request.json'), body)
                row = dict(arm=arm, started_at=time.time(), proposed_calls=[], status='requesting')
                result['cases'].append(row);result['real_model_requests'] += 1
                merged = {};start = time.monotonic();inflight = True
                async with session.post('http://127.0.0.1:18080/v1/chat/completions', data=data,
                    headers={'Content-Type': 'application/json'}) as reply:
                    reply.raise_for_status()
                    with (output / (arm + '-backend.jsonl')).open('x') as log:
                        async for event in read_sse(reply.content):
                            if event == '[DONE]': break
                            chunk = json.loads(event);log.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                            if chunk.get('error'): raise RuntimeError('Backend returned an error')
                            if chunk.get('usage'): row['usage'] = chunk['usage']
                            for choice in chunk.get('choices', []):
                                for call in choice.get('delta', {}).get('tool_calls', []):
                                    c = merged.setdefault(call.get('index', 0), dict(name='', arguments=''))
                                    for k in ('name', 'arguments'): c[k] += call.get('function', {}).get(k, '')
                                if choice.get('finish_reason'): row['finish_reason'] = choice['finish_reason']
                inflight = False
                for c in merged.values():
                    c['decoded_arguments'] = json.loads(c['arguments'])
                    conditions = c['decoded_arguments'].get('step', {}).get('conditions', [])
                    try: validate_checks(conditions);c['condition_fields_valid'] = True
                    except Exception: c['condition_fields_valid'] = False
                    c['field_selectors'] = [dict(condition=check, exact_visible_identity_matches=(
                        check.get('text') == expected_field['name'] and check.get('control_kind', 'entry') == 'entry'
                        if check.get('kind') == 'control_value' else
                        check.get('field_ref') == expected_field.get('field_ref') and bool(expected_field.get('field_ref'))))
                        for check in conditions if check.get('kind') in ('control_value', 'field_value')]
                    row['proposed_calls'].append(c)
                row.update(elapsed_seconds=time.monotonic()-start, finished_at=time.time(), status='finished')
                row['isolated_gpu_residency'] = not any(s['other_compute_pids'] for s in result['samples'] if
                    row['started_at'] <= s['at'] <= row['finished_at'])
                if not row['isolated_gpu_residency']: break
                save(output / 'result.json', result)
    finally:
        sampler.cancel();await asyncio.gather(sampler, return_exceptions=True)
        result.update(finished_at=time.time(), passed=len(result['cases']) == 2 and all(
            c.get('status') == 'finished' and c.get('isolated_gpu_residency') for c in result['cases']))
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'engine', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps(dict(passed=result['passed'], real_model_requests=result['real_model_requests'])))
