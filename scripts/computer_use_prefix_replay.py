#!/usr/bin/env python3
"""Replay a retained request sequence to diagnose guided KVMem prefix misses.

No desktop tools are dispatched. Recorded images and requests are immutable;
generated responses can differ, so reproduction must be observed separately
from successful request execution. Run only after the named GPU queue ends.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'apps/computer_use'
sys.path.insert(0, str(APP))
import aiohttp
from compare_replay import replay
from computer_use_vision_profile import terminal_unit
from windows_suite import stop


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


async def wait_ready(proc, endpoint):
    deadline = time.monotonic() + 240
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=3)) as client:
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError('Owned engine exited during startup')
            try:
                async with client.get(endpoint + '/health') as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, TimeoutError):
                await asyncio.sleep(1)
    raise TimeoutError('Owned engine startup deadline')


async def run(args):
    unit = terminal_unit(args.after_unit)
    manifest = json.loads(args.manifest.read_text())
    requests = manifest['requests']
    if not requests or requests[-1]['trace_tag'] != manifest['terminal_request_tag']:
        raise ValueError('Incomplete retained sequence manifest')
    for entry in requests:
        path = Path(entry['path'])
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Retained input changed: ' + str(path))
    users = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                    text=True, timeout=10).strip()
    if users:
        raise RuntimeError('GPU has existing compute processes; refusing to start')
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    endpoint = 'http://127.0.0.1:18080'
    state = dict(kind='retained_guided_prefix_sequence_replay', status='starting',
                 started_at=time.time(), requests=[], after_unit=args.after_unit,
                 prerequisite_state=unit, manifest=str(args.manifest.resolve()),
                 manifest_sha256=digest(args.manifest), script_sha256=digest(Path(__file__)),
                 tools_dispatched=False, reproduced_original_checkpoint_miss=False,
                 scope='Fixed recorded request sequence with KVMem prefix reuse and guided refresh. No VM or tool actions. Successful execution is not a live-task quality result or proof that original generated responses/checkpoint misses repeat.')
    def save():
        (output / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    proc = None
    try:
        env = dict(os.environ, PYTHONPATH=str(APP),
                   PATH='/home/chaidi/.local/bin:' + os.environ.get('PATH', '/usr/bin:/bin'))
        command = [sys.executable, str(APP / 'run_engine.py'), '--kvmem', 'on', '--method', 'dflash2',
                   '--guided-refresh', 'both', '--visual-history', 'retrievable', '--trace-retrieval',
                   '--lock-timeout', '0', '--run-dir', str(output / 'engine')]
        state['command'] = command
        save()
        with (output / 'launch.log').open('wb') as log:
            proc = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env, stdout=log, stderr=log)
        state['engine_launcher_pid'] = proc.pid
        save()
        await wait_ready(proc, endpoint)
        state['engine'] = json.loads((output / 'engine/manifest.json').read_text())
        original = manifest['engine_manifest']
        state['same_binary_as_retained'] = state['engine']['binary_sha256'] == original['binary_sha256']
        state['source_binary_sha256'] = original['binary_sha256']
        state['same_engine_controls_as_retained'] = state['engine']['command'][1:] == original['command'][1:]
        def comparison_env(engine):
            return {k: v for k, v in engine['qw3_env'].items() if k != 'QW3_KVMEM_DUMP_SCORES'}
        state['same_engine_environment_as_retained'] = comparison_env(state['engine']) == comparison_env(original)
        if not state['same_engine_controls_as_retained'] or not state['same_engine_environment_as_retained']:
            raise RuntimeError('Engine configuration differs from the retained experiment')
        state['status'] = 'running'
        save()
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=600)) as client:
            for number, entry in enumerate(requests):
                source = Path(entry['path'])
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError('Input changed after preflight')
                body = json.loads(raw)
                if body.get('kvmem_trace_tag') != entry['trace_tag']:
                    raise ValueError('Request identity differs from retained manifest')
                destination = output / f'{number:03d}'
                destination.mkdir()
                log_path = output / 'engine/engine.log'
                offset = log_path.stat().st_size
                result = await replay(client, endpoint, body, proc.pid, destination)
                with log_path.open('rb') as log:
                    log.seek(offset)
                    segment = log.read().decode()
                (destination / 'engine-segment.log').write_text(segment)
                item = dict(number=number, source=entry, **result,
                            checkpoint_misses=re.findall(r'prefix-cache MISS \(why\):.*', segment),
                            captures=re.findall(r'prefix-cache CAPTURE.*', segment),
                            ttft_ms=[float(v) for v in re.findall(r'response_ttft_ms=([\d.]+)', segment)])
                item['matches_retained_prompt_tokens'] = result.get('usage', {}).get('prompt_tokens') == entry['usage']['input_tokens']
                state['requests'].append(item)
                save()
                print(json.dumps({k: item.get(k) for k in ('number', 'status', 'elapsed_seconds',
                                                         'matches_retained_prompt_tokens', 'checkpoint_misses')}), flush=True)
                if result['status'] != 'completed' or not item['matches_retained_prompt_tokens']:
                    raise RuntimeError('Replay did not complete with the retained prompt token count')
        last = state['requests'][-1]
        state['reproduced_original_checkpoint_miss'] = any('reason=no_reusable_checkpoint' in line for line in last['checkpoint_misses'])
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        await stop(proc)
        state['finished_at'] = time.time()
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-unit', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
