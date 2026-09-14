"""Serial, resumable native-upstream measurements; preserve all attempts."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'formal'


def main():
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_CACHE='/data/huggingface/kvmem_upstream_decode_20260912/hub')
    for key in list(env):
        if key.lower() in ['http_proxy', 'https_proxy', 'all_proxy']:
            env.pop(key)
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [HERE/'run_upstream_flexgen.py', HERE/'run_upstream_infllm.py', HERE/'run_native_reference.py',
               HERE/'upstream_model_manifest.json', HERE/'frozen_configuration.json']}
    freeze = OUT/'frozen.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == frozen, 'Frozen benchmark input changed'
    else:
        freeze.write_text(json.dumps(frozen, indent=2))
    jobs = []
    python = str(HERE/'.venv-upstream/bin/python')
    # Alternate scheme ordering between repeats. All GPU ownership is serial.
    schemes = [('infinigen', 'cpu', 8), ('original', 'cpu', 4), ('original', 'gpu', 1)]
    for rep in range(3):
        for batch in [1, 4]:
            for method, kv, threads in (schemes if rep % 2 == 0 else list(reversed(schemes))):
                name = f'opt_{method}_{kv}_b{batch}_r{rep}'
                result = OUT/f'{name}.json'
                cmd = [python, '-u', str(HERE/'run_upstream_flexgen.py'), '--scheme', method, '--kv', kv,
                       '--batch', str(batch), '--threads', str(threads), '--gen', '128', '--decode-warmup', '32', '--out', str(result)]
                jobs.append((name, cmd, result))
    for method in ['infllm', 'dense']:
        name = f'mistral_{method}'
        result = OUT/f'{name}.json'
        cmd = [python, '-u', str(HERE/'run_upstream_infllm.py'), '--method', method, '--out', str(result),
               '--threads', '2', '--lengths', '8192', '32768', '122880', '--reps', '3', '--tokens', '128']
        if method == 'infllm':
            cmd += ['--sync-global']
        jobs.append((name, cmd, result))
    port = 18331
    for mtp in [0, 4]:
        for method in ['qw3', 'kvmem']:
            name = f'qwen_{method}_mtp{mtp}'
            directory = OUT/name
            cmd = [sys.executable, '-u', str(HERE/'run_native_reference.py'), '--method', method,
                   '--mtp', str(mtp), '--port', str(port), '--out', str(directory), '--reps', '3']
            jobs.append((name, cmd, directory/'result.json'))
            port += 1
    state = {'status': 'running', 'pid': os.getpid(), 'started': time.time(), 'jobs': len(jobs), 'completed': []}
    child = None
    def save():
        (OUT/'status.json').write_text(json.dumps(state, indent=2))
    def stop(sig, _):
        if child is not None and child.poll() is None:
            # The native-reference wrapper owns and cleans up its server on
            # KeyboardInterrupt. Keep the complete child process group owned.
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM, stop)
    save()
    try:
        for name, cmd, result in jobs:
            for p, sha in frozen.items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
            if result.exists():
                existing = json.loads(result.read_text())
                if existing.get('status') == 'complete':
                    state['completed'].append(name)
                    save()
                    continue
                raise RuntimeError(f'Inspect prior failed/incomplete result before retry: {result}')
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                raise RuntimeError('Another GPU workload is active')
            state.update(current=name, command=cmd, updated=time.time())
            save()
            print('START', name, time.time(), flush=True)
            with (OUT/f'{name}.log').open('w') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)
                state['child_pid'] = child.pid
                save()
                code = child.wait(timeout=1800)
            print('END', name, code, time.time(), flush=True)
            if code or not result.exists() or json.loads(result.read_text()).get('status') != 'complete':
                raise RuntimeError(f'Failed native benchmark: {name}, exit={code}')
            state['completed'].append(name)
            save()
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='needs_attention', error=repr(exc), traceback=traceback.format_exc())
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        state['ended'] = time.time()
        save()


if __name__ == '__main__':
    main()
