#!/usr/bin/env python3
"""Measure the existing CUDA vision frontend after a supervised GPU queue ends.

Uses the repository's GPU/CPU vision parity executable on two retained images.
This is a standalone frontend diagnostic, not a live-task latency benchmark.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from computer_use_phase_report import fields


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def terminal_unit(unit):
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'Result'], text=True, timeout=10)
    state = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    if state.get('ActiveState') not in ('inactive', 'failed') or state.get('MainPID') != '0':
        raise RuntimeError('GPU queue is not terminal: ' + unit)
    return state


def terminate_owned_group(proc, grace=5):
    """Stop our own session, including workers surviving their parent."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The parent can exit on TERM while a vision worker is still alive.
        # Finish the group even when waiting for the parent succeeded.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def validate_metrics(values):
    required = ('rows', 'dim', 'weights_mib', 'repeats', 'gpu_mean_ms', 'gpu_min_ms',
                'cpu_mean_ms', 'cpu_min_ms', 'partial_hit_ms', 'reordered_hit_ms',
                'mean_abs', 'rmse', 'max_abs', 'cosine')
    for key in required:
        value = values.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid vision metric: ' + key)
    if values['repeats'] != 1 or values['rows'] <= 0 or values['dim'] <= 0:
        raise ValueError('Unexpected vision shape/repetition count')
    if values['cosine'] <= .999 or values['mean_abs'] >= .02:
        raise ValueError('Vision GPU/CPU parity threshold failed')


def main(args):
    root = Path(__file__).resolve().parents[1]
    if not 1 <= args.repetitions <= 20:
        raise ValueError('repetitions must be between 1 and 20')
    state = {'kind': 'standalone_vision_frontend_profile', 'passed': False,
             'started_at': time.time(), 'trials': [], 'prerequisite_unit': args.after_unit,
             'unit_state': terminal_unit(args.after_unit),
             'scope': 'First-image timing includes Python image preprocessing and CUDA vision forward, after frontend/model construction. Append timing has one hit and one new image; reorder timing has two cached hits. No target/draft weights or live desktop actions are present. Timings do not replace live server latency or total application memory.'}
    with contextlib.ExitStack() as stack:
        shared = root.parent / 'qw3-dflash2/benchmark/dflash2/long_running_gpu.lock'
        locks = [Path('/tmp/qw3-computer-gpu.lock')] + ([shared] if shared.exists() else [])
        handles = []
        for path in locks:
            handle = stack.enter_context(path.open('rb' if path == shared else 'a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        users = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10).strip()
        if users:
            raise RuntimeError('GPU has existing compute processes; refusing to start')
        output = args.run_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        def save():
            (output / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
        try:
            state['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True, timeout=10).strip()
            build = (root / args.build_dir).resolve()
            with (output / 'build.log').open('wb') as log:
                subprocess.run(['cmake', '--build', str(build), '--target', 'qw3-vision-gpu-parity', '-j', '2'], cwd=root, stdout=log, stderr=log, check=True, timeout=600)
            binary = build / 'qw3-vision-gpu-parity'
            model = root / 'models/Qwen3.8-27B-BF16'
            index = model / 'model.safetensors.index.json'
            weights = json.loads(index.read_text())['weight_map']
            shards = sorted({name for key, name in weights.items() if key.startswith('model.visual.')})
            if not shards or any(Path(name).name != name for name in shards):
                raise ValueError('Unexpected visual weight index')
            sources = [Path(__file__), root / 'scripts/computer_use_phase_report.py',
                       root / 'tests/vision_gpu_parity.cpp', root / 'src/vision_gpu_frontend.cpp',
                       root / 'src/vision_cpu_frontend.cpp', root / 'scripts/qw3_vision_cpu_worker.py']
            state['identities'] = [identity(p) for p in [binary, build / 'libqw3_core.a', index, model / 'config.json', *[model / s for s in shards], *sources]]
            images = [p.resolve() for p in args.images]
            state['images'] = [identity(p) for p in images]
            if state['images'][0]['sha256'] == state['images'][1]['sha256']:
                raise ValueError('Two different images are required for the new-image case')
            env = {k: v for k, v in os.environ.items() if not k.startswith('QW3_')}
            # Leave CPU_CACHE unset: GPU host spill defaults to zero, whereas
            # the independent CPU reference defaults to a 512 MiB cache and
            # its parity assertions require per-image reuse.
            env.update(QW3_VISION_GPU_CACHE_MIB='512', QW3_VISION_PARITY_REPEATS='1')
            state['qw3_env'] = {k: v for k, v in env.items() if k.startswith('QW3_')}
            state['cache_capacities_bytes'] = {'gpu_features': 512 << 20, 'gpu_host_spill': 0, 'cpu_reference_features': 512 << 20}
            state['command'] = [str(binary), str(model), str(root / '.venv/bin/python'), str(root / 'scripts/qw3_vision_cpu_worker.py'), *map(str, images)]
            save()
            for repetition in range(args.repetitions):
                terminal_unit(args.after_unit)
                if identity(binary) != state['identities'][0] or [identity(p) for p in images] != state['images']:
                    raise RuntimeError('Frozen executable or input image changed')
                log_path = output / f'trial-{repetition}.log'
                trial = {'repetition': repetition, 'passed': False, 'started_at': time.time()}
                state['trials'].append(trial)
                save()
                with log_path.open('wb') as log:
                    proc = subprocess.Popen(state['command'], cwd=root, env=env, stdout=log, stderr=log,
                                            start_new_session=True, pass_fds=tuple(h.fileno() for h in handles))
                    trial['pid'] = proc.pid
                    save()
                    try:
                        trial['exit_code'] = proc.wait(timeout=300)
                    except BaseException:
                        # Own the process group created above, including both
                        # vision workers, on timeout or user interruption.
                        terminate_owned_group(proc)
                        raise
                lines = log_path.read_text().splitlines()
                summaries = [fields(line) for line in lines if line.startswith('rows=')]
                if trial['exit_code'] != 0 or len(summaries) != 1:
                    raise RuntimeError('Vision parity process failed; inspect ' + str(log_path))
                values = summaries[0]
                validate_metrics(values)
                trial.update(passed=True, finished_at=time.time(), metrics=values,
                             log_sha256=identity(log_path)['sha256'])
                save()
            state['passed'] = all(t['passed'] for t in state['trials'])
        except Exception as exc:
            state['error'] = repr(exc)
        finally:
            state['finished_at'] = time.time()
            save()
        print(json.dumps({k: state.get(k) for k in ('passed', 'error', 'trials')}, indent=2))
        return 0 if state['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-unit', required=True, help='Existing GPU queue must be terminal before this diagnostic starts')
    parser.add_argument('--images', type=Path, nargs=2, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=Path('build-dflash-integration'))
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    try:
        raise SystemExit(main(args))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
