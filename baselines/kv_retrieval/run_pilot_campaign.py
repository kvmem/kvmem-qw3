"""Run all frozen four tasks and three named conditions; preserve every attempt."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shared-gpu', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=a.resume)
    saved_plan = a.out / 'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes() != a.plan.read_bytes():
        raise ValueError('campaign plan changed; use a separate campaign directory')
    saved_plan.write_bytes(a.plan.read_bytes())
    plan = json.loads(saved_plan.read_text())
    if not plan.get('frozen'):
        raise ValueError('freeze and validate the plan before running full tasks')
    tasks = json.loads((ROOT / plan['task_manifest']).read_text())['tasks']
    methods = ['infllm', 'native_kvmem', 'native_dense']
    state = {'status': 'starting', 'started': time.time(), 'completed': [],
             'shared_gpu': a.shared_gpu, 'order': 'rotate method order across the four preselected tasks'}
    def save():
        tmp = a.out / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(a.out / 'state.json')
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for task_index, task in enumerate(tasks):
            offset = task_index % len(methods)
            for method in methods[offset:] + methods[:offset]:
                parent = a.out / task['task_id'] / method
                previous = sorted(parent.glob('attempt_*'))
                complete = [path for path in previous if (path / 'result.json').exists()
                    and json.loads((path / 'result.json').read_text()).get('status') == 'completed']
                if complete:
                    state['completed'].append(str(complete[-1]))
                    continue
                if previous and not a.retry_failed:
                    raise RuntimeError(f'inspect previous incomplete attempt before retry: {parent}')
                arm = parent / f'attempt_{len(previous) + 1:02d}'
                state.update(status='waiting_for_spare_gpu_memory', task=task['task_id'], method=method, arm=str(arm))
                save()
                while True:
                    free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'], text=True).strip()) / 1024
                    if free >= plan['methods'][method]['minimum_free_gpu_gib']:
                        break
                    state['free_gpu_gib'] = free
                    state['gpu_owners'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
                    save()
                    time.sleep(10)
                command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'run_pilot_arm.py'), '--plan', str(saved_plan),
                           '--out', str(arm), '--method', method, '--task-index', str(task_index)]
                if a.shared_gpu:
                    command.append('--shared-gpu')
                state['status'] = 'running_arm'
                save()
                parent.mkdir(parents=True, exist_ok=True)
                with (parent / f'attempt_{len(previous) + 1:02d}.controller.log').open('w') as log:
                    child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=log)
                    state['child_pid'] = child.pid
                    save()
                    code = child.wait()
                result = json.loads((arm / 'result.json').read_text()) if (arm / 'result.json').exists() else {}
                if code or result.get('status') != 'completed':
                    raise RuntimeError(f'arm needs inspection: {arm}; exit={code}, status={result.get("status")}')
                state['completed'].append(str(arm))
                save()
                subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'summarize_pilot.py'),
                    '--out', str(a.out / 'analysis'), *[x for path in state['completed'] for x in ('--run', path)]], check=True)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=55)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
