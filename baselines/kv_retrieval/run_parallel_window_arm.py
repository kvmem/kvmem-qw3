"""Run one later small-window trial with the same frozen arm harness.

The serial controller will skip this path if completed when it reaches it. If
it arrives earlier, its existing incomplete-attempt guard stops for inspection;
never start a duplicate or overwrite an attempt. This supervisor only schedules
and observes, without modifying the model, task, verifier or frozen scripts.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from compare_agent_windows import write_json, jsonl
from observe_pilot_memory import processes, kb_fields
from observe_window_campaign import process_identity, ClientProgress
from run_pilot_arm import clean_environment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--startup-timeout-s', type=int)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan_file = campaign/'plan_small.json'
    plan = json.loads(plan_file.read_text())
    tasks = json.loads((ROOT/plan['task_manifest']).read_text())['tasks']
    if args.task_index <= 0 or args.task_index >= len(tasks):
        raise ValueError('reserve a later task, not the active first task')
    task = tasks[args.task_index]['task_id']
    serial = json.loads((campaign/'state.json').read_text())
    if serial.get('status') != 'running_arm' or serial.get('task') == task:
        raise ValueError('serial controller must still be running an earlier task')
    serial_child = process_identity(serial.get('child_pid', 0))
    if (not serial_child or serial_child['state'] == 'Z'
            or str(serial.get('arm')) not in serial_child['command']
            or 'run_aligned_infllm_arm.py' not in serial_child['command']):
        raise ValueError('the earlier serial arm is not verified live')
    if args.attempt < 1:
        raise ValueError('attempt must be positive')
    parent=campaign/task/'small'
    previous=sorted(p for p in parent.glob('attempt_*') if p.is_dir())
    if args.attempt > 1:
        expected=parent/f'attempt_{args.attempt-1:02d}'
        if not previous or previous[-1]!=expected:
            raise ValueError('startup retry must follow the latest preserved attempt')
        old=json.loads((expected/'result.json').read_text())
        if (old.get('status')!='failed' or old.get('error')!="TimeoutError('model readiness')"
                or old.get('agent_started') or (expected/'cell/agent_summary.json').exists()
                or jsonl(expected/'server/requests.jsonl')):
            raise ValueError('only a readiness timeout before any agent request can retry here')
    elif previous:
        raise FileExistsError(previous[-1])
    arm = parent/f'attempt_{args.attempt:02d}'
    if arm.exists():
        raise FileExistsError(arm)
    for path in (campaign/'parallel_arms').glob('*.json'):
        earlier = json.loads(path.read_text())
        identity = process_identity(earlier.get('supervisor_pid', 0))
        if identity and identity['state'] != 'Z' and identity['starttime_ticks'] == earlier.get('supervisor_starttime_ticks'):
            raise RuntimeError('another reserved arm is still live; keep at most two model servers')
    identity = process_identity(__import__('os').getpid())
    status_file = campaign/'parallel_arms'/f'{task}.json'
    state = {'status': 'starting', 'task': task, 'arm': str(arm), 'port': args.port,
             'started': time.time(), 'shared_gpu': True,
             'supervisor_pid': identity['pid'], 'supervisor_starttime_ticks': identity['starttime_ticks'],
             'frozen_arm_harness': args.startup_timeout_s is None,
             'frozen_inference_and_task_harness': True,
             'serial_controller_may_require_resume_after_both_finish': True,
             'attempt':args.attempt,'startup_timeout_override_s':args.startup_timeout_s}
    write_json(status_file, state)
    arm.parent.mkdir(parents=True, exist_ok=True)
    runner='run_startup_retry_arm.py' if args.startup_timeout_s is not None else 'run_aligned_infllm_arm.py'
    command = [str(ROOT/'.venv/bin/python'), str(ROOT/runner),
               '--plan', str(plan_file), '--out', str(arm), '--method', 'infllm',
               '--task-index', str(args.task_index), '--port', str(args.port), '--shared-gpu']
    if args.startup_timeout_s is not None:
        if args.attempt==1 or not 240 < args.startup_timeout_s <= 1800:
            raise ValueError('readiness override is limited to a verified startup retry')
        command += ['--startup-timeout-s',str(args.startup_timeout_s)]
    child = None
    reader = ClientProgress()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    try:
        with (arm.parent/f'attempt_{args.attempt:02d}.controller.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT.parents[1], env=clean_environment(), stdout=log, stderr=log)
            state.update(child_pid=child.pid, command=command)
            while child.poll() is None:
                result_file = arm/'result.json'
                if result_file.exists():
                    result = json.loads(result_file.read_text())
                    server = process_identity(result.get('server_pid', 0))
                    server_live = bool(server and server['state'] != 'Z' and str(arm/'server') in server['command'])
                    state.update(status=result['status'], server_pid=result.get('server_pid'), server_process_verified_live=server_live)
                    if server_live:
                        sample = {'at': time.time(), 'arm': str(arm), 'campaign_status': 'parallel_arm',
                                  'host': kb_fields(Path('/proc/meminfo')), 'model_processes': processes(server['pid'])}
                        with (arm/'host_memory_samples.jsonl').open('a') as handle:
                            handle.write(json.dumps(sample)+'\n')
                    reader.update(arm/'client_events.jsonl')
                    rows = jsonl(arm/'server/requests.jsonl')
                    state.update(completed_model_requests=sum(r.get('status')=='completed' for r in rows),
                                 failed_model_requests=sum(r.get('status')!='completed' for r in rows),
                                 unique_tool_calls=len(reader.tool_ids), tool_result_errors=reader.tool_errors,
                                 last_client_event_at=reader.last_observed,
                                 current_response_thinking_characters=reader.thinking_chars)
                state['observed_at'] = time.time()
                write_json(status_file, state)
                time.sleep(10)
            state['exit_code'] = child.returncode
            result_file = arm/'result.json'
            result = json.loads(result_file.read_text()) if result_file.exists() else {}
            state.update(status=result.get('status', 'supervisor_failed'), result=result)
    except BaseException as exc:
        state.update(status='supervisor_failed', error=repr(exc))
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
        server = process_identity(state.get('server_pid', 0))
        state['server_process_verified_live'] = bool(server and server['state'] != 'Z'
            and str(arm/'server') in server['command'])
        write_json(status_file, state)


if __name__ == '__main__':
    main()
