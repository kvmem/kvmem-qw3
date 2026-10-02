"""Serial, evidence-preserving queue for unscored native integration attempts.

This entry point deliberately rejects formal trials until all ten task adapters
and the complete baseline configuration have been qualified and frozen.
"""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .native_run import DEFAULT_SNAPSHOT, run as run_attempt
from .scenarios import validate_task
from .source import ROOT, sha256
from .capabilities import TASKS as CAPABILITY_TASKS, INPUT_ROOT, input_sources


def validate(plan):
    if plan.get('formal_run') is not False:
        raise ValueError('This integration queue requires formal_run=false; V0 is not frozen')
    attempts = plan.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Queue needs explicit attempts')
    ids = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError('Malformed attempt')
        validate_task(attempt.get('task'))
        name = attempt.get('run', '')
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', name) or name in ids:
            raise ValueError('Invalid or duplicate attempt identity')
        ids.add(name)
        if attempt.get('kvmem') not in {'on', 'off'}:
            raise ValueError('Explicit KVMem policy required')
        for key, maximum in [('timeout_seconds', 14400), ('tool_call_limit', 10000)]:
            if type(attempt.get(key)) is not int or not 1 <= attempt[key] <= maximum:
                raise ValueError('Invalid explicit attempt budget: ' + key)
    if plan.get('visual_history') not in {'pin-all', 'retrievable'}:
        raise ValueError('One common visual history policy must be specified')


async def execute(plan, args, *, runner=run_attempt):
    validate(plan)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / 'queue.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(kind='windows_native_integration_queue', formal_run=False,
                  plan=plan, status='running', completed_attempts=0, trials=[])
    source_paths = list((ROOT / 'apps/computer_use/osworld_windows').glob('*.py'))
    if any(a['task'] in CAPABILITY_TASKS for a in plan['attempts']):
        source_paths += input_sources()
        if (plan.get('public_inputs_manifest_sha256') is not None
                and plan['public_inputs_manifest_sha256'] != sha256(INPUT_ROOT / 'manifest.json')):
            raise ValueError('Queue plan belongs to another public fixture corpus')
    sources = {p: sha256(p) for p in source_paths}
    report['source_sha256'] = {str(p.relative_to(ROOT)): value for p, value in sources.items()}

    def save():
        temporary = output / 'result.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.replace(output / 'result.json')

    try:
        (output / 'plan.json').write_text(json.dumps(plan, indent=2))
        save()
        for attempt in plan['attempts']:
            if any(sha256(path) != digest for path, digest in sources.items()):
                raise ValueError('Runner source changed during the queue; start a new versioned queue')
            entry = dict(attempt, status='running', result_path=str(output / attempt['run'] / 'result.json'))
            report['trials'].append(entry)
            save()
            options = SimpleNamespace(**vars(args))
            options.output = output / attempt['run']
            options.task = attempt['task']
            options.kvmem = attempt['kvmem']
            options.timeout = attempt['timeout_seconds']
            options.tool_limit = attempt['tool_call_limit']
            options.visual_history = plan['visual_history']
            result = await runner(options)
            entry.update(status=result['status'], pipeline_passed=result['pipeline_passed'],
                         queue_may_continue=result['queue_may_continue'],
                         strict_success=result.get('independent_grade', {}).get('strict_success'))
            report['completed_attempts'] += 1
            save()
            if result['queue_may_continue'] is not True:
                report['status'] = 'halted'
                break
        else:
            report['status'] = 'complete'
    except asyncio.CancelledError:
        report.update(status='interrupted', error='Queue cancelled; do not resume the same attempt')
        if report['trials'] and report['trials'][-1]['status'] == 'running':
            report['trials'][-1].update(status='interrupted', queue_may_continue=False)
        raise
    except Exception as exc:
        report.update(status='halted', error=f'{type(exc).__name__}: {exc}')
        if report['trials'] and report['trials'][-1]['status'] == 'running':
            report['trials'][-1].update(status='infra_invalid', queue_may_continue=False)
    finally:
        report['unstarted'] = plan['attempts'][len(report['trials']):]
        save()
        lock.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'apps/computer_use/.runtime/windows-native-client-13')
    parser.add_argument('--snapshot', default=DEFAULT_SNAPSHOT)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--grader-python', type=Path, default=ROOT / 'third_party/OSWorld-V2/.venv/bin/python')
    args = parser.parse_args()
    result = asyncio.run(execute(json.loads(args.plan.read_text()), args))
    print(json.dumps({key: result[key] for key in ('status', 'completed_attempts', 'formal_run')}))
    raise SystemExit(0 if result['status'] == 'complete' else 1)
