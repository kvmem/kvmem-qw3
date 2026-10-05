"""Run one frozen native-control pilot on its private, qualified service stack.

The serial controller supplies an already qualified owned engine. The unchanged
pilot owns its original deadline, prompt, initialization, grade and recorder.
Unknown delivery preserves the stack; it is never retried on another device.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import native_control_model_stack as stack
from osworld_linux import archive

STACK_SHA = '0688bfe05178a82fa26c1642a85f6877bcb942c599c8936d43d33d5aaaa8795e'
ADAPTER_SHA = '8e73897bf0324cf3c3307c593492f72fb3ffe118ce4eda7fe828c23ba2d12b41'
TOOLS = {'mcp__computer__observe', 'mcp__computer__act', 'mcp__computer__workflow'}


def terminal_result(result):
    if (type(result.get('uncertain_actions')) is not int or result['uncertain_actions'] != 0
            or result.get('owned_task_stop_unconfirmed') or result.get('owned_task_retired') is not True):
        raise stack.adapter.owned.SafetyHalt('Original task retirement or input delivery is unconfirmed')
    if (result.get('app_status') not in {'completed', 'failed', 'paused', 'stopped'}
            or any(type(result.get(key)) is not bool for key in ('artifact_passed', 'unassisted_passed', 'accepted', 'video_collection_passed'))):
        raise stack.adapter.owned.SafetyHalt('Require the original independent result, including failures')


def task_profile(task):
    expected = dict(thinking=True, model_output_tokens=16384, tool_call_limit=100,
                    execution_mode='verified', model_context_window=524288, device_platform='linux')
    if any(type(task.get(key)) is not type(value) or task.get(key) != value for key, value in expected.items()) or task.get('software_root') is not None:
        raise stack.adapter.owned.SafetyHalt('Actual task differs from the declared local GUI profile')
    used = task.get('tool_calls_used')
    if type(used) is not int or not 0 <= used <= 100:
        raise stack.adapter.owned.SafetyHalt('Require the original cumulative tool budget')


def wire_evidence(inference):
    inference = Path(inference)
    traces = {p.parent for pattern in ('*/request.json', '*/request.archive.json') for p in inference.glob(pattern)}
    if not traces:
        raise stack.adapter.owned.SafetyHalt('No real model requests; do not count service qualification as a model trial')
    evidence = []
    for trace in sorted(traces):
        data = archive.read_request(trace)
        body = json.loads(data)
        names = {tool.get('function', {}).get('name') for tool in body.get('tools', [])}
        if body.get('enable_thinking') is not True or body.get('max_tokens') != 16384 or names != TOOLS:
            raise stack.adapter.owned.SafetyHalt('Actual model wire changes thinking, output limit or GUI tool permissions')
        metrics = trace / 'metrics.json'
        if not metrics.is_file():
            raise stack.adapter.owned.SafetyHalt('A model request lacks a terminal metrics receipt')
        evidence.append(dict(trace=trace.name, request_sha256=archive.digest(data), bytes=len(data),
                             metrics_sha256=stack.adapter.owned.digest(metrics)))
    return evidence


def close_stack(service):
    """Include Codex/MCP descendants created after the initial idle startup."""
    client = service.record['services'].get('client')
    if client and not stack.retired(client['identity']):
        previous = service.record.get('client_descendants', [])
        current = stack.descendants(client['identity'])
        captured = {json.dumps(row, sort_keys=True):row for row in previous + current}
        service.save(client_descendants=list(captured.values()))
    service.close()


def cleanup_after_error(service, original_pilot_invoked, terminal_receipt_confirmed):
    if service is None:
        return False
    if original_pilot_invoked and not terminal_receipt_confirmed:
        # The client may already have died. Its absence must not clear an
        # unconfirmed task/action or destroy the device needed for reconciliation.
        return False
    close_stack(service)
    return True


def run(args):
    if stack.adapter.owned.digest(stack.__file__) != STACK_SHA or stack.adapter.owned.digest(stack.adapter.__file__) != ADAPTER_SHA:
        raise stack.adapter.owned.SafetyHalt('Qualified private stack or frozen adapter changed')
    study, row, framework, original = stack.adapter.validate_study(args.study, args.trial_id)
    engine = json.loads(args.engine_binding.read_text())
    actual = stack.adapter.owned.register_engine(Path(engine['runtime']), original['engine_off_identity'])
    if not stack.adapter.owned.same_process(engine['identity'], actual):
        raise stack.adapter.owned.SafetyHalt('Declared engine instance changed before task preparation')
    output = args.output.resolve()
    if not output.is_relative_to(stack.RUNTIME):
        raise ValueError('Use a fresh runtime owned by the goal')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copyfile(__file__, output / 'trial-source.py')
    result = dict(kind='one_frozen_private_native_control_qw3_trial', status='preparing',
        started_at=time.time(), trial_id=row['id'], arm=row['arm'], case=row['case'], repetition=row['repetition'],
        study_sha256=stack.adapter.owned.digest(args.study), controller_sha256=stack.adapter.owned.digest(__file__),
        model_configuration_or_task_or_grader_changed=False, outcome=None, termination_confirmed=False)
    def save(**values):
        result.update(values)
        stack.adapter.queue.save(output / 'result.json', result)
    service = None
    original_pilot_invoked = terminal_receipt_confirmed = False
    save()
    try:
        service = stack.NativeStack(args.study, row['id'], output / 'stack', args.port_base)
        state = service.start()
        if state.get('model_context_window') != 524288:
            raise stack.adapter.owned.SafetyHalt('Private client did not discover the original actual engine context')
        binding = service.binding(engine)
        binding_path = output / 'binding.json'
        stack.adapter.queue.save(binding_path, binding)
        pilot_output = output / (row['case'] + '-r' + str(row['repetition']))
        bound = stack.adapter.BoundPilot(args.study, row['id'], binding, pilot_output)
        save(status='running-original-pilot', binding_sha256=stack.adapter.owned.digest(binding_path))
        original_pilot_invoked = True
        outcome = bound.run()
        # Preserve this original grade even if later evidence collection fails.
        save(outcome=outcome, original_result_sha256=stack.adapter.owned.digest(pilot_output / 'integration-result.json'))
        terminal_result(outcome)
        terminal_receipt_confirmed = True
        task = json.loads((pilot_output / 'task.json').read_text())
        task_profile(task)
        manifest = json.loads((pilot_output / 'pilot-manifest.json').read_text())
        if (manifest['harness_sha256'] != original['helper_sha256']['platform_pilots.py']
                or manifest['deadline_seconds'] != row['seconds'] or manifest['model_output_tokens'] != 16384
                or manifest.get('no_model_access_to_initializer_or_grader') is not True
                or manifest['container_id'] != service.record['vm']['container_id']):
            raise stack.adapter.owned.SafetyHalt('Original prompt, deadline, material isolation or VM binding changed')
        if row['case'] == 'linux_text' and manifest['prompt'] != stack.adapter.pilot.LINUX_PROMPT:
            raise stack.adapter.owned.SafetyHalt('Original text prompt changed')
        inference = Path(service.client['runtime']) / 'inference'
        traces = wire_evidence(inference)
        close_stack(service)
        compacted = archive.archive_completed(inference)
        if wire_evidence(inference) != traces:
            raise stack.adapter.owned.SafetyHalt('Lossless model archive changed actual wire evidence')
        save(status='completed', termination_confirmed=True, finished_at=time.time(),
            prompt_sha256=hashlib.sha256(manifest['prompt'].encode()).hexdigest(),
            actual_task_sha256=stack.adapter.owned.digest(pilot_output / 'task.json'),
            actual_model_requests=traces, losslessly_archived_requests=compacted,
            stack_sha256=stack.adapter.owned.digest(service.output / 'stack.json'))
    except BaseException as error:
        save(status='halted-no-replay', error_kind=type(error).__name__, error=str(error), finished_at=time.time())
        if service is not None:
            try:
                closed = cleanup_after_error(service, original_pilot_invoked, terminal_receipt_confirmed)
                save(cleanup_confirmed=closed, live_work_preserved=not closed,
                     stack_sha256=stack.adapter.owned.digest(service.output / 'stack.json'))
            except Exception as cleanup_error:
                save(cleanup_error=str(cleanup_error), live_work_preserved=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'engine-binding', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--port-base', type=int, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key:result[key] for key in ('status', 'trial_id', 'outcome', 'termination_confirmed')}, ensure_ascii=False))
