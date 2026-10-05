"""Continue three untouched quota trials after a confirmed pre-task port failure.

Reuse the full frozen memory loop and exact engine profiles. Five completed
trials remain referenced unchanged. Wait for the original bind check before
each new launcher; never retry a launched engine or interrupt another owner.
"""
import argparse
import copy
import errno
import inspect
import os
from pathlib import Path
import signal
import socket
import sys
import time

import shared_quota_memory as previous

ROOT, PYTHON, owned, old = previous.ROOT, previous.PYTHON, previous.owned, previous.old
read, save, SafetyHalt = previous.read, owned.save, previous.SafetyHalt
MODULE = 'shared_quota_port_continuation'
HELPERS = (MODULE + '.py', 'tests/test_' + MODULE + '.py')
ERROR = 'SafetyHalt: Fresh engine exited; no model trial dispatched'
PORT_ERROR = 'run_engine.py: error: Requested engine port is already in use; select an unused port'
FAILED_ID = 'quota1-035-kvmem-off-r2'
PREFIX_IDS = ['quota1-035-kvmem-off-r1', 'quota1-035-kvmem-on-r1',
              'quota1-080-kvmem-on-r1', 'quota1-080-kvmem-off-r1', 'quota1-035-kvmem-on-r2']
PORT_WAIT_SECONDS = 120
RETIRED_REGISTRATIONS = {str(ROOT / 'apps/computer_use/.runtime' / name) for name in (
    'shared-quota-successors-20261005/sdk-registration.json',
    'shared-quota-successors-20261005/image-registration.json',
    'shared-quota-successors-20261005/thinking-registration.json',
    'shared-quota-successors-20261005/feedback-registration.json',
    'bounded-continuation-retest-launch-01/registration.json',
    'query-refresh-replay-launch-01/registration.json')}


def failure_scope(state, protocol, log, attempted_paths):
    prefix = state.get('trials', [])
    if (state.get('status') != 'halted-no-further-dispatch' or state.get('error') != ERROR
            or state.get('current_trial') != FAILED_ID or PORT_ERROR not in log or attempted_paths
            or [row.get('id') for row in prefix] != PREFIX_IDS
            or [row['id'] for row in protocol['trials'][:5]] != PREFIX_IDS
            or protocol['trials'][5]['id'] != FAILED_ID or len(protocol['trials']) != 8):
        raise SafetyHalt('Only the exact five-result pre-task port failure may continue')
    for spec, row in zip(protocol['trials'], prefix):
        result = row.get('result', {})
        if (row.get('kvmem') != spec['kvmem'] or row.get('exit_code') != 0
                or not result.get('finished_at') or type(result.get('uncertain_actions')) is not int
                or result['uncertain_actions'] != 0 or result.get('owned_task_stop_unconfirmed')):
            raise SafetyHalt('Completed prefix is not durably terminal and unchanged')
    return copy.deepcopy(prefix)


def wait_for_engine_port(port=18080, *, seconds=PORT_WAIT_SECONDS):
    if port != 18080 or seconds != PORT_WAIT_SECONDS:
        raise SafetyHalt('Keep the original engine port and bounded admission deadline')
    deadline = time.monotonic() + seconds
    attempts = 0
    while True:
        attempts += 1
        try:
            with socket.socket() as probe:
                # Match the original launcher exactly: no socket option changes.
                probe.bind(('127.0.0.1', port))
            return dict(port=port, original_bind_check_passed=True, probes=attempts,
                        other_processes_signalled=0, bounded_wait_seconds=seconds)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise SafetyHalt('Engine port probe failed outside the declared busy-port scope') from error
            if time.monotonic() >= deadline:
                raise SafetyHalt('Engine port remains busy; no launcher or task dispatched') from error
            time.sleep(.2)


def patch_loop(source):
    source = previous.patch_loop(source)
    replacements = (
        ('"status": "waiting-for-parent-forty", "trials": [],',
         '"status": "waiting-for-parent-forty", "trials": copy.deepcopy(completed_prefix),'),
        ('for row in p["trials"]:', 'for row in p["trials"][len(completed_prefix):]:'),
        ('                runtime = args.protocol.parent / "engines" / row["id"]\n',
         '                state["port_admission"] = wait_for_engine_port()\n'
         '                checkpoint()\n'
         '                runtime = args.protocol.parent / "engines" / row["id"]\n'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise SafetyHalt('Frozen port-continuation loop anchor changed')
        source = source.replace(before, after, 1)
    return source


def qualification(path):
    result = read(path)
    if (result.get('kind') != 'shared_quota_port_continuation_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run', 0) < 16
            or result.get('skipped') != 0 or result.get('model_calls') != 0 or result.get('device_inputs') != 0):
        raise SafetyHalt('Require the exact completed port-continuation qualification')
    for name in HELPERS:
        old.image.pinned(ROOT / 'apps/computer_use' / name, result['source_sha256'][name])


def validate(path):
    d = read(path)
    if (d.get('kind') != 'exact_shared_quota_pre_task_port_continuation'
            or d.get('python') != str(PYTHON) or d.get('completed_prefix_replayed') is not False
            or d.get('engine_profiles_changed') is not False or d.get('old_failures_overwritten') is not False
            or d.get('planned_trials') != 8 or d.get('remaining_model_trials') != 3
            or d.get('port_wait_seconds') != PORT_WAIT_SECONDS
            or len(d.get('retired_successors', [])) != 6
            or {item.get('registration') for item in d.get('retired_successors', [])} != RETIRED_REGISTRATIONS):
        raise SafetyHalt('Keep the declared frozen-profile three-trial continuation')
    for record in d['files']:
        old.image.pinned(record['path'], record['sha256'])
    qualification(d['qualification'])
    previous_declaration, canonical = previous.validate(Path(d['previous_declaration']))
    registration = read(d['previous_registration'])
    state = read(registration['queue'])
    if (registration.get('kind') != 'actual_shared_quota_memory_identity'
            or registration['declaration'] != d['previous_declaration']
            or registration['declaration_sha256'] != owned.digest(d['previous_declaration'])
            or registration['identity']['uid'] != os.getuid()
            or registration['identity']['command'] != [str(PYTHON), '-m', previous.MODULE, '--declaration', d['previous_declaration']]
            or owned.process_identity(registration['identity']['pid']) is not None
            or owned.process_identity(state['engine_launcher_pid']) is not None):
        raise SafetyHalt('Original failed controller or launcher remains live or differs')
    protocol = Path(previous_declaration['continued_protocol'])
    failed = protocol.parent / 'attempts' / FAILED_ID
    failed_engine = protocol.parent / 'engines' / FAILED_ID
    prefix = failure_scope(state, previous.validate_protocol(protocol), Path(d['failed_engine_log']).read_text(),
                           [str(item) for item in (failed, failed_engine) if item.exists()])
    if prefix != d['completed_prefix'] or d['profile'] != previous_declaration['profile']:
        raise SafetyHalt('Original matched prefix or exact common profile changed')
    for key in ('child', 'registered_engine'):
        if owned.process_identity(state[key]['pid']) is not None:
            raise SafetyHalt('Previous model runner or engine remains live')
    for item in d['retired_successors']:
        record = read(item['registration'])
        queue = read(record['queue'])
        if (queue.get('pid') != record['identity']['pid'] or queue.get('status') != 'halted-no-further-dispatch'
                or queue.get('trials') != [] or record['identity'].get('uid') != os.getuid()
                or record['identity'].get('cwd') != str(ROOT)
                or owned.process_identity(record['identity']['pid']) is not None):
            raise SafetyHalt('A previous successor dispatched work or remains live')
    clone = Path(d['continued_protocol'])
    if owned.digest(clone) != owned.digest(protocol):
        raise SafetyHalt('Original eight-trial protocol changed')
    previous.validate_protocol(clone)
    for spec, result in zip(read(protocol)['trials'], prefix):
        source = protocol.parent / 'attempts' / spec['id']
        reference = clone.parent / 'attempts' / spec['id']
        if (owned.digest(source / 'result.json') != result['result_sha256']
                or read(source / 'result.json') != result['result']
                or not reference.is_symlink() or reference.resolve() != source.resolve()):
            raise SafetyHalt('Matched prefix must reference the unchanged original evidence')
        owned.assert_no_services(source)
        owned.assert_vm_retired(source)
    for mode in ('off', 'on'):
        if (owned.digest(clone.parent / mode / 'acceptance.json')
                != owned.digest(protocol.parent / mode / 'acceptance.json')):
            raise SafetyHalt('Original task acceptance changed')
    owned.assert_no_services(protocol.parent / 'attempts')
    return d, canonical


def prepare(args):
    original, canonical = previous.validate(args.previous_declaration)
    registration = read(args.previous_registration)
    state = read(registration['queue'])
    protocol = Path(original['continued_protocol'])
    log = Path(registration['queue']).parent / (FAILED_ID + '-engine-launch.log')
    prefix = failure_scope(state, previous.validate_protocol(protocol), log.read_text(),
                           [str(item) for item in (protocol.parent / 'attempts' / FAILED_ID,
                                                  protocol.parent / 'engines' / FAILED_ID) if item.exists()])
    qualification(args.qualification)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    clone = output / 'protocol.json'
    clone.write_bytes(protocol.read_bytes())
    (output / 'attempts').mkdir()
    for row in prefix:
        (output / 'attempts' / row['id']).symlink_to(protocol.parent / 'attempts' / row['id'], target_is_directory=True)
    for mode in ('off', 'on'):
        (output / mode).mkdir()
        (output / mode / 'acceptance.json').write_bytes((protocol.parent / mode / 'acceptance.json').read_bytes())
    successors = [dict(registration=str(item.resolve())) for item in args.retired_registrations]
    files = [args.previous_declaration, args.previous_registration, Path(registration['queue']), protocol, clone,
             args.qualification, log, output / 'off/acceptance.json', output / 'on/acceptance.json']
    files += [protocol.parent / 'attempts' / row['id'] / 'result.json' for row in prefix]
    for item in successors:
        source = Path(item['registration'])
        files += [source, Path(read(source)['queue'])]
    d = dict(kind='exact_shared_quota_pre_task_port_continuation', declared_at=time.time(), python=str(PYTHON),
             previous_declaration=str(args.previous_declaration.resolve()), previous_registration=str(args.previous_registration.resolve()),
             original_followup=original['original_followup'], profile=original['profile'], continued_protocol=str(clone),
             state_dir=str(output / 'queue-runtime-01'), completed_prefix=prefix, planned_trials=8,
             remaining_model_trials=3, retired_successors=successors, failed_engine_log=str(log),
             qualification=str(args.qualification.resolve()), port_wait_seconds=PORT_WAIT_SECONDS,
             completed_prefix_replayed=False, engine_profiles_changed=False, old_failures_overwritten=False,
             files=[dict(path=str(item.resolve()), sha256=owned.digest(item)) for item in dict.fromkeys(files)],
             scope='Keep exact eight-trial protocol and five original results; execute only the remaining three in fresh paths. '
                   'Bounded original bind check before each launcher; no retries of launched engines or foreign signals. '
                   'The historical conflicting listener was not captured; current idle port does not prove TIME_WAIT as its cause.')
    path = output / 'continuation.json'
    save(path, d)
    validate(path)
    return path


def run(args):
    d, canonical = validate(args.declaration)
    if sys.executable != d['python']:
        raise SafetyHalt('Keep the qualified original task interpreter')
    args.protocol, args.state_dir, args.min_free_mib = Path(d['continued_protocol']), Path(d['state_dir']), 56000
    namespace = dict(vars(owned))
    namespace.update(declaration=d, declaration_sha=owned.digest(args.declaration), wrapper_sha=owned.digest(Path(__file__)),
                     old=old, owned=owned, canonical=canonical, copy=copy, completed_prefix=d['completed_prefix'],
                     PYTHON=PYTHON, wait_for_engine_port=wait_for_engine_port,
                     terminal_parent=lambda p, q: old.terminal_parent(p, q, Path(d['original_followup'])))
    def checked(path):
        validate(args.declaration)
        return previous.validate_protocol(path)
    namespace['validate_protocol'] = checked
    exec(compile(patch_loop(inspect.getsource(owned.run)), 'frozen-quota-port-continuation', 'exec'), namespace)
    return namespace['run'](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    for name in ('declaration', 'previous-declaration', 'previous-registration', 'qualification', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--retired-registrations', type=Path, nargs='+')
    parser.add_argument('--wait-seconds', type=int, default=172800)
    parser.add_argument('--resource-wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    if not 60 <= args.wait_seconds <= 172800 or not 60 <= args.resource_wait_seconds <= 86400:
        parser.error('Keep the original finite wait budgets')
    if args.prepare:
        if any(getattr(args, name) is None for name in ('previous_declaration', 'previous_registration', 'qualification', 'output', 'retired_registrations')):
            parser.error('Provide the exact failed producer, qualification and retired successors')
        print(str(prepare(args)))
        return
    if args.declaration is None:
        parser.error('Provide the independent immutable continuation declaration')
    args.declaration = args.declaration.resolve()
    def interrupted(*_):
        raise SafetyHalt('Port continuation interrupted; preserve live work without dispatch')
    signal.signal(signal.SIGTERM, interrupted)
    state = run(args)
    print(state['status'])


if __name__ == '__main__':
    main()
