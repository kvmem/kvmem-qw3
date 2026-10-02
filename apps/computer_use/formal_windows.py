"""Host-only formal enrollment around the shared Windows execution pipeline.

Enrollment is written before a new attempt starts. The raw runner keeps its
original transport report; this envelope never relabels an old preflight.
"""
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
import tomllib
from types import SimpleNamespace

from osworld_windows.source import ROOT, SELECTION
from osworld_windows.scenarios import TASKS, validate_task
from osworld_windows.launch import connection_waits

GATES = (
    'client_linux_and_windows_runtime_contracts', 'model_byte_inventory',
    'actual_on_near_1m_compaction', 'actual_off_near_1m_compaction_and_exit',
    'full_extended_wait_application_launcher', 'new_client_clean_snapshot_restore',
    '006_required_original_cv_available', '006_full_adapter_and_application_focus',
    '051_from_scratch_native_process_positive', 'all_ten_task_initialization_grading_and_reset',
    'formal_dispatch_implemented_and_qualified',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Evidence paths must be nonempty strings')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Evidence paths must be explicit workspace-relative paths')
    return ROOT / path


def reference(ref):
    if not isinstance(ref, dict) or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')):
        raise ValueError('Evidence requires a file SHA256')
    path = relative_path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('Preparation evidence changed: ' + ref['path'])
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    pointer = ref.get('pointer', '')
    if not isinstance(pointer, str):
        raise ValueError('Invalid evidence JSON pointer')
    if pointer:
        if not pointer.startswith('/'):
            raise ValueError('Invalid evidence JSON pointer')
        for key in pointer[1:].split('/'):
            key = key.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def validate_matrix(profile):
    attempts = profile.get('attempts', [])
    if not isinstance(attempts, list) or any(not isinstance(a, dict) for a in attempts):
        raise ValueError('Formal attempts must be explicit objects')
    if len(attempts) != 20 or Counter((a.get('task'), a.get('kvmem')) for a in attempts) != Counter(
            (task, mode) for task in TASKS for mode in ('on', 'off')):
        raise ValueError('A workflow round requires all ten original tasks and both KVMem modes, exactly once')
    names = [a.get('run', '') for a in attempts]
    if len(set(names)) != 20 or any(not isinstance(n, str) or not re.fullmatch('[a-zA-Z0-9_-]{8,60}', n) for n in names):
        raise ValueError('Formal attempt IDs must be unique and safe')
    for task in TASKS:
        pair = [a for a in attempts if a['task'] == task]
        for a in pair:
            for key, maximum in [('timeout_seconds', 14400), ('tool_call_limit', 10000)]:
                if type(a.get(key)) is not int or not 1 <= a[key] <= maximum:
                    raise ValueError('Invalid formal task budget')
        if any(pair[0][key] != pair[1][key] for key in ('timeout_seconds', 'tool_call_limit')):
            raise ValueError('The two KVMem modes require identical task budgets')
    if profile.get('visual_history') not in ('pin-all', 'retrievable'):
        raise ValueError('A shared visual-history policy is required')
    connection_waits(profile.get('backend_timeout_seconds'), profile.get('backend_read_timeout_seconds'))
    if profile.get('heartbeat_seconds') != 30 or profile.get('workflow_version', 'V0') not in ('V0', 'V1'):
        raise ValueError('Unsupported workflow or heartbeat policy')


def verify_sources(profile):
    bindings = profile.get('host_source_sha256', {})
    if not bindings or str(Path(__file__).relative_to(ROOT)) not in bindings:
        raise ValueError('Freeze must include the formal enrollment implementation')
    app = ROOT / 'apps/computer_use'
    from osworld_windows.native_run import runner_sources
    required = runner_sources()
    if not {str(p.relative_to(ROOT)) for p in required}.issubset(bindings):
        raise ValueError('Freeze does not cover every source used by the shared pipeline')
    for path, digest in bindings.items():
        if sha(relative_path(path)) != digest:
            raise ValueError('Frozen runner source changed: ' + path)
    if sha(SELECTION) != profile['selection_sha256']:
        raise ValueError('Original task selection changed')
    if sha(relative_path(profile['adaptation_contract_path'])) != profile['adaptation_contract_sha256']:
        raise ValueError('Frozen task interpretation changed')


def check_preparation(profile):
    missing = [key for key in GATES if profile.get('gates', {}).get(key) is not True]
    if missing:
        raise ValueError('Preparation incomplete: ' + ', '.join(missing))
    validate_matrix(profile)
    if profile.get('workflow_version', 'V0') == 'V1':
        base = reference(profile.get('baseline_round', {}))
        candidates = profile.get('optimization_candidates', [])
        if (base.get('formal_run') is not True or base.get('status') != 'complete'
                or base.get('valid_results') != 20 or base.get('profile', {}).get('workflow_version', 'V0') != 'V0'
                or not 1 <= len(candidates) <= 2 or any(not c.get('hypothesis') or not c.get('evidence') for c in candidates)):
            raise ValueError('V1 requires twenty completed V0 cells and one or two evidence-based candidates')
        common = ('selection_sha256', 'model_inventory_sha256', 'engine_binary_sha256', 'snapshot',
            'visual_history', 'backend_timeout_seconds', 'backend_read_timeout_seconds', 'heartbeat_seconds',
            'engine', 'recording', 'adaptation_contract_sha256', 'host_source_sha256')
        if any(base['profile'].get(k) != profile.get(k) for k in common):
            raise ValueError('V1 changed a common model, engine, task, grader, snapshot or host pipeline condition')
        before = {a['task']: (a['timeout_seconds'], a['tool_call_limit']) for a in base['profile']['attempts']}
        if any(before[a['task']] != (a['timeout_seconds'], a['tool_call_limit']) for a in profile['attempts']):
            raise ValueError('V1 changed a task budget')
    verify_sources(profile)
    for label in ('client_qualification', 'snapshot_qualification', 'actual_model_launcher_qualification'):
        result = reference(dict(path=profile[label + '_path'], sha256=profile[label + '_sha256']))
        if result.get('passed') is not True:
            raise ValueError('Native qualification is incomplete: ' + label)
        if label == 'snapshot_qualification' and result.get('snapshot') != profile['snapshot']:
            raise ValueError('Snapshot qualification belongs to another snapshot')
    bundle = relative_path(profile['client_bundle_path'])
    archive = json.loads((bundle / 'archive.json').read_text())
    if (archive['sha256'] != profile['client_archive_sha256'] or sha(Path(archive['path'])) != archive['sha256']
            or sha(bundle / 'manifest.json') != profile['client_manifest_sha256']):
        raise ValueError('Native client archive or installed-source manifest changed')
    readiness = profile.get('task_readiness', {})
    if set(readiness) != set(TASKS):
        raise ValueError('Each original task needs evidence for setup, reset and positive/negative grading')
    for task in TASKS:
        validate_task(task)  # In particular, do not bypass the unresolved 006 adapter.
        refs = readiness[task]
        if set(refs) != {'setup', 'reset', 'correct', 'initial', 'incorrect', 'missing'}:
            raise ValueError('Incomplete preparation evidence for ' + task)
        setup, reset = reference(refs['setup']), reference(refs['reset'])
        if setup.get('passed') is not True or reset.get('passed') is not True:
            raise ValueError('Task setup/reset not qualified: ' + task)
        correct = reference(refs['correct'])
        if task == '043':
            # Its existing working-hours interpretation is an explicitly
            # separate, contradictory diagnostic. Preserve strict=false;
            # qualify the published rubric and structural checks instead.
            positive = correct.get('original_compatible_score') == 1 and correct.get('benchmark_structural_success') is True
        else:
            positive = correct.get('strict_success') is True
        if not positive or any(reference(refs[key]).get('strict_success') is not False
                               for key in ('initial', 'incorrect', 'missing')):
            raise ValueError('Positive/negative grading evidence incomplete: ' + task)
        if task == '051':
            audit = correct.get('process_audit', {})
            if (correct.get('process_audit_status') != 'passed'
                    or audit.get('complete_trajectory_reviewed') is not True
                    or not correct.get('review_decision')
                    or correct.get('original_run_and_grade_modified') is not False):
                raise ValueError('051 requires an actual complete process review, not a synthetic binding control')
    inventory = reference(dict(path=profile['model_inventory_path'], sha256=profile['model_inventory_sha256']))
    if inventory.get('all_files_stable_during_hashing') is not True or inventory['binary_sha256'] != profile['engine_binary_sha256']:
        raise ValueError('Model inventory is unqualified')
    return inventory


def verify_model_stats(inventory):
    for member in inventory['model_assets']:
        path = relative_path(member['path'])
        stat = path.stat()
        if str(path.resolve()) != member['resolved_path'] or (stat.st_size, stat.st_mtime_ns) != (member['bytes'], member['mtime_ns']):
            raise ValueError('Frozen model asset changed: ' + member['path'])


def freeze(candidate, output):
    profile = copy.deepcopy(candidate)
    inventory = check_preparation(profile)
    if candidate.get('formal_run') is not False or candidate.get('v0_frozen') is not False:
        raise ValueError('Freeze requires a candidate, not an already-enrolled or legacy run')
    verify_model_stats(inventory)
    for member in inventory['model_assets']:
        if sha(relative_path(member['path'])) != member['sha256']:
            raise ValueError('Model asset bytes changed: ' + member['path'])
    verify_model_stats(inventory)
    if sha(ROOT / 'build-dflash-integration/qw3') != profile['engine_binary_sha256']:
        raise ValueError('Engine binary changed')
    profile.update(kind='windows_formal_campaign_freeze', status='frozen_workflow_round', formal_run=True, v0_frozen=True,
                   workflow_version=profile.get('workflow_version', 'V0'),
                   frozen_unix_ns=time.time_ns(), enrollment_implementation_sha256=sha(Path(__file__)))
    output = Path(output)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=False)
    return profile


def validate_frozen(profile):
    if profile.get('kind') != 'windows_formal_campaign_freeze' or profile.get('formal_run') is not True or profile.get('v0_frozen') is not True:
        raise ValueError('A complete, pre-existing formal freeze is required')
    if sha(Path(__file__)) != profile['enrollment_implementation_sha256']:
        raise ValueError('Formal enrollment implementation changed after freeze')
    return check_preparation(profile)


def grade_status(raw):
    grade = raw.get('independent_grade', {})
    native = raw.get('native_result', {})
    if (raw.get('pipeline_passed') is not True or raw.get('queue_may_continue') is not True
            or native.get('desktop_stopped') is not True):
        return dict(status=raw.get('status', 'infra_invalid'), valid=False, review_pending=False)
    if native.get('task', {}).get('outcome') == 'agent_failed' and raw.get('inference_failures'):
        return dict(status='backend_failed', valid=False, review_pending=False)
    if type(grade.get('strict_success')) is not bool:
        return dict(status='grading_failed', valid=False, review_pending=False)
    if (raw.get('task') == '051' and grade.get('artifact_checks_passed') is True
            and grade.get('process_audit_status') == 'not_audited'):
        return dict(status='process_review_pending', valid=False, review_pending=True)
    return dict(status=raw['status'], valid=True, review_pending=False)


def verify_execution(profile, attempt, raw, directory):
    if any(raw.get(key) != value for key, value in dict(task=attempt['task'], run=attempt['run'],
            kvmem=attempt['kvmem'], timeout_seconds=attempt['timeout_seconds'],
            tool_call_limit=attempt['tool_call_limit'], visual_history=profile['visual_history'],
            backend_timeout_seconds=profile['backend_timeout_seconds'],
            backend_read_timeout_seconds=profile['backend_read_timeout_seconds']).items()):
        raise ValueError('Execution differs from its predeclared formal cell')
    if raw.get('snapshot', {}).get('name') != profile['snapshot']:
        raise ValueError('Execution used a different snapshot')
    engine = json.loads((directory / 'engine/manifest.json').read_text())
    if (engine['binary_sha256'] != profile['engine_binary_sha256'] or engine['kvmem'] != attempt['kvmem']
            or engine['visual_history'] != profile['visual_history']):
        raise ValueError('Actual engine differs from the frozen engine')
    command = engine['command']
    for key, value in [('--ctx', '1048576'), ('--kv-dtype', 'fp8'), ('--speculative-method', 'dflash2'), ('--prefill-chunk', '2048')]:
        if key not in command or command[command.index(key) + 1] != value:
            raise ValueError('Actual engine parameter changed: ' + key)
    if engine.get('qw3_env', {}).get('QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS') != '7':
        raise ValueError('DFlash draft policy changed')
    if engine['prefix_reuse'] != (attempt['kvmem'] == 'on') or engine['guided_refresh'] != ('both' if attempt['kvmem'] == 'on' else 'off'):
        raise ValueError('Product KVMem/prefix/guided-refresh policy changed')
    if ('--kvmem' in command) != (attempt['kvmem'] == 'on'):
        raise ValueError('Actual KVMem switch differs from the declared switch')
    if attempt['kvmem'] == 'on':
        for key, value in [('--kvmem-block-tokens', '128'), ('--kvmem-budget', '131072'),
                ('--kvmem-gen-budget', '65536'), ('--kvmem-cpu-gb', '48'), ('--kvmem-prefill-budget', '131072')]:
            if key not in command or command[command.index(key) + 1] != value:
                raise ValueError('Actual KVMem budget changed: ' + key)
    config = json.loads((directory / 'launch/manifest.json').read_text())
    if config['bundle_sha256'] != profile['client_archive_sha256']:
        raise ValueError('Execution used another workflow bundle')
    if raw.get('initial_health', {}).get('platform') != 'windows' or raw.get('inference_request_receipts', 0) < 1:
        raise ValueError('No actual native Windows/model request evidence')
    codex_config = tomllib.loads((directory / 'guest/app/codex/config.toml').read_text())
    catalog = json.loads((directory / 'guest/app/codex/models.json').read_text())
    models = [m for m in catalog.get('models', []) if m.get('slug') == codex_config.get('model')]
    if (codex_config.get('model') != 'qw3-computer' or len(models) != 1
            or models[0].get('context_window') != 1048576
            or models[0].get('effective_context_window_percent') != 95
            or codex_config.get('model_catalog_json', '').replace('\\', '/').split('/')[-1] != 'models.json'
            or codex_config.get('model_auto_compact_token_limit') != 950000):
        raise ValueError('Actual native context/compaction configuration changed')
    receipts = {row['backend_trace_tag']: row for line in (directory / 'relay/requests.jsonl').read_text().splitlines()
        if (row := json.loads(line)).get('event') == 'request_terminal' and row.get('backend_trace_tag')}
    requests = list((directory / 'guest/app/inference').glob('*/request.json'))
    if len(requests) != raw['inference_request_receipts']:
        raise ValueError('Missing actual request bodies')
    for path in requests:
        body = json.loads(path.read_text())
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        if receipts.get(body.get('kvmem_trace_tag'), {}).get('canonical_request_sha256') != hashlib.sha256(canonical).hexdigest():
            raise ValueError('Actual model request does not bind its relay receipt')
    for path, digest in raw.get('source_sha256', {}).items():
        if profile['host_source_sha256'].get(path) != digest:
            raise ValueError('Actual runner source is not in the freeze: ' + path)


async def execute(profile, args, *, runner=None, preparation=validate_frozen, binding=verify_execution):
    inventory = preparation(profile)
    if runner is None:
        from osworld_windows.native_run import run as runner
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / 'campaign.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(kind='windows_formal_workflow_round', formal_run=True, status='running',
                  profile=profile, completed_executions=0, valid_results=0, trials=[])

    def save():
        temporary = output / 'result.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.replace(output / 'result.json')

    try:
        save()
        for attempt in profile['attempts']:
            preparation(profile)
            verify_model_stats(inventory)
            directory = output / attempt['run']
            if directory.exists():
                raise ValueError('Cannot enroll an existing attempt retroactively')
            enrollment = dict(kind='formal_attempt_enrollment', formal_run=True, declared_before_execution=True,
                declared_unix_ns=time.time_ns(), workflow_version=profile.get('workflow_version', 'V0'),
                attempt=attempt, freeze_sha256=sha(args.freeze), raw_directory=str(directory))
            context = output / (attempt['run'] + '-enrollment.json')
            with context.open('x') as handle:
                json.dump(enrollment, handle, indent=2)
            entry = dict(attempt, workflow_version=enrollment['workflow_version'], status='running',
                         valid=False, enrollment_path=str(context), enrollment_sha256=sha(context))
            report['trials'].append(entry)
            save()
            options = SimpleNamespace(**vars(args))
            options.output = directory
            options.task, options.kvmem = attempt['task'], attempt['kvmem']
            options.timeout, options.tool_limit = attempt['timeout_seconds'], attempt['tool_call_limit']
            options.bundle = relative_path(profile['client_bundle_path'])
            options.snapshot, options.visual_history = profile['snapshot'], profile['visual_history']
            options.backend_timeout = profile['backend_timeout_seconds']
            options.backend_read_timeout = profile['backend_read_timeout_seconds']
            raw = await runner(options)
            binding(profile, attempt, raw, directory)
            if sha(context) != entry['enrollment_sha256'] or sha(args.freeze) != enrollment['freeze_sha256']:
                raise ValueError('Formal enrollment or freeze changed during execution')
            preparation(profile)
            entry.update(grade_status(raw), raw_result_path=str(directory / 'result.json'),
                raw_result_sha256=sha(directory / 'result.json'), independent_grade=raw.get('independent_grade', {}),
                native_outcome=raw.get('native_result', {}).get('task', {}).get('outcome'))
            report['completed_executions'] += 1
            report['valid_results'] += int(entry['valid'])
            save()
            if raw.get('queue_may_continue') is not True or (not entry['valid'] and not entry['review_pending']):
                report['status'] = 'halted'
                break
        else:
            report['status'] = 'execution_complete_review_pending' if any(t['review_pending'] for t in report['trials']) else 'complete'
    except asyncio.CancelledError:
        report.update(status='interrupted', error='Owned formal queue cancelled; original attempt cannot be reused')
        if report['trials'] and report['trials'][-1]['status'] == 'running':
            report['trials'][-1].update(status='interrupted', valid=False)
        raise
    except Exception as exc:
        report.update(status='halted', error=f'{type(exc).__name__}: {exc}')
        if report['trials'] and report['trials'][-1]['status'] == 'running':
            report['trials'][-1].update(status='infra_invalid', valid=False)
    finally:
        report['unstarted'] = profile['attempts'][len(report['trials']):]
        save()
        lock.close()
    return report


def finalize_reviews(campaign, reviews, output, ffmpeg):
    """Apply bound full-process reviews without changing original execution data."""
    from osworld_windows.review051 import apply
    campaign, output = Path(campaign), Path(output)
    result = copy.deepcopy(json.loads(campaign.read_text()))
    if result.get('formal_run') is not True or result.get('status') not in ('execution_complete_review_pending', 'complete'):
        raise ValueError('Only a completed, enrolled workflow round can be finalized')
    pending = {t['run'] for t in result['trials'] if t.get('review_pending')}
    if set(reviews) != pending:
        raise ValueError('Provide exactly one complete review for each pending attempt')
    validate_frozen(result['profile'])
    output.mkdir(parents=True, exist_ok=False)
    result['original_campaign_path'] = str(campaign.resolve())
    result['original_campaign_sha256'] = sha(campaign)
    for trial in result['trials']:
        if not trial.get('review_pending'):
            continue
        directory = Path(trial['raw_result_path']).parent
        if sha(directory / 'result.json') != trial['raw_result_sha256']:
            raise ValueError('Original execution report changed')
        refs = reviews[trial['run']]
        package = reference(refs['package'])
        decision = reference(refs['decision'])
        package_path = relative_path(refs['package']['path']).parent
        if refs['package'].get('pointer', '') or refs['decision'].get('pointer', ''):
            raise ValueError('Review package and decision must bind complete files')
        if Path(package['run']).resolve() != directory.resolve() or package['preparation_tool_sha256'] != result['profile']['host_source_sha256']['apps/computer_use/osworld_windows/review051.py']:
            raise ValueError('Review package belongs to another attempt or preparer')
        if decision.get('package_sha256') != refs['package']['sha256']:
            raise ValueError('Review decision belongs to another package')
        applied = output / trial['run']
        grade = apply(package_path, relative_path(refs['decision']['path']), applied, ffmpeg)
        raw = json.loads((directory / 'result.json').read_text())
        graded = copy.deepcopy(raw)
        graded['independent_grade'] = grade
        outcome = raw['native_result']['task']['outcome']
        if outcome == 'agent_completed':
            graded['status'] = 'passed' if grade['strict_success'] else 'task_failed'
        updated = grade_status(graded)
        if updated['review_pending'] or not updated['valid']:
            raise ValueError('Process review did not establish a complete valid result')
        trial.update(updated, process_grade_path=str(applied / 'independent-process-grade.json'),
            process_grade_sha256=sha(applied / 'independent-process-grade.json'), independent_grade=grade,
            original_execution_outcome_preserved=True)
    result['valid_results'] = sum(t['valid'] for t in result['trials'])
    result['status'] = 'complete' if result['valid_results'] == 20 else 'incomplete'
    (output / 'reviewed-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'freeze'):
        p = sub.add_parser(command)
        p.add_argument('--candidate', type=Path, required=True)
        if command == 'freeze':
            p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path, required=True)
    p.add_argument('--grader-python', type=Path, default=ROOT / 'third_party/OSWorld-V2/.venv/bin/python')
    p = sub.add_parser('finalize')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        check_preparation(json.loads(args.candidate.read_text()))
        print(json.dumps(dict(ready=True, formal_run=False)))
    elif args.command == 'freeze':
        freeze(json.loads(args.candidate.read_text()), args.output)
        print(json.dumps(dict(frozen=True, path=str(args.output))))
    elif args.command == 'finalize':
        result = finalize_reviews(args.campaign, json.loads(args.reviews.read_text()), args.output, args.ffmpeg)
        print(json.dumps({k:result[k] for k in ('status', 'valid_results')}))
    else:
        result = asyncio.run(execute(json.loads(args.freeze.read_text()), args))
        print(json.dumps({k:result[k] for k in ('status', 'completed_executions', 'valid_results')}))
        raise SystemExit(0 if result['status'] in ('complete', 'execution_complete_review_pending') else 1)
