"""Shared unscored Windows attempt runner; formal execution requires a later freeze.

Restores the qualified disk, creates fresh external state, records continuously,
and grades final artifacts independently after native execution is stopped.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import aiohttp

from compare_replay import monitor
from windows_suite import command, stop
from .entries import prepare as prepare_entries
from .launch import prepare as prepare_launch
from .scenarios import IMPLEMENTED, SITES, collection, prepare as prepare_task, validate_task
from .sites import Site
from .source import ROOT, sha256

DEFAULT_SNAPSHOT = 'qw3-osworld-pdf-20261002'
SNAPSHOT_QUALIFICATIONS = {
    'qw3-osworld-clean-20261002': 'baseline-clean-qualification-01/roundtrip-result.json',
    DEFAULT_SNAPSHOT: 'pdf-handler-qualification-01/roundtrip-result.json',
}

def snapshot_qualification(app, name):
    if name not in SNAPSHOT_QUALIFICATIONS:
        raise ValueError('Snapshot has no registered restore qualification')
    path = app / '.runtime' / SNAPSHOT_QUALIFICATIONS[name]
    result = json.loads(path.read_text())
    if result.get('passed') is not True or result.get('snapshot', name) != name:
        raise ValueError('Selected snapshot has not passed its restore qualification')
    return dict(path=str(path), sha256=sha256(path), snapshot=name)

def terminal_status(report):
    native = report.get('native_result', {})
    task = native.get('task', {})
    outcome = task.get('outcome')
    grade = report.get('independent_grade', {})
    if report.get('grading_failure'):
        return dict(status='grading_failed', queue_may_continue=False)
    if (report.get('pipeline_passed') is not True or native.get('desktop_stopped') is not True
            or task.get('queue_may_continue') is not True):
        return dict(status='infra_invalid', queue_may_continue=False)
    if outcome in {'paused', 'paused_uncertain', 'interrupted'}:
        return dict(status='interrupted', queue_may_continue=False)
    if grade.get('grading_error') or not grade or grade.get('strict_success') is None:
        return dict(status='grading_failed', queue_may_continue=False)
    if outcome == 'time_limit':
        status = 'timed_out'
    elif outcome == 'tool_budget_exhausted':
        status = 'tool_budget_exhausted'
    elif outcome == 'agent_completed':
        status = 'passed' if grade.get('strict_success') is True else 'task_failed'
    elif outcome == 'agent_failed':
        status = 'task_failed'
    else:
        return dict(status='infra_invalid', queue_may_continue=False)
    return dict(status=status, queue_may_continue=True)


async def collect_browser022(bundle, run_id, share, output, wait_json):
    """Administrative read-only collection after acknowledged desktop stop."""
    archive = json.loads((Path(bundle) / 'archive.json').read_text())
    helper = ROOT / 'apps/computer_use/osworld_windows/collect022_guest.py'
    target = share / (run_id + '-collect022.py')
    shutil.copyfile(helper, target)
    script = r'''$ErrorActionPreference='Stop'
$source='\\host.lan\Data\RUN_ID-collect022.py'
if ((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'HELPER_SHA') {throw 'Collector changed'}
& 'C:\Qw3Computer\clients\CLIENT_ID\python\python.exe' $source --output '\\host.lan\Data\RUN_ID-browser-final.json' *> '\\host.lan\Data\RUN_ID-browser-collector.log'
if ($LASTEXITCODE -ne 0) {throw 'Browser collector failed'}
'''.replace('RUN_ID', run_id).replace('HELPER_SHA', sha256(helper)).replace('CLIENT_ID', archive['sha256'][:12])
    path = output / 'collect-browser.ps1'
    path.write_text(script)
    await command(sys.executable, ROOT / 'apps/computer_use/vm/provision.py', path)
    data = await wait_json(share / (run_id + '-browser-final.json'), 60)
    (output / 'browser-final.json').write_text(json.dumps(data, indent=2))


async def run(args):
    validate_task(args.task)
    if not 1 <= args.timeout <= 14400 or not 1 <= args.tool_limit <= 10000:
        raise ValueError("Invalid attempt budget")
    output = args.output.resolve()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', output.name):
        raise ValueError('Invalid attempt identity')
    output.mkdir(parents=True, exist_ok=False)
    run_id = output.name
    app = ROOT / 'apps/computer_use'
    share = app / '.runtime/windows/shared'
    lock = open('/tmp/qw3-osworld-vm.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = json.loads(await command('docker', 'inspect', 'qw3-computer-windows'))[0]
    if info['Config'].get('Labels', {}).get('com.qw3.computer-use.owner') != 'codex-goal-20260930':
        raise ValueError('Not the task-owned VM')
    token = (share / 'executor-token.txt').read_text().strip()
    env = dict(os.environ, PYTHONPATH=str(app))
    report = dict(kind='osworld_native_end_to_end_preflight', formal_run=False,
                  v0_frozen=False, qw3_backend_selected=True, task=args.task, run=run_id,
                  kvmem=args.kvmem, timeout_seconds=args.timeout, tool_call_limit=args.tool_limit, visual_history=args.visual_history,
                  pipeline_passed=False, task_success_claimed=False, events=[])
    engine = relay = recorder = sampler = reviews = None
    owns_desktop = False
    grading_started = False
    monitoring_stopped = asyncio.Event()
    site_initialized = False

    def event(stage):
        report['events'].append(dict(stage=stage, host_monotonic_ns=time.monotonic_ns(),
                                     host_unix_ns=time.time_ns()))
        (output / 'progress.json').write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    async def launch(name, argv):
        with (output / (name + '-launch.log')).open('wb') as log:
            return await asyncio.create_subprocess_exec(*map(str, argv), cwd=ROOT, env=env,
                                                        stdout=log, stderr=log)

    async def wait_json(path, timeout, process=None, predicate=lambda data: True):
        async with asyncio.timeout(timeout):
            while True:
                if process and process.returncode is not None:
                    raise RuntimeError('Owned process exited: ' + path.name)
                if path.exists():
                    try:
                        value = json.loads(path.read_text(encoding='utf-8-sig'))
                        if predicate(value):
                            return value
                    except ValueError:
                        pass
                await asyncio.sleep(.5)

    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=3)) as client:
        async def get(url, guest=False):
            async with client.get(url, headers={'Authorization': 'Bearer ' + token} if guest else {}) as response:
                response.raise_for_status()
                return await response.json()

        async def ready(url, guest=False, process=None):
            async with asyncio.timeout(240):
                while True:
                    if process and process.returncode is not None:
                        raise RuntimeError('Owned service exited at startup')
                    try:
                        return await get(url, guest)
                    except (aiohttp.ClientError, TimeoutError):
                        await asyncio.sleep(1)
        try:
            report['snapshot_qualification'] = snapshot_qualification(app, args.snapshot)
            health = await get('http://127.0.0.1:18082/health', True)
            if not health['paused']:
                raise ValueError('VM is active; refusing to restore it')
            try:
                await get('http://127.0.0.1:18080/health')
            except aiohttp.ClientConnectionError:
                pass
            else:
                raise ValueError('A model is already running')
            report['source_commit'] = (await command('git', 'rev-parse', 'HEAD')).strip()
            sources = list((app / 'osworld_windows').glob('*.py')) + [
                app / ('vm/' + name) for name in ('run_native_task.ps1', 'record_console.py',
                'model_relay.py', 'snapshot.py', 'archive_share.py', 'client_resources.ps1')]
            sources.append(app / 'run_engine.py')
            report['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
            report['snapshot'] = json.loads((share.parent / 'snapshots' / args.snapshot / 'manifest.json').read_text())
            owns_desktop = True
            event('restoring_clean_base')
            await command('docker', 'stop', '-t', '120', 'qw3-computer-windows')
            event('vm_stopped')
            archived = await command(sys.executable, app / 'vm/archive_share.py', 'shared-archive-' + run_id)
            (output / 'prior-share-archive.json').write_text(archived)
            event('prior_share_archived')
            await command(sys.executable, app / 'vm/snapshot.py', 'restore', '--name', args.snapshot)
            event('disk_snapshot_restored')
            await command('docker', 'start', 'qw3-computer-windows')
            event('vm_started_waiting_for_executor')
            report['initial_health'] = await ready('http://127.0.0.1:18082/health', True)
            if not report['initial_health']['paused'] or (await get('http://127.0.0.1:18082/files', True))['files']:
                raise ValueError('Restored desktop is not empty and paused')
            event('clean_desktop_ready')
            manifest = await asyncio.to_thread(prepare_task, args.task, run_id, output / 'task', args.bundle)
            site_initialized = bool(SITES.get(args.task))
            staged = share / (run_id + '-inputs')
            staged.mkdir(exist_ok=True)
            prepare_entries(args.task, run_id, manifest, output / 'entries', staged)
            combined = output / 'initialize.ps1'
            combined.write_text((output / 'task/initialize.ps1').read_text() + '\n' +
                                (output / 'entries/install.ps1').read_text())
            await command(sys.executable, app / 'vm/provision.py', combined)
            report['initialization'] = await wait_json(share / (run_id + '-setup-result.json'), 120)
            report['application_entries'] = await wait_json(share / (run_id + '-entry-result.json'), 120)
            if not report['initialization']['passed'] or not report['application_entries']['passed']:
                raise ValueError('Native scenario initialization failed')
            event('scenario_initialized')
            event('starting_qw3')
            engine = await launch('engine', [sys.executable, app / 'run_engine.py', '--method', 'dflash2',
                '--kvmem', args.kvmem, '--guided-refresh', 'both', '--visual-history', args.visual_history,
                '--lock-timeout', '30', '--run-dir', output / 'engine'])
            sampler = asyncio.create_task(monitor(engine.pid, output / 'engine-resources.jsonl', monitoring_stopped))
            await ready('http://127.0.0.1:18080/health', process=engine)
            relay = await launch('relay', [sys.executable, app / 'vm/model_relay.py', '--run-dir', output / 'relay'])
            relay_info = await wait_json(output / 'relay/manifest.json', 20, relay)
            await asyncio.sleep(.5)
            if relay.returncode is not None:
                raise RuntimeError('Relay failed to bind')
            shutil.copyfile(output / 'relay/model-token.txt', share / 'native-model-token.txt')
            (share / 'native-model-token.txt').chmod(0o600)
            prepare_launch(args.task, run_id, manifest['instruction'], collection(args.task), args.bundle, staged,
                output / 'launch', timeout=args.timeout, tool_limit=args.tool_limit,
                backend=f"http://{relay_info['bind']}:18084/v1")
            recorder = await launch('recorder', [sys.executable, app / 'vm/record_console.py',
                '--output', output / 'video', '--ffmpeg', args.ffmpeg, '--fps', '10'])
            await wait_json(output / 'video/recording.json', 20, recorder,
                            lambda value: value.get('status') == 'recording')
            if args.task == '035':
                from .reviews035 import serve
                reviews = asyncio.create_task(serve(Site('teamchat', run_id), share / (run_id + '-reviews'),
                    output / 'reviews', args.timeout + 300, formal_run=False, model_used=True))
            event('recording_started_before_task_launch')
            await command(sys.executable, app / 'vm/provision.py', output / 'launch/launch.ps1')
            event('native_task_launched')
            native = await wait_json(share / (run_id + '-result.json'), args.timeout + 300)
            report['native_result'] = native
            if native.get('run') != run_id:
                raise ValueError('Native result identity mismatch')
            event('native_task_terminal')
            export = share / 'native-results' / run_id
            if export.exists():
                await asyncio.to_thread(shutil.copytree, export, output / 'guest')
            metrics = list((output / 'guest/app/inference').glob('*/metrics.json'))
            report['inference_request_receipts'] = len(metrics)
            if not native.get('desktop_stopped'):
                raise RuntimeError('Desktop stop not acknowledged; no final state grading')
            for site in SITES.get(args.task, ()):
                final_state = await asyncio.to_thread(Site(site, run_id).state)
                (output / (site + '-final.json')).write_text(json.dumps(final_state, indent=2))
            if args.task == '022':
                await collect_browser022(args.bundle, run_id, share, output, wait_json)
            event('independent_grading_started')
            grading_started = True
            grader = await launch('grader', [args.grader_python, '-m', 'osworld_windows.grade_run',
                                            '--task', args.task, '--directory', output])
            try:
                async with asyncio.timeout(240):
                    if await grader.wait():
                        raise RuntimeError('Host grader failed; see grader-launch.log')
            finally:
                await stop(grader)
            report['independent_grade'] = json.loads((output / 'independent-grade.json').read_text())
            grading_started = False
            report['task_success_claimed'] = report['independent_grade']['strict_success']
            report['pipeline_passed'] = native.get('infrastructure_ok') is True
            event('independent_grading_complete')
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            if grading_started:
                report['grading_failure'] = report['error']
        finally:
            try:
                if not owns_desktop:
                    raise ValueError('No desktop ownership established; no pause sent')
                async with client.post('http://127.0.0.1:18082/control',
                        headers={'Authorization': 'Bearer ' + token}, json={'command': 'pause'}) as response:
                    response.raise_for_status()
                    report['final_desktop_paused'] = (await response.json()).get('paused') is True
                report['pipeline_passed'] &= report['final_desktop_paused']
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                report['final_pause_error'] = str(exc)
                report['pipeline_passed'] = False
            if reviews:
                if not reviews.done():
                    reviews.cancel()
                try:
                    await reviews
                except asyncio.CancelledError:
                    report['review_delivery_complete'] = False
                except Exception as exc:
                    report['review_error'] = f'{type(exc).__name__}: {exc}'
                    report['pipeline_passed'] = False
                else:
                    report['review_delivery_complete'] = True
            # Stop the recorder cooperatively; its MP4/MKV and capture hashes
            # survive even when the task or native transport fails.
            if recorder and recorder.returncode is None:
                recorder.terminate()
                try:
                    await asyncio.wait_for(recorder.wait(), 90)
                except TimeoutError:
                    recorder.kill(); await recorder.wait()
            if (output / 'video/recording.json').exists():
                report['recording'] = json.loads((output / 'video/recording.json').read_text())
                report['pipeline_passed'] &= bool(report['recording'].get('complete'))
            else:
                report['pipeline_passed'] = False
            await stop(relay)
            await stop(engine)
            monitoring_stopped.set()
            if sampler:
                await sampler
            report['private_site_initialized'] = site_initialized
            event('owned_host_services_stopped')
            report.update(terminal_status(report))
            (output / 'result.json').write_text(json.dumps(report, indent=2))
            lock.close()
    print(json.dumps({k: report[k] for k in ('pipeline_passed', 'task_success_claimed', 'formal_run')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=IMPLEMENTED, required=True)
    parser.add_argument('--tool-limit', type=int, default=300)
    parser.add_argument('--visual-history', choices=('pin-all', 'retrievable'), default='pin-all')
    parser.add_argument('--grader-python', type=Path, default=ROOT / 'third_party/OSWorld-V2/.venv/bin/python')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'apps/computer_use/.runtime/windows-native-client-13')
    parser.add_argument('--snapshot', default=DEFAULT_SNAPSHOT)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--kvmem', choices=('on', 'off'), default='on')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args))['pipeline_passed'] else 1)
