"""Kill the dedicated VM executor during a selected text action and verify recovery.

Run only after other VM/model work is terminal and the reviewed executor has
been separately deployed. This probe does not deploy or modify the executor itself.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import aiohttp

from qw3_computer.desktop import Desktop, DesktopError

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'apps/computer_use'


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shared = args.shared.resolve()
    candidate = APP / 'vm/executor.ps1'
    expected_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    result = {'kind': 'real_windows_text_dispatch_process_death', 'passed': False,
              'executor_sha256': expected_hash, 'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'simulated_fixture_reconciliation': True}
    desktop = Desktop(args.desktop, (shared / 'executor-token.txt').read_text().strip(), output / 'desktop')
    started = time.monotonic()
    action_task = None
    setup_number = 0
    notepad_window = 0

    async def setup(script):
        nonlocal setup_number
        setup_number += 1
        with (output / f'setup-{setup_number}.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(sys.executable, str(APP / 'vm/provision.py'), str(script),
                                                          cwd=ROOT, stdout=log, stderr=log)
            assert await process.wait() == 0, 'VM fixture setup failed'

    async def wait_json(name, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        path = shared / name
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    value = json.loads(path.read_text(encoding='utf-8-sig'))
                    if predicate(value):
                        return value
                except (OSError, ValueError):
                    pass
            await asyncio.sleep(.05)
        raise TimeoutError('Fixture report: ' + name)

    async def focus_notepad():
        source = (APP / 'vm/focus_notepad_fixture.ps1').read_text()
        source = source.replace('param([long]$WindowHandle = 0)', f'param([long]$WindowHandle = {notepad_window})')
        tag = 'focus-' + uuid.uuid4().hex
        report_name = tag + '-result.json'
        source = source.replace('input-focus-ready.json', report_name)
        script = output / (tag + '.ps1')
        script.write_text(source)
        await setup(script)
        report = await wait_json(report_name, lambda x: x.get('kind') == 'dedicated_notepad_focus_setup')
        assert report['passed'], 'Notepad focus fixture failed'
        (output / (tag + '-result.json')).write_text(json.dumps(report, indent=2))

    async def clipboard(operation):
        # Unique SMB paths prevent a rapid seed/check sequence from reading the
        # preceding operation or accepting another invocation's result.
        tag = 'clipboard-' + operation + '-' + uuid.uuid4().hex
        request_name, report_name = tag + '-request.json', tag + '-result.json'
        (shared / request_name).write_text(json.dumps({'operation': operation}))
        source = (APP / 'vm/test_clipboard.ps1').read_text()
        source = source.replace('clipboard-check-request.json', request_name).replace('clipboard-check-result.json', report_name)
        script = output / (tag + '.ps1')
        script.write_text(source)
        await setup(script)
        report = await wait_json(report_name, lambda x: x.get('operation') == operation)
        (output / f'clipboard-{operation}.json').write_text(json.dumps(report, indent=2))
        assert report['passed'] and report['executor_sha256'] == expected_hash, 'Clipboard fixture or deployed source mismatch'
        await focus_notepad()
        return report

    async def act(action, action_id=None):
        return await desktop.act({'desktop_id': desktop.latest['desktop_id'],
                                  'observation_id': desktop.latest['observation_id'],
                                  'action_id': action_id or uuid.uuid4().hex, 'action': action})

    async def key(name):
        return await act({'kind': 'key', 'key': name})

    async def click(x, y):
        return await act({'kind': 'click', 'x': x, 'y': y, 'button': 'left', 'clicks': 1})

    async def control(name, kind):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            observation = await desktop.observe()
            matches = [c for c in observation.get('accessibility', {}).get('controls', [])
                       if c['name'] == name and c['kind'] == kind]
            if matches:
                return matches[0]
            await asyncio.sleep(.2)
        raise TimeoutError('Missing visible control: ' + name)

    async def notepad_focus():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            await desktop.observe()
            front = desktop.latest['foreground']
            if front['process'].lower() == 'notepad' and (notepad_window == 0 or int(front['handle']) == notepad_window):
                await click(500, 400)
                return
            await asyncio.sleep(.2)
        raise TimeoutError('The prepared Notepad window did not regain focus')

    try:
        queue = APP / '.runtime/type-native-long-queue-01.json'
        if queue.exists():
            assert json.loads(queue.read_text())['status'] in ('passed', 'failed'), 'Frozen VM queue must be terminal'
        paired = APP / '.runtime/archive-paired-off-followup-01.json'
        if paired.exists():
            assert json.loads(paired.read_text())['status'] in ('passed', 'failed'), 'Paired archive queue must be terminal'
        repaired = APP / '.runtime/long-repair-queue-01.json'
        if repaired.exists():
            assert json.loads(repaired.read_text())['status'] in ('passed', 'failed'), 'Repaired archive queue must be terminal'
        supervised = APP / '.runtime/long-repair-queue-02.json'
        if supervised.exists():
            assert json.loads(supervised.read_text())['status'] in ('passed', 'failed'), 'Supervised archive queue must be terminal'
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=4)) as client:
            for url in ('http://127.0.0.1:18080/health', 'http://127.0.0.1:18083/api/state'):
                try:
                    async with client.get(url):
                        raise RuntimeError('Another model or desktop application is running')
                except aiohttp.ClientConnectionError:
                    pass
        result['clipboard_seed'] = await clipboard('seed')
        await desktop.start()
        await desktop.control('resume')
        await desktop.observe()
        await act({'kind': 'open_app', 'app': 'notepad'})
        await notepad_focus()
        await key('CTRL+N')
        await act({'kind': 'type', 'text': 'BASE-'})
        notepad_window = int(desktop.latest['foreground']['handle'])
        before = await desktop.request('GET', '/health')
        action_id = 'input-death-' + uuid.uuid4().hex
        request_name = action_id + '-request.json'
        ready_name = action_id + '-ready.json'
        result_name = action_id + '-result.json'
        worker_name = action_id + '-worker.ps1'
        receipt_id = desktop.action_receipt_id(action_id)
        (shared / request_name).write_text(json.dumps({'action_id': receipt_id}))
        helper_text = (APP / 'vm/kill_on_dispatch.ps1').read_text()
        for old, new in [('input-kill-request.json', request_name),
                         ('input-kill-ready.json', ready_name),
                         ('input-kill-result.json', result_name),
                         ('input-kill-worker.ps1', worker_name)]:
            helper_text = helper_text.replace(old, new)
        helper = output / worker_name
        helper.write_text(helper_text)
        shutil.copyfile(helper, shared / worker_name)
        await setup(helper)
        ready = await wait_json(ready_name, lambda x: x.get('action_id') == receipt_id)
        assert ready['executor_pid'] == before['process_id'], 'Unexpected executor identity'
        await focus_notepad()
        await notepad_focus()
        arguments = {'desktop_id': desktop.latest['desktop_id'], 'observation_id': desktop.latest['observation_id'],
                     'action_id': action_id, 'action': {'kind': 'type', 'text': 'ONCE'}}
        result['selected_action_id'] = action_id
        result['selected_receipt_id'] = receipt_id
        action_task = asyncio.create_task(desktop.act(arguments))
        killed = await wait_json(result_name, lambda x: x.get('action_id') == receipt_id)
        result['fault'] = killed
        assert killed['passed'] and killed['executor_sha256'] == expected_hash, 'Fault injection did not hit the candidate dispatch window'
        assert killed['status_after_kill'] == 'dispatching' and killed['clipboard_sequence_unchanged']
        try:
            await asyncio.wait_for(action_task, 30)
        except DesktopError as exc:
            result['client_delivery_error'] = str(exc)
        else:
            raise AssertionError('A killed dispatch unexpectedly returned a completed action')
        assert desktop.paused and [a['action_id'] for a in desktop.unresolved()] == [action_id]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                after = await desktop.request('GET', '/health')
                if after['process_id'] != before['process_id']:
                    break
            except DesktopError:
                pass
            await asyncio.sleep(.5)
        else:
            raise TimeoutError('Owned executor did not restart')
        assert after['paused'] and before['desktop_id'] == after['desktop_id']
        result['restart'] = {'before': before, 'after': after}
        receipt = await desktop.request('GET', '/actions/' + receipt_id)
        assert receipt['status'] == 'dispatching'
        result['retained_remote_status'] = receipt['status']
        try:
            await desktop.request('POST', '/act', arguments | {'action_id': receipt_id})
        except DesktopError as exc:
            assert 'uncertain' in str(exc).lower()
            result['remote_duplicate_rejected'] = True
        else:
            raise AssertionError('Executor replayed an unresolved action')
        try:
            await desktop.control('resume')
        except DesktopError:
            assert desktop.paused and desktop.unresolved()
            result['resume_requires_reconciliation'] = True
        else:
            raise AssertionError('Uncertain action was resumed without reconciliation')
        result['clipboard_after_kill'] = await clipboard('check')
        # This controlled fixture acknowledges the known interrupted input;
        # production callers must have the user inspect the current screen.
        await desktop.control('resume', acknowledge_uncertain=True)
        await notepad_focus()
        try:
            await desktop.act(arguments)
        except DesktopError as exc:
            assert 'user_reconciled' in str(exc)
            result['local_duplicate_rejected_after_reconciliation'] = True
        else:
            raise AssertionError('Client reused a reconciled action identifier')
        filename = action_id + '.txt'
        await key('CTRL+S')
        field = await control('File name:', 'Edit')
        await click(field['center_x'], field['center_y'])
        await key('CTRL+A')
        await act({'kind': 'type', 'text': 'C:\\Qw3Computer\\outputs\\' + filename})
        button = await control('Save', 'Button')
        await click(button['center_x'], button['center_y'])
        files = (await desktop.request('GET', '/files'))['files']
        actual = next(f for f in files if f['name'] == filename)
        raw = actual['content'].encode('utf-8')
        assert actual['content'] in ('BASE-', 'BASE-ONCE'), 'Input was duplicated or corrupted'
        assert actual['size'] == len(raw) and actual['sha256'] == hashlib.sha256(raw).hexdigest()
        result['verified_file'] = actual
        result['input_effect_observed'] = actual['content'] == 'BASE-ONCE'
        result['passed'] = True
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        if action_task and not action_task.done():
            action_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, DesktopError):
                await action_task
        if desktop.session:
            with contextlib.suppress(DesktopError):
                await desktop.control('pause')
        await desktop.close()
        result['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result.get(k) for k in ('passed', 'elapsed_seconds', 'error')}), flush=True)
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--shared', type=Path, default=APP / '.runtime/windows/shared')
    parser.add_argument('--desktop', default='http://127.0.0.1:18082')
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
