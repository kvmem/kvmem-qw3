"""Physical Mac tool qualification; no model calls and no personal-file edits.

Runs the existing Desktop and Gateway around the separate Mac driver. Creates
only new files in the explicitly declared QW3 Mac test directory. It retains
all action receipts and real native screenshot samples, including failures.
"""
import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import time

from aiohttp import web

from macos_agent_driver import MacDriver
from macos_gateway_entry import create_mac_app
from qw3_computer.desktop import Desktop, DesktopError, DesktopRejected, DesktopValidationError
from qw3_computer.device_gateway import Gateway
from qw3_computer.device_lease import DeviceLease


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


async def ssh(connection, command, content=None):
    argv = ['ssh', '-F', '/dev/null', '-i', connection['identity_file'], '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile=' + connection['known_hosts'],
            connection['user'] + '@' + connection['host'], command]
    child = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, error = await asyncio.wait_for(child.communicate(content), timeout=30)
    if child.returncode:
        raise RuntimeError('Task-scoped SSH operation failed: ' + error.decode(errors='replace'))
    return out


async def run(args):
    connection = json.loads(args.connection.read_text())
    if 'QW3ComputerUseTest/tasks/' not in connection['remote_task_dir']:
        raise ValueError('Qualification must use the dedicated Mac test directory')
    if args.output.exists():
        raise ValueError('Use a fresh qualification output; existing evidence is immutable')
    args.output.mkdir(parents=True, mode=0o700)
    (args.output / 'frames').mkdir()
    remote_dir = connection['remote_task_dir'] + '/' + args.output.name
    main_file = remote_dir + '/QW3-Mac-20261006-textedit.txt'
    unrelated_file = remote_dir + '/Unrelated-QW3-test.txt'
    text = 'QW3 Mac qualification.\n中文输入与保存测试。\nUnicode: café, λ, 😀.\n'
    report = {'scope': 'Physical Mac desktop tools, not model business success', 'model_calls': 0,
              'checks': [], 'actions': [], 'frames': [], 'started_at': time.time(), 'status': 'preparing'}
    driver = MacDriver(connection['endpoint'], connection['device_id'], connection['native_token_path'],
                       args.output / 'native-dispatches')
    lease = DeviceLease('macos:' + connection['device_id']).acquire()
    gateway = desktop = runner = None
    recording = None
    recording_done = asyncio.Event()

    def check(name, passed, evidence=None):
        report['checks'].append({'name': name, 'passed': bool(passed), 'evidence': evidence})
        save(args.output / 'result.json', report)
        if not passed:
            raise RuntimeError('Mac qualification failed: ' + name)

    async def status():
        return await driver.request('GET', '/status')

    async def record():
        while not recording_done.is_set():
            began = time.monotonic()
            try:
                frame = await driver.request('GET', '/frame')
                png = base64.b64decode(frame.pop('png'), validate=True)
                number = len(report['frames'])
                path = args.output / 'frames' / f'{number:05d}.png'
                path.write_bytes(png)
                report['frames'].append(frame | {'path': str(path.resolve()),
                    'sha256': hashlib.sha256(png).hexdigest(), 'received_at': time.time()})
            except Exception as exc:
                report.setdefault('recording_errors', []).append(type(exc).__name__ + ': ' + str(exc))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(recording_done.wait(), timeout=max(0.01, 1 - (time.monotonic() - began)))

    async def action(value, name, space=None):
        latest = desktop.latest
        request = {'desktop_id': latest['desktop_id'], 'observation_id': latest['observation_id'],
                   'action_id': name, 'action': value}
        if space:
            request['coordinate_space'] = space
        began = time.monotonic()
        result = await desktop.act(request)
        evidence = {k: v for k, v in result.items() if k != 'observation'}
        observation = {k: v for k, v in result['observation'].items() if k != 'image'}
        save(args.output / (name + '.json'), {'request': request, 'result': evidence, 'observation': observation})
        report['actions'].append({'name': name, 'elapsed_seconds': time.monotonic() - began,
                                   'native_receipt': result.get('driver_receipt'), 'status': result['status']})
        return request, result

    async def open_document(path):
        await ssh(connection, '/usr/bin/open -a TextEdit ' + shlex.quote(path))
        for _ in range(12):
            observation = await desktop.observe()
            foreground = observation['foreground']
            if foreground.get('process') == 'com.apple.TextEdit' and foreground.get('title', '').startswith(Path(path).stem):
                return observation
            await asyncio.sleep(0.3)
        raise RuntimeError('Prepared test document did not become foreground')

    try:
        await driver.start()
        initial = await status()
        save(args.output / 'native-initial.json', initial)
        check('native-permissions-and-paused-claim', initial['ax_trusted'] and initial['screen_capture_allowed'] and initial['paused'], initial)
        gateway = Gateway(driver, args.output / 'gateway')
        token = secrets.token_hex(32)
        runner = web.AppRunner(create_mac_app(gateway, token)); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        desktop = Desktop(f'http://127.0.0.1:{port}', token, args.output / 'desktop')
        # Create only new task-owned plaintext files. Never replace an existing document.
        command = '\n'.join(['set -eu', 'umask 077', 'mkdir -p ' + shlex.quote(remote_dir),
            'test ! -e ' + shlex.quote(main_file), 'test ! -e ' + shlex.quote(unrelated_file),
            '/usr/bin/printf %s ' + shlex.quote('Independent focus guard fixture.\n') + ' > ' + shlex.quote(unrelated_file),
            '/bin/cat > ' + shlex.quote(main_file)])
        await ssh(connection, command, b'QW3 Mac initial fixture.\n')
        await desktop.start()
        await open_document(main_file)
        before = await status()
        try:
            await action({'kind': 'key', 'key': 'CTRL+A'}, 'paused-attempt')
        except DesktopError as exc:
            if 'paused' not in str(exc).lower():
                raise
        else:
            raise RuntimeError('Paused input was unexpectedly accepted')
        check('default-pause-sends-zero-input', (await status())['synthetic_actions'] == before['synthetic_actions'])
        recording = asyncio.create_task(record())
        await desktop.control('resume')
        observation = await desktop.observe(); geometry = observation['geometry']
        check('real-screenshot-retina-scale', observation['width'] / geometry['point_width'] == 2
              and observation['height'] / geometry['point_height'] == 2, geometry)
        controls = observation['accessibility']['controls']
        areas = [c for c in controls if c['kind'] == 'AXTextArea' and c.get('control_id')]
        check('native-visible-textarea', bool(areas), {'provider': observation['accessibility'].get('provider'), 'controls': len(controls)})
        area = areas[0]
        # A native recording frame must leave this exact action capture usable.
        await driver.request('GET', '/frame')
        x = round(area['center_x'] * 1000 / observation['width'])
        y = round(area['center_y'] * 1000 / observation['height'])
        _, clicked = await action({'kind': 'click', 'x': x, 'y': y, 'button': 'left', 'clicks': 1},
                                  'normalized-retina-click', 'normalized_1000')
        dispatched = clicked['coordinate_mapping']['dispatched_action']
        pointer = (await status())['pointer']
        expected = {'x': geometry['point_x'] + dispatched['x'] * geometry['point_width'] / geometry['width'],
                    'y': geometry['point_y'] + dispatched['y'] * geometry['point_height'] / geometry['height']}
        check('normalized-pixel-point-click-with-recording', abs(pointer['x'] - expected['x']) <= 1
              and abs(pointer['y'] - expected['y']) <= 1, {'pointer': pointer, 'expected': expected})
        observation = await desktop.observe()
        area = next(c for c in observation['accessibility']['controls'] if c['kind'] == 'AXTextArea' and c.get('control_id'))
        _, controlled = await action({'kind': 'click_control', 'control_id': area['control_id']}, 'native-control-click')
        pointer = (await status())['pointer']; target = controlled['coordinate_mapping']['dispatched_action']
        check('accessibility-center-to-retina-point', abs(pointer['x'] - target['x'] / 2 - geometry['point_x']) <= 1
              and abs(pointer['y'] - target['y'] / 2 - geometry['point_y']) <= 1, pointer)
        before = await status(); await desktop.observe()
        try:
            await action({'kind': 'click_control', 'control_id': area['control_id']}, 'stale-control-attempt')
        except DesktopValidationError:
            pass
        else:
            raise RuntimeError('Stale control was unexpectedly accepted')
        check('stale-control-no-input', (await status())['synthetic_actions'] == before['synthetic_actions'])
        await desktop.observe(); before = await status()
        await ssh(connection, '/usr/bin/open -a TextEdit ' + shlex.quote(unrelated_file))
        await asyncio.sleep(0.4)
        try:
            await action({'kind': 'key', 'key': 'CTRL+A'}, 'focus-changed-attempt')
        except DesktopRejected:
            pass
        else:
            raise RuntimeError('Changed focus was unexpectedly accepted')
        check('focus-change-no-input', (await status())['synthetic_actions'] == before['synthetic_actions'])
        await open_document(unrelated_file); before = await status()
        try:
            await action({'kind': 'type', 'text': 'must not be inserted'}, 'outside-scope-attempt')
        except DesktopRejected:
            pass
        else:
            raise RuntimeError('Out-of-scope window was unexpectedly accepted')
        check('fresh-outside-task-window-no-input', (await status())['synthetic_actions'] == before['synthetic_actions'])
        other = await ssh(connection, '/bin/cat ' + shlex.quote(unrelated_file))
        check('independent-other-file-preserved', other == b'Independent focus guard fixture.\n')
        await open_document(main_file)
        await action({'kind': 'key', 'key': 'CTRL+A'}, 'select-task-document')
        request, typed = await action({'kind': 'type', 'text': text}, 'unicode-type')
        before = await status(); duplicate = await desktop.act(request)
        check('duplicate-receipt-no-second-input', duplicate.get('duplicate') and
              (await status())['synthetic_actions'] == before['synthetic_actions'])
        await action({'kind': 'key', 'key': 'CTRL+S'}, 'save-task-document')
        actual = b''
        for _ in range(10):
            actual = await ssh(connection, '/bin/cat ' + shlex.quote(main_file))
            if actual == text.encode():
                break
            await asyncio.sleep(0.3)
        (args.output / 'saved-document.txt').write_bytes(actual)
        check('independent-utf8-file-save', actual == text.encode(),
              {'expected_sha256': hashlib.sha256(text.encode()).hexdigest(), 'actual_sha256': hashlib.sha256(actual).hexdigest()})
        await action({'kind': 'key', 'key': 'CTRL+W'}, 'close-saved-document')
        observation = await open_document(main_file)
        values = [c.get('value') for c in observation['accessibility']['controls'] if c['kind'] == 'AXTextArea']
        check('reopened-document-content', text in values)
        await desktop.control('pause'); before = await status()
        try:
            await action({'kind': 'type', 'text': 'must not be inserted'}, 'final-paused-attempt')
        except DesktopError as exc:
            if 'paused' not in str(exc).lower():
                raise
        else:
            raise RuntimeError('Final paused input was unexpectedly accepted')
        check('final-pause-no-input', before['paused'] and (await status())['synthetic_actions'] == before['synthetic_actions'])
        check('native-recording-captured', len(report['frames']) >= 2 and not report.get('recording_errors'),
              {'frames': len(report['frames']), 'target_hz': 1, 'method': 'Native sampled screenshots; recording endpoint never changes action capture'})
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        recording_done.set()
        if recording:
            await recording
        if desktop and desktop.session:
            with contextlib.suppress(Exception):
                await desktop.control('pause')
            await desktop.close()
        if runner:
            await runner.cleanup()
        if gateway:
            await gateway.close()
        else:
            await driver.close()
        lease.close()
        report['finished_at'] = time.time()
        report['elapsed_seconds'] = report['finished_at'] - report['started_at']
        report['business_model_success_claimed'] = False
        save(args.output / 'result.json', report)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--connection', type=Path, required=True)
    value.add_argument('--output', type=Path, required=True)
    return value


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
