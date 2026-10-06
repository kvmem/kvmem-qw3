"""Exercise actual Appium expiry in a new, CPU-rendered owned emulator."""
import argparse
import asyncio
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time

import aiohttp
from aiohttp import web

import android_idle_recovery as recovery
import kvmem_osworld_queue as owned
import qualify_android_idle_recovery as cpu
from live_android_recording import LiveAndroidRecording

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'apps/computer_use/.runtime'
BASE = RUNTIME / 'android'
IMAGE = 'sha256:40ad7f7dade5d2e0eda1e6dcc5f041e0a4be377cc6fd404c77e842f3460571b2'
LABEL = 'android-idle-recovery-20261006'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def command(arguments, timeout=30):
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def container_info(identifier):
    template = '{"id":{{json .Id}},"image":{{json .Image}},"name":{{json .Name}},"labels":{{json .Config.Labels}},"state":{{json .State}}}'
    return json.loads(command(['docker', 'inspect', '--format', template, identifier]))


def require_container(identifier, name):
    info = container_info(identifier)
    if (info['id'] != identifier or info['name'] != '/' + name or info['image'] != IMAGE
            or info['labels'].get('qw3.goal') != 'cross-device-20261003'
            or info['labels'].get('qw3.study') != LABEL):
        raise ValueError('Exact owned emulator container changed')
    return info


class TrackedDriver(recovery.IdleRecoveryDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.events = []

    async def create_session_once(self):
        self.events.append({'kind': 'session-post-start', 'at': time.time()})
        result = await super().create_session_once()
        self.events.append({'kind': 'session-post-confirmed', 'session_id': result, 'at': time.time()})
        return result

    async def request(self, method, route, body=None):
        event = {'kind': 'session-command', 'method': method, 'route': route, 'at': time.time()}
        self.events.append(event)
        try:
            result = await super().request(method, route, body)
            event['status'] = 'completed'; return result
        except recovery.AppiumError as error:
            event.update(status='rejected', code=error.code, http_status=error.status)
            raise


async def actual_checks(directory, identifier, name, port, result):
    def adb(*arguments):
        require_container(identifier, name)
        return command(['docker', 'exec', identifier, str(BASE / 'sdk/platform-tools/adb'),
                        '-s', 'emulator-5556', *arguments], timeout=30)

    async def identity():
        info = require_container(identifier, name)
        if not info['state']['Running']:
            raise ValueError('Owned emulator stopped')
        boot = await asyncio.to_thread(adb, 'shell', 'cat', '/proc/sys/kernel/random/boot_id')
        return {'device_id': name, 'serial': 'emulator-5556',
                'server_instance': identifier + ':' + info['state']['StartedAt'], 'device_boot': boot}

    expected = await identity()
    # First boot initializes media defaults asynchronously after boot_completed.
    # Wait for an explicit completed baseline, keeping full setting equality as
    # the acceptance criterion rather than ignoring fields that changed.
    def settings_map(raw):
        return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    print(json.dumps({'phase': 'waiting-same-emulator-first-boot-settings'}), flush=True)
    deadline = time.monotonic() + 120
    previous = None; stable = 0; baseline_samples = []
    while True:
        raw = await asyncio.to_thread(adb, 'shell', 'settings', 'list', 'system')
        settings = settings_map(raw)
        baseline_samples.append({'at': time.time(), 'settings': settings})
        initialized = all(settings.get(key) == '1' for key in ('ringtone_set', 'alarm_alert_set', 'notification_sound_set'))
        stable = stable + 1 if initialized and settings == previous else 0
        if stable >= 2:
            break
        if time.monotonic() >= deadline:
            save(directory / 'first-boot-settings.json', baseline_samples)
            raise TimeoutError('First-boot settings did not settle; preserve the same emulator')
        previous = settings
        await asyncio.sleep(3)
    save(directory / 'first-boot-settings.json', baseline_samples)
    driver = TrackedDriver(f'http://127.0.0.1:{port}', name, 'emulator-5556', command_timeout=5, system_port=18221)
    gateway = runner = recorder = None
    result['expected_identity'] = expected
    await driver.start()
    try:
        gateway = recovery.IdleRecoveryGateway(driver, directory / 'gateway', expected, identity)
        token = secrets.token_hex(24)
        runner = web.AppRunner(recovery.create_app(gateway, token)); await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', port + 1).start()
        recorder = LiveAndroidRecording(['docker', 'exec', identifier, str(BASE / 'sdk/platform-tools/adb'),
                                        '-s', 'emulator-5556'], directory)
        await asyncio.to_thread(recorder.start)
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=150)) as session:
            async def api(route, body=None):
                async with session.request('GET' if body is None else 'POST', f'http://127.0.0.1:{port+1}' + route,
                                           headers={'Authorization': 'Bearer ' + token}, json=body) as response:
                    return response.status, await response.json()

            status, first = await api('/observe'); assert status == 200 and first['paused']
            (directory / 'before.png').write_bytes(base64.b64decode(first['image']))
            save(directory / 'before-observation.json', {k: v for k, v in first.items() if k != 'image'})
            before_settings = await asyncio.to_thread(adb, 'shell', 'settings', 'list', 'system')
            result['initial_session'] = driver.session_id
            print(json.dumps({'phase': 'actual-emulator-ready', 'session_started': True}), flush=True)
            await asyncio.sleep(7)
            # An active gateway must not silently reconnect, even after a
            # genuine Appium idle expiry. This does not send Android input.
            assert (await api('/control', {'command': 'resume'}))[0] == 200
            status, active = await api('/observe')
            save(directory / 'active-expiry-rejection.json', {'http_status': status, 'body': active})
            assert status == 409 and gateway.recovery_rows() == []
            assert len([e for e in driver.events if e['kind'] == 'session-post-start']) == 1
            assert (await api('/control', {'command': 'pause'}))[0] == 200
            status, recovered = await api('/observe'); assert status == 200 and recovered['paused']
            assert recovered['session_id'] != first['session_id'] and recovered['desktop_id'] != first['desktop_id']
            assert driver.session_id != result['initial_session']
            assert recovered['session_recovery']['input_replayed'] is False
            (directory / 'after.png').write_bytes(base64.b64decode(recovered['image']))
            save(directory / 'after-observation.json', {k: v for k, v in recovered.items() if k != 'image'})
            assert len(gateway.recovery_rows()) == 1 and gateway.recovery_rows()[0]['status'] == 'observed'
            after_settings = await asyncio.to_thread(adb, 'shell', 'settings', 'list', 'system')
            save(directory / 'settings.json', {'before': before_settings, 'after': after_settings})
            assert settings_map(before_settings) == settings_map(after_settings)
            assert first['foreground'] == recovered['foreground']
            assert (await api('/control', {'command': 'resume'}))[0] == 200
            stale = {'action_id': 'stale-capture-check-01', 'desktop_id': first['desktop_id'],
                     'observation_id': first['observation_id'], 'action': {'kind': 'key', 'key': 'ANDROID_HOME'}}
            status, rejected = await api('/act', stale)
            save(directory / 'stale-capture-rejection.json', {'http_status': status, 'body': rejected})
            assert status == 409
            assert (await api('/control', {'command': 'pause'}))[0] == 200
            assert gateway.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0] == 0
            # Explicit local receipt fixture; no input was ever dispatched.
            gateway.db.execute("INSERT INTO actions VALUES ('fixture-unsettled','{}','uncertain',NULL)"); gateway.db.commit()
            await asyncio.sleep(7)
            status, unsettled = await api('/observe')
            save(directory / 'unsettled-expiry-rejection.json', {'http_status': status, 'body': unsettled})
            assert status == 409 and len(gateway.recovery_rows()) == 1
            assert len([e for e in driver.events if e['kind'] == 'session-post-start']) == 2
            assert not any(e['kind'] == 'session-command' and e['method'] == 'POST' for e in driver.events)
            result['identity_after'] = await identity()
            result['checks'] = {'actual_idle_expiry_on_read': any(e.get('code') == 'invalid session id' for e in driver.events),
                'active_session_does_not_reconnect': True, 'paused_session_recovered_once': True,
                'gateway_and_appium_sessions_rotated': True, 'fresh_observation_available': True,
                'old_capture_rejected_before_reservation': True, 'zero_input_commands': True,
                'foreground_preserved': True, 'system_settings_preserved': True,
                'synthetic_unsettled_receipt_blocks_real_expiry_recovery': True,
                'same_container_and_boot_preserved': result['identity_after'] == expected}
            assert all(result['checks'].values())
            result['actual_session_posts'] = 2
            print(json.dumps({'phase': 'actual-expiry-checks-passed', 'checks': len(result['checks'])}), flush=True)
    finally:
        if runner:
            await runner.cleanup()
        if recorder:
            result['recording'] = await asyncio.to_thread(recorder.finish)
        if gateway:
            result['recovery_receipts'] = gateway.recovery_rows()
            result['sqlite_action_counts'] = gateway.db.execute('SELECT status,COUNT(*) FROM actions GROUP BY status').fetchall()
            await gateway.close()
        else:
            await driver.close()
        result['appium_trace'] = driver.events
        result['driver_client_closed'] = driver.session.closed
        result['recorder_thread_retired'] = recorder is None or not recorder.worker.is_alive()


def run(args):
    directory = args.output.resolve()
    if not directory.is_relative_to(RUNTIME) or not 19000 <= args.port <= 65000:
        raise ValueError('Require a new goal-owned directory and private port')
    qualification = json.loads(args.cpu.read_text())
    if not qualification.get('passed') or qualification.get('tests_run') != 14:
        raise ValueError('Require complete CPU/HTTP recovery qualification')
    for name, digest in qualification['source_sha256'].items():
        if owned.digest(ROOT / 'apps/computer_use' / name) != digest:
            raise ValueError('Qualified recovery source changed: ' + name)
    import android_idle_recovery_candidate as candidate
    build_path = args.build.resolve(); build = json.loads(build_path.read_text())
    framework = build_path.parent / 'framework'
    if (build.get('kind') != 'full_framework_with_opt_in_android_idle_recovery'
            or build.get('added_files') != ['android_idle_recovery.py'] or build.get('changed_existing_files') != []
            or candidate.inventory(framework) != build['candidate_files_sha256']
            or build['cpu_qualification']['sha256'] != owned.digest(args.cpu)
            or Path(recovery.__file__).resolve() != framework / 'android_idle_recovery.py'):
        raise ValueError('Require the actual qualified private recovery package')
    actual_paths = {'android_idle_recovery.py': recovery.__file__}
    for name, digest in qualification['source_sha256'].items():
        if name.startswith('qw3_computer/'):
            module_name = name[:-3].replace('/', '.')
            module_name = 'qw3_computer' if module_name.endswith('.__init__') else module_name
            actual = Path(importlib.import_module(module_name).__file__).resolve()
            if actual != framework / name or owned.digest(actual) != digest:
                raise ValueError('Actual gateway import differs from the qualified private package: ' + name)
            actual_paths[name] = str(actual)
    if shutil.disk_usage(ROOT).free < 30 * 2**30:
        raise ValueError('Insufficient space for an isolated emulator')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    if available < 20 * 2**20:
        raise ValueError('Insufficient host memory for an isolated emulator')
    for port in (args.port, args.port + 1):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    if command(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE]) != IMAGE:
        raise ValueError('Require the existing fixed Android image; never download')
    directory.mkdir(mode=0o700, exist_ok=False)
    shutil.copyfile(__file__, directory / 'qualifier-source.py')
    name = 'qw3-' + directory.name
    if len(name) > 63:
        raise ValueError('Use a short unique test directory name')
    result = {'kind': 'actual_isolated_android_idle_session_recovery_qualification', 'started_at': time.time(),
              'passed': False, 'producer': owned.process_identity(os.getpid()), 'container_name': name,
              'image': IMAGE, 'cpu_qualification': {'path': str(args.cpu.resolve()), 'sha256': owned.digest(args.cpu)},
              'source_sha256': qualification['source_sha256'] | {Path(__file__).name: owned.digest(__file__)},
              'private_candidate': {'path': str(build_path), 'sha256': owned.digest(build_path)},
              'actual_private_imports': actual_paths,
              'model_requests': 0, 'real_device_inputs': 0, 'original_controllers_modified': False,
              'fault_injection': 'New isolated emulator, Appium newCommandTimeout=5; production default remains 3600.',
              'video_scope': 'Live ADB sampling loop at 1 Hz; not a historical reconstruction or full-frame-rate recording.'}
    identifier = None
    try:
        for folder in ('avd/qw3-local-api35.avd', 'user', 'storage'):
            (directory / folder).mkdir(parents=True)
        seed = BASE / 'user/avd/qw3-local-api35.avd/config.ini'
        config = seed.read_text().replace('disk.dataPartition.path=<temp>', 'disk.dataPartition.path=/avd/qw3-local-api35.avd/userdata-qemu.img')
        (directory / 'avd/qw3-local-api35.avd/config.ini').write_text(config)
        (directory / 'avd/qw3-local-api35.ini').write_text('avd.ini.encoding=UTF-8\npath=/avd/qw3-local-api35.avd\ntarget=android-35\n')
        # This inactive template is distinct from the running container-avd.
        # The installed system image lacks userdata.img; retain that fact and
        # copy the existing seed into a new writable disk instead of guessing
        # an image path, downloading, or copying the live device's disk.
        for filename in ('userdata-qemu.img', 'encryptionkey.img', 'cache.img'):
            shutil.copyfile(seed.parent / filename, directory / 'avd/qw3-local-api35.avd' / filename)
        entry = ROOT / 'apps/computer_use/containers/android/entrypoint.sh'
        shutil.copyfile(entry, directory / 'entrypoint.sh')
        result['initialization_sources'] = {str(path): owned.digest(path) for path in
            (seed, entry, *(seed.parent / name for name in ('userdata-qemu.img', 'encryptionkey.img', 'cache.img')),
             BASE / 'sdk/system-images/android-35/google_apis/x86_64/source.properties')}
        launch = ['docker', 'run', '--pull=never', '--detach', '--name', name,
                  '--label', 'qw3.goal=cross-device-20261003', '--label', 'qw3.study=' + LABEL,
                  '--memory=12g', '--cpus=4', '--shm-size=1g', '--device=/dev/kvm',
                  '--group-add', str(os.stat('/dev/kvm').st_gid),
                  '-p', f'127.0.0.1:{args.port}:18200', '-e', 'QW3_ANDROID_BASE=' + str(BASE)]
        for source, target, mode in [(directory/'avd', '/avd', 'rw'), (directory/'user', '/emulator-user', 'rw'),
              (directory/'storage', '/storage', 'rw'), (directory/'entrypoint.sh', '/qw3-entrypoint.sh', 'ro'),
              (Path('/home/chaidi/.nvm/versions/node/v20.20.2'), '/node', 'ro')]:
            launch += ['-v', f'{source}:{target}:{mode}']
        for folder in ('sdk', 'jdk', 'appium', 'appium-home'):
            launch += ['-v', f'{BASE/folder}:{BASE/folder}:ro']
        launch.append(IMAGE)
        save(directory/'launch.json', {'command': launch, 'cwd': str(ROOT)})
        identifier = command(launch, timeout=60); result['container_id'] = identifier
        save(directory/'manifest.json', result)
        require_container(identifier, name)
        deadline = time.monotonic() + 600
        while True:
            info = require_container(identifier, name)
            if not info['state']['Running']:
                raise RuntimeError('New emulator exited before readiness')
            boot = subprocess.run(['docker','exec',identifier,str(BASE/'sdk/platform-tools/adb'),'-s','emulator-5556',
                                   'shell','getprop','sys.boot_completed'],capture_output=True,text=True,timeout=15)
            if boot.returncode == 0 and boot.stdout.strip() == '1':
                try:
                    import urllib.request
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(f'http://127.0.0.1:{args.port}/status', timeout=3) as response:
                        status = json.loads(response.read())
                    if status['value'].get('ready') is True:
                        break
                except (OSError, ValueError, KeyError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Same emulator readiness observation deadline reached; no restart')
            time.sleep(3)
        result['ready_at'] = time.time()
        asyncio.run(actual_checks(directory, identifier, name, args.port, result))
        result['passed'] = True
    except BaseException as error:
        result.update(error_kind=type(error).__name__, error=str(error)); raise
    finally:
        if identifier:
            result['container_before_retirement'] = require_container(identifier, name)
            logs = subprocess.run(['docker', 'logs', identifier], capture_output=True, text=True, timeout=15)
            save(directory/'container-log.json', {'returncode': logs.returncode, 'stdout': logs.stdout, 'stderr': logs.stderr})
            # One stop of this exact owned instance. Observation after it is
            # bounded and never becomes another stop or new launch.
            command(['docker', 'stop', '--time', '30', identifier], timeout=45)
            retired = require_container(identifier, name)
            result['container_retired'] = retired['state']['Running'] is False
            result['container_terminal_state'] = retired['state']
            if result['container_retired']:
                command(['docker', 'rm', identifier])
                missing = subprocess.run(['docker','container','inspect',identifier],capture_output=True,text=True)
                save(directory/'container-absence-probe.json', {'returncode': missing.returncode, 'stdout': missing.stdout, 'stderr': missing.stderr})
                result['container_absent'] = missing.returncode != 0 and ('No such object' in missing.stderr or 'No such container' in missing.stderr)
        result['finished_at'] = time.time(); save(directory/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19680)
    result = run(parser.parse_args())
    print(json.dumps({k:result.get(k) for k in ('passed','container_absent','actual_session_posts','checks','recording')}))
