"""Read original captures, Appium logs, receipts and video independently."""
import argparse
import json
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import time

import kvmem_osworld_queue as owned
import android_idle_recovery_candidate as candidate
import qualify_android_idle_recovery_live as live


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': owned.digest(path)}


def read(path):
    return json.loads(Path(path).read_text())


def require_absent(identifier):
    probe = subprocess.run(['docker', 'container', 'inspect', identifier], capture_output=True, text=True)
    if not (probe.returncode != 0 and identifier in probe.stderr and 'No such container' in probe.stderr):
        raise ValueError('Exact emulator absence was not confirmed')
    return {'returncode': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr}


def run(directory, output):
    import av
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if not directory.is_relative_to(live.RUNTIME) or not output.is_relative_to(live.RUNTIME):
        raise ValueError('Use goal-owned source and fresh audit directories')
    output.mkdir(mode=0o700, exist_ok=False)
    shutil.copyfile(__file__, output / 'auditor-source.py')
    result_path = directory / 'result.json'; result = read(result_path)
    if (result.get('kind') != 'actual_isolated_android_idle_session_recovery_qualification'
            or result.get('passed') is not True or result.get('image') != live.IMAGE
            or result.get('model_requests') != 0 or result.get('real_device_inputs') != 0
            or len(result.get('checks', {})) != 11 or not all(result['checks'].values())):
        raise ValueError('Require all actual isolated recovery checks')
    qualification = read(result['cpu_qualification']['path'])
    if owned.digest(result['cpu_qualification']['path']) != result['cpu_qualification']['sha256'] or not qualification['passed']:
        raise ValueError('CPU qualification changed')
    build_ref = result['private_candidate']; build_path = Path(build_ref['path']); build = read(build_path)
    framework = build_path.parent / 'framework'
    if (owned.digest(build_path) != build_ref['sha256'] or build['added_files'] != ['android_idle_recovery.py']
            or build['changed_existing_files'] != [] or candidate.inventory(framework) != build['candidate_files_sha256']):
        raise ValueError('Full tested private package changed')
    original_path = Path(build['base_build']['path']); original = read(original_path)
    if (owned.digest(original_path) != candidate.BASE_SHA
            or build['candidate_files_sha256'] != original['candidate_files_sha256'] | {
                'android_idle_recovery.py': qualification['source_sha256']['android_idle_recovery.py']}):
        raise ValueError('An existing private framework file changed')
    expected_imports = {'android_idle_recovery.py'} | {name for name in qualification['source_sha256'] if name.startswith('qw3_computer/')}
    if set(result['actual_private_imports']) != expected_imports:
        raise ValueError('Private import proof is incomplete')
    for name, path in result['actual_private_imports'].items():
        if Path(path) != framework / name or owned.digest(path) != qualification['source_sha256'][name]:
            raise ValueError('Actual private import does not match the qualified source')
    source_checks = []
    for name, digest in result['source_sha256'].items():
        if owned.digest(live.ROOT / 'apps/computer_use' / name) != digest:
            raise ValueError('Tested source changed: ' + name)
        archive = directory / 'qualifier-source.py' if name == Path(live.__file__).name else Path(result['cpu_qualification']['path']).parent / 'source' / name
        if owned.digest(archive) != digest:
            raise ValueError('Actual source archive does not match: ' + name)
        source_checks.append(reference(archive))
    manifest = read(directory / 'manifest.json')
    if result['producer'] != manifest['producer'] or result['container_id'] != manifest['container_id']:
        raise ValueError('Actual producer or container identity changed')
    actual = owned.process_identity(result['producer']['pid'])
    if actual is not None and owned.same_process(result['producer'], actual) and actual.get('state') != 'Z':
        raise ValueError('Original producer is still live')
    expected = result['expected_identity']
    if result['identity_after'] != expected or expected['server_instance'] != result['container_id'] + ':' + result['container_before_retirement']['state']['StartedAt']:
        raise ValueError('Android boot or Appium instance changed')
    absence = require_absent(result['container_id'])
    live.save(output / 'container-absence.json', absence)
    launch = read(directory / 'launch.json')['command']
    ports = launch[launch.index('-p') + 1].split(':')
    port = int(ports[1])
    for selected in (port, port + 1):
        try:
            with socket.create_connection(('127.0.0.1', selected), timeout=2):
                raise ValueError('Owned service is still reachable')
        except ConnectionRefusedError:
            pass
    before, after = read(directory / 'before-observation.json'), read(directory / 'after-observation.json')
    for observation, filename in ((before, 'before.png'), (after, 'after.png')):
        if owned.digest(directory / filename) != observation['sha256'] or not observation['paused']:
            raise ValueError('Actual capture hash or paused state changed')
    if (before['session_id'] == after['session_id'] or before['desktop_id'] == after['desktop_id']
            or before['foreground'] != after['foreground'] or after['session_recovery']['input_replayed'] is not False):
        raise ValueError('Recovery capture identity or foreground is invalid')
    settings = read(directory / 'settings.json')
    parse = lambda raw: dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    if parse(settings['before']) != parse(settings['after']):
        raise ValueError('A system setting changed')
    baseline = read(directory / 'first-boot-settings.json')
    if len(baseline) < 3 or not baseline[-1]['settings'] == baseline[-2]['settings'] == baseline[-3]['settings']:
        raise ValueError('First-boot baseline was not stable')
    for key in ('ringtone_set', 'alarm_alert_set', 'notification_sound_set'):
        if baseline[-1]['settings'].get(key) != '1':
            raise ValueError('First-boot default is not initialized')
    for name in ('active-expiry-rejection.json', 'stale-capture-rejection.json', 'unsettled-expiry-rejection.json'):
        if read(directory / name)['http_status'] != 409:
            raise ValueError('Actual recovery rejection missing: ' + name)
    db_path = directory / 'gateway/receipts.sqlite'
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        actions = db.execute('SELECT * FROM actions').fetchall()
        receipts = [dict(zip(('expired_session', 'before_epoch', 'status', 'new_session', 'evidence'), row))
                    for row in db.execute('SELECT * FROM recoveries ORDER BY rowid')]
    if actions != [('fixture-unsettled', '{}', 'uncertain', None)]:
        raise ValueError('Unexpected gateway input reservation; preserve all evidence')
    if receipts != result['recovery_receipts'] or len(receipts) != 1 or receipts[0]['status'] != 'observed':
        raise ValueError('Actual recovery receipt differs')
    evidence = json.loads(receipts[0]['evidence'])
    if evidence['identity'] != expected or evidence['creation_posts'] != 1 or evidence['input_replayed'] is not False:
        raise ValueError('Actual recovery authority or creation count differs')
    trace = result['appium_trace']
    if len([e for e in trace if e['kind'] == 'session-post-start']) != 2 or any(
            e['kind'] == 'session-command' and e['method'] == 'POST' for e in trace):
        raise ValueError('Actual driver input or extra session creation occurred')
    server_commands = re.findall(r'\[HTTP\] --> (GET|POST|DELETE) (\S+)', (directory / 'user/appium.log').read_text())
    posts = [route for method, route in server_commands if method == 'POST']
    if posts != ['/session', '/session']:
        raise ValueError('Actual Appium server logged unexpected POST commands')
    if not any(e.get('code') == 'invalid session id' and e.get('method') == 'GET' and e.get('http_status') == 404 for e in trace):
        raise ValueError('Actual structured idle expiry was not exercised')
    recording = read(directory / 'recording.json')
    if (recording != result['recording'] or recording.get('passed') is not True
            or recording.get('historical_reconstruction') is not False or recording.get('sample_hz') != 1):
        raise ValueError('Original live recording receipt changed')
    video = directory / recording['video_file']
    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (recording['width'], recording['height']):
                raise ValueError('Native video dimensions changed')
            frames.append(float(frame.pts * frame.time_base))
    captures = [json.loads(line) for line in (directory / 'frame-captures.jsonl').read_text().splitlines()]
    if len(frames) != recording['frames'] or len(captures) != len(frames) or len(frames) < 7 or not captures[-1]['last_capture']:
        raise ValueError('Continuous sampling loop or decoded frames are incomplete')
    if not all(left < right for left, right in zip(frames, frames[1:])):
        raise ValueError('Video timestamps are not monotonic')
    if not result['driver_client_closed'] or not result['recorder_thread_retired']:
        raise ValueError('Original local resources did not retire')
    final = {'kind': 'independent_actual_android_idle_session_recovery_audit', 'passed': True,
             'checked_at': time.time(), 'original_result': reference(result_path), 'cpu_qualification': result['cpu_qualification'],
             'source_archives_checked': source_checks, 'producer_retired': True, 'container_absent': True,
             'private_candidate': build_ref, 'actual_private_imports_checked': len(expected_imports),
             'ports_refused': [port, port + 1], 'system_setting_keys_unchanged': len(parse(settings['before'])),
             'gateway_recovery_receipts': len(receipts), 'actual_appium_session_posts': 2,
             'actual_appium_input_posts': 0, 'synthetic_unsettled_receipts': 1,
             'video': reference(video), 'video_receipt': reference(directory / 'recording.json'),
             'decoded_frames': len(frames), 'native_video_dimensions': [recording['width'], recording['height']],
             'model_requests': 0, 'real_device_inputs': 0, 'business_success_gain_proven': False,
             'scope': 'Real isolated Appium idle expiry and read-only recovery. The unsettled receipt is an explicit local fixture; no QW3 task or unknown input was created.'}
    final['auditor_source'] = reference(output / 'auditor-source.py')
    live.save(output / 'result.json', final)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'system_setting_keys_unchanged', 'actual_appium_input_posts', 'decoded_frames', 'container_absent')}))
