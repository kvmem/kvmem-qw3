"""Private fixtures and saved submission checks for GUI-only browser pilots."""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from osworld_windows.source import ROOT


SERVER = Path(__file__).with_name('browser_pilot_server.js')
PORT = 18770
ANDROID_CONTAINER_ROOT = ROOT / 'apps/computer_use/.runtime/android/container-user'


def device_container(platform):
    if platform not in {'android','linux'}:
        raise ValueError('Unsupported fixture device')
    info=json.loads(subprocess.check_output(['docker','inspect','qw3-computer-'+platform]))[0]
    if info['Config'].get('Labels',{}).get('qw3.goal')!='cross-device-20261003':
        raise ValueError('Fixture device owner changed')
    return info['Id']


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('Fixture identity must be a short simple directory name')
    return value


def expected(identity_value):
    value = identity(identity_value)
    return {'name': 'QW3 Test ' + value, 'email': value.lower() + '@example.invalid',
            'topic': 'python', 'delivery': 'local',
            'notes': 'Prepare Python for the local demo. Keep existing projects.', 'consent': 'yes'}


def instruction(url, fields):
    return (f'In the browser, open {url} and fill the offline support form. Full name: {fields["name"]}. '
            f'Email address: {fields["email"]}. Topic: Python environment. Preferred delivery: Local notification. '
            f'Details: {fields["notes"]} Check the confirmation that these are test details. '
            'Review all six fields, then save the request once and verify the saved confirmation and details. '
            'Use ordinary visible GUI interactions only; do not use a terminal, browser console, scripts or hidden state APIs.')


def grade_records(text, fields):
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (ValueError, TypeError) as exc:
        return {'kind': 'independent-saved-browser-submission', 'passed': False, 'error': type(exc).__name__}
    valid = len(rows) == 1 and isinstance(rows[0], dict) and isinstance(rows[0].get('values'), dict)
    values = rows[0]['values'] if valid else {}
    checks = {key: valid and values.get(key) == value for key, value in fields.items()}
    checks.update(exactly_one_submission=len(rows) == 1, exact_fields=valid and set(values) == set(fields))
    return {'kind': 'independent-saved-browser-submission', 'passed': all(checks.values()),
            'checks': checks, 'submission_count': len(rows), 'submissions': rows,
            'grading_scope': 'Saved ordinary form fields, exactly one final submission; no model access to grader'}


def initialize(platform, identity_value, directory, linux_execute=None):
    value = identity(identity_value)
    source = SERVER.read_bytes()
    fields = expected(value)
    container_id=device_container(platform)
    if platform == 'android':
        target = ANDROID_CONTAINER_ROOT / ('browser-' + value)
        target.mkdir(exist_ok=False)
        (target / 'server.js').write_bytes(source)
        guest = '/emulator-user/' + target.name
        command = ['docker', 'exec', '-d', 'qw3-computer-android', '/node/bin/node', guest + '/server.js', guest, str(PORT)]
        subprocess.run(command, check=True, capture_output=True)
        url = f'http://10.0.2.2:{PORT}/'
    elif platform == 'linux':
        if linux_execute is None:
            raise ValueError('Linux administrative fixture transport is required')
        guest = '/home/user/.qw3-browser-pilots/' + value
        code = ('from pathlib import Path;import base64,subprocess;'
                f'p=Path({guest!r});p.mkdir(parents=True,exist_ok=False);'
                f'(p/"server.js").write_bytes(base64.b64decode({base64.b64encode(source).decode()!r}));'
                'f=(p/"server.log").open("wb");'
                f'subprocess.Popen(["node",str(p/"server.js"),str(p),{str(PORT)!r}],stdout=f,stderr=f,start_new_session=True)')
        linux_execute(code)
        target = None
        url = f'http://127.0.0.1:{PORT}/'
    else:
        raise ValueError('Browser pilots support Linux and Android')
    fixture = {'platform': platform, 'identity': value, 'guest_directory': guest, 'port': PORT, 'url': url,
               'source_sha256': hashlib.sha256(source).hexdigest(), 'fields': fields,
               'administrative_fixture_only': True, 'model_used': False,'container_id':container_id}
    if target:
        fixture['host_directory'] = str(target)
    (Path(directory) / 'browser-fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        ready = read_guest(fixture, 'server-ready.json', linux_execute)
        if ready is not None:
            metadata = json.loads(ready)
            if metadata.get('source') != guest + '/server.js' or metadata.get('port') != PORT:
                raise ValueError('Unexpected fixture readiness identity')
            return fixture, instruction(url, fields)
        time.sleep(.2)
    raise TimeoutError('Local form fixture did not become ready; no browser input sent')


def read_guest(fixture, name, linux_execute=None):
    if name not in {'server-ready.json', 'submissions.jsonl'}:
        raise ValueError('Unsupported fixture artifact')
    if fixture['platform'] == 'android':
        target = Path(fixture['host_directory'])
        if target.parent != ANDROID_CONTAINER_ROOT or target.name != 'browser-' + identity(fixture['identity']):
            raise ValueError('Fixture host path identity changed')
        path = target / name
        return path.read_text() if path.exists() else None
    if linux_execute is None:
        raise ValueError('Linux administrative result transport is required')
    if fixture['guest_directory'] != '/home/user/.qw3-browser-pilots/' + identity(fixture['identity']):
        raise ValueError('Fixture guest path identity changed')
    path = fixture['guest_directory'] + '/' + name
    result = linux_execute('from pathlib import Path;import json;'
                           f'p=Path({path!r});print(json.dumps(p.read_text() if p.exists() else None))')
    return json.loads(result['output'])


def grade(fixture, linux_execute=None):
    return grade_records(read_guest(fixture, 'submissions.jsonl', linux_execute) or '', fixture['fields'])


def stop(fixture, linux_execute=None):
    """Signal only the fixture process whose live cmdline has the recorded path."""
    if device_container(fixture['platform'])!=fixture['container_id']:
        raise ValueError('Fixture container identity changed; no process signalled')
    ready = read_guest(fixture, 'server-ready.json', linux_execute)
    if ready is None:
        raise RuntimeError('Fixture service has no identity receipt; do not stop an unknown process')
    ready = json.loads(ready)
    pid = ready['pid']
    source = fixture['guest_directory'] + '/server.js'
    if type(pid) is not int or pid <= 1 or ready.get('source') != source:
        raise ValueError('Fixture service identity receipt does not match')
    if fixture['platform'] == 'android':
        # Node reads only this live process identity and sends one SIGTERM.
        code = ('const fs=require("fs");const p=' + str(pid) + ';'
                'const raw=fs.readFileSync("/proc/"+p+"/cmdline","utf8").split("\\0").filter(Boolean);'
                'if(raw.length!==4||raw[1]!==' + json.dumps(source) + '||raw[2]!==' + json.dumps(fixture['guest_directory']) +
                '||raw[3]!==' + json.dumps(str(fixture['port'])) + ')throw Error("Fixture process identity changed");'
                'process.kill(p,"SIGTERM");')
        subprocess.run(['docker', 'exec', 'qw3-computer-android', '/node/bin/node', '-e', code], check=True, capture_output=True)
    else:
        code = ('from pathlib import Path;import os,signal;'
                f'p={pid};raw=Path(f"/proc/{{p}}/cmdline").read_bytes().split(b"\\0");'
                'args=[v.decode() for v in raw if v];'
                f'assert len(args)==4 and args[1:]=={[source,fixture["guest_directory"],str(fixture["port"])]!r};'
                'os.kill(p,signal.SIGTERM)')
        linux_execute(code)
    return {'stopped_pid': pid, 'source': source, 'owned_process_identity_checked': True}
