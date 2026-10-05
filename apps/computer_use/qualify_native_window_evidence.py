"""Cross-check fixed Xlib reads against X tools in the owned VM.

This does not take input ownership, activate windows or call the model. Every
sample is retained, including state changes during the read-only observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener

import native_window_evidence as native
from platform_pilots import owned_container
import shared_quota_staged_successors as staged


FIELDS = ('window_id', 'root_id', 'display_name', 'display_size', 'title', 'wm_class',
          'bounds', 'reported_owner', 'focus_window_id', 'focus_ancestry')


def probe_script():
    return native.guest_script().removesuffix('import json\nprint(json.dumps(read_native_window()))\n') + '''
import json,os,re,subprocess
first=read_native_window()
independent={}
if first['status']=='ok':
 def query(arguments):
  return subprocess.check_output(['xdotool']+arguments,text=True,timeout=2).strip()
 legacy_geometry=dict(line.split('=',1) for line in query(['getwindowgeometry','--shell',str(first['window_id'])]).splitlines())
 raw_geometry=subprocess.check_output(['xwininfo','-id',str(first['window_id'])],text=True,timeout=2,
  env=dict(os.environ,LC_ALL='C'))
 def field(label):
  values=re.findall(r'^\\s*'+re.escape(label)+r':\\s*(-?\\d+)\\s*$',raw_geometry,re.MULTILINE)
  assert len(values)==1,'Missing unique xwininfo geometry field'
  return int(values[0])
 independent=dict(window_id=int(query(['getactivewindow'])),
  focus_window_id=int(query(['getwindowfocus','-f'])),
  reported_pid=int(query(['getwindowpid',str(first['window_id'])])),
  bounds=[field(key) for key in ['Absolute upper-left X','Absolute upper-left Y','Width','Height']],
  bounds_provider='xwininfo-absolute-client-origin',
  legacy_xdotool_bounds=[int(legacy_geometry[key]) for key in ['X','Y','WIDTH','HEIGHT']],
  legacy_xdotool_version=query(['version']),raw_xwininfo=raw_geometry)
last=read_native_window()
print(json.dumps(dict(first=first,independent=independent,last=last)))
'''


def check_sample(sample):
    first, last, independent = sample['first'], sample['last'], sample['independent']
    if first.get('status') != 'ok' or last.get('status') != 'ok':
        return dict(passed=False, reason='native_read_unavailable_or_changed')
    if any(first.get(key) != last.get(key) for key in FIELDS):
        return dict(passed=False, reason='window_or_focus_changed_between_sources')
    classes = first.get('wm_class')
    if not isinstance(classes, list) or len(classes) != 2:
        return dict(passed=False, reason='native_class_unavailable')
    foreground = dict(handle=hex(first['window_id']), process=classes[-1], title=first['title'])
    if native.valid_native_window(first, foreground, *first['display_size']) is None:
        return dict(passed=False, reason='native_evidence_not_positive')
    expected = dict(window_id=first['window_id'], focus_window_id=first['focus_window_id'],
                    reported_pid=first['reported_owner']['pid'], bounds=first['bounds'])
    if (independent.get('bounds_provider') != 'xwininfo-absolute-client-origin'
            or {key:independent.get(key) for key in expected} != expected):
        return dict(passed=False, reason='independent_native_tools_mismatch', expected=expected)
    return dict(passed=True, reason='stable_xlib_xwininfo_geometry_and_xdotool_identity_focus_match',
                real_raw_focus=True, input_authorized=False,
                legacy_xdotool_geometry_differs=independent.get('legacy_xdotool_bounds') != first['bounds'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(output, declaration):
    q = staged.RUNTIME
    d = q.read(declaration)
    registration = q.read(d['sdk_registration'])
    q.base.require_identity(registration['identity'])
    sdk = q.read(registration['queue'])
    linux = [row for row in sdk['new_gateways'] if row['catalogue_id'] == 'linux-vm']
    if len(linux) != 1:
        raise RuntimeError('Expected one registered owned Linux gateway')
    q.base.require_identity(linux[0]['process_identity'])
    container = owned_container('qw3-computer-linux', 'qw3.goal', 'cross-device-20261003')
    if container['Id'] != sdk['containers']['linux'] or not container['State']['Running']:
        raise RuntimeError('The registered owned Linux VM changed or stopped')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='readonly_actual_native_window_and_focus_qualification', started_at=time.time(),
                  passed=False, model_requests=0, device_inputs=0, window_activations=0,
                  production_driver_or_policy_changed=False, business_success_proven=False,
                  owned_container_id=container['Id'], gateway_identity=linux[0]['process_identity'],
                  existing_gateway_session_id=linux[0]['session_id'],
                  sources={name:digest(Path(__file__).with_name(name)) for name in
                           ('native_window_evidence.py', 'qualify_native_window_evidence.py')},
                  test_source_sha256=digest(Path(__file__).parent / 'tests/test_native_window_evidence.py'),
                  reference_sources=[
                      'https://github.com/python-xlib/python-xlib/blob/master/Xlib/display.py',
                      'https://github.com/python-xlib/python-xlib/blob/master/Xlib/xobject/drawable.py',
                      'https://github.com/jordansissel/xdotool/blob/main/xdotool.pod',
                      'https://github.com/jordansissel/xdotool/blob/v3.20160805.1/xdo.c'],
                  samples=[])
    opener = build_opener(ProxyHandler({}))
    script = probe_script()
    (output / 'guest-readonly.py').write_text(script)
    report['fixed_guest_script_sha256'] = digest(output / 'guest-readonly.py')

    def save():
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        for index in range(6):
            began = time.time()
            body = json.dumps(dict(command=['python', '-c', script], shell=False, timeout=8)).encode()
            request = Request('http://127.0.0.1:18126/execute', data=body,
                              headers={'Content-Type': 'application/json'})
            try:
                with opener.open(request, timeout=10) as response:
                    value = json.load(response)
                if value.get('returncode', 0) != 0 or value.get('status') == 'error':
                    raise RuntimeError('Fixed read-only native query failed: ' + str(value.get('error', value.get('stderr', ''))))
                sample = json.loads(value['output'])
                sample.update(index=index, started_at=began, seconds=time.time() - began)
                sample['check'] = check_sample(sample)
            except Exception as error:
                sample = dict(index=index, started_at=began, seconds=time.time() - began,
                              error_kind=type(error).__name__, error=str(error),
                              check=dict(passed=False, reason='readonly_probe_failed'))
            report['samples'].append(sample)
            save()
            time.sleep(.1)
        q.base.require_identity(registration['identity'])
        q.base.require_identity(linux[0]['process_identity'])
        report['stable_samples_passed'] = sum(sample['check']['passed'] for sample in report['samples'])
        report['passed'] = report['stable_samples_passed'] >= 3
        report['status'] = 'completed' if report['passed'] else 'qualification_failed'
        report['limitations'] = [
            'Observed current task windows, not an isolated WPS keyboard recovery or GUI task acceptance.',
            'X11 PID properties are reported correlations, not cryptographic ownership guarantees.',
            'No input permission is granted; target semantics/pixels and immediate pre-dispatch state still need qualification.',
            'Two equal reads do not rule out a transient intervening change.']
    finally:
        report['finished_at'] = time.time()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.output.resolve(), args.declaration.resolve())
    print(json.dumps({key:report[key] for key in ('passed', 'stable_samples_passed', 'model_requests', 'device_inputs')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
