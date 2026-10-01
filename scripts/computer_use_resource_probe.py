#!/usr/bin/env python3
"""Qualify Windows process-identity sampling in the idle, owned test VM.

This setup-only probe tests synthetic PID reuse and real Windows process birth
times. It does not run a model task or qualify application RAM measurements.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

from computer_use_vision_profile import terminal_unit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'apps/computer_use/.runtime'
CHECKS = {'pid_reuse_excludes_old_system_branch', 'old_parent_edge_has_reason',
          'descendant_parent_birth_retained', 'root_pid_reuse_refused', 'missing_root_refused',
          'missing_child_creation_not_assumed', 'actual_windows_root_identity',
          'actual_windows_child_included', 'actual_windows_child_birth_matches',
          'actual_windows_only_current_descendants'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    prerequisites = {args.after_unit: terminal_unit(args.after_unit),
                     'qw3-computer-final-qualification-04.service':
                         terminal_unit('qw3-computer-final-qualification-04.service')}
    if args.candidate_dir:
        candidate = args.candidate_dir.resolve()
        prepared = json.loads((candidate / 'preparation.json').read_text())
        for name in ('helpers.ps1', 'test_client_resources.ps1', 'client_resources.py'):
            if digest(candidate / name) != prepared['source_files'][name]:
                raise ValueError('Candidate source changed: ' + name)
        helper_source = (candidate / 'helpers.ps1').read_text()
        probe_source = (candidate / 'test_client_resources.ps1').read_text()
        validator_path = candidate / 'client_resources.py'
    else:
        # Extract the exact embedded observer so the standalone probe and
        # native task sampler cannot silently test different implementations.
        driver = (ROOT / 'apps/computer_use/vm/run_native_case.ps1').read_text()
        begin = '# Read-only process accounting.'
        end = '\nfunction Sample-ClientProcesses'
        if driver.count(begin) != 1 or driver.count(end) != 1:
            raise ValueError('Native resource helper boundaries are ambiguous')
        helper_source = driver[driver.index(begin):driver.index(end)]
        probe_source = (ROOT / 'apps/computer_use/vm/test_client_resources.ps1').read_text()
        validator_path = ROOT / 'apps/computer_use/client_resources.py'
    info = json.loads(subprocess.check_output(['docker', 'inspect', 'qw3-computer-windows'],
                                              text=True, timeout=10))[0]
    if info['Config']['Labels'].get('com.qw3.computer-use.owner') != 'codex-goal-20260930':
        raise RuntimeError('VM ownership label differs')
    if not info['State']['Running']:
        raise RuntimeError('Owned VM is not running; prepare it before this probe')
    shared = next(Path(m['Source']) for m in info['Mounts'] if m['Destination'] == '/shared')
    token = (shared / 'executor-token.txt').read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def health():
        request = urllib.request.Request('http://127.0.0.1:18082/health',
                                        headers={'Authorization': 'Bearer ' + token})
        with opener.open(request, timeout=10) as response:
            data = json.load(response)
        if data['platform'] != 'windows' or not data['paused']:
            raise RuntimeError('The idle Windows desktop must already be paused')
        return data
    before = health()
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(kind='actual_windows_client_resource_identity_probe', passed=False,
                 started_at=time.time(), prerequisites=prerequisites, initial_health=before,
                 candidate_sources={'helpers.ps1': hashlib.sha256(helper_source.encode()).hexdigest(),
                                    'test_client_resources.ps1': hashlib.sha256(probe_source.encode()).hexdigest(),
                                    'client_resources.py': digest(validator_path)},
                 native_application_memory_measured=False)
    identifier = 'client-resource-' + uuid.uuid4().hex
    helper_name = identifier + '-helpers.ps1'
    report_name = identifier + '.json'
    try:
        (shared / helper_name).write_text(helper_source)
        source = probe_source.replace('client-resource-helpers.ps1', helper_name).replace('client-resource-probe.json', report_name)
        script = output / 'native-probe.ps1'
        script.write_text(source)
        state['staged_probe_sha256'] = digest(script)
        state['shared_report_name'] = report_name
        with (output / 'setup.log').open('wb') as log:
            subprocess.run([sys.executable, str(ROOT / 'apps/computer_use/vm/provision.py'), str(script)],
                           cwd=ROOT, stdout=log, stderr=log, timeout=90, check=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                report = json.loads((shared / report_name).read_text(encoding='utf-8-sig'))
                break
            except (FileNotFoundError, ValueError):
                time.sleep(.5)
        else:
            raise TimeoutError('Native process identity probe did not return a report')
        (output / 'native-report.json').write_text(json.dumps(report, indent=2) + '\n')
        state['native_report'] = report
        if not report['passed'] or report['source_sha256'] != state['candidate_sources']['helpers.ps1']:
            raise RuntimeError('Native probe failed or tested another source')
        checks = report['checks']
        if len(checks) != len(CHECKS) or {c['name'] for c in checks} != CHECKS or not all(c['passed'] for c in checks):
            raise RuntimeError('Native probe contract coverage is incomplete')
        spec = importlib.util.spec_from_file_location('candidate_client_resources', validator_path)
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        validator.validate_sample(report['real_sample'])
        state['python_identity_audit_passed'] = True
        state['final_health'] = health()
        if state['final_health']['desktop_id'] != before['desktop_id']:
            raise RuntimeError('The desktop changed during the setup-only probe')
        state['passed'] = True
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        state['error'] = repr(exc)
    finally:
        state['finished_at'] = time.time()
        (output / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({k: state.get(k) for k in ('passed', 'error', 'python_identity_audit_passed',
                                              'native_application_memory_measured')}))
    return state['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-unit', required=True)
    parser.add_argument('--candidate-dir', type=Path, help='Optional isolated candidate; defaults to the checked-in observer')
    parser.add_argument('--run-dir', type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
