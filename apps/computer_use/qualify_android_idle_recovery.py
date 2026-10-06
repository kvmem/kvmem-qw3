"""Qualify the opt-in recovery component with explicit CPU/HTTP fixtures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILES = ('android_idle_recovery.py', 'qualify_android_idle_recovery.py', 'android_idle_recovery_candidate.py',
         'tests/test_android_idle_recovery.py', 'qw3_computer/android_driver.py',
         'qw3_computer/device_gateway.py', 'qw3_computer/action_validation.py',
         'qw3_computer/device_protocol.py', 'qw3_computer/device_lease.py',
         'qw3_computer/tool_schema.py', 'qw3_computer/__init__.py')


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'apps/computer_use/.runtime'):
        raise ValueError('Use a new goal-owned qualification directory')
    output.mkdir(mode=0o700, exist_ok=False)
    hashes = {}
    for name in FILES:
        source = ROOT / 'apps/computer_use' / name; target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('idle_recovery_tests', ROOT / 'apps/computer_use/tests/test_android_idle_recovery.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (output / 'checks.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record = {'kind': 'opt_in_android_idle_recovery_cpu_http_qualification', 'checked_at': time.time(),
              'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
              'errors': len(result.errors), 'failures': len(result.failures), 'skipped': len(result.skipped),
              'source_sha256': hashes, 'model_calls': 0, 'real_device_inputs': 0,
              'scope': 'Explicit CPU/loopback HTTP fixtures; not a real Appium expiry or QW3 task.'}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in ('passed', 'tests_run', 'errors', 'failures')}))
    if not result['passed']:
        raise SystemExit(1)
