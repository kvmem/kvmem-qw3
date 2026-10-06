"""Qualify new routing with the unchanged native and full predecessor checks."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import time
import unittest

import resource_window_resume_scope as scope
import resource_window_native as native
import resource_window_condition as condition


def run(output):
    output = Path(output).resolve()
    if output != scope.QUALIFICATION.parent:
        raise ValueError('Use the declared new qualification directory')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    hashes = {}
    for name in scope.sources():
        source = scope.ROOT/'apps/computer_use'/name
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = scope.owned.digest(source)
    suite = unittest.TestSuite()
    for name, replacement in (('test_native_control_model_queue', native.NAMESPACE),
                              ('test_condition_schema_model_queue', condition.NAMESPACE),
                              ('test_resource_window_resume', None)):
        path = scope.ROOT/'apps/computer_use/tests'/f'{name}.py'
        spec = importlib.util.spec_from_file_location('resume_qualification_'+name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if replacement is not None:
            module.q = replacement
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    with (output/'checks.log').open('x') as log:
        r = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    record = dict(kind='resource_window_serial_routing_cpu_qualification', checked_at=time.time(),
        passed=r.wasSuccessful(), tests_run=r.testsRun, errors=len(r.errors), failures=len(r.failures),
        skipped=len(r.skipped), source_sha256=hashes, old_audit=scope.ref(scope.AUDIT),
        image_declaration=scope.ref(scope.DECLARATION),
        original_qualifications={name:scope.ref(path) for name,path in scope.ORIGINAL_QUALIFICATIONS.items()},
        model_calls=0, device_inputs=0,
        scope='CPU queue fixtures on the new namespaces; original full receipt, unknown-action, no-replay and same-child timeout checks. Not model task success.')
    scope.stable.old.save(output/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print({key:result[key] for key in ('passed','tests_run','errors','failures')})
    if not result['passed']:
        raise SystemExit(1)
