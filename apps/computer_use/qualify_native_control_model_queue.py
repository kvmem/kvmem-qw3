"""Run isolated CPU queue/receipt qualification; no device or model dispatch."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import time
import unittest

import native_control_model_queue as queue


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(queue.RUNTIME):raise ValueError('Use a fresh goal-owned runtime')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    source_sha = {}
    for name in queue.HELPERS:
        source = queue.ROOT/'apps/computer_use'/name
        target = output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);source_sha[name] = queue.owned.digest(source)
    spec = importlib.util.spec_from_file_location('native_queue_qualification_tests',
        queue.ROOT/'apps/computer_use/tests/test_native_control_model_queue.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (output/'checks.log').open('x') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record = dict(kind='native_control_model_queue_cpu_qualification',checked_at=time.time(),
        passed=result.wasSuccessful(),tests_run=result.testsRun,skipped=len(result.skipped),
        errors=len(result.errors),failures=len(result.failures),source_sha256=source_sha,
        model_calls=0,device_inputs=0,qualification_scope='CPU receipt/queue fixtures; no real model trial')
    queue.save(output/'result.json',record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result = run(parser.parse_args().output);print({k:result[k] for k in ('passed','tests_run','errors','failures')})
    if not result['passed']:raise SystemExit(1)
