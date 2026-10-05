"""CPU trial/queue fixtures only; actual terminal native handoff stays pending."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import time
import unittest

import condition_schema_model_queue as queue


def run(output):
    output=Path(output).resolve()
    if not output.is_relative_to(queue.RUNTIME):raise ValueError('Use a fresh goal-owned qualification directory')
    output.mkdir(parents=True,exist_ok=False,mode=0o700);hashes={}
    for name in queue.HELPERS:
        source=queue.ROOT/'apps/computer_use'/name;target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);hashes[name]=queue.owned.digest(source)
    suite=unittest.TestSuite()
    try:
        for name in ('test_condition_schema_model_trial','test_condition_schema_model_queue'):
            spec=importlib.util.spec_from_file_location('qualified_'+name,queue.ROOT/'apps/computer_use/tests'/f'{name}.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    except BaseException as error:
        queue.save(output/'result.json',dict(kind='condition_schema_model_queue_cpu_qualification',checked_at=time.time(),
            passed=False,tests_run=0,errors=1,failures=0,skipped=0,source_sha256=hashes,error_kind=type(error).__name__,
            error=str(error),model_calls=0,device_inputs=0,scope='Suite import failed before any test ran.'))
        raise
    with (output/'checks.log').open('x') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(kind='condition_schema_model_queue_cpu_qualification',checked_at=time.time(),passed=r.wasSuccessful(),
        tests_run=r.testsRun,errors=len(r.errors),failures=len(r.failures),skipped=len(r.skipped),source_sha256=hashes,
        model_calls=0,device_inputs=0,scope='Original failed-grade preservation, archived wire schemas and queue/parent inventory fixtures. No real terminal native eight or QW3 dispatch.')
    queue.save(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args().output);print({k:result[k] for k in ('passed','tests_run','errors','failures')})
    if not result['passed']:raise SystemExit(1)
