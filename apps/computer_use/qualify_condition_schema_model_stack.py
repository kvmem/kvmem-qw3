"""Reuse the original read-only VM stack qualification for the separate study."""
import argparse
import json
from pathlib import Path
import shutil

import condition_schema_model_study as study
import qualify_native_control_model_stack as original


def perform(args):
    if study.candidate.digest(original.__file__)!=study.QUALIFIER_SHA:
        raise study.adapter.owned.SafetyHalt('Original read-only stack qualifier changed')
    study.validate_study(args.study,args.trial_id)
    with study.scoped_stack(args.study,args.trial_id):
        result=original.perform(args)
    for name in ('condition_schema_model_study.py','qualify_condition_schema_model_stack.py'):
        shutil.copyfile(Path(__file__).with_name(name),args.output/'source'/name)
    receipt=dict(kind='separate_condition_schema_readonly_vm_stack_qualification',
        study=study.ref(args.study),original_result=study.ref(args.output/'result.json'),
        original_qualifier_source=study.ref(Path(original.__file__)),
        new_helper_sources={name:study.ref(args.output/'source'/name) for name in
            ('condition_schema_model_study.py','qualify_condition_schema_model_stack.py')},
        passed=result['passed'],model_requests=0,device_inputs=0,production_installed=False,
        actual_engine_bound=False,model_context_not_qualified_without_engine=True)
    study.adapter.queue.save(args.output/'condition-stack-qualification.json',receipt)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--trial-id',required=True);parser.add_argument('--port-base',required=True,type=int)
    result=perform(parser.parse_args())
    print(json.dumps({key:result.get(key) for key in ('passed','checks','error','cleanup_error','video')},ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
