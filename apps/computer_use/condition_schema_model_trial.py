"""Run an original pilot under the separate, immutable condition-wire study."""
import argparse
import json
from pathlib import Path
import shutil
import time

import condition_schema_model_study as study
from condition_schema_wire import codex_normalized_items
import native_control_model_trial as original

ORIGINAL_TRIAL_SHA = 'cca356568a9964ba0e622b256ca81267acd759051a099dc43a5061b2f11f4953'


def wire_conditions(inference,arm):
    if arm not in study.ARMS:raise study.adapter.owned.SafetyHalt('Unknown condition comparison arm')
    from qw3_computer.tool_schema import WORKFLOW_TOOLS
    declared=WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    base=original.wire_evidence(inference)
    rows=[]
    for record in base:
        trace=Path(inference)/record['trace']
        body=json.loads(original.archive.read_request(trace));metric=json.loads((trace/'metrics.json').read_text())
        if metric.get('status') not in ('completed','cancelled','failed'):
            raise study.adapter.owned.SafetyHalt('Actual model request remains nonterminal')
        parameters=next(t['function']['parameters'] for t in body['tools'] if t['function']['name']=='mcp__computer__act')
        condition=parameters['properties']['step']['properties']['conditions']
        if arm=='condition-repair':
            if condition!=declared:raise study.adapter.owned.SafetyHalt('Actual repair request lacks its trusted full inline definition')
            shape='trusted-full-definition'
        else:
            items=condition.get('items')
            if items not in ({},declared['items'],codex_normalized_items(declared['items'])):
                raise study.adapter.owned.SafetyHalt('Unknown original wire condition definition')
            shape='opaque' if items=={} else 'declared' if items==declared['items'] else 'known-codex-normalization'
        rows.append(dict(record,metric_status=metric['status'],condition_shape=shape))
    return rows


def run(args):
    if study.candidate.digest(original.__file__)!=ORIGINAL_TRIAL_SHA:
        raise study.adapter.owned.SafetyHalt('Original task/grade/cleanup implementation changed')
    protocol,row,framework,_=study.validate_study(args.study,args.trial_id)
    output=args.output.resolve()
    protected=[Path(args.study).resolve().parent,Path(protocol['source_original_native_study']['path']).parent,
        Path(protocol['original_sdk_protocol']['path']).parent]
    protected += [Path(protocol[key]['path']).parent for key in ('control_build','candidate_build')]
    if (not output.is_relative_to(study.candidate.RUNTIME) or output.exists()
            or any(output.is_relative_to(path) for path in protected)):
        raise study.adapter.owned.SafetyHalt('Use a fresh trial outside all immutable source/attempt directories')
    args.output=output
    receipt=dict(kind='separate_condition_schema_original_pilot_receipt',status='preparing',started_at=time.time(),
        study=study.ref(args.study),trial_id=row['id'],arm=row['arm'],case=row['case'],repetition=row['repetition'],
        original_trial_source=study.ref(Path(original.__file__)),wrapper_source=study.ref(__file__),
        original_task_material_budget_grader_and_cleanup_changed=False,business_improvement_proven=False)
    try:
        with study.scoped_stack(args.study,args.trial_id):
            result=original.run(args)
        # The original grade is already written and stays authoritative, even
        # if this additional evidence collector fails after the task stopped.
        receipt['original_trial_result']=study.ref(args.output/'result.json')
        service=json.loads((args.output/'stack/stack.json').read_text())
        if service['framework']!=str(framework):raise study.adapter.owned.SafetyHalt('Actual selected private framework changed')
        rows=wire_conditions(Path(service['services']['client']['runtime'])/'inference',row['arm'])
        receipt.update(status='completed',termination_confirmed=True,actual_condition_wire=rows,
            original_business_outcome=result['outcome'],finished_at=time.time())
    except BaseException as error:
        receipt.update(status='halted-no-replay',error_kind=type(error).__name__,error=str(error),finished_at=time.time())
        raise
    finally:
        if args.output.is_dir():
            shutil.copyfile(__file__,args.output/'condition-trial-source.py')
            study.adapter.queue.save(args.output/'condition-trial-receipt.json',receipt)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','engine-binding','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--trial-id',required=True);parser.add_argument('--port-base',required=True,type=int)
    result=run(parser.parse_args());print(json.dumps(dict(status=result['status'],termination_confirmed=result['termination_confirmed'])))
