"""Refresh goal evidence from terminal receipts, including retained failures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'apps/computer_use/.runtime'


def read(path):return json.loads(path.read_text())


def formal_attempts(campaign):
    rows=[]
    for path in campaign.glob('*/result.json'):
        result=read(path)
        if not result.get('finished_at'):continue
        task,arm,repetition=path.parent.name.split('-')
        valid=result.get('status')=='scored' and type(result.get('score')) in (int,float)
        rows.append(dict(task=task,arm=arm,repetition=int(repetition[1:]),status=result['status'],
            score=result.get('score'),agent_status=result.get('agent_status'),seconds=result.get('seconds'),
            tool_calls=result.get('tool_calls'),uncertain_actions=result.get('uncertain_actions'),
            turn_error=result.get('turn_error'),unassisted_passed=result.get('unassisted_passed'),
            infrastructure_error=result.get('error'),model_score_valid=valid,path=str(path.relative_to(ROOT)),
            finished_at=result['finished_at']))
    return sorted(rows,key=lambda row:row['finished_at'])


def formal_summary(rows):
    counts=Counter(row['status'] for row in rows)
    return dict(ended_attempts=len(rows),valid_original_scores=sum(row['model_score_valid'] for row in rows),
        infrastructure_failures=counts['infrastructure-failed'],unavailable_original_evaluators=counts['evaluator-unavailable'],
        successes=sum(row['model_score_valid'] and row['score']>=1 for row in rows),
        improvement_demonstrated=False)


def product_row(path):
    result=read(path)
    if not result.get('finished_at'):return None
    guest=result.get('guest_result',{})
    task=guest.get('task',{})
    state=guest.get('terminal_state',{})
    accepted=(result.get('passed') is True and result.get('model_used') is True
              and result.get('recording',{}).get('complete') is True
              and guest.get('desktop_stopped') is True and not guest.get('needs_reconciliation'))
    return dict(id=path.parent.name,task=result['task'],status=result['status'],passed=result.get('passed'),
        accepted=accepted,model_used=result.get('model_used'),pure_gui=False,
        seconds=task.get('elapsed_seconds'),tool_calls=state.get('tool_calls_used'),
        workflow_mode=result.get('workflow_mode'),path=str(path.relative_to(ROOT)))


def refresh(progress, campaign):
    result=read(progress)
    rows=formal_attempts(campaign)
    prefix=str(campaign.relative_to(ROOT))+'/'
    result['formal_attempts']=[row for row in result['formal_attempts'] if not row['path'].startswith(prefix)]+rows
    result['formal_summary']=formal_summary(rows)
    result['frozen_campaign']['finished']=len(rows)
    result['frozen_campaign']['current']=[job['id'] for job in read(campaign/'plan.json')['jobs']
                                        if job['status']=='running' and not (campaign/job['id']/'result.json').exists()]
    by_id={row['id']:row for row in result['integration_pilots']}
    pilot_paths=[*RUNTIME.glob('candidate-interlude-*/pilot/integration-result.json'),
                 *RUNTIME.glob('xdev-*/batch/*/integration-result.json')]
    for path in pilot_paths:
        value=read(path);identity=path.parent.parent.name if path.parent.name=='pilot' else path.parent.name
        by_id[identity]=by_id.get(identity,{}) | dict(id=identity,formal=False,path=str(path.relative_to(ROOT)),
            status=value.get('app_status',value.get('status')),seconds=value.get('seconds'),
            tool_calls=value.get('tool_calls'),artifact_passed=value.get('artifact_passed'),
            accepted=value.get('accepted',False),unassisted_passed=value.get('unassisted_passed',False),
            video_collection_passed=value.get('video_collection_passed'),uncertain_actions=value.get('uncertain_actions'),
            fault_condition=value.get('fault_condition'),fault_response_passed=value.get('fault_response_passed'),
            fault_accepted=value.get('fault_accepted'),owned_task_retired=value.get('owned_task_retired'))
    result['integration_pilots']=list(by_id.values())
    products=[row for path in RUNTIME.glob('xdev-*/*-trial/result.json') if (row:=product_row(path))]
    result['windows_product_attempts']=sorted(products,key=lambda row:row['id'])
    if 'windows_product_protocol' in result:
        result['windows_product_protocol'].update(ended=len(products),
            accepted=sum(row['accepted'] for row in products),started=sum(row['model_used'] is True for row in products))
    interludes=[]
    for path in RUNTIME.glob('*/interlude.json'):
        value=read(path)
        if value.get('status') in {'waiting-for-current-formal-attempt','running-separate-pilot','needs-reconciliation'}:
            interludes.append((value.get('started_at',0),path.parent.name,value['status']))
    if interludes:
        _,identity,status=max(interludes)
        result['frozen_campaign'].update(separate_pilot_queued=identity,separate_pilot_status=status)
    else:
        result['frozen_campaign'].update(separate_pilot_queued=None,separate_pilot_status=None)
    result['updated_at']=time.time()
    progress.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress',type=Path,default=ROOT/'docs/computer_use_cross_device_progress_20261003.json')
    parser.add_argument('--campaign',type=Path,default=RUNTIME/'linux/campaign-v2-20261004')
    args=parser.parse_args()
    value=refresh(args.progress,args.campaign.resolve())
    print(dict(formal=value['formal_summary'],windows=value.get('windows_product_protocol')))
