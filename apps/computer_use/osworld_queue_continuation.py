"""Continue unexecuted frozen trials after one independently retired VM failure.

The original halted queue, result, unknown receipt and trial directory are never
updated or replayed. A separate queue carries their exact terminal rows forward.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import kvmem_osworld_queue as owned
from osworld_linux.runner import save


CASE = '022-legacy-r1'
RECEIPT = 'q3_b4b87fef8706f9283e1d8290703162c93da84a6989815808235b7b61147b0913'
CONTROLLER = {'pid':2468729,'uid':1001,'start_ticks':365383716,
    'command':[str(owned.ROOT/'third_party/OSWorld-V2/.venv/bin/python'),'/tmp/qw3-candidate49b-osworld-after-receipt.py'],
    'cwd':str(owned.ROOT)}
WINDOWS_EXECUTOR_SHA = '59111c5fc460e0372438e6d1271e7ce1a41160d1fc6730843a58af22d2f7a869'


def read(path):return json.loads(Path(path).read_text())


def ordered_trials(protocol):
    first=protocol['diagnostic_first_trials']
    rows=protocol['trials'];by_id={r['id']:r for r in rows}
    if len(rows)!=40 or len(by_id)!=40 or len(first)!=len(set(first)) or set(first)-set(by_id):
        raise owned.SafetyHalt('Frozen forty-trial order is invalid')
    return [by_id[k] for k in first]+[r for r in rows if r['id'] not in first]


def retired_controller(identity):
    actual=owned.process_identity(identity['pid'])
    if actual is not None:
        if not owned.same_process(identity,actual):raise owned.SafetyHalt('Prior controller PID reused')
        if actual['state']!='Z':raise owned.SafetyHalt('Prior controller still running')


def original_rows(protocol, state, stored_results):
    expected=ordered_trials(protocol)
    rows=state.get('trials',[])
    if (state.get('status')!='halted-no-further-model-dispatch' or len(rows)!=12
            or state.get('current_trial')!=CASE or state.get('pid')!=CONTROLLER['pid']
            or [r['id'] for r in rows]!=[r['id'] for r in expected[:12]]
            or rows[-1]['id']!=CASE):
        raise owned.SafetyHalt('Use the exact twelve-row halted checkpoint')
    for row in rows:
        result=stored_results.get(row['id'])
        if (not isinstance(result,dict) or result!=row.get('result') or not result.get('finished_at')
                or type(result.get('uncertain_actions')) is not int
                or result['uncertain_actions']!=(1 if row['id']==CASE else 0)
                or result.get('owned_task_stop_unconfirmed')
                or result.get('accepted') is not False):
            raise owned.SafetyHalt('Original terminal receipt altered or another unknown found')
    return copy.deepcopy(rows)


def validate(declaration, *, live=True):
    d=read(declaration)
    if (d.get('kind')!='exact_retired_osworld_trial_continuation'
            or d.get('case')!=CASE or d.get('receipt_id')!=RECEIPT
            or d.get('original_unknown_reclassified') is not False
            or d.get('original_trial_replayed') is not False
            or d.get('controller_identity')!=CONTROLLER):
        raise owned.SafetyHalt('Unrecognized retired-trial continuation')
    for reference in d['files']:
        if owned.digest(reference['path'])!=reference['sha256']:raise owned.SafetyHalt('Frozen continuation evidence changed')
    if owned.digest(Path(__file__))!=d['recipe_sha256']:raise owned.SafetyHalt('Continuation source changed')
    campaign=Path(d['campaign']);protocol=read(campaign/'protocol.json');state=read(campaign/'queue.json')
    if state['protocol_sha256']!=owned.digest(campaign/'protocol.json'):raise owned.SafetyHalt('Original protocol identity changed')
    results={r['id']:read(campaign/'attempts'/r['id']/'result.json') for r in state['trials']}
    original=original_rows(protocol,state,results)
    attempt=campaign/'attempts'/CASE
    vm=read(attempt/'vm.json');agent=read(attempt/'final_agent_state.json')
    if vm.get('container_id')!=d['container_id'] or vm.get('name')!='qw3-linux-'+CASE:
        raise owned.SafetyHalt('Failed VM identity changed')
    with sqlite3.connect((attempt/'gateway/receipts.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT id,status,request,result FROM actions WHERE status NOT IN ("completed","rejected")').fetchall()
    if len(rows)!=1 or rows[0][0]!=RECEIPT or rows[0][1]!='uncertain' or rows[0][3] is not None:
        raise owned.SafetyHalt('Uncertain gateway row is not the declared original row')
    request=json.loads(rows[0][2])
    if hashlib.sha256(rows[0][2].encode()).hexdigest()!=d['request_sha256']:
        raise owned.SafetyHalt('Original unknown request changed')
    uncertain=agent.get('uncertain_actions',[])
    if (len(uncertain)!=1 or uncertain[0].get('receipt_id')!=RECEIPT
            or uncertain[0].get('status')!='uncertain' or agent.get('desktop_paused') is not True
            or request.get('desktop_id')!=d['desktop_id']):
        raise owned.SafetyHalt('Original paused client unknown does not match')
    # Verify all current and successor frozen sources through their unchanged
    # source validators; no new model/config/task/evaluator package is selected.
    import execution_retest_queue_v2 as successor
    _,kp=successor.validate(Path(d['sdk51_protocol']))
    if (campaign!=Path(kp['parent_candidate_protocol']).resolve().parent
            or Path(d['continued_queue'])!=campaign/'continued_queue.json'
            or d.get('remaining_trial_count')!=28 or d.get('original_queue_changed') is not False):
        raise owned.SafetyHalt('Continuation output or frozen campaign changed')
    if owned.digest(owned.ROOT/'apps/computer_use/vm/executor.ps1')!=WINDOWS_EXECUTOR_SHA:
        raise owned.SafetyHalt('Windows input executor changed')
    if live:
        retired_controller(CONTROLLER)
        owned.assert_no_services(campaign)
        for row in original:owned.assert_vm_retired(campaign/'attempts'/row['id'])
    return d,protocol,original


def prepare(campaign, output, sdk51_protocol):
    campaign,output,sdk51_protocol=map(lambda p:Path(p).resolve(),(campaign,output,sdk51_protocol))
    state=read(campaign/'queue.json');protocol=read(campaign/'protocol.json')
    results={r['id']:read(campaign/'attempts'/r['id']/'result.json') for r in state['trials']}
    original=original_rows(protocol,state,results);attempt=campaign/'attempts'/CASE
    with sqlite3.connect((attempt/'gateway/receipts.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute('SELECT request FROM actions WHERE id=?',(RECEIPT,)).fetchone()
    if not row:raise owned.SafetyHalt('Original unknown row is missing')
    paths=[campaign/'queue.json',campaign/'protocol.json',Path(CONTROLLER['command'][1]),
           attempt/'vm.json',attempt/'final_agent_state.json',sdk51_protocol]
    paths += [campaign/'attempts'/r['id']/'result.json' for r in original]
    d={'kind':'exact_retired_osworld_trial_continuation','at':time.time(),'campaign':str(campaign),
       'case':CASE,'receipt_id':RECEIPT,'desktop_id':json.loads(row[0])['desktop_id'],
       'request_sha256':hashlib.sha256(row[0].encode()).hexdigest(),
       'container_id':read(attempt/'vm.json')['container_id'],'controller_identity':CONTROLLER,
       'recipe_sha256':owned.digest(Path(__file__)),'sdk51_protocol':str(sdk51_protocol),
       'files':[{'path':str(path),'sha256':owned.digest(path)} for path in paths],
       'original_unknown_reclassified':False,'original_trial_replayed':False,
       'original_queue_changed':False,'remaining_trial_count':28,
       'continued_queue':str(campaign/'continued_queue.json'),
       'scope':'Exactly one retained unknown in the retired disposable 022 legacy VM. No receipt recovery claimed; fresh unexecuted trials only.'}
    if output.exists():raise owned.SafetyHalt('Declaration already exists')
    save(output,d);validate(output)
    return d


def run(declaration):
    d,p,original=validate(declaration);campaign=Path(d['campaign']);queue=Path(d['continued_queue'])
    if queue.exists():raise owned.SafetyHalt('Continuation queue already exists; do not replay')
    state={'kind':'frozen_osworld_continuation_after_retired_trial','pid':os.getpid(),'started_at':time.time(),
           'status':'running','planned_trials':40,'protocol_sha256':owned.digest(campaign/'protocol.json'),
           'original_halted_queue_sha256':owned.digest(campaign/'queue.json'),
           'continuation_declaration':str(Path(declaration).resolve()),'continuation_declaration_sha256':owned.digest(declaration),
           'controller_sha256':owned.digest(Path(__file__)),'trials':original,
           'historical_unknowns_preserved':1,'model_concurrency':1}
    def checkpoint():save(queue,state)
    checkpoint()
    try:
        for trial in ordered_trials(p)[len(original):]:
            validate(declaration);output=campaign/'attempts'/trial['id']
            if output.exists():raise owned.SafetyHalt('Unexecuted trial directory already exists')
            state.update(current_trial=trial['id']);checkpoint()
            name=trial['framework_name']
            command=[sys.executable,'-m','osworld_office_candidate_run','--task',trial['task'],'--arm',trial['arm'],
                '--output',str(output),'--framework',p['frameworks'][name],'--engine-runtime',p['engine_runtime'],
                '--formal-campaign',p['formal_campaign'],'--acceptance-manifest',str(campaign/name/'acceptance.json'),
                '--pilot-app',p['pilot_app'],'--model-output-tokens','16384']
            environment=dict(os.environ,PYTHONPATH=str(owned.ROOT/'apps/computer_use')+':'+str(owned.ROOT/'apps/computer_use/.runtime/media-qualification'))
            with (campaign/(trial['id']+'.continuation-launch.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=owned.ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                state['child']=owned.register_child(child.pid,command);checkpoint();code=child.wait(timeout=5100)
            path=output/'result.json'
            if not path.is_file():raise owned.SafetyHalt('No actual terminal result; stop next dispatch')
            result=read(path);state['trials'].append({'id':trial['id'],'exit_code':code,'result':result});checkpoint()
            if (not result.get('finished_at') or type(result.get('uncertain_actions')) is not int
                    or result['uncertain_actions']!=0 or result.get('owned_task_stop_unconfirmed')):
                raise owned.SafetyHalt('New unknown or unconfirmed terminal task; no next dispatch')
            owned.assert_no_services(output);owned.assert_vm_retired(output)
        state.update(status='completed',finished_at=time.time());checkpoint()
    except BaseException as exc:
        state.update(status='halted-no-further-model-dispatch',error=type(exc).__name__+': '+str(exc),finished_at=time.time());checkpoint()
        raise
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--declaration',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--campaign',type=Path);p.add_argument('--sdk51-protocol',type=Path)
    args=p.parse_args()
    result=prepare(args.campaign,args.declaration,args.sdk51_protocol) if args.prepare else run(args.declaration)
    print(json.dumps({'status':result.get('status','declared'),'trials':len(result.get('trials',[]))}))


if __name__=='__main__':main()
