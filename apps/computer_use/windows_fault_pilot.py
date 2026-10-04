"""Shared host Codex/QW3 runtime, native Windows input and recovery fixture."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from browser_pilot import grade_records
from fault_browser_pilot import FIELDS,prompt,response_grade
from linux_recording import LinuxRecording
from osworld_linux.runner import request,save
from platform_pilots import owned_container,retire_task
from windows_capture_candidate import health as executor_health
import windows_fault_fixture as fixture_transport


class WindowsRecording(LinuxRecording):
    def inspect(self):
        info=owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')
        if info['Id']!=self.container_id or not info['State']['Running']:
            raise ValueError('Windows recording device changed')
        return info


def independent_grade(fixture,directory):
    root=directory/'windows-fault-final'
    records=(root/'submissions.jsonl').read_text() if (root/'submissions.jsonl').exists() else ''
    result=grade_records(records,FIELDS)
    state=json.loads((root/'fault-state.json').read_text())
    names=[event['name'] for event in state['events']]
    condition=fixture['condition']
    exercised={'slow_loading':'loading_visible' in names,'focus_dialog':'notice_shown' in names,
               'no_progress':'saving_pending_visible' in names,'normal':bool(state['save_attempts'])}[condition]
    result.update(condition=condition,fault_exercised=exercised,save_attempt_count=len(state['save_attempts']),
                  saved_count=state['saved_count'],saved_visible='saved_visible' in names,
                  notice_closed='notice_closed' in names,private_fixture_state=state)
    return result


def admitted_child(path,pid=None):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        try:admission=json.loads(path.read_text())
        except (OSError,ValueError):time.sleep(.05);continue
        if admission.get('pilot_pid')==(os.getpid() if pid is None else pid):
            if admission.get('status')=='running-separate-pilot':return admission
            if admission.get('status')=='running-standalone-pilot':
                from completed_campaign_pilot import completed_campaign
                proof=completed_campaign(Path(admission['formal_completion']['campaign']))
                if proof!=admission['formal_completion']:
                    raise ValueError('Completed formal campaign identity changed')
                return admission
        if admission.get('pilot_pid') is not None:break
        time.sleep(.05)
    raise ValueError('Use the identified owned interlude child, never a concurrent model workload')


def run(args):
    admitted_child(args.admission)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result={'kind':'integration-pilot-not-formal','platform':'windows','formal':False,'model_used':False,
            'accepted':False,'fault_accepted':False,'started_at':time.time()}
    fixture=None;video=None;task_id=None;deadline_reached=False
    with open('/tmp/qw3-osworld-vm.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state=request(args.app,'/api/state')
            if (state['status']!='idle' or state.get('uncertain_actions') or state.get('thinking') is not True
                    or state.get('model_output_tokens')!=16384 or state.get('tool_call_limit')!=100):
                raise ValueError('Use the idle qualified local thinking candidate')
            vm=owned_container('qw3-computer-windows','com.qw3.computer-use.owner','codex-goal-20260930')
            native=executor_health()
            if native.get('paused') is not True:
                raise ValueError('The native Windows input owner must be paused before setup')
            state=request(args.app,'/api/device',{'device_id':'windows-vm'})
            candidate=json.loads((args.runtime/'candidate-manifest.json').read_text())
            for filename,expected in candidate['files'].items():
                if hashlib.sha256((args.runtime/'framework/qw3_computer'/filename).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Candidate core differs from its frozen manifest')
            instruction=prompt('http://127.0.0.1:8086/',FIELDS)
            manifest={'kind':'integration-pilot-not-formal','case':'windows_browser','fault_condition':args.fault,
                'prompt':instruction,'container_id':vm['Id'],'deadline_seconds':args.seconds,
                'model_output_tokens':16384,'thinking':True,'tool_call_limit':100,'advisor':False,'kvmem':'off',
                'app_runtime':str(args.runtime.resolve()),'video_mode':'vnc','no_model_access_to_initializer_or_grader':True,
                'native_executor':{key:native.get(key) for key in ('process_id','desktop_id','version',
                    'capture_profile','source_sha256','base_source_sha256')},
                'candidate_manifest_sha256':hashlib.sha256((args.runtime/'candidate-manifest.json').read_bytes()).hexdigest(),
                'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ['windows_fault_pilot.py','windows_fault_fixture.py','fault_browser_windows_guest.py','fault_browser_server.js']}}
            save(output/'pilot-manifest.json',manifest)
            fixture=fixture_transport.initialize(output.name,args.fault,output,args.bundle)
            event_source=args.runtime/'codex-events.jsonl';event_offset=event_source.stat().st_size if event_source.exists() else 0
            video=WindowsRecording(output,vm['Id'],'qw3-computer-windows');video.start()
            fixture_transport.arm(fixture,output)
            started=time.monotonic();result['model_dispatch_attempted']=True
            state=request(args.app,'/api/task',{'prompt':instruction});task_id=state['id'];result['model_used']=True
            manifest.update(task_id=task_id,started_at=time.time(),codex_event_offset_bytes=event_offset)
            save(output/'pilot-manifest.json',manifest)
            with (output/'progress.jsonl').open('w') as log:
                while True:
                    state=request(args.app,'/api/state')
                    if state.get('id')!=task_id:raise RuntimeError('Windows probe task was replaced')
                    log.write(json.dumps({'at':time.time(),'status':state['status'],'tools':state['tool_calls_used']})+'\n');log.flush()
                    if state['status'] in {'completed','failed','paused','stopped'}:break
                    if time.monotonic()-started>=args.seconds:
                        deadline_reached=True;state=request(args.app,'/api/control',{'command':'stop'});break
                    time.sleep(2)
            save(output/'task.json',state)
            task_seconds=time.monotonic()-started
            if state.get('uncertain_actions') or state.get('desktop_paused') is not True:
                raise RuntimeError('Windows probe input stop/delivery requires reconciliation')
            fixture_transport.collect(fixture,output);fixture=None
            # Read the retained fixture descriptor, independent of agent claims.
            descriptor=json.loads((output/'windows-fault-fixture.json').read_text())
            grade=independent_grade(descriptor,output);save(output/'independent-browser.json',grade)
            response=response_grade(grade,state,deadline_reached=deadline_reached)
            save(output/'independent-fault-response.json',response)
            result.update(app_status=state['status'],seconds=task_seconds,tool_calls=state['tool_calls_used'],
                turn_error=state.get('turn_error'),uncertain_actions=len(state.get('uncertain_actions',[])),
                artifact_passed=grade['passed'],fault_condition=args.fault,fault_exercised=grade['fault_exercised'],
                fault_response_passed=response['passed'],unassisted_passed=response['unassisted_business_passed'])
            try:result['video_collection_passed']=video.finish()['complete']
            except Exception as exc:result.update(video_collection_passed=False,video_collection_error=type(exc).__name__+': '+str(exc))
            result.update(accepted=result['unassisted_passed'] and result['video_collection_passed'],
                          fault_accepted=response['passed'] and result['video_collection_passed'])
            if (output/'continuous.mp4').exists():result['video_bytes']=(output/'continuous.mp4').stat().st_size
            if event_source.exists():
                with event_source.open('rb') as log:log.seek(event_offset);(output/'codex-events.jsonl').write_bytes(log.read())
        except Exception as exc:
            result.update(status='infrastructure-failed',error=type(exc).__name__+': '+str(exc))
            if result.get('model_dispatch_attempted') and task_id is None:result['model_used']=None
        finally:
            if task_id:
                try:
                    current=request(args.app,'/api/state')
                    if current.get('id')==task_id and current['status'] in {'starting','running','verifying'}:
                        request(args.app,'/api/control',{'command':'stop'})
                    save(output/'task-retirement.json',retire_task(args.app,task_id,'linux-vm'))
                    result['owned_task_retired']=True
                except Exception as exc:
                    result.update(owned_task_stop_unconfirmed=True,accepted=False,fault_accepted=False,
                                  retirement_error=type(exc).__name__+': '+str(exc))
            if fixture:
                try:fixture_transport.collect(fixture,output)
                except Exception as exc:result['fixture_cleanup_error']=type(exc).__name__+': '+str(exc)
            if video and not video.collected:
                try:video.finish('interrupted.mp4')
                except Exception as exc:result['recording_cleanup_error']=type(exc).__name__+': '+str(exc)
            result['finished_at']=time.time();save(output/'integration-result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--fault',choices=['slow_loading','focus_dialog','no_progress'],required=True)
    parser.add_argument('--seconds',type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.seconds<=900:parser.error('Use60..900seconds')
    outcome=run(args)
    print({key:outcome.get(key) for key in ['accepted','fault_accepted','error','owned_task_retired']},flush=True)
