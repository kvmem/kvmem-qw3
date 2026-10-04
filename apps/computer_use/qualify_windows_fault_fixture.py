"""Actual Windows/Edge qualification of the shared fixture, with no model.

Only this fixture's UI is operated. Expected fields stay in this administrative
probe and never become agent instructions or independent grader substitutions.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import qw3_computer

from fault_browser_pilot import FIELDS
from qw3_computer.desktop import Desktop, DesktopRejected
from qw3_computer.execution import Execution
import windows_fault_fixture as fixture
from windows_fault_pilot import independent_grade


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


async def qualify(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report={'kind':'actual_windows_fault_fixture_qualification','real_model':False,
            'desktop_solving_success_claimed':False,'started_at':time.time(), 'trials':[], 'passed':False}
    sources=args.output/'host-source';sources.mkdir()
    inventory={}
    for name in ('qualify_windows_fault_fixture.py','windows_fault_fixture.py',
                 'windows_fault_pilot.py','fault_browser_windows_guest.py','fault_browser_server.js'):
        source=Path(__file__).with_name(name);shutil.copyfile(source,sources/name)
        inventory[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    report['host_source_sha256']=inventory
    framework=Path(qw3_computer.__file__).resolve().parent
    frozen=args.output/'framework/qw3_computer'
    shutil.copytree(framework,frozen,ignore=shutil.ignore_patterns('__pycache__'))
    report['framework_source_sha256']={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in frozen.rglob('*') if p.is_file()}
    bundle_manifest=json.loads((args.bundle/'manifest.json').read_text())
    for filename,digest in report['framework_source_sha256'].items():
        if bundle_manifest['source_files'].get('qw3_computer/'+filename)!=digest:
            raise ValueError('Administrative probe core differs from the qualified native bundle: '+filename)
    with Path('/tmp/qw3-osworld-vm.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for condition in ('normal','slow_loading','focus_dialog','no_progress'):
            directory=args.output/(args.output.name[:24]+'-'+condition.replace('_','-'))
            directory.mkdir()
            desktop=None;execution=None;descriptor=None
            trial={'condition':condition,'passed':False,'started_at':time.time()}
            try:
                descriptor=fixture.initialize(directory.name,condition,directory,args.bundle)
                desktop=Desktop('http://127.0.0.1:18082',
                    (fixture.SHARE/'executor-token.txt').read_text().strip(),directory/'desktop')
                await desktop.start()
                desktop.task_id=directory.name
                execution=Execution(desktop,directory);desktop.execution=execution
                await desktop.control('resume')
                fixture.arm(descriptor,directory)
                # The administrative arming script changes focus/capture state.
                # Never submit an input using the earlier setup observation.
                await desktop.observe()
                sequence=0

                async def act(action, step=None):
                    nonlocal sequence
                    sequence+=1
                    arguments={'desktop_id':desktop.latest['desktop_id'],
                        'observation_id':desktop.latest['observation_id'],
                        'action_id':'qualification_'+str(sequence),'action':action}
                    if step is not None:arguments['step']=step
                    if action['kind']=='click_control':arguments['coordinate_space']='normalized_1000'
                    return await desktop.act(arguments)

                async def click(name, kinds=None, step=None):
                    # Retries only proven non-deliveries, with newly observed IDs.
                    deadline=time.monotonic()+30
                    rejected=0
                    while time.monotonic()<deadline and rejected<3:
                        await desktop.observe()
                        controls=[c for c in desktop.latest.get('accessibility',{}).get('controls',[])
                                  if c.get('name')==name and (not kinds or c.get('kind','').lower() in kinds)]
                        if len(controls)>1:raise ValueError('Ambiguous visible control: '+name)
                        if not controls:
                            await asyncio.sleep(.2)
                            continue
                        try:return await act({'kind':'click_control','control_id':controls[0]['control_id']},step)
                        except DesktopRejected:rejected+=1
                    raise RuntimeError('Target remained unstable without input')

                await act({'kind':'open_app','app':'edge'},
                    {'goal':'Show this request desk','conditions':[{'kind':'foreground_title','text':'Request desk'}]})
                pending=[s for s in execution.steps() if s['status']=='pending']
                if pending:
                    checked=await execution.invoke({'operation':'wait','timeout_seconds':20})
                    if not checked['verified']:raise ValueError('Fixture browser did not become foreground')
                ready_started=time.monotonic()
                await execution.invoke({'operation':'begin_step','step_id':'form-ready',
                    'goal':'Form becomes ready','conditions':[{'kind':'control_text','text':'Name'},
                                                            {'kind':'control_text','text':'Reference'}]})
                while True:
                    checked=await execution.invoke({'operation':'wait','step_id':'form-ready','timeout_seconds':20})
                    if checked['verified']:break
                    if time.monotonic()-ready_started>120:raise TimeoutError('Form never became ready')
                trial['form_ready_seconds']=time.monotonic()-ready_started
                for name,value in (('Name',FIELDS['name']),('Reference',FIELDS['reference'])):
                    await desktop.observe()
                    candidates=[c for c in desktop.latest['accessibility']['controls']
                        if c.get('name')==name and c.get('kind','').lower() in {'edit','entry','text-box','textbox'}]
                    if len(candidates)!=1:raise ValueError('Input field cannot be uniquely identified: '+name)
                    definition={'goal':'Enter '+name,'conditions':[{'kind':'control_value','text':name,
                                'value':value,'control_kind':candidates[0]['kind']}]}
                    await click(name,{'edit','entry','text-box','textbox'},definition)
                    if condition=='focus_dialog' and name=='Name':
                        await click('Continue form',{'button','push-button'})
                        focused=[c for c in desktop.latest.get('accessibility',{}).get('controls',[])
                                 if c.get('name')==name and c.get('kind','').lower() in {'edit','entry','text-box','textbox'}
                                 and c.get('focused') is True]
                        # Closing a modal can restore the field's focus. Only
                        # send another focus click if the fresh receipt does
                        # not prove this, rather than creating a new transition.
                        if len(focused)!=1:await click(name,{'edit','entry','text-box','textbox'})
                    typed=await act({'kind':'type','text':value})
                    value_check=typed['step_check']
                    if not value_check['verified']:
                        value_check=await execution.invoke({'operation':'wait','timeout_seconds':20})
                    if not value_check['verified']:
                        raise ValueError('Actual field value was not verified: '+name)
                saved=await click('Save request',{'button','push-button'},
                    {'goal':'Save the request','conditions':[{'kind':'control_text','text':'Saved'}]})
                checked=saved['step_check']
                if not checked['verified']:
                    checked=await execution.invoke({'operation':'wait','timeout_seconds':2 if condition=='no_progress' else 20})
                if condition=='no_progress' and checked['verified']:raise ValueError('Permanent saving incorrectly completed')
                if condition!='no_progress' and not checked['verified']:raise ValueError('No visible save evidence')
                await desktop.control('pause')
                fixture.collect(descriptor,directory);descriptor=None
                grade=independent_grade(json.loads((directory/'windows-fault-fixture.json').read_text()),directory)
                save(directory/'independent.json',grade)
                trial['passed']=(grade['fault_exercised'] and not desktop.unresolved()
                    and (grade['saved_count']==0 and not grade['passed'] if condition=='no_progress'
                         else grade['passed'] and grade['saved_visible']
                         and (condition!='focus_dialog' or grade['notice_closed'])))
                trial['input_dispatches']=sequence
            except Exception as exc:
                trial['error']=type(exc).__name__+': '+str(exc)
            finally:
                if desktop:
                    try:await desktop.control('pause')
                    except Exception as exc:trial['pause_error']=type(exc).__name__+': '+str(exc)
                    if desktop.unresolved():
                        trial['needs_reconciliation']=True
                        trial['uncertain_action_ids']=[row['action_id'] for row in desktop.unresolved()]
                if descriptor:
                    try:fixture.collect(descriptor,directory)
                    except Exception as exc:trial['cleanup_error']=type(exc).__name__+': '+str(exc)
                if execution:execution.close()
                if desktop:await desktop.close()
                trial['finished_at']=time.time();report['trials'].append(trial);save(args.output/'result.json',report)
            if trial.get('needs_reconciliation') or any(key in trial for key in ('pause_error','cleanup_error')):break
    report.update(passed=len(report['trials'])==4 and all(t['passed'] for t in report['trials']),
                  finished_at=time.time())
    save(args.output/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(qualify(args))
    print({'passed':result['passed'],'trials':result['trials']},flush=True)
