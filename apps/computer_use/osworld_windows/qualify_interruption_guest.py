"""Windows control-plane diagnostic with real Codex and a scripted backend.

Pause during an incomplete tool call, restart only the owned client, then
resume its persisted task. This helper does not inject controller state.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import aiohttp


async def run(output, backend, token_file, partial_ready):
    if os.name != 'nt' or os.environ.get('USERNAME') != 'Qw3Test':
        raise ValueError('Only the dedicated Windows test user may run this probe')
    output.mkdir(exist_ok=False)
    runtime=Path(r'C:\Qw3Computer\InterruptionQualification')/output.name
    runtime.mkdir(parents=True,exist_ok=False)
    app_runtime=runtime/'app'
    report=dict(kind='native_pause_restart_resume_qualification',passed=False,formal_run=False,
                qw3_inference_used=False,scripted_backend=True,states=[],steps=[],checks={})
    process=None
    def save(stage):
        report['steps'].append(dict(stage=stage,guest_monotonic_ns=time.monotonic_ns()))
        (output/'progress.json').write_text(json.dumps(report,indent=2))
    def actions():
        path=app_runtime/'desktop/desktop.sqlite'
        if not path.exists():return []
        with sqlite3.connect(path) as db:
            return [dict(id=row[0],request=json.loads(row[1]),status=row[2],created=row[3])
                    for row in db.execute('SELECT id,request,status,created FROM actions ORDER BY created')]
    def launch(label):
        log=(runtime/(label+'.log')).open('wb')
        return subprocess.Popen([sys.executable,'-X','utf8','-m','qw3_computer.app',
            '--desktop','http://127.0.0.1:5000','--desktop-token',r'C:\Qw3Computer\executor-token.txt',
            '--backend',backend,'--backend-token',str(token_file),'--runtime',str(app_runtime),
            '--tool-call-limit','20','--output-policy','task'],stdout=log,stderr=log)
    def kill_owned():
        if process is not None and process.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],capture_output=True,check=True)
            process.wait(timeout=10)
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=20)) as session:
        async def api(path,body=None):
            async with session.request('POST' if body is not None else 'GET','http://127.0.0.1:18083'+path,json=body) as response:
                value=await response.json()
                if response.status!=200:raise ValueError('Owned client HTTP '+str(response.status))
                return value
        async def ready():
            async with asyncio.timeout(60):
                while True:
                    if process.poll() is not None:raise ValueError('Owned client exited')
                    try:
                        state=await api('/api/state')
                        if state.get('ready'):return state
                    except (aiohttp.ClientError,TimeoutError):pass
                    await asyncio.sleep(.2)
        async def state(stage):
            value=await api('/api/state');report['states'].append(dict(stage=stage,state=value));return value
        try:
            desktop_token=Path(r'C:\Qw3Computer\executor-token.txt').read_text().strip()
            async with session.get('http://127.0.0.1:5000/health',headers={'Authorization':'Bearer '+desktop_token}) as response:
                response.raise_for_status()
                if not (await response.json())['paused']:raise ValueError('Desktop must begin paused')
            process=launch('initial-client');await ready();save('initial_client_ready')
            initial=await api('/api/task',{'prompt':'Control-plane diagnostic: observe once, open a blank Notepad, and await interruption. Do not type or save files.'})
            async with asyncio.timeout(60):
                while not partial_ready.exists():
                    if process.poll() is not None:raise ValueError('Client exited before partial call')
                    await asyncio.sleep(.1)
            before=await state('partial_call_pending')
            assert before['status']=='running' and before['tool_calls_used']==2
            before_actions=actions();assert len(before_actions)==1 and before_actions[0]['status']=='completed'
            assert before_actions[0]['request']['action']==dict(kind='open_app',app='notepad')
            pause_started=time.monotonic_ns()
            paused=await api('/api/control',{'command':'pause'})
            report['pause_acknowledgment_ms']=(time.monotonic_ns()-pause_started)/1e6
            report['states'].append(dict(stage='pause_acknowledged',state=paused))
            assert paused['status']=='paused' and paused['desktop_paused']
            await asyncio.sleep(1)
            settled=await state('paused_settled')
            stable_actions=actions();assert stable_actions==before_actions
            assert settled['tool_calls_used']==2 and settled['turn_status']=='interrupted'
            save('partial_tool_call_interrupted_without_dispatch')
            kill_owned();process=launch('restarted-client');restored=await ready()
            report['states'].append(dict(stage='restarted_client_ready',state=restored))
            report['checks']['paused_task_restored']=(restored['status']=='paused' and restored['desktop_paused'])
            report['checks']['task_and_thread_preserved']=(restored['id']==before['id'] and restored['thread_id']==before['thread_id'])
            report['checks']['budget_preserved_after_restart']=(restored['tool_calls_used']==2 and restored['tool_call_limit']==20)
            resumed=await api('/api/control',{'command':'resume'})
            report['states'].append(dict(stage='resume_acknowledged',state=resumed))
            async with asyncio.timeout(60):
                while True:
                    final=await api('/api/state')
                    if final['status'] in ('completed','failed','paused','stopped'):break
                    await asyncio.sleep(.2)
            report['states'].append(dict(stage='resumed_terminal',state=final))
            report['checks']['same_task_completed_after_resume']=(final['status']=='completed' and final['id']==before['id'] and final['thread_id']==before['thread_id'])
            report['checks']['new_turn_same_history']=(final['turn_id']!=before['turn_id'])
            report['checks']['lifetime_budget_incremented']=(final['tool_calls_used']==3 and final['tool_call_limit']==20)
            report['checks']['completed_desktop_paused']=(final['desktop_paused'] and not final['uncertain_actions'])
            report['checks']['no_partial_type_dispatched']=actions()==before_actions
            report['checks']['pause_settled_before_restart']=settled['turn_status']=='interrupted'
            report['passed']=all(report['checks'].values());save('restarted_task_resumed_and_completed')
        except BaseException as exc:
            report['error']=repr(exc);raise
        finally:
            try:
                if process is not None and process.poll() is None:
                    stopped=await api('/api/control',{'command':'stop'})
                    report['final_desktop_stopped']=stopped['desktop_paused'] and stopped['status']=='stopped'
                    if not report['final_desktop_stopped']:report['passed']=False
                kill_owned()
                (output/'actions.json').write_text(json.dumps(actions(),indent=2))
                for name in ('task.json','codex-events.jsonl'):
                    if (app_runtime/name).exists():shutil.copyfile(app_runtime/name,output/name)
                for path in runtime.glob('*.log'):shutil.copyfile(path,output/path.name)
            except BaseException as cleanup:
                report['cleanup_error']=repr(cleanup);report['passed']=False
            finally:(output/'result.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--partial-ready',type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(run(args.output,args.backend,args.token_file,args.partial_ready))
    raise SystemExit(0 if result['passed'] else 1)
