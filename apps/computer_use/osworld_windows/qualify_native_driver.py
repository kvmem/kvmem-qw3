"""Real Windows/Codex/desktop and 035 handoff, with a scripted backend.

This is infrastructure qualification, never a QW3 solving run. No known task
answer is given to the agent; the tiny collection fixture is explicitly seeded.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

from aiohttp import web

from vm.model_relay import create_app as relay_app
from .events035 import schedule
from .launch import prepare
from .reviews035 import serve
from .sites import Site
from .source import ROOT, sha256


async def wait_json(path, timeout=240):
    async with asyncio.timeout(timeout):
        while not path.is_file():
            await asyncio.sleep(.5)
        return json.loads(path.read_text(encoding='utf-8-sig'))


async def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    share=ROOT/'apps/computer_use/.runtime/windows/shared'
    run_id=output.name
    staged=share/(run_id+'-inputs');staged.mkdir(exist_ok=False)
    info=json.loads(subprocess.check_output(['docker','inspect','qw3-computer-windows']))[0]
    if info['Config']['Labels'].get('com.qw3.computer-use.owner')!='codex-goal-20260930':
        raise ValueError('VM ownership mismatch')
    network=info['NetworkSettings']['Networks']['bridge']
    token=secrets.token_hex(32)
    # Caller archives this credential out of SMB before a formal snapshot.
    (share/'native-model-token.txt').write_text(token);(share/'native-model-token.txt').chmod(0o600)
    initial,_=schedule();site=Site('teamchat',run_id);site.initialize(initial)
    (output/'site-before.json').write_text(json.dumps(site.state(),indent=2))
    calls=[]
    async def chunk(response,delta=None,finish=None):
        await response.write(('data: '+json.dumps({'choices':[{'index':0,'delta':delta or {},'finish_reason':finish}]})+'\n\n').encode())
    async def backend(request):
        body=await request.json();calls.append(body)
        (output/f'backend-request-{len(calls):02}.json').write_text(json.dumps(body))
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
        if len(calls)%2:
            name=next(t['function']['name'] for t in body['tools'] if t['function']['name'].endswith('observe'))
            await chunk(response,{'tool_calls':[{'index':0,'id':f'qualify_{len(calls)}','type':'function','function':{'name':name,'arguments':'{}'}}]})
            await chunk(response,finish='tool_calls')
        else:
            await chunk(response,{'content':f'Infrastructure review marker {len(calls)//2} complete.'})
            await chunk(response,finish='stop')
        await response.write(b'data: [DONE]\n\n');return response
    app=web.Application(client_max_size=64*1024*1024);app.router.add_post('/v1/chat/completions',backend)
    backend_runner=web.AppRunner(app);await backend_runner.setup()
    listener=web.TCPSite(backend_runner,'127.0.0.1',0);await listener.start()
    backend_url='http://127.0.0.1:'+str(listener._server.sockets[0].getsockname()[1])
    relay=web.AppRunner(relay_app(backend_url,network['IPAddress'],token));await relay.setup()
    event_task=None
    report={'kind':'native_campaign_driver_qualification','formal_run':False,'qw3_inference_used':False,
            'scripted_backend':True,'actual_codex_and_windows_desktop':True,'passed':False,
            'task_solving_success_claimed':False,'continuous_video_recorded':False}
    try:
        await web.TCPSite(relay,network['Gateway'],18084).start()
        fixture='Native binary collection qualification. This is a seeded input, not an agent output.\n'
        (staged/'qualification-input.txt').write_text(fixture)
        collection={'files':[{'name':'qualification-input.txt','source':rf'C:\Users\Qw3Test\Documents\{run_id}\input.txt'},
                             {'name':'missing-output.txt','source':rf'C:\Users\Qw3Test\Documents\{run_id}\missing.txt'}]}
        prepare('035',run_id,'Infrastructure qualification: observe the desktop once and finish this review. This is not a purchase-order solving task.',
                collection,args.bundle,staged,output/'launch',timeout=180,tool_limit=20,backend=f"http://{network['Gateway']}:18084/v1")
        setup=(f"$ErrorActionPreference='Stop'\n$root=Join-Path \"$env:USERPROFILE\\Documents\" '{run_id}'\n"
               "if (Test-Path $root) {throw 'Qualification root already exists'}\nNew-Item -ItemType Directory -Path $root | Out-Null\n"
               f"Copy-Item '\\\\host.lan\\Data\\{run_id}-inputs\\qualification-input.txt' \"$root\\input.txt\"\n"+
               (output/'launch/launch.ps1').read_text())
        (output/'launch-with-fixture.ps1').write_text(setup)
        event_task=asyncio.create_task(serve(site,share/(run_id+'-reviews'),output/'reviews',210,formal_run=False,model_used=False))
        process=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'apps/computer_use/vm/provision.py'),str(output/'launch-with-fixture.ps1'))
        if await process.wait()!=0:raise RuntimeError('Native qualification launch failed')
        result=await wait_json(share/(run_id+'-result.json'))
        (output/'native-result.json').write_text(json.dumps(result,indent=2))
        if not result['infrastructure_ok']:raise RuntimeError('Native launcher reported an infrastructure error')
        await event_task
        shutil.copytree(share/'native-results'/run_id,output/'guest')
        terminal=result['task'];reviews=terminal['completed_reviews']
        receipt=json.loads((output/'guest/artifacts/receipt.json').read_text())
        after=site.state();(output/'site-after.json').write_text(json.dumps(after,indent=2))
        _,plan=schedule();expected={e['messageId'] for group in plan['batches'] for e in group}
        present=[m['messageId'] for conversation in after['data']['teamchat']['messages'].values() for m in conversation if m.get('messageId') in expected]
        history=json.dumps(calls[-1]['messages'])
        checks={'four_reviews_same_thread':len(reviews)==4 and len({r['thread_id'] for r in reviews})==1,
                'four_distinct_turns':len({r['turn_id'] for r in reviews})==4,
                'lifetime_budget_retained':[r['tool_calls_used'] for r in reviews]==[1,2,3,4] and all(r['tool_call_limit']==20 for r in reviews),
                'three_public_updates_in_history':all(f'TeamChat update batch {i} of 3' in history for i in (1,2,3)),
                'previous_review_text_retained':all(f'Infrastructure review marker {i} complete.' in history for i in (1,2,3)),
                'desktop_observed_once_per_review':len(calls)==8,
                'all_original_events_once':len(present)==len(expected)==23 and set(present)==expected,
                'terminal_queue_continues':terminal['outcome']=='agent_completed' and terminal['queue_may_continue'],
                'native_export_preserves_fixture':sha256(output/'guest/artifacts/qualification-input.txt')==hashlib.sha256(fixture.encode()).hexdigest(),
                'missing_artifact_not_infrastructure':receipt['infrastructure_ok'] and [r['status'] for r in receipt['files']]==['copied','missing'],
                'credential_not_exported':not (output/'guest/model-token.txt').exists()}
        report.update(checks=checks,passed=all(checks.values()),native_bundle=json.loads((args.bundle/'archive.json').read_text())['sha256'])
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if event_task and not event_task.done():event_task.cancel()
        if event_task:await asyncio.gather(event_task,return_exceptions=True)
        await relay.cleanup();await backend_runner.cleanup()
        (output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args))['passed'] else 1)
