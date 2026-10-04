import asyncio
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from PIL import Image

from receipt_model_pilot import recovery_checks, run
from osworld_windows.source import ROOT
from test_bridge import chunk, serve


class ReceiptModelAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.before = {'id':'task-one','status':'paused','desktop_paused':True,'turn_status':'interrupted',
            'tool_calls_used':7,'tool_call_limit':100,'uncertain_actions':[{'receipt_id':'original'}]}
        self.after = self.before | {'uncertain_actions':[]}
        self.fault = {'selected_action_id':'original','events':[{'event':'completed_receipt_withheld'}],
            'dispatch_counts':{'focus-name':1,'original':1},'receipt_queries':['original']}
        self.evidence = dict(backend_calls_before=4, backend_calls_after=4,
            ledger_receipt_id='original', ledger_status='completed',main_idle=True,
            dispatch_counts_before=dict(self.fault['dispatch_counts']))

    def check(self):
        return recovery_checks(self.before,self.after,self.fault,**self.evidence)

    def test_safe_restart_and_original_receipt_query_is_accepted(self):
        self.assertTrue(all(self.check().values()))

    def test_second_input_or_a_new_intent_during_recovery_is_rejected(self):
        for dispatches in [{'focus-name':1,'original':2}, {'focus-name':1,'original':1,'new-input':1}]:
            with self.subTest(dispatches=dispatches):
                self.fault['dispatch_counts']=dispatches
                self.assertFalse(all(self.check().values()))

    def test_restart_cannot_grant_budget_change_task_or_resume_inference(self):
        for changed in [{'tool_calls_used':0},{'tool_call_limit':101},{'id':'new-task'},
                        {'status':'running'},{'desktop_paused':False},{'turn_status':'inProgress'}]:
            with self.subTest(changed=changed):
                self.after = self.before | {'uncertain_actions':[]} | changed
                self.assertFalse(all(self.check().values()))

    def test_unresolved_wrong_receipt_or_extra_model_call_is_rejected(self):
        variants = [({'uncertain_actions':[{'receipt_id':'original'}]}, {}),
                    ({}, {'ledger_receipt_id':'another'}), ({}, {'ledger_status':'uncertain'}),
                    ({}, {'backend_calls_after':5}), ({}, {'main_idle':False})]
        for state, evidence in variants:
            with self.subTest(state=state,evidence=evidence):
                before = copy.deepcopy(self.after); prior = dict(self.evidence)
                self.after.update(state); self.evidence.update(evidence)
                self.assertFalse(all(self.check().values()))
                self.after = before; self.evidence = prior

    def test_unexercised_fault_or_unconfirmed_model_stop_cannot_pass(self):
        self.fault['events']=[]
        self.assertFalse(all(self.check().values()))
        self.fault['events']=[{'event':'completed_receipt_withheld'}]
        self.before['turn_status']='inProgress'
        self.assertFalse(all(self.check().values()))


@unittest.skipUnless(os.environ.get('QW3_TEST_CODEX') == '1', 'Actual Codex integration is opt-in')
class ReceiptHarnessIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_codex_fault_timeout_restart_and_query_without_model_or_vm(self):
        """Actual driver/transport/client, deterministic backend, fixture media."""
        from types import SimpleNamespace
        sdk=ROOT/'apps/computer_use/.runtime/windows-native-client-49'
        if not (sdk/'qualification.json').exists(): self.skipTest('Qualified candidate49 bundle is required')
        receipts={}; effects=[]; current={}; sequence=0; paused=True; backend_calls=[]
        pixels=io.BytesIO();Image.new('RGB',(128,96),'blue').save(pixels,format='PNG');png=pixels.getvalue()
        def observe():
            nonlocal sequence,current
            sequence+=1
            current={'desktop_id':'fixture-linux:session','observation_id':str(sequence),
                'width':128,'height':96,'image':base64.b64encode(png).decode(),
                'sha256':hashlib.sha256(png).hexdigest(),'paused':paused,
                'foreground':{'handle':'one','title':'Fixture form','process':'fixture'},
                'accessibility':{'status':'ok','truncated':False,'controls':[]}}
            return current
        async def gateway(request):
            nonlocal paused
            self.assertEqual(request.headers.get('Authorization'),'Bearer fixture-token')
            if request.path=='/observe': return web.json_response(observe())
            if request.path=='/control':
                paused=(await request.json())['command']!='resume';return web.json_response({'paused':paused})
            if request.path.startswith('/actions/'):
                return web.json_response(receipts.get(request.path.rsplit('/',1)[1],{'status':'unknown'}))
            if request.path=='/act':
                body=await request.json();effects.append(body['action_id'])
                result={'action_id':body['action_id'],'status':'completed','observation':observe()}
                receipts[body['action_id']]={'status':'completed','result':result}
                return web.json_response(result)
            return web.json_response({'files':[]})
        async def backend(request):
            if request.path=='/v1/models': return web.json_response({'data':[{'context_window':524288}]})
            body=await request.json();backend_calls.append(body)
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
            name=next(t['function']['name'] for t in body['tools'] if t['function']['name'].endswith('__act'))
            arguments={'desktop_id':current['desktop_id'],'observation_id':current['observation_id'],
                'action_id':'first-type','action':{'kind':'type','text':'Requested value'},
                'step':{'goal':'Fill the requested visible field','conditions':[{'kind':'screen_changed'}]}}
            await chunk(response,{'tool_calls':[{'index':0,'id':'fixture-call','type':'function',
                'function':{'name':name,'arguments':json.dumps(arguments)}}]})
            await chunk(response,finish='tool_calls');await response.write(b'data: [DONE]\n\n');return response
        selected='linux-vm'
        def main_state(): return {'status':'idle','desktop_paused':True,'uncertain_actions':[],
            'thinking':True,'model_output_tokens':16384,'tool_call_limit':100,'devices':{'selected':selected}}
        async def main_api(request):
            nonlocal selected
            if request.path=='/api/device': selected=(await request.json())['device_id']
            return web.json_response(main_state())
        apps=[]
        for handler in (gateway,backend,main_api):
            app=web.Application();app.router.add_route('*','/{path:.*}',handler);apps.append(await serve(app))
        try:
            with tempfile.TemporaryDirectory(prefix='qw3-receipt-harness-test-') as tmp:
                directory=Path(tmp);runtime=directory/'main';runtime.mkdir()
                shutil.copytree(sdk/'app/qw3_computer',runtime/'framework/qw3_computer',ignore=shutil.ignore_patterns('__pycache__'))
                files={str(p.relative_to(runtime/'framework/qw3_computer')):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (runtime/'framework/qw3_computer').rglob('*') if p.is_file()}
                (runtime/'candidate-manifest.json').write_text(json.dumps({'files':files}))
                token=directory/'token.txt';token.write_text('fixture-token')
                registry=directory/'registry.json';registry.write_text(json.dumps({'devices':[
                    {'id':'linux-vm','label':'Fixture','platform':'linux','endpoint':apps[0][1],'token_file':str(token)},
                    {'id':'windows-vm','label':'Other fixture','platform':'windows','endpoint':'http://127.0.0.1:1','token_file':str(token)}]}))
                class FixtureRecording:
                    def __init__(self,output,*_args):self.output=output;self.collected=False
                    def start(self):pass
                    def finish(self,name='continuous.mp4'):
                        self.collected=True;(self.output/name).write_bytes(b'fixture media only');return {'complete':True}
                original_popen=subprocess.Popen
                original_open=open
                def isolated_locks(path,*args,**kwargs):
                    if str(path) in {'/tmp/qw3-cross-device-model-queue.lock','/tmp/qw3-osworld-vm.lock'}:
                        path=directory/Path(path).name
                    return original_open(path,*args,**kwargs)
                def fixture_backend(argv,*args,**kwargs):
                    if 'qw3_computer.app' in argv and '--backend' not in argv:
                        argv+=['--backend',apps[1][1]+'/v1']
                    return original_popen(argv,*args,**kwargs)
                options=SimpleNamespace(app=apps[2][1],runtime=runtime,registry=registry,device='linux-vm',
                    bundle=sdk,output=directory/'receipt-fixture-test',seconds=60,record_gpu=False)
                with patch('receipt_model_pilot.owned_container',return_value={'Id':'fixture','State':{'Running':True}}), \
                     patch('receipt_model_pilot.linux_execute',return_value={}), \
                     patch('browser_pilot.initialize',return_value=({'url':'http://fixture.invalid'},'Fill one visible form field.')), \
                     patch('browser_pilot.grade',return_value={'passed':False}),patch('browser_pilot.stop',return_value={}), \
                     patch('linux_recording.LinuxRecording',FixtureRecording), \
                     patch('receipt_model_pilot.open',side_effect=isolated_locks,create=True), \
                     patch('receipt_model_pilot.subprocess.Popen',side_effect=fixture_backend):
                    outcome=await asyncio.wait_for(run(options),120)
                self.assertTrue(outcome['fault_accepted'],outcome)
                self.assertFalse(outcome['accepted'])
                self.assertTrue(outcome['owned_task_retired'])
                self.assertEqual(len(effects),1)
                self.assertEqual(len(backend_calls),1)
                self.assertTrue(paused)
        finally:
            for runner,_ in apps:await runner.cleanup()
