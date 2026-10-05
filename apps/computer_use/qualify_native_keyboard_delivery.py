"""Real input/receipt qualification in one dedicated, owned Linux VM.

Original GUI tasks and running devices are untouched. Inputs go through the
private candidate's actual Desktop -> HTTP Gateway -> driver -> guest path.
The GUI is a fixed diagnostic fixture; no QW3/business score is claimed.
"""
import argparse
import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import time
import types
from unittest.mock import patch
import uuid

import requests
from aiohttp import web

from bounded_linux_observation import BoundedOfficeLinuxDriver
from osworld_linux import environment
from osworld_windows.source import ROOT
from qw3_computer import desktop as desktop_module
from qw3_computer.desktop import Desktop, DesktopRejected
from qw3_computer.device_gateway import Gateway, create_app
from qw3_computer.device_lease import DeviceLease
from qw3_computer.execution import Execution
import recoverable_linux_recording as recording


PORTS=dict(server=18960,chromium=18961,vnc=18962,vlc=18963)
GATEWAY_PORT=18964
ENDPOINT='http://127.0.0.1:'+str(PORTS['server'])
AGE_SECONDS=55
INITIAL_TEXT='QW3 native café 测试'
RECOVERY_TEXT='恢复正常 — QW3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_vm(name,cid):
    values=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=3))
    if (len(values)!=1 or values[0]['Id']!=cid or values[0]['Name']!='/'+name
            or values[0]['Image']!=environment.IMAGE or not values[0]['State']['Running']
            or values[0]['Config'].get('Labels',{}).get('qw3.goal')!='cross-device-20261003'):
        raise RuntimeError('Dedicated input VM identity changed')
    return values[0]


def guest(code,seconds=8):
    with requests.Session() as session:
        session.trust_env=False
        response=session.post(ENDPOINT+'/execute',json=dict(command=['python','-c',code],shell=False,timeout=seconds),timeout=seconds+2)
    response.raise_for_status();result=response.json()
    if result.get('returncode',0)!=0 or result.get('status')=='error':
        raise RuntimeError('Owned fixture operation failed: '+str(result.get('error','')))
    return json.loads(result['output'])


def checked_fixture_source(directory,expected,nonce):
    """Literal fixed read, bound to the exact fixture generation and nonce."""
    return f'''import json,os
from pathlib import Path
expected={expected!r};process=Path('/proc')/str(expected['pid'])
assert process.stat().st_uid==expected['uid']
assert int((process/'stat').read_text().rpartition(')')[2].split()[19])==expected['start_ticks']
assert [v.decode() for v in (process/'cmdline').read_bytes().split(b'\\0') if v]==expected['command']
assert os.readlink(process/'cwd')==expected['cwd'] and os.readlink(process/'exe')==expected['executable']
state=json.loads((Path({str(directory)!r})/'state.json').read_text())
assert state['nonce']=={nonce!r} and state['identity']==expected
'''


def focus_source(directory,expected,nonce,index):
    if index not in (0,1):raise ValueError('Only an observed owned fixture entry is supported')
    return checked_fixture_source(directory,expected,nonce)+f'''
from Xlib import X
from Xlib.display import Display
connection=Display();root=connection.screen().root
value=root.get_full_property(connection.intern_atom('_NET_ACTIVE_WINDOW',only_if_exists=True),X.AnyPropertyType)
assert value is not None and len(value.value)==1 and int(value.value[0])==state['native_window_id']
target=connection.create_resource_object('window',state['entry_window_ids'][{index}]);current=target
for _ in range(16):
 if current.id==state['native_window_id']:break
 current=current.query_tree().parent
else:raise RuntimeError('Owned entry is not within the observed active fixture window')
target.set_input_focus(X.RevertToParent,X.CurrentTime);connection.sync()
assert connection.get_input_focus().focus.id==target.id
connection.close()
print(json.dumps(dict(input_focus_changed=True,keyboard_input_sent=False,target_window_id=target.id)))
'''


class Probe:
    def __init__(self,args):
        self.args=args;self.output=args.output.resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.nonce=uuid.uuid4().hex;self.name='qw3-native-keyboard-'+self.nonce[:12]
        self.env=self.recorder=self.driver=self.gateway=self.runner=self.desktop=self.execution=self.lease=None
        self.cid=self.fixture_identity=self.fixture_directory=None
        self.fault=None
        self.record=dict(kind='actual_native_keyboard_and_focus_fault_qualification',status='preparing',passed=False,
            started_at=time.time(),producer=recording.owned.process_identity(os.getpid()),
            model_requests=0,real_model_retest=False,business_success_claimed=False,
            existing_devices_or_input_owners_changed=False,inputs=[],checks={},
            fixture_scope='Controlled native GUI, exact candidate framework and real HTTP/SQLite/input receipts. Not an OSWorld business score.',
            age_wait_seconds=AGE_SECONDS,recipe_sha256=digest(__file__),fixture_sha256=digest(Path(__file__).with_name('native_keyboard_fixture.py')))
        self.save()

    def save(self,**values):
        self.record.update(values)
        temporary=self.output/'state.tmp';temporary.write_text(json.dumps(self.record,ensure_ascii=False,indent=2)+'\n');temporary.replace(self.output/'result.json')

    def fixture_state(self):
        owned_vm(self.name,self.cid)
        return guest(checked_fixture_source(self.fixture_directory,self.fixture_identity,self.nonce)+'\nprint(json.dumps(state,ensure_ascii=False))')

    def focus(self,index):
        return guest(focus_source(self.fixture_directory,self.fixture_identity,self.nonce,index))

    async def wait_values(self,expected,seconds=3):
        deadline=time.monotonic()+seconds
        while True:
            state=self.fixture_state()
            if state['values']==expected:return state
            if time.monotonic()>=deadline:raise RuntimeError('Native entry values did not verify; no input retry')
            await asyncio.sleep(.1)

    async def capture(self,name):
        result=await self.desktop.observe()
        metadata={key:value for key,value in result.items() if key!='image'}
        (self.output/(name+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        (self.output/(name+'.png')).write_bytes(base64.b64decode(result['image']))
        if result.get('native_window_capture',{}).get('screenshot_bracket_matched') is not True:
            raise RuntimeError('Actual fixture screenshot lacks a positive native bracket')
        return result

    def request(self,label,action):
        return dict(desktop_id=self.desktop.latest['desktop_id'],observation_id=self.desktop.latest['observation_id'],
            action_id='native-'+label,action=action,
            step=dict(goal='Qualification input; independent fixture state must verify the text',
                      conditions=[dict(kind='control_value',text='Input A',value=RECOVERY_TEXT)]))

    async def send(self,label,action,*,reject=False):
        request=self.request(label,action)
        row=dict(label=label,action_kind=action['kind'],request=request,status='pending',started_at=time.time())
        self.record['inputs'].append(row);self.save()
        try:
            result=await self.desktop.act(request)
            row.update(status='completed',result={k:v for k,v in result.items() if k!='observation'},
                       observation_id=result['observation']['observation_id'])
            if reject:raise RuntimeError('Faulted input was delivered instead of rejected')
            self.save();return request,result
        except DesktopRejected as error:
            row.update(status='rejected',error=str(error));self.save()
            if not reject:raise
            return request,None
        except BaseException as error:
            row.update(error_kind=type(error).__name__,error=str(error));self.save();raise

    def verify_build(self):
        build=json.loads(self.args.candidate.joinpath('build.json').read_text())
        package=Path(desktop_module.__file__).resolve().parent
        if (build.get('kind')!='isolated_sdk51_native_keyboard_execution_candidate'
                or package!=(self.args.candidate/'framework/qw3_computer').resolve()):
            raise ValueError('Select the declared private candidate with PYTHONPATH')
        for name,expected in build['candidate_files_sha256'].items():
            if digest(self.args.candidate/'framework'/name)!=expected:raise ValueError('Candidate bytes changed')
        self.save(candidate_build_sha256=digest(self.args.candidate/'build.json'),candidate_framework=str(package.parent),
                  candidate_files_sha256=build['candidate_files_sha256'])

    async def start_stack(self):
        self.driver=BoundedOfficeLinuxDriver(ENDPOINT,'native-keyboard-'+self.nonce)
        await self.driver.start()
        original_dispatch=self.driver.dispatch;original_execute=self.driver.execute

        async def dispatch(action):
            if self.fault=='after-preflight':
                self.focus(1)
            return await original_dispatch(action)

        async def execute(code,*,read_timeout=None):
            if self.fault=='after-clipboard':
                code='import os\nos.environ["PATH"]='+repr(self.wrapper_directory+':')+"+os.environ.get('PATH','')\n"+code
            return await original_execute(code,read_timeout=read_timeout)

        # Fixed qualification fault injectors only; candidate files stay intact.
        self.driver.dispatch=dispatch;self.driver.execute=execute
        self.gateway=Gateway(self.driver,self.output/'gateway')
        token=secrets.token_hex(32)
        self.runner=web.AppRunner(create_app(self.gateway,token),handler_cancellation=True)
        await self.runner.setup();await web.TCPSite(self.runner,'127.0.0.1',GATEWAY_PORT).start()
        self.desktop=Desktop('http://127.0.0.1:'+str(GATEWAY_PORT),token,self.output/'desktop')
        self.desktop.task_id='native-qualification-'+self.nonce
        await self.desktop.start();await self.desktop.control('resume')
        self.execution=Execution(self.desktop,self.output/'desktop');self.desktop.execution=self.execution
        self.save(gateway_session_id=self.gateway.session_id,status='input-stack-ready')

    def launch_fixture(self):
        directory='/home/user/.qw3-native-keyboard-'+self.nonce
        self.fixture_directory=directory+'/evidence'
        data=Path(__file__).with_name('native_keyboard_fixture.py').read_bytes()
        code=f'''import base64,json,os,subprocess,time
from pathlib import Path
p=Path({directory!r});p.mkdir(exist_ok=False);source=p/'fixture.py';source.write_bytes(base64.b64decode({base64.b64encode(data).decode()!r}))
log=(p/'fixture.log').open('xb')
child=subprocess.Popen(['python',str(source),'--output',{self.fixture_directory!r},'--nonce',{self.nonce!r}],cwd=p,stdout=log,stderr=log)
deadline=time.monotonic()+30
while not (Path({self.fixture_directory!r})/'state.json').is_file():
 if child.poll() is not None or time.monotonic()>=deadline:
  print(json.dumps(dict(fixture_startup_failed=True,returncode=child.poll(),log=(p/'fixture.log').read_text())))
  raise SystemExit(0)
 time.sleep(.1)
print((Path({self.fixture_directory!r})/'state.json').read_text())'''
        (self.output/'guest-fixture-launch.py').write_text(code)
        state=guest(code,seconds=35)
        if state.get('fixture_startup_failed'):
            self.save(fixture_startup_failure=state)
            raise RuntimeError('Owned fixture failed before input: '+state['log'][-2000:])
        self.fixture_identity=state['identity']
        self.save(fixture_identity=self.fixture_identity,fixture_initial_state=state)
        self.wrapper_directory=directory+'/clipboard-fault'
        wrapper=('#!/usr/bin/env python\nimport os,subprocess,sys\n'
            "subprocess.run(['/usr/bin/xclip']+sys.argv[1:],input=sys.stdin.buffer.read(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=3)\n"
            +focus_source(self.fixture_directory,self.fixture_identity,self.nonce,1).replace('print(json.dumps(dict(input_focus_changed=True,keyboard_input_sent=False,target_window_id=target.id)))',''))
        guest(f"from pathlib import Path;import json;p=Path({self.wrapper_directory!r});p.mkdir();f=p/'xclip';f.write_text({wrapper!r});f.chmod(0o700);print(json.dumps(dict(wrapper_created=True)))")
        self.focus(0)

    async def original_policy_rejection(self,initial):
        source=self.args.baseline/'qw3_computer/desktop.py'
        expected=json.loads(self.args.acceptance.read_text())['framework_files']['qw3_computer/desktop.py']
        if digest(source)!=expected:raise ValueError('Original SDK51 desktop source changed')
        module=types.ModuleType('qw3_computer._readonly_native_keyboard_baseline')
        module.__package__='qw3_computer';exec(compile(source.read_bytes(),'exact-original-sdk51-desktop','exec'),module.__dict__)
        baseline=module.Desktop('http://127.0.0.1:'+str(GATEWAY_PORT),'not-used',self.output/'baseline-policy')
        baseline.task_id='readonly-native-baseline';baseline.paused=False
        baseline.execution=Execution(baseline,self.output/'baseline-policy')
        baseline.remember(copy.deepcopy(initial))

        async def only_observe(method,route,body=None):
            if (method,route)!=('GET','/observe'):raise RuntimeError('Readonly baseline may never send input')
            return await self.desktop.request(method,route,body)
        baseline.request=only_observe
        request=dict(desktop_id=initial['desktop_id'],observation_id=initial['observation_id'],action_id='readonly-baseline',
                     action=dict(kind='type',text=INITIAL_TEXT),
                     step=dict(goal='Diagnostic input',conditions=[dict(kind='control_value',text='Input A',value=RECOVERY_TEXT)]))
        try:
            await baseline.act(request)
            raise RuntimeError('Original policy did not reject the clock-changed fixture')
        except module.DesktopRejected as error:
            evidence=baseline.db.execute('SELECT evidence FROM revalidations').fetchall()
            self.save(original_policy_rejection=dict(error=str(error),revalidations=[json.loads(row[0]) for row in evidence],
                readonly_only=True,device_inputs=0,source_sha256=expected))
        finally:
            baseline.execution.close();await baseline.close()

    async def tests(self):
        await self.start_stack()
        initial=await self.capture('initial')
        self.save(status='waiting-for-real-observation-age',initial_captured_at=initial['captured_at'])
        # The wait is real, not a rewritten timestamp. The fixture's separate
        # clock changes outside the unchanged client; its entry caret is static.
        for second in range(AGE_SECONDS):
            await asyncio.sleep(1)
            if second%10==9:self.save(real_age_wait_elapsed_seconds=second+1)
        await self.original_policy_rejection(initial)
        self.save(status='testing-actual-input-and-receipts')
        request,result=await self.send('unicode',dict(kind='type',text=INITIAL_TEXT))
        state=await self.wait_values([INITIAL_TEXT,''])
        evidence=result['observation_revalidation']
        self.record['checks'].update(aged_native_keyboard_path_exercised=evidence.get('native_keyboard_context_unchanged') is True,
            whole_screenshot_changed=evidence['same_view'] is False,actual_unicode_paste_correct=True)
        count=len(state['key_events']);duplicate=await self.desktop.act(request);after=self.fixture_state()
        self.record['checks']['duplicate_returns_saved_receipt_without_input']=duplicate.get('duplicate') is True and after['values']==state['values'] and len(after['key_events'])==count
        await self.capture('before-home');await self.send('home',dict(kind='key',key='HOME'))
        await asyncio.sleep(.2);state=self.fixture_state()
        self.record['checks']['home_key_moves_native_entry_cursor']=state['insert_indices'][0]==0
        await self.capture('before-focus-fault');before=self.fixture_state();self.fault='after-preflight'
        _,result=await self.send('focus-fault',dict(kind='key',key='ENTER'),reject=True)
        self.fault=None;after=self.fixture_state()
        self.record['checks']['focus_fault_rejected_without_key_or_text']=result is None and after['values']==before['values'] and len(after['key_events'])==len(before['key_events'])
        self.focus(0);await self.capture('before-clipboard-fault');before=self.fixture_state();self.fault='after-clipboard'
        _,result=await self.send('clipboard-fault',dict(kind='type',text='This text must never be inserted'),reject=True)
        self.fault=None;after=self.fixture_state()
        self.record['checks']['clipboard_focus_fault_prevents_paste']=result is None and after['values']==before['values'] and len(after['key_events'])==len(before['key_events'])
        self.focus(0);await self.capture('before-recovery-home');await self.send('recovery-home',dict(kind='key',key='HOME'))
        await self.capture('before-recovery-select');await self.send('recovery-select',dict(kind='key',key='SHIFT+END'))
        await asyncio.sleep(.3);selected=self.fixture_state()
        expected_selection=dict(first=0,last=len(INITIAL_TEXT),text=INITIAL_TEXT)
        self.record['checks']['recovery_selection_verified_before_paste']=selected['selections']==[expected_selection,None]
        self.save(recovery_selection_state=selected)
        if not self.record['checks']['recovery_selection_verified_before_paste']:
            raise RuntimeError('Selection did not cover the initial text; no recovery paste sent')
        # Native widgets may paste beside a selection. Delete explicitly and
        # verify the empty field before sending new text; do not assume that
        # a successfully delivered shortcut establishes its intended result.
        await self.capture('before-recovery-delete');await self.send('recovery-delete',dict(kind='key',key='DELETE'))
        cleared=await self.wait_values(['',''])
        self.record['checks']['recovery_empty_field_verified_before_paste']=True
        self.save(recovery_cleared_state=cleared)
        await self.capture('before-recovery-type');await self.send('recovery-type',dict(kind='type',text=RECOVERY_TEXT))
        final=await self.wait_values([RECOVERY_TEXT,'']);await self.capture('final')
        self.record['checks']['fresh_capture_recovers_and_pastes_correctly']=True
        self.record['checks']['delivery_unknowns_zero']=self.desktop.unresolved()==[] and not self.gateway.paused
        rows=self.gateway.db.execute('SELECT status,count(*) FROM actions GROUP BY status').fetchall()
        self.record['checks']['durable_gateway_receipts_match_inputs']=dict(rows)=={'completed':6,'rejected':2}
        self.record['checks']['unobserved_completion_not_claimed']=self.execution.completion() is None
        self.save(final_fixture_state=final,gateway_receipt_counts=dict(rows))

    async def run(self):
        try:
            self.verify_build()
            if shutil.disk_usage(self.output).free<50*1024**3:raise RuntimeError('Less than50GiB free for dedicated VM')
            with subprocess.Popen(['docker','image','inspect',environment.IMAGE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) as child:
                if child.wait(timeout=5)!=0:raise RuntimeError('Existing pinned VM image unavailable; no pull allowed')
            for port in [*PORTS.values(),GATEWAY_PORT]:
                with socket.socket() as sock:sock.bind(('127.0.0.1',port))
            self.lease=DeviceLease(ENDPOINT+':desktop').acquire()
            seed=environment.DISK.stat();self.seed_before=(seed.st_dev,seed.st_ino,seed.st_size,seed.st_mtime_ns)
            self.save(status='booting-dedicated-vm',container_name=self.name,ports=PORTS,environment_source_sha256=digest(environment.__file__))
            print('Booting the dedicated native input VM',flush=True)
            with patch.object(environment,'PORTS',PORTS):self.env=environment.create(self.output,self.name)
            self.cid=json.loads((self.output/'vm.json').read_text())['container_id']
            info=owned_vm(self.name,self.cid)
            mounts=[v for v in info['Mounts'] if v['Destination']=='/System.qcow2']
            if len(mounts)!=1 or mounts[0]['RW'] or mounts[0]['Source']!=str(environment.DISK):raise RuntimeError('VM seed is not the declared read-only disk')
            self.save(container_id=self.cid,seed_read_only_mount_verified=True,status='starting-recorder')
            self.recorder=recording.RecoverableLinuxRecording(self.output,self.cid,name=self.name,
                profile=recording.RecordingProfile(duration_seconds=240))
            self.recorder.start();self.launch_fixture();await self.tests()
            self.record['passed']=all(self.record['checks'].values())
            self.save(status='passed' if self.record['passed'] else 'failed')
        except BaseException as error:
            self.save(status='failed',passed=False,error_kind=type(error).__name__,error=str(error))
        finally:
            self.fault=None
            if self.env is not None and self.fixture_directory is not None:
                try:
                    owned_vm(self.name,self.cid)
                    raw=guest(f"import base64,json;from pathlib import Path;p=Path({str(Path(self.fixture_directory).parent)!r});print(json.dumps({{name:base64.b64encode((p/name).read_bytes()).decode() for name in ['fixture.log','evidence/state.json'] if (p/name).is_file()}}))")
                    for name,data in raw.items():
                        if name not in {'fixture.log','evidence/state.json'}:raise RuntimeError('Unexpected fixture evidence')
                        (self.output/('guest-'+name.replace('/','-'))).write_bytes(base64.b64decode(data,validate=True))
                    self.save(guest_fixture_logs_collected=list(raw))
                    diagnostic=guest("import json,subprocess;print(json.dumps(dict(active=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True),clients=subprocess.check_output(['wmctrl','-lp'],text=True))))")
                    self.save(final_native_window_diagnostic=diagnostic)
                except Exception as error:self.save(guest_fixture_log_error=str(error))
            if self.desktop is not None:
                try:await self.desktop.control('pause');await self.desktop.close()
                except Exception as error:self.save(desktop_cleanup_error=str(error),passed=False)
            if self.execution is not None:self.execution.close()
            if self.runner is not None:await self.runner.cleanup()
            if self.gateway is not None:await self.gateway.close()
            elif self.driver is not None:await self.driver.close()
            if self.recorder is not None:
                try:
                    receipt=self.recorder.finish('continuous.mp4' if self.record.get('passed') else 'interrupted.mp4')
                    self.save(recording=receipt,video_collection_passed=True)
                except Exception as error:self.save(recording_error=str(error),video_collection_passed=False,passed=False)
            if self.env is not None:
                try:
                    owned_vm(self.name,self.cid);self.env.close()
                    retired=subprocess.run(['docker','inspect',self.cid],capture_output=True,timeout=5).returncode!=0
                    self.save(dedicated_vm_retired=retired)
                    if not retired:self.record['passed']=False
                except Exception as error:self.save(vm_cleanup_error=str(error),dedicated_vm_retired=False,passed=False)
            if self.lease is not None:self.lease.close()
            if hasattr(self,'seed_before'):
                stat=environment.DISK.stat();unchanged=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==self.seed_before
                self.save(seed_metadata_unchanged=unchanged)
                if not unchanged:self.record['passed']=False
            self.save(finished_at=time.time())
        return self.record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,required=True)
    args=parser.parse_args();result=asyncio.run(Probe(args).run())
    print(json.dumps({key:result.get(key) for key in ('status','passed','checks','gateway_receipt_counts','dedicated_vm_retired','video_collection_passed')}))
    if not result.get('passed'):raise SystemExit(1)


if __name__=='__main__':main()
