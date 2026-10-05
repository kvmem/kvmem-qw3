"""Actual Gedit native Save invocation through Desktop/HTTP/Gateway receipts.

This is a fixed diagnostic in a dedicated VM, not a QW3 task or OSWorld score.
The historical policy is evaluated read-only on the same aged capture. All
input delivery and file outcomes are checked independently, with no retries.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import time
import types

from qualify_native_keyboard_delivery import Probe as KeyboardProbe, owned_vm, guest, digest
from qw3_computer import desktop as desktop_module
from qw3_computer.execution import Execution
from qw3_computer.native_control_evidence import native_button
from qw3_computer.linux_driver import FOREGROUND_SCRIPT


TEXT='QW3 native button qualification'


class Probe(KeyboardProbe):
    def __init__(self,args):
        super().__init__(args)
        self.name='qw3-native-control-'+self.nonce[:12]
        self.destination='/home/user/qw3-native-control-'+self.nonce+'.txt'
        self.next_conditions=[dict(kind='foreground_title',text='Save As')]
        self.next_goal='Open Save As using the uniquely observed native button'
        self.actual_dispatches=[]
        self.record.pop('fixture_sha256',None)
        self.save(kind='actual_native_gedit_button_invocation_and_file_qualification',
                  recipe_sha256=digest(__file__),
                  fixture_scope='Fixed actual Gedit task in a dedicated VM. Native button, focus fault, durable receipts and independently saved file. No QW3 or OSWorld business score.',
                  fixed_setup_inputs=[],requested_file=self.destination)

    def verify_build(self):
        build=json.loads(self.args.candidate.joinpath('build.json').read_text())
        package=Path(desktop_module.__file__).resolve().parent
        if (build.get('kind')!='isolated_native_button_invocation_candidate'
                or package!=(self.args.candidate/'framework/qw3_computer').resolve()):
            raise ValueError('Select the private native-button candidate with PYTHONPATH')
        for name,expected in build['candidate_files_sha256'].items():
            if digest(self.args.candidate/'framework'/name)!=expected:raise ValueError('Candidate bytes changed')
        self.save(candidate_build_sha256=digest(self.args.candidate/'build.json'),
                  candidate_framework=str(package.parent),candidate_files_sha256=build['candidate_files_sha256'])

    def setup_key(self,*keys):
        owned_vm(self.name,self.cid)
        self.record['fixed_setup_inputs'].append(dict(kind='key',keys=list(keys),at=time.time()))
        self.save()
        return guest('import json,pyautogui;pyautogui.hotkey('+','.join(repr(k) for k in keys)+');print(json.dumps(dict(fixed_setup_key_sent=True)))')

    def launch_fixture(self):
        owned_vm(self.name,self.cid)
        result=guest("import json,subprocess;child=subprocess.Popen(['gedit','--standalone'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(json.dumps(dict(gedit_pid=child.pid)))")
        deadline=time.monotonic()+30
        while guest(FOREGROUND_SCRIPT)['process']!='Gedit':
            if time.monotonic()>=deadline:raise RuntimeError('Gedit did not become foreground')
            time.sleep(.2)
        self.record['fixed_setup_inputs'].append(dict(kind='type',text=TEXT,at=time.time()))
        self.save(gedit_launch=result)
        guest("import json,pyautogui;pyautogui.write("+repr(TEXT)+",interval=.02);print(json.dumps(dict(fixed_setup_edit=True)))")

    def focus(self,index):
        if index!=1:raise ValueError('Only the declared fixed focus-fault injector is supported')
        return self.setup_key('alt','f2')

    async def start_stack(self):
        await super().start_stack()
        original=self.driver.dispatch
        async def counted(action):
            row=dict(kind=action['kind'],started_at=time.time(),status='calling')
            self.actual_dispatches.append(row);self.save(actual_dispatches=self.actual_dispatches)
            try:
                result=await original(action)
                row.update(status='returned',receipt=result);self.save(actual_dispatches=self.actual_dispatches)
                return result
            except BaseException as error:
                row.update(status='raised',error_kind=type(error).__name__,error=str(error))
                self.save(actual_dispatches=self.actual_dispatches);raise
        self.driver.dispatch=counted

    def request(self,label,action):
        return dict(desktop_id=self.desktop.latest['desktop_id'],observation_id=self.desktop.latest['observation_id'],
                    action_id='native-control-'+label,action=action,
                    step=dict(goal=self.next_goal,conditions=self.next_conditions))

    def save_button(self,observation):
        matches=[c for c in observation['accessibility']['controls'] if c['name']=='Save' and native_button(c) is not None]
        if len(matches)!=1:raise RuntimeError('Exactly one observed native Save button required')
        return matches[0]

    async def send_save(self,label,**options):
        return await self.send(label,dict(kind='click_control',control_id=self.save_button(self.desktop.latest)['control_id']),**options)

    async def original_policy_rejection(self,initial):
        source=self.args.baseline/'qw3_computer/desktop.py'
        expected=json.loads(self.args.acceptance.read_text())['framework_files']['qw3_computer/desktop.py']
        if digest(source)!=expected:raise ValueError('Frozen original SDK51 source changed')
        module=types.ModuleType('qw3_computer._readonly_native_button_baseline')
        module.__package__='qw3_computer'
        exec(compile(source.read_bytes(),'exact-original-sdk51-desktop','exec'),module.__dict__)
        baseline=module.Desktop(self.desktop.endpoint,'unused',self.output/'baseline-policy')
        baseline.task_id='readonly-native-button-baseline';baseline.paused=False
        baseline.execution=Execution(baseline,self.output/'baseline-policy');baseline.remember(copy.deepcopy(initial))
        async def only_observe(method,route,body=None):
            if (method,route)!=('GET','/observe'):raise RuntimeError('Readonly baseline must never dispatch')
            return await self.desktop.request(method,route,body)
        baseline.request=only_observe
        request=dict(desktop_id=initial['desktop_id'],observation_id=initial['observation_id'],action_id='readonly-native-button-baseline',
                     action=dict(kind='click_control',control_id=self.save_button(initial)['control_id']),
                     step=dict(goal=self.next_goal,conditions=self.next_conditions))
        try:
            await baseline.act(request)
            raise RuntimeError('Original whole-content policy did not reject the diagnostic')
        except module.DesktopRejected as error:
            rows=baseline.db.execute('SELECT evidence FROM revalidations').fetchall()
            self.save(original_policy_rejection=dict(error=str(error),readonly_only=True,device_inputs=0,
                       revalidations=[json.loads(row[0]) for row in rows],source_sha256=expected))
        finally:
            baseline.execution.close();await baseline.close()

    def independent_file(self):
        owned_vm(self.name,self.cid)
        return guest('import json;from pathlib import Path;p=Path('+repr(self.destination)+');print(json.dumps(dict(exists=p.is_file(),text=p.read_text() if p.is_file() else None)))')

    async def tests(self):
        await self.start_stack();initial=await self.capture('initial')
        self.save(initial_captured_at=initial['captured_at'],status='waiting-for-real-observation-age')
        for second in range(55):
            await asyncio.sleep(1)
            if second%10==9:self.save(real_age_wait_elapsed_seconds=second+1)
        # An explicit fixed caret move reproduces the irrelevant editor pixel
        # change. It does not alter document text, button state or raw focus.
        self.setup_key('left')
        await self.original_policy_rejection(initial)
        self.save(status='testing-real-native-control-input')
        request,result=await self.send_save('aged-save')
        proof=result['observation_revalidation'];receipt=result['driver_receipt']
        self.record['checks'].update(aged_native_control_path_exercised=proof.get('native_control_valid') is True,
            unchanged_target_despite_changed_view=proof['same_view'] is False and proof['target_control_valid'] is True,
            actual_native_action_invoked_once=receipt['backend']=='atspi-native-button' and receipt['native_invocation']['invocation_accepted'] is True,
            actual_save_as_transition_verified=result['step_check']['verified'] is True)
        before=guest(FOREGROUND_SCRIPT);count=len(self.actual_dispatches)
        duplicate=await self.desktop.act(request);after=guest(FOREGROUND_SCRIPT)
        self.record['checks']['duplicate_returns_saved_receipt_without_second_input']=duplicate.get('duplicate') is True and len(self.actual_dispatches)==count and before==after
        self.next_goal='Cancel the observed file dialog and return to the unsaved document'
        self.next_conditions=[dict(kind='foreground_title',text='Untitled Document 1')]
        await self.capture('before-cancel');await self.send('cancel-dialog',dict(kind='key',key='ESC'))
        await self.capture('before-focus-fault')
        self.next_goal='Open Save As using the uniquely observed native button'
        self.next_conditions=[dict(kind='foreground_title',text='Save As')]
        self.fault='after-preflight'
        _,fault_result=await self.send_save('focus-fault',reject=True)
        self.fault=None
        self.record['checks']['actual_focus_fault_rejected_before_native_invocation']=fault_result is None and not self.independent_file()['exists'] and self.actual_dispatches[-1]['status']=='raised'
        self.setup_key('esc');await self.capture('after-focus-recovery')
        # The button intent was already reserved for its earlier successful
        # transition and the confirmed focus rejection. Preserve the repeat
        # limit; use an observed, allowed alternative shortcut after recovery.
        _,result=await self.send('recovered-save-shortcut',dict(kind='key',key='CTRL+S'))
        self.record['checks']['fresh_observation_recovers_via_alternative_shortcut']=result['driver_receipt']['backend']=='pyautogui' and result['step_check']['verified'] is True
        await self.capture('save-as-controls')
        self.next_goal='Expose the file chooser location input'
        self.next_conditions=[dict(kind='foreground_title',text='Save As')]
        await self.send('location',dict(kind='key',key='CTRL+L'))
        self.next_goal='Enter the declared unique file path'
        await self.capture('before-location-type');await self.send('location-type',dict(kind='type',text=self.destination))
        self.next_goal='Navigate to the declared file destination'
        await self.capture('before-location-enter');await self.send('location-enter',dict(kind='key',key='ENTER'))
        if self.independent_file()['exists']:
            self.record['checks']['file_created_by_location_enter']=True
        else:
            self.next_goal='Save the document and return to its named window'
            self.next_conditions=[dict(kind='foreground_title',text=Path(self.destination).name)]
            await self.capture('before-final-save');await self.send_save('final-save')
        deadline=time.monotonic()+3
        final=self.independent_file()
        while not final['exists'] and time.monotonic()<deadline:
            await asyncio.sleep(.2);final=self.independent_file()
        self.record['checks']['saved_file_independently_correct']=final['exists'] and final['text']==TEXT+'\n'
        self.record['checks']['no_uncertain_inputs']=self.desktop.unresolved()==[] and not self.gateway.paused
        counts=dict(self.gateway.db.execute('SELECT status,count(*) FROM actions GROUP BY status').fetchall())
        delivered=sum(row['status']=='completed' for row in self.record['inputs'])
        rejected=sum(row['status']=='rejected' for row in self.record['inputs'])
        self.record['checks']['durable_receipts_match_actual_requests']=counts==dict(completed=delivered,rejected=rejected)
        self.record['checks']['unobserved_business_completion_not_claimed']=self.execution.completion() is None
        self.save(final_file_verification=final,gateway_receipt_counts=counts)
        await self.capture('final')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--acceptance',type=Path,required=True)
    args=p.parse_args();d=asyncio.run(Probe(args).run())
    print(json.dumps({key:d.get(key) for key in ('status','passed','checks','gateway_receipt_counts','dedicated_vm_retired','video_collection_passed','error')}))
    if not d.get('passed'):raise SystemExit(1)
