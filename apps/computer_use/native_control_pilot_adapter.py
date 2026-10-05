"""Bind unchanged development pilots to one explicitly owned private device.

Only administrative transport, ownership checks and the recorder binding are
adapted in this child process. Prompts, deadlines, graders, model instructions,
tool budgets and frozen pilot source files are unchanged. The text initializer
skips its broad process-name signal only after a positive empty-process read.
"""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from unittest.mock import patch

import execution_retest_queue as queue
import kvmem_osworld_queue as owned
import linux_recording
from osworld_linux import environment
import platform_pilots as pilot
from qw3_computer import desktop as desktop_module
from qw3_computer.device_registry import load_registry

ORIGINAL_CONTAINER = pilot.owned_container


RESET_SOURCE = ("import subprocess,time; subprocess.run(['pkill','-TERM','-x','gedit'],check=False); time.sleep(1); "
                "subprocess.Popen(['gedit'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)")
EMPTY_PROCESS_READ = '''import json,os
from pathlib import Path
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  name=(p/'comm').read_text().strip()
  command=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  if name=='gedit' or command and Path(command[0]).name=='gedit':
   matches.append(dict(pid=int(p.name),uid=p.stat().st_uid,
       start_ticks=int((p/'stat').read_text().rpartition(')')[2].split()[19]),command=command))
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(gedit_processes=matches)))
'''


def checked_ref(record):
    if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
            or not re.fullmatch('[a-f0-9]{64}', record.get('sha256', ''))):
        raise owned.SafetyHalt('Require an exact frozen source reference')
    path=Path(record['path']).resolve()
    if owned.digest(path)!=record['sha256']:
        raise owned.SafetyHalt('Frozen pilot dependency changed')
    return path


def endpoint(value):
    parsed=urlparse(value)
    if (parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.port is None
            or parsed.path not in ('','/') or parsed.username or parsed.query or parsed.fragment):
        raise owned.SafetyHalt('Require the explicit owned loopback endpoint')
    return 'http://127.0.0.1:'+str(parsed.port)


def option(command, name):
    if command.count(name) != 1 or command.index(name) + 1 >= len(command):
        raise owned.SafetyHalt('Require exactly one explicit service option: '+name)
    return command[command.index(name)+1]


def validate_study(path, trial_id):
    study=json.loads(Path(path).read_text())
    if (study.get('kind')!='prepared_native_button_matched_qw3_model_retest'
            or study.get('status')!='prepared-not-registered-not-dispatched'):
        raise owned.SafetyHalt('Require the immutable prepared eight-trial study')
    rows=study.get('trials',[])
    if len(rows)!=8 or len({row['id'] for row in rows})!=8:
        raise owned.SafetyHalt('Eight unique matched trial specifications required')
    for row in rows:
        if (row.get('arm') not in {'native-keyboard-control','native-button-candidate'}
                or row.get('case') not in {'linux_text','linux_calc'} or type(row.get('repetition')) is not int
                or row.get('repetition') not in (1,2)
                or row.get('seconds')!=(360 if row['case']=='linux_text' else 900) or row.get('tools')!=100):
            raise owned.SafetyHalt('Keep the original tasks, repetitions and budgets')
    expected={(arm,case,rep) for arm in ['native-keyboard-control','native-button-candidate']
              for case in ['linux_text','linux_calc'] for rep in [1,2]}
    if {(row['arm'],row['case'],row['repetition']) for row in rows}!=expected:
        raise owned.SafetyHalt('Matched conditions are incomplete or duplicated')
    order=[(arm,case,rep) for rep in (1,2)
           for arm in (['native-keyboard-control','native-button-candidate'] if rep==1
                       else ['native-button-candidate','native-keyboard-control'])
           for case in ('linux_text','linux_calc')]
    if [(r['arm'],r['case'],r['repetition']) for r in rows]!=order:
        raise owned.SafetyHalt('Preserve the frozen reversal of treatment order')
    matches=[row for row in rows if row['id']==trial_id]
    if len(matches)!=1:raise owned.SafetyHalt('Trial is not declared exactly once')
    row=matches[0]
    frameworks={}
    for arm,key in [('native-keyboard-control','control_build'),('native-button-candidate','candidate_build')]:
        build_path=checked_ref(study[key]);build=json.loads(build_path.read_text())
        selected=build_path.parent/'framework';declared=build['candidate_files_sha256']
        actual={str(p.relative_to(selected)):owned.digest(p) for p in selected.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
        if actual!=declared:raise owned.SafetyHalt('Private framework inventory changed')
        frameworks[arm]=selected
    framework=frameworks[row['arm']]
    for key in ['cpu_qualification','actual_component_qualification','independent_audit',
                'exact_recorded_policy_diagnostic','predecessor_query_registration']:
        checked_ref(study[key])
    original=json.loads(checked_ref(study['original_sdk_protocol']).read_text())
    for name,reference in study['helpers'].items():
        if checked_ref(reference)!=queue.ROOT/'apps/computer_use'/name:
            raise owned.SafetyHalt('Pilot helper runtime path changed')
    for name in ['platform_pilots.py','calc_pilot.py','browser_pilot.py','browser_pilot_server.js',
                 'android_contacts_pilot.py','linux_recording.py','live_android_recording.py',
                 'vm/record_console.py','vm/vnc_admin.py']:
        checked_ref(dict(path=str(queue.ROOT/'apps/computer_use'/name),sha256=original['helper_sha256'][name]))
    profile=dict(thinking=True,dflash2=True,kvmem='off',model_output_tokens=16384,
        model_context_tokens=524288,gui_only=True,advisor=False,software=False,prefix_reuse=False,
        visual_history='pin-all',model_concurrency=1,ordinary_tasks_tools=100)
    if study['model_profile']!=profile:raise owned.SafetyHalt('Native pilot model profile changed')
    return study,row,framework,original


class OwnedGuestTransport:
    """One owned device for original administrative initialization and reads."""
    def __init__(self,vm,output,audit):
        self.vm=vm;self.output=Path(output).resolve();self.audit=audit
        self.guest_endpoint=endpoint(vm['endpoint'])

    def save(self):
        if self.output.exists():queue.save(self.output/'private-device-adapter.json',self.audit)

    def owned_vm(self):
        info=ORIGINAL_CONTAINER(self.vm['name'],'qw3.goal','cross-device-20261003')
        if (info['Id']!=self.vm['container_id'] or info['Name']!='/'+self.vm['name']
                or info['Image']!=environment.IMAGE or not info['State']['Running']
                or not self.vm['name'].startswith('qw3-native-control-model-')):
            raise owned.SafetyHalt('Private pilot VM identity changed')
        expected_port=str(urlparse(self.guest_endpoint).port)
        bindings=info['NetworkSettings']['Ports'].get('5000/tcp')
        if bindings!=[dict(HostIp='127.0.0.1',HostPort=expected_port)]:
            raise owned.SafetyHalt('Private VM transport is not the declared loopback binding')
        mounts=[m for m in info['Mounts'] if m['Destination']=='/System.qcow2']
        if len(mounts)!=1 or mounts[0]['RW'] is not False or mounts[0]['Source']!=str(environment.DISK):
            raise owned.SafetyHalt('Private VM seed is not the pinned read-only image')
        return info

    def execute(self,code):
        self.owned_vm()
        if code==RESET_SOURCE:
            read=self.transport(EMPTY_PROCESS_READ)
            snapshot=json.loads(read['output'])
            if set(snapshot)!={'gedit_processes'} or snapshot['gedit_processes']!=[]:
                self.audit['empty_editor_process_reads'].append(snapshot);self.save()
                raise owned.SafetyHalt('A prior editor process exists; no signal or initializer replacement is permitted')
            self.audit['empty_editor_process_reads'].append(snapshot)
            self.audit['administrative_adaptations'].append(dict(
                operation='skip_broad_editor_signal_after_positive_empty_process_read',
                no_existing_process_signaled=True,blank_document_and_wait_unchanged=True))
            self.save()
            code="import subprocess,time; time.sleep(1); subprocess.Popen(['gedit'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)"
        return self.transport(code)

    def transport(self,code):
        # One request; no uncertain initialization or input is automatically retried.
        import hashlib
        operations=self.audit.setdefault('guest_operations',[])
        row=dict(sequence=len(operations)+1,source_sha256=hashlib.sha256(code.encode()).hexdigest(),status='requesting')
        operations.append(row);self.save()
        try:
            result=pilot.request(self.guest_endpoint,'/execute',
                dict(command=['python','-c',code],shell=False),timeout=30)
        except BaseException as error:
            row.update(status='reply-unconfirmed',error_kind=type(error).__name__);self.save();raise
        if self.output.exists():
            directory=self.output/'guest-operations';directory.mkdir(exist_ok=True)
            path=directory/(str(row['sequence'])+'.json')
            if path.exists():raise owned.SafetyHalt('Never overwrite an administrative operation reply')
            queue.save(path,result);row['response']=dict(path=str(path),sha256=owned.digest(path))
        row.update(status='returned',returncode=result.get('returncode'),guest_status=result.get('status'));self.save()
        if result.get('returncode')!=0 or result.get('status')!='success':
            raise RuntimeError('Private pilot guest operation failed')
        return result


class BoundPilot(OwnedGuestTransport):
    def __init__(self,study_path,trial_id,binding,output):
        self.study_path=Path(study_path).resolve();self.study,self.row,self.framework,self.original=validate_study(self.study_path,trial_id)
        self.binding=binding
        if Path(output).name!=self.row['case']+'-r'+str(self.row['repetition']):
            raise owned.SafetyHalt('Both arms require the same neutral fixture basename')
        audit=dict(kind='owned_private_native_control_pilot_adapter',study_sha256=owned.digest(self.study_path),
            adapter_sha256=owned.digest(__file__),trial_id=trial_id,model_or_task_or_grading_changed=False,
            private_transport=True,administrative_adaptations=[],empty_editor_process_reads=[])
        super().__init__(binding['vm'],output,audit)
        self.client=binding['client'];self.gateway=binding['gateway']
        self.app=endpoint(self.client['endpoint']);endpoint(self.gateway['endpoint'])

    def validate_services(self):
        if Path(desktop_module.__file__).resolve().parent!=self.framework/'qw3_computer':
            raise owned.SafetyHalt('Interpreter selected a different candidate package')
        for role in [self.client,self.gateway]:
            queue.require_identity(role['identity'])
            command=role['identity']['command']
            if (Path(option(command,'--runtime')).resolve()!=Path(role['runtime']).resolve()
                    or option(command,'--port')!=str(urlparse(role['endpoint']).port)):
                raise owned.SafetyHalt('Private service runtime or endpoint changed')
        command=self.gateway['identity']['command']
        if (command[1:3]!=['-m','device_gateway_entry']
                or option(command,'--driver-endpoint')!=self.guest_endpoint
                or option(command,'--platform')!='linux'):
            raise owned.SafetyHalt('Gateway targets a different input device')
        registry=checked_ref(self.binding['registry']);devices=load_registry(registry)
        if (set(devices)!={'linux-vm'} or devices['linux-vm']['platform']!='linux'
                or devices['linux-vm']['endpoint']!=self.gateway['endpoint']
                or devices['linux-vm']['token_file']!=Path(self.gateway['runtime']).resolve()/'token.txt'):
            raise owned.SafetyHalt('Private model registry exposes an undeclared device')
        command=self.client['identity']['command']
        if (command[1:3]!=['-m','qw3_computer.app'] or Path(option(command,'--device-registry')).resolve()!=registry
                or option(command,'--selected-device')!='linux-vm'
                or option(command,'--backend')!='http://127.0.0.1:18080/v1'
                or option(command,'--execution-mode')!='verified'
                or option(command,'--model-output-tokens')!='16384'
                or option(command,'--tool-call-limit')!='100'
                or option(command,'--output-policy')!='task' or command.count('--thinking')!=1
                or any(v in command for v in ('--software-root','--backend-token','--image-history'))):
            raise owned.SafetyHalt('Private model client command changes the frozen task profile')
        if (Path(self.client['runtime'])/'device-dependencies-canonical-20261004.json').exists():
            raise owned.SafetyHalt('Do not route this private pilot through the historical shared catalogue')
        engine=self.binding['engine']
        actual=owned.register_engine(Path(engine['runtime']),self.original['engine_off_identity'])
        if not owned.same_process(engine['identity'],actual):
            raise owned.SafetyHalt('QW3 engine generation changed')
        state=pilot.request(self.app,'/api/state')
        if (state.get('status')!='idle' or state.get('desktop_paused') is not True
                or state.get('uncertain_actions') or state.get('thinking') is not True
                or state.get('model_output_tokens')!=16384 or state.get('tool_call_limit')!=100):
            raise owned.SafetyHalt('Model client is not idle with the declared settings')
        self.owned_vm()

    def run(self):
        self.validate_services()
        if self.output.exists():raise owned.SafetyHalt('Never reuse a pilot attempt directory')
        adapter=self
        def recorder(directory,container_id):
            return PrivateRecorder(adapter,directory,container_id)
        def mapped_container(name,label,value):
            if (name,label,value)!=('qw3-computer-linux','qw3.goal','cross-device-20261003'):
                raise owned.SafetyHalt('Pilot tried to reach an undeclared device')
            return self.owned_vm()
        args=argparse.Namespace(case=self.row['case'],output=self.output,
            runtime=Path(self.client['runtime']),app=self.app,seconds=self.row['seconds'],
            linux_video_mode='vnc',video_mode='live',fault=None,retire_task=True)
        # Hooks are scoped to this newly owned trial process, not live shared
        # modules/files. The original pilot retains its own independent grader.
        with patch.object(pilot,'linux_execute',self.execute),patch.object(pilot,'owned_container',mapped_container),\
             patch.object(linux_recording,'LinuxRecording',recorder):
            # owned_vm uses the original function to avoid recursive routing.
            result=pilot.run(args)
        self.save()
        return result


class PrivateRecorder(linux_recording.LinuxRecording):
    """Retain the existing read-only recorder, with actual worker registration."""
    def __init__(self,adapter,directory,container_id):
        if container_id!=adapter.vm['container_id']:raise owned.SafetyHalt('Recorder VM identity differs')
        self.adapter=adapter
        super().__init__(directory,container_id,name=adapter.vm['name'])
        self.native_worker_identity=None

    def inspect(self):
        self.adapter.owned_vm()
        return super().inspect()

    def start(self):
        try:
            super().start()
        finally:
            # Readiness failures may leave this same worker alive. Establish
            # its identity before the pilot's cleanup; never spawn a replacement.
            if self.worker is not None and self.worker.poll() is None:
                command=json.loads((self.directory/'recorder-launch.json').read_text())['command']
                self.native_worker_identity=owned.register_child(self.worker.pid,command)
                queue.save(self.directory/'private-recorder-identity.json',self.native_worker_identity)

    def finish(self,name='continuous.mp4'):
        if self.worker is not None and self.worker.poll() is None:
            if self.native_worker_identity is None:raise owned.SafetyHalt('Recorder ownership is not established')
            queue.require_identity(self.native_worker_identity)
        return super().finish(name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--trial-id',required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();binding=json.loads(args.binding.read_text())
    result=BoundPilot(args.study,args.trial_id,binding,args.output).run()
    print(json.dumps(result,ensure_ascii=False))
