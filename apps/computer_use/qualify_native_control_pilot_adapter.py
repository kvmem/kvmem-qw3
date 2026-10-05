"""Real owned VM qualification of the original task materials and transport.

No model client, inference request, gateway activation or GUI input is sent.
The original administrative initializers open the apps; their independent
saved-file checks must still reject unsolved tasks. Each invocation owns a
fresh VM and records its actual console. This is not QW3 task acceptance.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from unittest.mock import patch
import uuid

from bounded_linux_observation import BoundedOfficeLinuxDriver
import calc_pilot
import native_control_pilot_adapter as adapter
from qw3_computer.device_lease import DeviceLease
from qw3_computer.linux_driver import FOREGROUND_SCRIPT


PORTS=dict(server=18970,chromium=18971,vnc=18972,vlc=18973)


def port_map(base):
    if type(base) is not int or not 1024<=base<=65532:raise ValueError('Declare four consecutive explicit unprivileged ports')
    return dict(zip(('server','chromium','vnc','vlc'),range(base,base+4)))


class Qualification:
    def __init__(self,args):
        self.args=args;self.ports=port_map(getattr(args,'port_base',18970))
        self.output=args.output.resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.task_output=self.output/(args.case+'-r1');self.task_output.mkdir()
        study=json.loads(args.study.read_text())
        row=next(r for r in study['trials'] if (r['arm'],r['case'],r['repetition'])==(args.arm,args.case,1))
        self.study,self.row,self.framework,self.original=adapter.validate_study(args.study,row['id'])
        self.env=self.transport=self.driver=self.recorder=self.lease=None
        self.name='qw3-native-control-model-fixture-'+uuid.uuid4().hex[:12]
        self.record=dict(kind='real_private_native_pilot_material_and_transport_qualification',
            started_at=time.time(),passed=False,status='preparing',case=args.case,arm=args.arm,
            trial_id=row['id'],study_sha256=adapter.owned.digest(args.study),
            producer=adapter.owned.process_identity(os.getpid()),model_requests=0,
            model_services_started=False,gui_inputs=0,gateway_activations=0,business_success_claimed=False,
            original_task_recipe_and_grader_preserved=True,existing_devices_changed=False,
            checks={},administrative_adaptations=[],empty_editor_process_reads=[])
        names=['native_control_pilot_adapter.py','qualify_native_control_pilot_adapter.py','platform_pilots.py',
               'calc_pilot.py','linux_recording.py','osworld_linux/environment.py','vm/record_console.py','vm/vnc_admin.py']
        self.record['source_sha256']={}
        for name in names:
            source=adapter.queue.ROOT/'apps/computer_use'/name;target=self.output/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            self.record['source_sha256'][name]=adapter.owned.digest(source)
        self.save()

    def save(self,**values):
        self.record.update(values);adapter.queue.save(self.output/'result.json',self.record)

    def read(self,code):
        return json.loads(self.transport.execute(code)['output'])

    async def observe_ready(self):
        desired='Gedit' if self.args.case=='linux_text' else 'libreoffice-calc'
        deadline=time.monotonic()+45
        while True:
            foreground=self.read(FOREGROUND_SCRIPT)
            if foreground['process']==desired:break
            if time.monotonic()>=deadline:raise RuntimeError('Original app did not become foreground within45seconds')
            await asyncio.sleep(.2)
        self.driver=BoundedOfficeLinuxDriver(self.transport.guest_endpoint,'private-pilot-material-'+self.name)
        await self.driver.start()
        # Starting Office can change its client bounds and child input focus
        # during a screenshot. Retain each actual read and wait only for a
        # positively bracketed observation; no input or initializer is retried.
        startup_reads=self.task_output/'startup-reads';startup_reads.mkdir()
        deadline=time.monotonic()+30;count=0
        while True:
            observation=await self.driver.observe();count+=1
            adapter.queue.save(startup_reads/(str(count)+'.json'),{k:v for k,v in observation.items() if k!='png'})
            (startup_reads/(str(count)+'.png')).write_bytes(observation['png'])
            if (observation['foreground']['process']==desired
                    and observation.get('native_window_capture',{}).get('screenshot_bracket_matched') is True):break
            if time.monotonic()>=deadline:raise RuntimeError('App window did not settle within30seconds; no input sent')
            await asyncio.sleep(.2)
        self.record['startup_observation_reads']=count
        metadata={k:v for k,v in observation.items() if k!='png'}
        adapter.queue.save(self.task_output/'initial-observation.json',metadata)
        (self.task_output/'initial-observation.png').write_bytes(observation['png'])
        self.record['checks']['original_app_is_foreground']=observation['foreground']['process']==desired
        self.record['checks']['positive_native_window_bracket']=observation.get('native_window_capture',{}).get('screenshot_bracket_matched') is True
        return observation

    async def initialize_and_read(self):
        if self.args.case=='linux_text':
            with patch.object(adapter.pilot,'linux_execute',self.transport.execute):
                adapter.pilot.initialize_linux_text(self.task_output)
            prompt=adapter.pilot.LINUX_PROMPT
            observation=await self.observe_ready()
            before=self.read(adapter.EMPTY_PROCESS_READ)
            try:
                self.transport.execute(adapter.RESET_SOURCE)
                raise RuntimeError('Repeated initializer unexpectedly accepted an existing editor')
            except adapter.owned.SafetyHalt as error:
                self.save(existing_editor_rejection=str(error))
            after=self.read(adapter.EMPTY_PROCESS_READ)
            self.record['checks']['existing_editor_refused_without_signal_or_restart']=bool(before['gedit_processes']) and before==after
            artifact=self.read("from pathlib import Path;import json,base64;p=Path('/home/user/Desktop/qw3-pilot-note.txt');print(json.dumps(dict(exists=p.exists(),data=base64.b64encode(p.read_bytes()).decode() if p.exists() else None)))")
            self.record['checks']['unsolved_text_file_missing']=artifact==dict(exists=False,data=None)
            self.save(original_prompt=prompt,initial_file_read=artifact,editor_identity_before=before,editor_identity_after=after,
                      observed_controls=len(observation['accessibility']['controls']))
        else:
            fixture,prompt=calc_pilot.initialize(self.transport.execute,self.task_output)
            adapter.queue.save(self.task_output/'calc-fixture.json',fixture)
            observation=await self.observe_ready()
            initial=Path(fixture['initial_file']).read_bytes()
            grade=calc_pilot.grade(self.transport.execute,fixture,self.task_output)
            saved=(self.task_output/'saved-workbook.xlsx').read_bytes()
            self.record['checks']['guest_workbook_bytes_match_actual_original_material']=saved==initial
            self.record['checks']['all_initial_cells_match_original_recipe']=calc_pilot.cells(saved)==calc_pilot.cells(calc_pilot.initial_workbook())
            self.record['checks']['original_grader_rejects_unsolved_workbook']=grade['passed'] is False
            self.record['checks']['partial_office_observation_remains_partial']=observation['accessibility']['truncated'] is True and observation['accessibility']['scope']=='foreground_office_chrome'
            self.save(original_prompt=prompt,initial_cells=calc_pilot.cells(initial),independent_unsolved_workbook_grade=grade,
                initial_guest_file=fixture['guest_file'],actual_initial_bytes_sha256=adapter.owned.digest(fixture['initial_file']),
                observed_controls=len(observation['accessibility']['controls']))
        adapter.queue.save(self.task_output/'material-manifest.json',dict(prompt=prompt,
            deadline_seconds=self.row['seconds'],tools=self.row['tools'],model_output_tokens=16384,
            case=self.row['case'],neutral_fixture_basename=self.task_output.name,
            original_pilot_sha256=adapter.owned.digest(adapter.pilot.__file__)))

    async def run(self):
        try:
            from qw3_computer import desktop
            if Path(desktop.__file__).resolve().parent!=self.framework/'qw3_computer':
                raise RuntimeError('Interpreter did not load the declared private candidate')
            if shutil.disk_usage(self.output).free<50*1024**3:raise RuntimeError('Less than50GiB free for the owned VM')
            memory={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
            if memory['MemAvailable']<12*1024**2:raise RuntimeError('Less than12GiB available host memory')
            self.save(host_mem_available_kib=memory['MemAvailable'],gpu_snapshot_before=adapter.owned.gpu_snapshot())
            subprocess.run(['docker','image','inspect',adapter.environment.IMAGE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=5)
            for port in self.ports.values():
                with socket.socket() as sock:sock.bind(('127.0.0.1',port))
            endpoint='http://127.0.0.1:'+str(self.ports['server'])
            self.lease=DeviceLease(endpoint+':desktop').acquire()
            stat=adapter.environment.DISK.stat();self.seed_before=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            self.save(status='booting-one-owned-material-vm',container_name=self.name,ports=self.ports)
            print('Starting one private task-material VM; no model inference',flush=True)
            with patch.object(adapter.environment,'PORTS',self.ports):self.env=adapter.environment.create(self.output,self.name)
            cid=json.loads((self.output/'vm.json').read_text())['container_id'];self.save(container_id=cid)
            self.transport=adapter.OwnedGuestTransport(dict(name=self.name,container_id=cid,endpoint=endpoint),self.task_output,self.record)
            self.transport.owned_vm();self.record['checks']['actual_owned_vm_and_readonly_seed']=True
            self.recorder=adapter.PrivateRecorder(self.transport,self.task_output,cid);self.recorder.start()
            self.save(status='checking-original-material')
            await self.initialize_and_read()
            self.record['passed']=all(self.record['checks'].values())
        except BaseException as error:
            self.save(passed=False,error_kind=type(error).__name__,error=str(error))
        finally:
            if self.driver is not None:await self.driver.close()
            if self.recorder is not None:
                try:
                    receipt=self.recorder.finish('continuous.mp4' if self.record['passed'] else 'interrupted.mp4')
                    import av
                    movie=self.task_output/('continuous.mp4' if self.record['passed'] else 'interrupted.mp4')
                    with av.open(str(movie)) as stream:frames=sum(1 for _ in stream.decode(video=0))
                    if frames<1:raise RuntimeError('Recording decoded no frames')
                    self.save(video_complete=receipt['complete'],decoded_frames=frames,recording_sha256=adapter.owned.digest(movie))
                except Exception as error:self.save(passed=False,video_error=str(error))
            if self.env is not None:
                try:
                    self.transport.owned_vm();self.env.close()
                    retired=subprocess.run(['docker','inspect',self.record['container_id']],capture_output=True,timeout=5).returncode!=0
                    self.save(dedicated_vm_retired=retired)
                    if not retired:self.record['passed']=False
                except Exception as error:self.save(passed=False,vm_cleanup_error=str(error))
            if self.lease is not None:self.lease.close()
            if hasattr(self,'seed_before'):
                stat=adapter.environment.DISK.stat();same=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==self.seed_before
                self.save(seed_metadata_unchanged=same)
                if not same:self.record['passed']=False
            self.save(status='passed' if self.record['passed'] else 'failed',finished_at=time.time())
        return self.record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['native-keyboard-control','native-button-candidate'],required=True)
    p.add_argument('--case',choices=['linux_text','linux_calc'],required=True)
    p.add_argument('--port-base',type=int,default=18970,help='Four explicit owned VM ports; collision fails before creating a VM')
    args=p.parse_args();result=asyncio.run(Qualification(args).run())
    print(json.dumps({k:result.get(k) for k in ('passed','checks','error','decoded_frames','dedicated_vm_retired')},ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
