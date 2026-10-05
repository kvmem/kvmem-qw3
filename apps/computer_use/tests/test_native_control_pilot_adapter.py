"""Private pilot transport and unchanged task/grade/recorder boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,Mock,patch

import native_control_pilot_adapter as a
import calc_pilot
import qualify_native_control_pilot_adapter as material


STUDY=a.queue.ROOT/'apps/computer_use/.runtime/native-control-model-retest-20261005-01/protocol.json'


def adapter(output):
    obj=object.__new__(a.BoundPilot)
    obj.output=Path(output)
    obj.vm=dict(name='qw3-native-control-model-test',container_id='owned-id',endpoint='http://127.0.0.1:19000')
    obj.guest_endpoint=obj.vm['endpoint'];obj.app='http://127.0.0.1:19004'
    obj.audit=dict(administrative_adaptations=[],empty_editor_process_reads=[])
    return obj


def vm_info(obj):
    return dict(Id=obj.vm['container_id'],Name='/'+obj.vm['name'],Image=a.environment.IMAGE,
                State=dict(Running=True),Mounts=[dict(Destination='/System.qcow2',RW=False,Source=str(a.environment.DISK))],
                NetworkSettings=dict(Ports={'5000/tcp':[dict(HostIp='127.0.0.1',HostPort='19000')]}))


def services(obj,directory):
    folder=Path(directory);obj.framework=folder/'framework';obj.original={'engine_off_identity':{'binary_sha256':'pinned'}}
    client_runtime=folder/'client';gateway_runtime=folder/'gateway'
    client_command=['python','-m','qw3_computer.app','--runtime',str(client_runtime),'--port','19004',
        '--device-registry',str(folder/'registry.json'),'--selected-device','linux-vm',
        '--backend','http://127.0.0.1:18080/v1','--execution-mode','verified','--thinking',
        '--model-output-tokens','16384','--tool-call-limit','100','--output-policy','task']
    gateway_command=['python','-m','device_gateway_entry','--runtime',str(gateway_runtime),'--port','19005',
        '--driver-endpoint',obj.guest_endpoint,'--platform','linux','--device-id','private-linux']
    obj.client=dict(runtime=str(client_runtime),endpoint=obj.app,identity={'pid':1,'command':client_command})
    obj.gateway=dict(runtime=str(gateway_runtime),endpoint='http://127.0.0.1:19005',identity={'pid':2,'command':gateway_command})
    registry=folder/'registry.json'
    registry.write_text(json.dumps(dict(devices=[dict(id='linux-vm',label='Private Linux',platform='linux',
        endpoint=obj.gateway['endpoint'],token_file=str(gateway_runtime/'token.txt'))])))
    engine=dict(pid=3,uid=1001,start_ticks=4,command=['owned-qw3'],cwd=str(a.queue.ROOT))
    obj.binding=dict(registry=dict(path=str(registry),sha256=a.owned.digest(registry)),engine=dict(runtime=str(folder/'engine'),identity=engine))
    obj.owned_vm=Mock()
    return dict(status='idle',desktop_paused=True,uncertain_actions=[],thinking=True,model_output_tokens=16384,tool_call_limit=100),engine


class PrivatePilotTests(unittest.TestCase):
    def test_guest_process_read_is_executable_python(self):
        compile(a.EMPTY_PROCESS_READ,'exact-empty-editor-process-read','exec')

    def test_material_vm_ports_are_explicit_and_bounded(self):
        self.assertEqual(material.port_map(18980),dict(server=18980,chromium=18981,vnc=18982,vlc=18983))
        for value in (True,1023,65533):
            with self.assertRaises(ValueError):material.port_map(value)

    def test_all_eight_real_frozen_trials_and_original_sources_validate(self):
        study=json.loads(STUDY.read_text())
        for row in study['trials']:
            _,actual,framework,original=a.validate_study(STUDY,row['id'])
            self.assertEqual(actual,row)
            self.assertEqual(original['helper_sha256']['platform_pilots.py'],a.owned.digest(a.pilot.__file__))
            self.assertTrue((framework/'qw3_computer/desktop.py').is_file())

    def test_reordered_or_changed_budgets_are_rejected_before_transport(self):
        study=json.loads(STUDY.read_text())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'study.json'
            for mutation in ('budget','order','duplicate','bool_repetition'):
                changed=copy.deepcopy(study)
                if mutation=='budget':changed['trials'][0]['seconds']=361
                if mutation=='order':changed['trials'][0],changed['trials'][1]=changed['trials'][1],changed['trials'][0]
                if mutation=='duplicate':changed['trials'][1]=changed['trials'][0]
                if mutation=='bool_repetition':changed['trials'][0]['repetition']=True
                path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation),self.assertRaises(a.owned.SafetyHalt):
                    a.validate_study(path,study['trials'][0]['id'])

    def test_old_device_mapper_does_not_recursively_call_itself(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d)
            with patch.object(a,'ORIGINAL_CONTAINER',return_value=vm_info(obj)) as original,\
                 patch.object(a.pilot,'owned_container',side_effect=AssertionError('recursive routing')):
                self.assertEqual(obj.owned_vm()['Id'],'owned-id')
            original.assert_called_once_with(obj.vm['name'],'qw3.goal','cross-device-20261003')

    def test_vm_replacement_nonlocal_port_and_writable_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d)
            for mutation in ('id','port','seed','image','running'):
                info=vm_info(obj)
                if mutation=='id':info['Id']='replacement-id'
                if mutation=='port':info['NetworkSettings']['Ports']['5000/tcp'][0]['HostIp']='0.0.0.0'
                if mutation=='seed':info['Mounts'][0]['RW']=True
                if mutation=='image':info['Image']='other-image'
                if mutation=='running':info['State']['Running']=False
                with self.subTest(mutation=mutation),patch.object(a,'ORIGINAL_CONTAINER',return_value=info),\
                     self.assertRaises(a.owned.SafetyHalt):obj.owned_vm()

    def test_existing_editor_blocks_initializer_without_any_signal(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);obj.owned_vm=Mock()
            obj.transport=Mock(return_value=dict(output=json.dumps(dict(gedit_processes=[dict(pid=12)]))))
            with self.assertRaises(a.owned.SafetyHalt):obj.execute(a.RESET_SOURCE)
            self.assertEqual(obj.transport.call_args_list[0].args,(a.EMPTY_PROCESS_READ,))
            self.assertEqual(obj.transport.call_count,1)
            self.assertEqual(obj.audit['administrative_adaptations'],[])

    def test_empty_editor_preserves_wait_and_blank_launch_without_broad_signal(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);obj.owned_vm=Mock()
            obj.transport=Mock(side_effect=[dict(output='{"gedit_processes":[]}'),dict(status='success')])
            obj.execute(a.RESET_SOURCE)
            launch=obj.transport.call_args_list[1].args[0]
            self.assertIn('time.sleep(1)',launch)
            self.assertIn("subprocess.Popen(['gedit']",launch)
            self.assertNotIn('pkill',launch)
            self.assertNotIn('kill',launch)

    def test_missing_or_unreadable_empty_process_proof_does_not_launch(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);obj.owned_vm=Mock()
            for reply in ('{}','null','broken'):
                obj.transport=Mock(return_value=dict(output=reply))
                with self.subTest(reply=reply),self.assertRaises((a.owned.SafetyHalt,TypeError,json.JSONDecodeError)):
                    obj.execute(a.RESET_SOURCE)
                self.assertEqual(obj.transport.call_count,1)

    def test_transport_is_owned_and_one_attempt_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);obj.owned_vm=Mock()
            with patch.object(a.pilot,'request',side_effect=TimeoutError('lost reply')) as request:
                with self.assertRaises(TimeoutError):obj.execute('print(1)')
            obj.owned_vm.assert_called_once();request.assert_called_once_with(obj.guest_endpoint,'/execute',
                dict(command=['python','-c','print(1)'],shell=False),timeout=30)

    def test_arm_basename_cannot_leak_into_calc_task_filename(self):
        with self.assertRaises(a.owned.SafetyHalt):
            a.BoundPilot(STUDY,json.loads(STUDY.read_text())['trials'][0]['id'],{},Path('/tmp/native-control-arm-name'))
        with tempfile.TemporaryDirectory() as d:
            prompts=[];materials=[]
            for arm in ('control','candidate'):
                out=Path(d)/arm/'linux_calc-r1';out.mkdir(parents=True)
                fixture,prompt=calc_pilot.initialize(Mock(),out)
                prompts.append(prompt);materials.append(calc_pilot.cells(Path(fixture['initial_file']).read_bytes()))
            self.assertEqual(prompts[0],prompts[1]);self.assertEqual(materials[0],materials[1])

    def test_original_unsolved_workbook_is_not_a_success(self):
        initial=calc_pilot.initial_workbook()
        grade=calc_pilot.grade_bytes(initial,initial)
        self.assertFalse(grade['passed'])
        self.assertFalse(grade['checks']['beta_hours'])
        self.assertFalse(grade['checks']['summary_formula'])

    def test_scoped_pilot_hooks_restore_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(Path(d)/'linux_text-r1');obj.validate_services=Mock();obj.owned_vm=Mock(return_value={'Id':'owned-id'})
            obj.row=dict(case='linux_text',seconds=360);obj.client=dict(runtime=d)
            original_execute=a.pilot.linux_execute;original_container=a.pilot.owned_container;original_recorder=a.linux_recording.LinuxRecording
            def fail(args):
                self.assertEqual(args.seconds,360);self.assertTrue(args.retire_task)
                self.assertEqual(a.pilot.owned_container('qw3-computer-linux','qw3.goal','cross-device-20261003')['Id'],'owned-id')
                with self.assertRaises(a.owned.SafetyHalt):a.pilot.owned_container('other','qw3.goal','cross-device-20261003')
                raise RuntimeError('pilot failure')
            with patch.object(a.pilot,'run',side_effect=fail),self.assertRaisesRegex(RuntimeError,'pilot failure'):obj.run()
            self.assertIs(a.pilot.linux_execute,original_execute);self.assertIs(a.pilot.owned_container,original_container)
            self.assertIs(a.linux_recording.LinuxRecording,original_recorder)

    def test_recorder_generation_change_prevents_signal(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);rec=a.PrivateRecorder(obj,d,obj.vm['container_id'])
            rec.worker=Mock();rec.worker.poll.return_value=None;rec.native_worker_identity=dict(pid=123)
            with patch.object(a.queue,'require_identity',side_effect=a.owned.SafetyHalt('PID reused')),\
                 self.assertRaises(a.owned.SafetyHalt):rec.finish()
            rec.worker.terminate.assert_not_called()

    def test_startup_timeout_registers_same_worker_for_later_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);rec=a.PrivateRecorder(obj,d,obj.vm['container_id'])
            child=Mock(pid=123);child.poll.return_value=None
            command=['python','-m','vm.record_console']
            def start(_):
                rec.worker=child;Path(d,'recorder-launch.json').write_text(json.dumps(dict(command=command)))
                raise TimeoutError('readiness')
            with patch.object(a.PrivateRecorder.__bases__[0],'start',start),\
                 patch.object(a.owned,'register_child',return_value=dict(pid=123)) as register,\
                 self.assertRaises(TimeoutError):rec.start()
            register.assert_called_once_with(123,command)
            self.assertEqual(rec.native_worker_identity,dict(pid=123));child.terminate.assert_not_called()

    def test_loopback_and_unique_options_required(self):
        for value in ('https://127.0.0.1:19000','http://localhost:19000','http://127.0.0.1:19000/?x=1'):
            with self.assertRaises(a.owned.SafetyHalt):a.endpoint(value)
        with self.assertRaises(a.owned.SafetyHalt):a.option(['--port','19000','--port','19001'],'--port')

    def test_services_require_the_private_registry_profile_and_engine_generation(self):
        with tempfile.TemporaryDirectory() as d:
            obj=adapter(d);state,engine=services(obj,d)
            with patch.object(a.desktop_module,'__file__',str(obj.framework/'qw3_computer/desktop.py')),\
                 patch.object(a.queue,'require_identity') as require,\
                 patch.object(a.owned,'register_engine',return_value=engine),\
                 patch.object(a.pilot,'request',return_value=state):obj.validate_services()
            self.assertEqual(require.call_count,2);obj.owned_vm.assert_called_once()

    def test_wrong_backend_budget_device_or_busy_client_prevents_admission(self):
        for mutation in ('backend','budget','software','gateway','busy','generation','registry'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                obj=adapter(d);state,engine=services(obj,d)
                client=obj.client['identity']['command'];gateway=obj.gateway['identity']['command']
                if mutation=='backend':client[client.index('--backend')+1]='https://external.invalid/v1'
                if mutation=='budget':client[client.index('--tool-call-limit')+1]='250'
                if mutation=='software':client.extend(['--software-root',d])
                if mutation=='gateway':gateway[gateway.index('--driver-endpoint')+1]='http://127.0.0.1:18126'
                if mutation=='busy':state['desktop_paused']=False
                if mutation=='generation':engine=engine|dict(start_ticks=5)
                if mutation=='registry':Path(obj.binding['registry']['path']).write_text('{}')
                with patch.object(a.desktop_module,'__file__',str(obj.framework/'qw3_computer/desktop.py')),\
                     patch.object(a.queue,'require_identity'),patch.object(a.owned,'register_engine',return_value=engine),\
                     patch.object(a.pilot,'request',return_value=state),self.assertRaises(a.owned.SafetyHalt):obj.validate_services()
                obj.owned_vm.assert_not_called()


class RawDriverObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_material_capture_accepts_the_real_raw_png_interface(self):
        with tempfile.TemporaryDirectory() as d:
            probe=object.__new__(material.Qualification)
            probe.args=Mock(case='linux_text');probe.task_output=Path(d);probe.name='private-test'
            probe.transport=Mock(guest_endpoint='http://127.0.0.1:19000')
            probe.record={'checks':{}};probe.read=Mock(return_value=dict(process='Gedit'))
            raw=dict(png=b'raw-png-bytes',foreground=dict(process='Gedit'),accessibility=dict(controls=[]),
                     native_window_capture=dict(screenshot_bracket_matched=True))
            driver=Mock();driver.start=AsyncMock();driver.observe=AsyncMock(return_value=raw)
            with patch.object(material,'BoundedOfficeLinuxDriver',return_value=driver):await probe.observe_ready()
            self.assertEqual((Path(d)/'initial-observation.png').read_bytes(),raw['png'])
            self.assertNotIn('png',json.loads((Path(d)/'initial-observation.json').read_text()))

    async def test_transient_startup_window_is_retained_before_positive_capture(self):
        with tempfile.TemporaryDirectory() as d:
            probe=object.__new__(material.Qualification)
            probe.args=Mock(case='linux_calc');probe.task_output=Path(d);probe.name='private-test'
            probe.transport=Mock(guest_endpoint='http://127.0.0.1:19000')
            probe.record={'checks':{}};probe.read=Mock(return_value=dict(process='libreoffice-calc'))
            old=dict(png=b'old-png',foreground=dict(process='libreoffice-calc'),accessibility=dict(controls=[]),
                native_window_capture=dict(screenshot_bracket_matched=False))
            fresh=old|dict(png=b'fresh-png',native_window_capture=dict(screenshot_bracket_matched=True))
            driver=Mock();driver.start=AsyncMock();driver.observe=AsyncMock(side_effect=[old,fresh])
            with patch.object(material,'BoundedOfficeLinuxDriver',return_value=driver),\
                 patch.object(material.asyncio,'sleep',new=AsyncMock()):await probe.observe_ready()
            self.assertEqual((Path(d)/'startup-reads/1.png').read_bytes(),old['png'])
            self.assertEqual((Path(d)/'initial-observation.png').read_bytes(),fresh['png'])
            self.assertEqual(probe.record['startup_observation_reads'],2)

    async def test_startup_timeout_does_not_reinitialize_or_send_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            probe=object.__new__(material.Qualification)
            probe.args=Mock(case='linux_calc');probe.task_output=Path(d);probe.name='private-test'
            probe.transport=Mock(guest_endpoint='http://127.0.0.1:19000')
            probe.record={'checks':{}};probe.read=Mock(return_value=dict(process='libreoffice-calc'))
            raw=dict(png=b'old-png',foreground=dict(process='libreoffice-calc'),
                native_window_capture=dict(screenshot_bracket_matched=False))
            driver=Mock();driver.start=AsyncMock();driver.observe=AsyncMock(return_value=raw)
            with patch.object(material,'BoundedOfficeLinuxDriver',return_value=driver),\
                 patch.object(material,'time',Mock(monotonic=Mock(side_effect=[0,0,0,31]))),\
                 self.assertRaisesRegex(RuntimeError,'did not settle'):await probe.observe_ready()
            probe.transport.execute.assert_not_called();driver.dispatch.assert_not_called()


if __name__=='__main__':unittest.main()
