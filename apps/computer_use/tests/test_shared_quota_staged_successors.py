"""Exact original receipts and recovery of the recorded partial SDK deployment."""
from contextlib import ExitStack, nullcontext
import copy
import errno
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import shared_quota_staged_successors as adapter

q = adapter.RUNTIME
spec = importlib.util.spec_from_file_location("retained_staged_inventory", Path(__file__).with_name("test_shared_quota_handoff_successors.py"))
retained = importlib.util.module_from_spec(spec); spec.loader.exec_module(retained)
retained.q = retained.retained.q = retained.retained.wait_tests.q = retained.retained.wait_tests.previous_tests.q = retained.retained.terminal_tests.q = q


class SuccessorProducerTests(retained.SuccessorProducerTests):
    pass


class SuccessorRegistrationTests(retained.SuccessorRegistrationTests):
    pass


class ExactInventoryTests(retained.ExactInventoryTests):
    pass


class ServicePortTests(unittest.TestCase):
    def probe(self, errors, times=(0,)):
        socket = Mock(); socket.__enter__ = Mock(return_value=socket); socket.__exit__ = Mock(return_value=False)
        socket.bind.side_effect = errors
        with patch.object(adapter.socket,"socket",return_value=socket),patch.object(adapter.time,"monotonic",side_effect=times),patch.object(adapter.time,"sleep"):
            return adapter.admit_service_port(18204),socket

    def test_busy_then_free_keeps_same_port_and_has_no_process_action(self):
        result,socket=self.probe([OSError(errno.EADDRINUSE,'busy'),None],(0,1))
        self.assertEqual(result['port'],18204);self.assertEqual(socket.bind.call_count,2)

    def test_permanently_busy_does_not_launch_a_child(self):
        with self.assertRaises(q.SafetyHalt):self.probe([OSError(errno.EADDRINUSE,'busy')],(0,121))

    def test_other_probe_error_is_not_retried(self):
        with self.assertRaises(q.SafetyHalt):self.probe([OSError(errno.EACCES,'denied')])

    def test_port_role_cannot_change(self):
        with self.assertRaises(q.SafetyHalt):adapter.admit_service_port(18100)


class StageScopeTests(unittest.TestCase):
    def setUp(self):
        self.audit=copy.deepcopy(q.read(adapter.AUDIT))
        self.declaration=q.read(adapter.FAILED_DECLARATION)
        self.protocol=q.read(self.declaration['sdk_protocol'])
        self.states={key:q.read(row['queue']) for key,row in self.audit['successors'].items()}
        self.records={key:q.read(row['registration']) for key,row in self.audit['successors'].items()}
        self.actual={}
        self.engine=copy.deepcopy(self.audit['sdk_partial_service_checkpoint']['registered_engine'])
        self.container=dict(Id=self.audit['android_recovery']['container_id'],State=copy.deepcopy(self.audit['android_recovery']['actual_state']),HostConfig=dict(Memory=12*1024**3))
        self.appium_ready=True

    def execute(self,live=False):
        documents={str(adapter.AUDIT):self.audit}
        for key,row in self.audit['successors'].items():documents[row['queue']]=self.states[key]
        def retired(identity):
            actual=self.actual.get(identity['pid'])
            if actual is None:return True
            if not q.owned.same_process(identity,actual):raise q.SafetyHalt('PID reused')
            return False
        def register(path,stage,*args):return self.records[stage]
        with ExitStack() as stack:
            for item in [patch.object(adapter.previous.RUNTIME.image,'pinned'),
                         patch.object(adapter.previous.RUNTIME,'read',side_effect=lambda path:copy.deepcopy(documents[str(path)])),
                         patch.object(adapter.previous.RUNTIME,'validate',return_value=(self.declaration,{},{})),
                         patch.object(adapter.previous.RUNTIME,'successor_registration',side_effect=register),
                         patch.object(adapter.previous.RUNTIME,'frozen_protocol',return_value=(self.protocol,{})),
                         patch.object(q.base,'process_retired',side_effect=retired),
                         patch.object(q.owned,'register_engine',return_value=self.engine),
                         patch.object(q.base,'require_identity'),
                         patch.object(adapter,'owned_container',return_value=self.container),
                         patch.object(q.base,'request',return_value=dict(value=dict(ready=self.appium_ready)))]:stack.enter_context(item)
            return adapter.stage_scope(live=live)

    def test_exact_zero_trial_partial_deployment_is_accepted(self):
        self.assertEqual(self.execute()['partial'],self.audit['sdk_partial_service_checkpoint'])

    def test_live_sdk_or_reused_pid_is_not_a_finished_checkpoint(self):
        identity=self.records['sdk']['identity']
        for actual in [identity,dict(identity,start_ticks=-1)]:
            self.actual={identity['pid']:actual}
            with self.assertRaises(q.SafetyHalt):self.execute()

    def test_dispatched_task_cannot_be_replayed_as_setup(self):
        self.states['sdk']['trials']=[dict(id='already-dispatched')]
        with self.assertRaises(q.SafetyHalt):self.execute()

    def test_already_started_client_is_not_partial_android_setup(self):
        self.states['sdk']['model_client']=dict(pid=100)
        with self.assertRaises(q.SafetyHalt):self.execute()

    def test_changed_partial_gateway_receipt_is_rejected(self):
        self.states['sdk']['new_gateways'][0]['session_id']='other-session'
        with self.assertRaises(q.SafetyHalt):self.execute()

    def test_live_original_service_retirement_cannot_be_replayed(self):
        identity=self.protocol['previous_app_identity'];self.actual={identity['pid']:identity}
        with self.assertRaises(q.SafetyHalt):self.execute()

    def test_restored_dependency_must_have_exact_identity_and_capacity(self):
        self.assertEqual(self.execute(live=True)['protocol'],self.protocol)

    def test_replaced_live_engine_is_not_reused(self):
        self.engine['start_ticks']=-1
        with self.assertRaises(q.SafetyHalt):self.execute(live=True)

    def test_other_android_container_is_not_reused(self):
        self.container['Id']='other'
        with self.assertRaises(q.SafetyHalt):self.execute(live=True)

    def test_android_capacity_revision_must_match_qualification(self):
        self.container['HostConfig']['Memory']=8*1024**3
        with self.assertRaises(q.SafetyHalt):self.execute(live=True)

    def test_unready_appium_prevents_phase_recovery(self):
        self.appium_ready=False
        with self.assertRaises(q.SafetyHalt):self.execute(live=True)


class StagedDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.runtime=self.root/'runtime';self.runtime.mkdir()
        self.client=self.root/'client';self.client.mkdir()
        self.protocol=self.root/'sdk/protocol.json';self.protocol.parent.mkdir()
        self.catalogue=self.runtime/'cross-device-catalogue.json'
        q.save(self.catalogue,dict(devices=[dict(id='windows-vm',token_file='original-windows'),dict(id='linux-vm',token_file='old-linux'),dict(id='android-emulator',token_file='old-android')]))
        self.linux=dict(catalogue_id='linux-vm',pid=120,process_identity=dict(pid=120),device_id='linux-vm',session_id='same-linux',paused=True)
        self.partial=dict(new_gateways=[self.linux],service_retirements=dict(app=dict(retired=True)),containers={name:name for name in ['windows','linux','android']},engine_runtime='same-engine',registered_engine=dict(pid=121))
        p=dict(previous_catalogue_sha256=q.owned.digest(self.catalogue),app_runtime=str(self.client),framework='pinned-framework',app='fixture-app',
               helper_sha256={'device_gateway_entry.py':'pinned'},previous_gateways={'linux-vm':dict(item={}), 'android-emulator':dict(item={})})
        self.state={};self.calls=[]
        def checkpoint(**values):self.state.update(values)
        def spawn(command,name,framework):
            self.calls.append((command,name,framework));pid=130+len(self.calls)
            return types.SimpleNamespace(pid=pid),dict(pid=pid)
        def ready(child,getter,predicate,*args):
            observed=getter()
            if not predicate(observed):raise q.SafetyHalt('ready predicate failed')
            return observed
        self.controller=types.SimpleNamespace(p=p,args=types.SimpleNamespace(protocol=self.protocol),checkpoint=checkpoint,spawn=spawn,ready=ready)
        self.observe_linux=dict(paused=True,device_id='linux-vm',session_id='same-linux')
        self.observe_android=dict(paused=True,device_id='android-api35',session_id='fresh-android')
        self.app_state=dict(ready=True,status='idle',desktop_paused=True,uncertain_actions=0,thinking=True,model_output_tokens=16384,tool_call_limit=100)

    def execute(self,port_error=False):
        self.retire=Mock(side_effect=AssertionError('No original retirement may be replayed'))
        self.lock=Mock(side_effect=nullcontext)
        def observe(item,route):
            self.assertEqual(route,'/observe')
            return self.observe_android if 'gateway-05' in item['token_file'] else self.observe_linux
        with ExitStack() as stack:
            for item in [patch.object(adapter,'stage_scope',return_value=dict(partial=self.partial)),
                         patch.object(q.owned,'model_lock',self.lock),patch.object(q.owned,'retire_registered',self.retire),
                         patch.object(q.base,'RUNTIME',self.runtime),patch.object(q.base,'require_identity'),patch.object(q.base,'ledger_terminal',return_value={}),
                         patch.object(q.base,'gateway_api',side_effect=observe),patch.object(q.base,'request',return_value=self.app_state),
                         patch.object(adapter,'owned_container',side_effect=lambda name,*args:dict(Id=name.removeprefix('qw3-computer-'),State=dict(Running=True))),
                         patch.object(adapter,'admit_service_port',side_effect=q.SafetyHalt('busy') if port_error else lambda port:dict(port=port))]:stack.enter_context(item)
            adapter.staged_deploy(self.controller)

    def test_complete_deployment_only_starts_missing_android_and_client(self):
        self.execute()
        self.assertEqual([name for _,name,_ in self.calls],['android-gateway','model-client'])
        self.retire.assert_not_called()
        self.assertTrue(self.state['linux_gateway_reused'])
        self.assertTrue(self.state['staged_deployment_completed'])
        android=self.calls[0][0];self.assertIn('gateway-05',android[android.index('--runtime')+1])
        client=self.calls[1][0];self.assertEqual(client[client.index('--model-output-tokens')+1],'16384')
        self.assertEqual(client[client.index('--tool-call-limit')+1],'100')
        registry=q.read(self.state['registry']);self.assertEqual(registry['devices'][0]['token_file'],'original-windows')
        self.assertTrue(registry['devices'][1]['token_file'].endswith('linux/gateway-04/token.txt'))
        self.assertTrue(registry['devices'][2]['token_file'].endswith('android/gateway-05/token.txt'))

    def test_existing_registry_or_gateway_prevents_any_spawn(self):
        for path in [self.protocol.parent/'registry.json',self.runtime/'android/gateway-05']:
            path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            with self.assertRaises(q.SafetyHalt):self.execute()
            self.assertEqual(self.calls,[]);path.unlink()

    def test_busy_service_port_prevents_catalogue_change_or_child(self):
        before=self.catalogue.read_bytes()
        with self.assertRaises(q.SafetyHalt):self.execute(True)
        self.assertEqual(self.calls,[]);self.assertEqual(self.catalogue.read_bytes(),before)

    def test_changed_linux_session_prevents_android_creation(self):
        self.observe_linux['session_id']='changed'
        with self.assertRaises(q.SafetyHalt):self.execute()
        self.assertEqual(self.calls,[])

    def test_unready_android_preserves_partial_checkpoint_without_client(self):
        self.observe_android['paused']=False
        with self.assertRaises(q.SafetyHalt):self.execute()
        self.assertEqual([name for _,name,_ in self.calls],['android-gateway'])
        self.assertNotIn('registry',self.state)

    def test_existing_dependency_receipt_is_not_overwritten(self):
        path=self.client/'device-dependencies-canonical-20261004.json';path.write_text('original')
        with self.assertRaises(q.SafetyHalt):self.execute()
        self.assertEqual(path.read_text(),'original');self.assertEqual(self.calls,[])

    def test_restore_reuses_engine_without_any_retirement_or_launch(self):
        controller=types.SimpleNamespace(checkpoint=Mock())
        with patch.object(q.owned,'model_lock',side_effect=nullcontext),patch.object(adapter,'stage_scope',return_value=dict(partial=self.partial)),\
             patch.object(adapter.previous,'failure_scope',return_value=dict(final_engine=dict(pid=125))),patch.object(q.owned,'retire_registered') as retire:
            adapter.staged_restore(controller,dict(pid=125))
        retire.assert_not_called();self.assertEqual(controller.checkpoint.call_args.kwargs['engine_runtime'],'same-engine')

    def test_restore_rejects_other_memory_predecessor(self):
        with patch.object(q.owned,'model_lock',side_effect=nullcontext),patch.object(adapter,'stage_scope',return_value=dict(partial=self.partial)),\
             patch.object(adapter.previous,'failure_scope',return_value=dict(final_engine=dict(pid=125))):
            with self.assertRaises(q.SafetyHalt):adapter.staged_restore(self.controller,dict(pid=126))


if __name__ == '__main__':
    unittest.main()
