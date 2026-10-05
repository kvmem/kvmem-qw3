"""Actual binding and cleanup failure modes of the private native pilot stack."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import native_control_model_stack as s


def instance():
    obj=object.__new__(s.NativeStack)
    obj.original={'engine_off_identity':{'profile':'unchanged'}}
    obj.record=dict(status='ready-idle',vm={'id':'private'},registry={'path':'private'},
        services=dict(client={'identity':{'pid':111},'endpoint':'http://127.0.0.1:19115','runtime':'/tmp/private/client'},
                      gateway={'identity':{'pid':112},'endpoint':'http://127.0.0.1:19114','runtime':'/tmp/private/gateway'}))
    obj.children={};obj.save=Mock();obj.env=None
    return obj


class NativeStackTests(unittest.TestCase):
    def test_unprivileged_six_port_scope(self):
        self.assertEqual(s.port_map(19110),dict(server=19110,chromium=19111,vnc=19112,vlc=19113,gateway=19114,client=19115))
        for value in (True,1023,65531,'19110'):
            with self.subTest(value=value),self.assertRaises(ValueError):s.port_map(value)

    def test_descendant_scan_skips_init_process_and_checks_exact_parent(self):
        child=dict(pid=2,uid=os.getuid(),state='S',command=['codex','app-server'])
        with patch.object(s.adapter.queue,'require_identity'),\
             patch.object(s.Path,'iterdir',return_value=[Path('/proc/1'),Path('/proc/2')]),\
             patch.object(s.Path,'read_text',return_value='2 (codex) S 111 0 0'),\
             patch.object(s.adapter.owned,'process_identity',return_value=child) as identity:
            self.assertEqual(s.descendants({'pid':111}),[child])
        identity.assert_called_once_with(2)

    def test_idle_client_profile_rejects_activity_unknown_or_changed_permissions(self):
        good=dict(ready=True,status='idle',desktop_paused=True,uncertain_actions=[],thinking=True,
            model_output_tokens=16384,tool_call_limit=100,tool_calls_used=0,software=None,devices={'selected':'linux-vm'})
        self.assertTrue(s.idle_profile(good))
        for key,value in dict(ready=False,status='running',desktop_paused=False,uncertain_actions=[1],thinking=False,
                             model_output_tokens=4096,tool_calls_used=1,software={'root':'other'},devices={'selected':'windows-vm'}).items():
            with self.subTest(key=key):self.assertFalse(s.idle_profile(good|{key:value}))

    def test_no_task_binding_without_real_registered_engine(self):
        obj=instance();engine={'runtime':'/tmp/private/engine','identity':{'pid':113}}
        with patch.object(s.adapter.owned,'register_engine',side_effect=s.adapter.owned.SafetyHalt('no manifest')) as register:
            with self.assertRaises(s.adapter.owned.SafetyHalt):obj.binding(engine)
        register.assert_called_once_with(Path(engine['runtime']),obj.original['engine_off_identity'])

    def test_engine_generation_replacement_blocks_binding(self):
        obj=instance();engine={'runtime':'/tmp/private/engine','identity':{'pid':113}}
        with patch.object(s.adapter.owned,'register_engine',return_value={'pid':114}),\
             patch.object(s.adapter.owned,'same_process',return_value=False),self.assertRaises(s.adapter.owned.SafetyHalt):obj.binding(engine)

    def test_only_ready_stack_exposes_binding(self):
        obj=instance();obj.record['status']='booting'
        with patch.object(s.adapter.owned,'register_engine') as register,self.assertRaises(s.adapter.owned.SafetyHalt):obj.binding({})
        register.assert_not_called()

    def test_unknown_task_preserves_all_services_and_vm_before_any_signal(self):
        obj=instance();obj.env=Mock()
        with patch.object(s,'retired',return_value=False),patch.object(s.adapter.queue,'require_identity'),\
             patch.object(s.adapter.owned,'assert_idle',side_effect=s.adapter.owned.SafetyHalt('unknown')),\
             patch.object(s.adapter.owned,'retire_registered') as signal, self.assertRaises(s.adapter.owned.SafetyHalt):obj.close()
        signal.assert_not_called();obj.env.close.assert_not_called();obj.save.assert_not_called()

    def test_unknown_gateway_receipt_blocks_all_retirement(self):
        obj=instance();obj.env=Mock()
        with patch.object(s,'retired',return_value=False),patch.object(s.adapter.queue,'require_identity'),\
             patch.object(s.adapter.owned,'assert_idle'),\
             patch.object(s.adapter.queue,'ledger_terminal',side_effect=s.adapter.owned.SafetyHalt('uncertain')) as ledger,\
             patch.object(s.adapter.owned,'retire_registered') as signal,self.assertRaises(s.adapter.owned.SafetyHalt):obj.close()
        ledger.assert_called_once_with(Path('/tmp/private/gateway'));signal.assert_not_called();obj.env.close.assert_not_called()

    def test_unpaused_gateway_is_not_retired(self):
        obj=instance()
        with patch.object(s,'retired',return_value=False),patch.object(s.adapter.queue,'require_identity'),\
             patch.object(s.adapter.owned,'assert_idle'),patch.object(s.adapter.queue,'ledger_terminal',return_value={}),\
             patch.object(s.adapter.queue,'gateway_api',return_value={'paused':False}),\
             patch.object(s.adapter.owned,'retire_registered') as signal,self.assertRaises(s.adapter.owned.SafetyHalt):obj.close()
        signal.assert_not_called()

    def test_unregistered_live_child_is_preserved(self):
        obj=instance();child=Mock();child.poll.return_value=None;obj.children={'other':child}
        with patch.object(s.adapter.owned,'retire_registered') as signal,self.assertRaises(s.adapter.owned.SafetyHalt):obj.close()
        signal.assert_not_called();child.terminate.assert_not_called()

    def test_source_study_digest_changes_are_rejected_before_output_or_transport(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'changed.json';path.write_text('{}')
            out=Path(d)/'never-created'
            with patch.object(s.adapter,'validate_study') as validate,self.assertRaises(s.adapter.owned.SafetyHalt):
                s.NativeStack(path,'undeclared',out,19110)
            self.assertFalse(out.exists());validate.assert_not_called()

    def test_same_vm_is_observed_until_actual_disappearance_without_second_close(self):
        vm={'container_id':'same-id','name':'owned-name'}
        live=[dict(Id='same-id',Name='/owned-name',Image=s.adapter.environment.IMAGE,
                   Config={'Labels':{'qw3.goal':'cross-device-20261003'}},State={'Running':False})]
        replies=[Mock(returncode=0,stdout=json.dumps(live),stderr=''),
                 Mock(returncode=1,stdout='[]',stderr='Error response from daemon: No such container: same-id')]
        with tempfile.TemporaryDirectory() as d,patch.object(s.subprocess,'run',side_effect=replies) as run,\
             patch.object(s.time,'sleep'):
            self.assertEqual(s.wait_vm_retired(vm,Path(d)/'reads'),dict(retired=True,observations=2))
            self.assertEqual(run.call_count,2)
            self.assertTrue(all(c.args[0]==['docker','inspect','--type','container','same-id'] for c in run.call_args_list))
            self.assertEqual(len(list((Path(d)/'reads').glob('*.json'))),2)

    def test_docker_permission_error_is_not_disappearance(self):
        reply=Mock(returncode=1,stdout='',stderr='permission denied')
        with tempfile.TemporaryDirectory() as d,patch.object(s.subprocess,'run',return_value=reply),\
             self.assertRaises(s.adapter.owned.SafetyHalt):s.wait_vm_retired({'container_id':'id'},Path(d)/'reads')

    def test_changed_vm_identity_blocks_wait_without_any_mutation(self):
        reply=Mock(returncode=0,stdout=json.dumps([{'Id':'other'}]),stderr='')
        with tempfile.TemporaryDirectory() as d,patch.object(s.subprocess,'run',return_value=reply) as run,\
             self.assertRaises(s.adapter.owned.SafetyHalt):
            s.wait_vm_retired({'container_id':'id','name':'own'},Path(d)/'reads')
        self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
