"""Frozen wait-loop checks plus exact new memory inventory/routing admission."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_quota_successors as adapter

q = adapter.RUNTIME
spec = importlib.util.spec_from_file_location("retained_successor_contracts", Path(__file__).with_name("test_runner_runtime_successors.py"))
previous_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous_tests)
previous_tests.q = q


class SuccessorProducerTests(previous_tests.ProducerTests):
    """Retain all fourteen original actual-process admission checks."""


class SuccessorRegistrationTests(previous_tests.RegistrationTests):
    def setUp(self):
        super().setUp()
        self.r["kind"] = "actual_shared_quota_successor_identity"


class FrozenLoops(previous_tests.FrozenLoopTests):
    def fixture(self, root, stage):
        args,d,canonical,mr,p = super().fixture(root,stage)
        memory_protocol=Path(d["memory_protocol"]);memory_protocol.parent.mkdir();q.save(memory_protocol,{})
        return args,d,canonical,mr,p

    def test_sdk_real_wait_loop_does_not_deploy_when_memory_process_missing(self):
        with patch.object(q.quota,"validate_protocol",return_value={}):
            super().test_sdk_real_wait_loop_does_not_deploy_when_memory_process_missing()

    def test_sdk_trial_command_changes_only_runner_interpreter(self):
        with patch.object(q.quota,"validate_protocol",return_value={}):
            super().test_sdk_trial_command_changes_only_runner_interpreter()

    def test_sdk_physical_receipts_route_to_new_memory_directory_without_profile_change(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol=Path(folder)/"protocol.json";q.save(protocol,{})
            original=dict(parent_osworld_queue="old-os",parent_kvmem_queue="old-memory",parent_kvmem_protocol="old-protocol",
                          parent_kvmem_protocol_sha256="old-digest",predecessor_processes={},trials=[dict(id="sdk-frozen")],
                          model_output_tokens=16384,model_context_tokens=524288,thinking=True,kvmem="off")
            d=dict(memory_queue="new-queue",memory_protocol=str(protocol))
            canonical=dict(continued_osworld_queue="continued-os",continuation_controller_identity={"pid":119})
            before=copy.deepcopy(original);actual=q.sdk_protocol(original,d,canonical,{"identity":{"pid":120}})
            self.assertEqual(original,before)
            self.assertEqual(actual["parent_kvmem_protocol_sha256"],q.owned.digest(protocol))
            for key in ("trials","model_output_tokens","model_context_tokens","thinking","kvmem"):
                self.assertEqual(actual[key],before[key])


class FreshMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.declaration=root/"declaration.json";self.protocol=root/"protocol.json";self.queue=root/"state/queue.json"
        self.queue.parent.mkdir()
        for p in (self.declaration,self.protocol):q.save(p,{})
        self.identity=dict(pid=120,uid=os.getuid(),start_ticks=12,cwd=str(q.ROOT),state="S",
            command=[str(q.quota.PYTHON),"-m",q.quota.MODULE,"--declaration",str(self.declaration)])
        self.runtime=dict(kind="fresh_shared_quota_kvmem_cohort",python=str(q.quota.PYTHON),continued_protocol=str(self.protocol),state_dir=str(self.queue.parent))
        self.reg=dict(kind="actual_shared_quota_memory_identity",declaration=str(self.declaration),declaration_sha256=q.owned.digest(self.declaration),
            protocol=str(self.protocol),protocol_sha256=q.owned.digest(self.protocol),queue=str(self.queue),identity=self.identity,old_business_prefix_reused=False,
            source_sha256={name:q.owned.digest(q.ROOT/"apps/computer_use"/name) for name in q.quota.HELPERS})
        self.regpath=root/"registration.json"
        self.state=dict(pid=120,status="running",trials=[],shared_quota_declaration_sha256=q.owned.digest(self.declaration),
            protocol_sha256=q.owned.digest(self.protocol),shared_quota_wrapper_sha256=q.owned.digest(q.quota.__file__),
            qualified_runner_python=str(q.quota.PYTHON),common_gpu_quota_ratio=1.0,previous_profile_prefix_separate=1)

    def registration(self):
        q.save(self.regpath,self.reg)
        return q.memory_registration(self.regpath,self.declaration,self.runtime)

    def queue_admission(self,actual):
        q.save(self.queue,self.state)
        with patch.object(q.owned,"process_identity",return_value=actual):return q.memory_queue(self.reg,self.declaration)

    def test_exact_new_registration_and_live_actual_queue_are_admitted(self):
        self.assertEqual(self.registration(),self.reg);self.assertEqual(self.queue_admission(self.identity),self.state)

    def test_old_capacity_registration_cannot_replace_the_new_producer(self):
        self.reg["kind"]="actual_kvmem_capacity_continuation_identity"
        with self.assertRaises(q.SafetyHalt):self.registration()

    def test_old_prefix_cannot_count_as_a_new_pair(self):
        self.reg["old_business_prefix_reused"]=True
        with self.assertRaises(q.SafetyHalt):self.registration()

    def test_changed_source_or_protocol_is_rejected(self):
        name=q.quota.HELPERS[0];old=self.reg["source_sha256"][name];self.reg["source_sha256"][name]="changed"
        with self.assertRaises(q.SafetyHalt):self.registration()
        self.reg["source_sha256"][name]=old;self.state["protocol_sha256"]="changed"
        with self.assertRaises(q.SafetyHalt):self.queue_admission(self.identity)

    def test_missing_actual_or_reused_pid_cannot_be_treated_as_waiting(self):
        for actual in (None,dict(self.identity,start_ticks=13)):
            with self.subTest(actual=actual),self.assertRaises(q.SafetyHalt):self.queue_admission(actual)

    def test_halted_or_wrong_quota_queue_never_admits_next_stage(self):
        self.state["status"]="halted-no-further-dispatch"
        with self.assertRaises(q.SafetyHalt):self.queue_admission(None)
        self.state["status"]="running";self.state["common_gpu_quota_ratio"]=0.5
        with self.assertRaises(q.SafetyHalt):self.queue_admission(self.identity)

    def test_completed_actual_retired_queue_can_reach_terminal_inventory_checks(self):
        self.state["status"]="completed"
        self.assertEqual(self.queue_admission(None),self.state)

    def test_interpreter_path_is_not_resolved_out_of_its_qualified_venv(self):
        self.reg["identity"]["command"][0]=str(q.quota.PYTHON.resolve())
        with self.assertRaises(q.SafetyHalt):self.registration()


if __name__=="__main__":unittest.main()
