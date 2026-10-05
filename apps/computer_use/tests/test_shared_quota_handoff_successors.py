"""Retired-engine receipts, exact inventories and pre-spawn port admission."""
import argparse
from contextlib import ExitStack, nullcontext
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import shared_quota_handoff_successors as adapter

q = adapter.RUNTIME
spec = importlib.util.spec_from_file_location("retained_handoff_contracts", Path(__file__).with_name("test_shared_quota_port_successors.py"))
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)
retained.q = retained.wait_tests.q = retained.wait_tests.previous_tests.q = retained.terminal_tests.q = q
spec = importlib.util.spec_from_file_location("retained_port_probes", Path(__file__).with_name("test_shared_quota_port_continuation.py"))
port_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port_tests)


class PortProbeTests(unittest.TestCase):
    test_busy_port_waits_same_probe_scope_until_free = port_tests.PortContinuationTests.test_busy_port_waits_same_probe_scope_until_free
    test_permanently_busy_port_stops_without_new_launcher = port_tests.PortContinuationTests.test_permanently_busy_port_stops_without_new_launcher
    test_permission_error_not_retried = port_tests.PortContinuationTests.test_permission_error_not_retried
    test_original_port_and_deadline_cannot_be_changed = port_tests.PortContinuationTests.test_original_port_and_deadline_cannot_be_changed


class SuccessorProducerTests(retained.SuccessorProducerTests):
    pass


class SuccessorRegistrationTests(retained.SuccessorRegistrationTests):
    def setUp(self):
        super().setUp()
        self.r["kind"] = "actual_shared_quota_handoff_successor_identity"


class ExactInventoryTests(retained.TerminalHandoffTests):
    def execute(self, stage, producer_live=None):
        scope = dict(audit=dict(memory=dict(queue=str(self.memory_queue), protocol=str(self.memory_protocol))),
                     memory=copy.deepcopy(self.mq), final_engine=copy.deepcopy(self.engine),
                     failed_sdk=dict(engine_retirement=dict(pid=self.engine["pid"], retired=True)))
        original = q.base.process_retired
        def retired(identity):
            if stage == "sdk" and identity == self.engine:
                return True  # Fixture's memory engine was retired by its failed SDK.
            return original(identity)
        with patch.object(adapter, "failure_scope", return_value=scope), patch.object(q.base, "process_retired", side_effect=retired):
            return super().execute(stage, producer_live)


class SpawnAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.controller = types.SimpleNamespace(checkpoint=Mock())
        self.spawn = Mock(return_value=("child", "identity"))
        self.command = [str(q.CONTROLLER_PYTHON), str(q.ROOT / "apps/computer_use/run_engine.py"), "--kvmem", "off"]

    def test_original_command_waits_before_one_spawn(self):
        events = []
        self.spawn.side_effect = lambda *args: events.append("spawn") or ("child", "identity")
        with patch.object(adapter.previous.port, "wait_for_engine_port", side_effect=lambda: events.append("port") or {"probes": 2}):
            self.assertEqual(adapter.admitted_spawn(self.controller, self.command, "engine", None, self.spawn), ("child", "identity"))
        self.assertEqual(events, ["port", "spawn"])
        self.spawn.assert_called_once_with(self.command, "engine", None)

    def test_busy_port_does_not_spawn_or_retry_child(self):
        with patch.object(adapter.previous.port, "wait_for_engine_port", side_effect=q.SafetyHalt("busy")):
            with self.assertRaises(q.SafetyHalt):
                adapter.admitted_spawn(self.controller, self.command, "engine", None, self.spawn)
        self.spawn.assert_not_called()

    def test_non_engine_children_are_unchanged(self):
        command = ["python", "-m", "task"]
        with patch.object(adapter.previous.port, "wait_for_engine_port") as wait:
            adapter.admitted_spawn(self.controller, command, "task", "framework", self.spawn)
        wait.assert_not_called()
        self.spawn.assert_called_once_with(command, "task", "framework")

    def test_all_four_controllers_use_pre_spawn_admission(self):
        source = adapter.patched(Path(adapter.previous.prior.original.__file__).read_text())
        self.assertEqual(source.count("return admitted_spawn(self, command, name, framework, super().spawn)"), 4)
        self.assertIn("return retired_handoff_ready(original_ready, p, kp, d)", source)

    def real_restore(self, fail_port):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = argparse.Namespace(declaration=root / "declaration.json", protocol=root / "protocol.json",
                                      state_dir=root, min_free_mib=56000, resource_wait_seconds=60)
            for path in (args.declaration,args.protocol): q.save(path,{})
            declaration = dict(original_followup=str(args.declaration), runner_python=str(q.quota.PYTHON),
                               memory_protocol=str(args.protocol), memory_queue="fixture-memory")
            canonical = dict(continued_osworld_queue="fixture-os",continuation_controller_identity=dict(pid=119))
            protocol = dict(quarantine_declaration=str(args.declaration),previous_app="fixture",engine_off_identity={})
            child = types.SimpleNamespace(pid=123,poll=lambda:None)
            popen = Mock(return_value=child)
            def invoke(_):
                # Use the production subclass and full original restore/spawn
                # methods; only host/GPU/network operations are fixture probes.
                controller = object.__new__(q.base.Controller)
                controller.args,controller.p,controller.state,controller.children = args,protocol,{},[]
                controller.restore_engine(dict(pid=122))
                return controller.state
            with ExitStack() as stack:
                for item in [patch.object(q,"validate"),patch.object(q,"frozen_protocol",return_value=(protocol,{})),
                             patch.object(q.quota,"validate_protocol",return_value={}),
                             patch.object(q.owned,"model_lock",side_effect=nullcontext),
                             patch.object(q.owned,"assert_apps_idle"),patch.object(q.owned,"retire_registered",return_value=dict(retired=True)),
                             patch.object(q.owned,"gpu_snapshot",return_value=dict(free_mib=60000)),
                             patch.object(q.owned,"register_child",return_value=dict(pid=123)),
                             patch.object(q.owned,"register_engine",return_value=dict(pid=123)),
                             patch.object(q.base,"request",return_value=dict(data=[dict(context_window=524288)])),
                             patch.object(q.base.subprocess,"Popen",popen),patch.object(q.base,"run",side_effect=invoke),
                             patch.object(adapter.previous.port,"wait_for_engine_port",side_effect=q.SafetyHalt("busy") if fail_port else lambda:dict(probes=2))]:
                    stack.enter_context(item)
                if fail_port:
                    with self.assertRaises(q.SafetyHalt): q.run_sdk(args,declaration,canonical,dict(identity=dict(pid=120)))
                    popen.assert_not_called()
                else:
                    state = q.run_sdk(args,declaration,canonical,dict(identity=dict(pid=120)))
                    self.assertEqual(state['engine_port_admission'],dict(probes=2))
                    self.assertEqual(popen.call_count,1)
                    command = popen.call_args.args[0]
                    self.assertEqual(command[1],str(q.ROOT/'apps/computer_use/run_engine.py'))
                    self.assertEqual(command[command.index('--context-tokens')+1],'524288')

    def test_full_sdk_restore_launches_original_command_once_after_admission(self):
        self.real_restore(False)

    def test_full_sdk_restore_busy_port_prevents_any_child(self):
        self.real_restore(True)


class RetiredReceiptTests(unittest.TestCase):
    def setUp(self):
        self.engine = dict(pid=122, uid=1001, start_ticks=10, command=["owned"], cwd=str(q.ROOT))
        self.state = dict(current_engine_runtime="exact-memory-runtime", final_registered_engine=self.engine)
        self.scope = dict(audit=dict(memory=dict(queue="exact-queue", protocol="exact-protocol")),
                          memory=self.state, final_engine=self.engine,
                          failed_sdk=dict(engine_retirement=dict(pid=122, retired=True)))
        self.d = dict(memory_queue="exact-queue", memory_protocol="exact-protocol")
        self.kp = dict(engine_identities=dict(on=dict(profile="on")))

    def call(self, getter, retired=True):
        with patch.object(adapter, "failure_scope", return_value=self.scope), patch.object(q.base, "process_retired", return_value=retired):
            return adapter.retired_handoff_ready(getter, {}, self.kp, self.d)

    def test_retirement_is_explicit_and_original_registration_is_restored(self):
        original = q.owned.register_engine
        def getter(p, kp):
            engine = q.owned.register_engine(Path("exact-memory-runtime"), kp["engine_identities"]["on"])
            return dict(registered_engine=engine, kvmem=["all-eight"])
        answer = self.call(getter)
        self.assertEqual(answer["registered_engine"], self.engine)
        self.assertTrue(answer["registered_engine_retired_before_recovery"])
        self.assertEqual(answer["kvmem"], ["all-eight"])
        self.assertIs(q.owned.register_engine, original)

    def test_live_or_reused_engine_is_not_accepted(self):
        getter = lambda p,kp: q.owned.register_engine(Path("exact-memory-runtime"), kp["engine_identities"]["on"])
        with self.assertRaises(q.SafetyHalt): self.call(getter, False)

    def test_other_runtime_or_profile_cannot_use_retired_receipt(self):
        for runtime,expected in [("other",self.kp["engine_identities"]["on"]),("exact-memory-runtime",dict(profile="off"))]:
            with self.subTest(runtime=runtime, expected=expected), self.assertRaises(q.SafetyHalt):
                self.call(lambda p,kp: q.owned.register_engine(Path(runtime), expected))

    def test_other_memory_producer_is_rejected_before_inventory_callback(self):
        callback = Mock(); self.d["memory_queue"] = "other"
        with self.assertRaises(q.SafetyHalt): self.call(callback)
        callback.assert_not_called()

    def test_inventory_exception_is_preserved_and_registration_restored(self):
        original = q.owned.register_engine
        with self.assertRaisesRegex(q.SafetyHalt, "eighth missing"):
            self.call(Mock(side_effect=q.SafetyHalt("eighth missing")))
        self.assertIs(q.owned.register_engine, original)

    def test_pending_inventory_stays_pending(self):
        self.assertIsNone(self.call(lambda *args: None))


class FailureScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audit = copy.deepcopy(q.read(adapter.FAILURE_AUDIT))
        # CPU fixtures read declarations without probing host services/VMs.
        # Actual full admission is recorded separately before startup.
        self.d = q.read(adapter.FAILED_DECLARATION)
        self.mr = q.read(self.d["memory_registration"])
        self.memory = q.read(self.mr["queue"])
        self.states = {key: q.read(row["queue"]) for key,row in self.audit["successors"].items()}
        self.records = {key: q.read(row["registration"]) for key,row in self.audit["successors"].items()}
        self.runtime = self.root / "sdk/engine-off"
        self.d["sdk_protocol"] = str(self.runtime.parent / "protocol.json")
        child = self.audit["sdk_startup_failure"]["current_child"]
        child["command"][child["command"].index("--run-dir")+1] = str(self.runtime)
        self.states["sdk"]["current_child"] = copy.deepcopy(child)
        self.audit_path = self.root / "audit.json"

    def execute(self, actual=None, absent=True):
        q.save(self.audit_path, self.audit)
        documents = {str(self.audit_path):self.audit}
        for stage,row in self.audit["successors"].items():
            documents[row["registration"]] = self.records[stage]; documents[row["queue"]] = self.states[stage]
        original_read = adapter.previous.port.read
        with ExitStack() as stack:
            for item in [patch.object(adapter,"FAILURE_AUDIT",self.audit_path),
                         patch.object(adapter,"FAILURE_SHA",q.owned.digest(self.audit_path)),
                         patch.object(adapter.previous.port,"read",side_effect=lambda p:copy.deepcopy(documents[str(p)]) if str(p) in documents else original_read(p)),
                         patch.object(adapter.previous.RUNTIME,"read",side_effect=lambda p:copy.deepcopy(documents[str(p)]) if str(p) in documents else original_read(p)),
                         patch.object(adapter.previous.RUNTIME,"validate",return_value=(self.d,{},self.mr)),
                         patch.object(adapter.previous,"memory_queue",return_value=self.memory),
                         patch.object(adapter.previous.RUNTIME,"successor_registration"),
                         patch.object(adapter.stopped_deferred,"deferred_registration"),
                         patch.object(q.owned,"process_identity",side_effect=lambda pid: actual if actual and pid==actual['pid'] else None)]:
                stack.enter_context(item)
            return adapter.failure_scope(require_absent=absent)

    def test_exact_recorded_retirement_and_zero_trial_failures_are_accepted(self):
        self.assertEqual(self.execute()["final_engine"],self.memory["final_registered_engine"])

    def test_started_task_is_not_a_pre_task_failure(self):
        self.states["sdk"]["trials"] = [dict(id="started")]
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_other_startup_error_is_not_admitted(self):
        self.states["sdk"]["error"] = "different failure"
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_wrong_retired_engine_is_rejected(self):
        self.states["sdk"]["engine_retirement"]["pid"] += 1
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_launcher_profile_change_is_rejected(self):
        self.audit["sdk_startup_failure"]["current_child"]["command"].append("--other")
        self.states["sdk"]["current_child"] = copy.deepcopy(self.audit["sdk_startup_failure"]["current_child"])
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_existing_runtime_cannot_be_restarted(self):
        self.runtime.mkdir(parents=True)
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_live_final_engine_or_launcher_is_not_admitted(self):
        for actual in [self.memory["final_registered_engine"],self.audit["sdk_startup_failure"]["current_child"]]:
            with self.subTest(pid=actual["pid"]), self.assertRaises(q.SafetyHalt): self.execute(actual)

    def test_reused_final_engine_pid_is_preserved_and_rejected(self):
        actual = dict(self.memory["final_registered_engine"],start_ticks=-1)
        with self.assertRaises(q.SafetyHalt): self.execute(actual)

    def test_missing_memory_trial_is_rejected(self):
        self.memory["trials"].pop()
        with self.assertRaises(q.SafetyHalt): self.execute()

    def test_post_start_declaration_validation_keeps_historical_failure(self):
        self.runtime.mkdir(parents=True)
        self.assertEqual(self.execute(absent=False)["failed_sdk"]["trials"],[])


if __name__ == "__main__":
    unittest.main()
