"""Actual producer admission and frozen successor terminal/wait paths."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_quota_port_successors as adapter

q = adapter.RUNTIME


def retained_tests(name):
    spec = importlib.util.spec_from_file_location("port_retained_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.q = q
    if hasattr(module, "previous_tests"):
        module.previous_tests.q = q
    return module


wait_tests = retained_tests("test_shared_quota_successors")
terminal_tests = retained_tests("test_shared_quota_terminal_handoff")


class SuccessorProducerTests(wait_tests.SuccessorProducerTests):
    pass


class SuccessorRegistrationTests(wait_tests.SuccessorRegistrationTests):
    def setUp(self):
        super().setUp()
        self.r["kind"] = "actual_shared_quota_port_successor_identity"


class FrozenLoops(wait_tests.FrozenLoops):
    pass


class TerminalHandoffTests(terminal_tests.TerminalHandoffTests):
    def setUp(self):
        super().setUp()
        ids = [*adapter.port.PREFIX_IDS, "quota1-035-kvmem-off-r2",
               "quota1-080-kvmem-off-r2", "quota1-080-kvmem-on-r2"]
        self.kp["trials"] = [dict(id=name) for name in ids]
        self.mq["trials"] = self.results(self.kp["trials"], self.root / "port-memory")
        # The original inventory validators read actual result files here.
        for row in self.mq["trials"]:
            source = self.root / "port-memory/attempts" / row["id"] / "result.json"
            target = self.memory_protocol.parent / "attempts" / row["id"] / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        q.save(self.declaration, dict(completed_prefix=copy.deepcopy(self.mq["trials"][:5])))
        self.mr["kind"] = "actual_shared_quota_port_continuation_identity"
        self.mq.update(planned_trials=8, shared_quota_declaration_sha256=q.owned.digest(self.declaration))
        self.sq["runtime_successor_declaration_sha256"] = q.owned.digest(self.declaration)
        self.write_queues()


class PortMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.declaration, self.protocol = root / "declaration.json", root / "protocol.json"
        self.queue, self.regpath = root / "state/queue.json", root / "registration.json"
        self.queue.parent.mkdir()
        self.prefix = [dict(id=name, result={"uncertain_actions": 0}) for name in adapter.port.PREFIX_IDS]
        q.save(self.declaration, dict(completed_prefix=self.prefix))
        q.save(self.protocol, {})
        self.identity = dict(pid=120, uid=os.getuid(), start_ticks=12, cwd=str(q.ROOT), state="S",
            command=[str(q.quota.PYTHON), "-m", q.quota.MODULE, "--declaration", str(self.declaration)])
        self.runtime = dict(kind="exact_shared_quota_pre_task_port_continuation", python=str(q.quota.PYTHON),
                            continued_protocol=str(self.protocol), state_dir=str(self.queue.parent))
        self.reg = dict(kind="actual_shared_quota_port_continuation_identity", declaration=str(self.declaration),
            declaration_sha256=q.owned.digest(self.declaration), protocol=str(self.protocol),
            protocol_sha256=q.owned.digest(self.protocol), queue=str(self.queue), identity=self.identity,
            planned_trials=8, completed_matched_prefix_retained=5, remaining_model_trials=3,
            original_failed_launch_and_controller_preserved=True,
            source_sha256={name:q.owned.digest(q.ROOT / "apps/computer_use" / name) for name in q.quota.HELPERS})
        self.state = dict(pid=120, status="running", trials=copy.deepcopy(self.prefix), planned_trials=8,
            shared_quota_declaration_sha256=q.owned.digest(self.declaration), protocol_sha256=q.owned.digest(self.protocol),
            shared_quota_wrapper_sha256=q.owned.digest(q.quota.__file__), qualified_runner_python=str(q.quota.PYTHON),
            common_gpu_quota_ratio=1.0, previous_profile_prefix_separate=1)

    def registration(self):
        q.save(self.regpath, self.reg)
        return q.memory_registration(self.regpath, self.declaration, self.runtime)

    def admit(self, actual):
        q.save(self.queue, self.state)
        with patch.object(q.owned, "process_identity", return_value=actual):
            return q.memory_queue(self.reg, self.declaration)

    def test_exact_five_result_prefix_and_actual_producer_are_admitted_readonly(self):
        self.assertEqual(self.registration(), self.reg)
        before = copy.deepcopy(self.state)
        self.assertEqual(self.admit(self.identity), before)

    def test_old_memory_kind_cannot_substitute_for_current_producer(self):
        self.reg["kind"] = "actual_shared_quota_memory_identity"
        with self.assertRaises(q.SafetyHalt): self.registration()

    def test_declared_three_remaining_trials_and_preserved_failure_are_required(self):
        for field, value in (("planned_trials", 3), ("remaining_model_trials", 8),
                             ("completed_matched_prefix_retained", 0), ("original_failed_launch_and_controller_preserved", False)):
            with self.subTest(field=field):
                saved = self.reg[field]
                self.reg[field] = value
                with self.assertRaises(q.SafetyHalt): self.registration()
                self.reg[field] = saved

    def test_original_fresh_cohort_declaration_is_not_the_port_continuation(self):
        self.runtime["kind"] = "fresh_shared_quota_kvmem_cohort"
        with self.assertRaises(q.SafetyHalt): self.registration()

    def test_changed_source_protocol_and_venv_are_rejected(self):
        for field in ("protocol_sha256", "declaration_sha256"):
            saved = self.reg[field]; self.reg[field] = "changed"
            with self.assertRaises(q.SafetyHalt): self.registration()
            self.reg[field] = saved
        self.reg["source_sha256"][q.quota.HELPERS[0]] = "changed"
        with self.assertRaises(q.SafetyHalt): self.registration()

    def test_interpreter_symlink_must_keep_the_qualified_venv(self):
        self.identity["command"][0] = str(q.quota.PYTHON.resolve())
        with self.assertRaises(q.SafetyHalt): self.registration()

    def test_dead_zombie_and_reused_pid_never_become_waiting_producers(self):
        for actual in (None, dict(self.identity, state="Z"), dict(self.identity, start_ticks=13)):
            with self.subTest(actual=actual), self.assertRaises(q.SafetyHalt): self.admit(actual)

    def test_halted_producer_is_rejected_before_dispatch(self):
        self.state["status"] = "halted-no-further-dispatch"
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_retained_prefix_content_order_and_length_cannot_change(self):
        for rows in (self.prefix[:-1], list(reversed(self.prefix)), [dict(self.prefix[0], result={}), *self.prefix[1:]]):
            self.state["trials"] = copy.deepcopy(rows)
            with self.subTest(rows=rows), self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_profile_and_source_metadata_cannot_change(self):
        for field, value in (("common_gpu_quota_ratio", .5), ("planned_trials", 3),
                             ("shared_quota_wrapper_sha256", "changed"), ("protocol_sha256", "changed")):
            saved = self.state[field]; self.state[field] = value
            with self.subTest(field=field), self.assertRaises(q.SafetyHalt): self.admit(self.identity)
            self.state[field] = saved

    def test_completed_retired_producer_only_reaches_full_terminal_inventory_checks(self):
        self.state["status"] = "completed"
        self.assertEqual(self.admit(None), self.state)
        # Admission does not replace the downstream exact-eight validation.
        # TerminalHandoffTests verifies that a missing eighth result fails.


if __name__ == "__main__":
    unittest.main()
