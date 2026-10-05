"""Exercise the immutable SDK/image success path with fresh eight-case receipts.

Only static protocol admission, the historical forty-case proof and host probes
use fixtures. The fresh memory/SDK inventories, result digests, queue identities,
scoped callbacks and service handoff checks run their production validators.
No model, subprocess or device operation is allowed in this qualification.
"""
import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_quota_successors as adapter

q = adapter.RUNTIME


class TerminalHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.declaration = self.root / "declaration.json"
        self.followup = self.root / "followup.json"
        self.memory_protocol = self.root / "memory/protocol.json"
        self.sdk_protocol = self.root / "sdk/protocol.json"
        self.image_protocol = self.root / "image/protocol.json"
        self.sdk_registration = self.root / "sdk-registration.json"
        self.parent_registration = self.root / "parent-registration.json"
        self.registry = self.root / "registry.json"
        for path in (self.declaration, self.followup, self.memory_protocol, self.sdk_protocol,
                     self.image_protocol, self.sdk_registration, self.registry):
            path.parent.mkdir(parents=True, exist_ok=True)
            q.save(path, {})
        self.os_identity, self.memory_identity, self.sdk_identity = (self.identity(i) for i in (119, 120, 121))
        self.engine, self.client, self.gateway = (self.identity(i) for i in (122, 123, 124))
        q.save(self.parent_registration, dict(identity=self.os_identity))
        self.os_queue = self.root / "osworld/queue.json"
        self.memory_queue = self.root / "memory/queue.json"
        self.sdk_queue = self.root / "sdk/queue.json"
        self.osq = dict(status="completed", trials=[])
        self.os_queue.parent.mkdir()
        q.save(self.os_queue, self.osq)
        self.kp = dict(trials=[dict(id=f"quota1-case-{i}") for i in range(8)],
                       engine_identities=dict(on={"qualified": "on"}))
        self.old_kp = dict(trials=[dict(id=f"old-case-{i}") for i in range(8)])
        self.sp = dict(trials=[dict(id=f"sdk-case-{i}", kind="original-osworld") for i in range(10)],
                       quarantine_declaration=str(self.declaration), parent_kvmem_protocol_sha256="old-digest")
        self.ip = dict(trials=[], sdk_protocol=str(self.sdk_protocol), sdk_protocol_sha256=q.owned.digest(self.sdk_protocol),
                       memory_protocol_sha256="old-digest", parent_app="fixture-app", engine_off_identity={"qualified": "off"})
        self.mr = dict(kind="actual_shared_quota_memory_identity", identity=self.memory_identity,
                       queue=str(self.memory_queue), protocol_sha256=q.owned.digest(self.memory_protocol))
        self.sr = dict(identity=self.sdk_identity, queue=str(self.sdk_queue),
                       protocol_sha256=q.owned.digest(self.sdk_protocol))
        self.mq = dict(pid=120, status="completed", protocol_sha256=self.mr["protocol_sha256"],
                       shared_quota_declaration_sha256=q.owned.digest(self.declaration),
                       shared_quota_wrapper_sha256=q.owned.digest(q.quota.__file__),
                       qualified_runner_python=str(q.quota.PYTHON), common_gpu_quota_ratio=1.0,
                       previous_profile_prefix_separate=1, final_engine_policy="on",
                       current_engine_runtime=str(self.root / "engine-on"), final_registered_engine=self.engine,
                       trials=self.results(self.kp["trials"], self.memory_protocol.parent))
        self.sq = dict(pid=121, status="completed", protocol_sha256=self.sr["protocol_sha256"],
                       runtime_successor_declaration_sha256=q.owned.digest(self.declaration),
                       runtime_successor_wrapper_sha256=q.owned.digest(q.__file__),
                       continued_successor_declaration_sha256=q.owned.digest(self.followup),
                       termination_confirmed=True, new_services_preserved=True, model_client=self.client,
                       registry=str(self.registry), registry_sha256=q.owned.digest(self.registry),
                       new_gateways=[dict(process_identity=self.gateway, command=["fixture", "--runtime", str(self.root / "gateway")])],
                       engine_runtime=str(self.root / "engine-off"), registered_engine=self.engine,
                       trials=self.results(self.sp["trials"], self.sdk_protocol.parent))
        self.d = dict(memory_continuation=str(self.declaration), original_followup=str(self.followup),
                      memory_protocol=str(self.memory_protocol), memory_queue=str(self.memory_queue),
                      runner_python=str(q.quota.PYTHON), sdk_registration=str(self.sdk_registration))
        self.canonical = dict(continued_osworld_queue=str(self.os_queue), continuation_controller_identity=self.os_identity,
                              parent_registration=str(self.parent_registration))
        self.write_queues()

    @staticmethod
    def identity(pid):
        return dict(pid=pid, uid=os.getuid(), start_ticks=pid * 10, command=["fixture", str(pid)], cwd=str(q.ROOT), state="S")

    def results(self, declared, directory):
        rows = []
        for spec in declared:
            result = dict(status="evaluated", score=0, passed=False, finished_at=1, uncertain_actions=0)
            path = directory / "attempts" / spec["id"] / "result.json"
            path.parent.mkdir(parents=True)
            q.save(path, result)
            rows.append(dict(id=spec["id"], result=result, result_sha256=q.owned.digest(path)))
        return rows

    def write_queues(self):
        q.save(self.memory_queue, self.mq)
        q.save(self.sdk_queue, self.sq)

    def execute(self, stage, producer_live=None):
        args = argparse.Namespace(declaration=self.declaration,
                                  protocol=self.sdk_protocol if stage == "sdk" else self.image_protocol)
        identities = {r["pid"]: r for r in (self.engine, self.client, self.gateway)}
        identities.update(producer_live or {})
        def frozen(name, path):
            return (copy.deepcopy(self.sp), copy.deepcopy(self.old_kp)) if name == "sdk" else copy.deepcopy(self.ip)
        def terminal(kp, state, path):
            self.assertEqual(kp, self.kp)
            self.assertEqual(path, self.followup)
            return [{"id": "historical-forty-qualification"}]
        def invoke_sdk(a):
            p, kp = q.base.validate(a.protocol)
            self.assertEqual(kp, self.kp)
            self.assertEqual(p["parent_kvmem_protocol_sha256"], q.owned.digest(self.memory_protocol))
            return q.base.predecessors_ready(p, kp)
        def invoke_image(a):
            p = q.image.validate(a.protocol)
            self.assertEqual(p["memory_protocol_sha256"], q.owned.digest(self.memory_protocol))
            return q.image.predecessors_ready(p)
        with ExitStack() as stack:
            patches = [
                patch.object(q, "validate", return_value=(self.d, self.canonical, self.mr)),
                patch.object(q, "frozen_protocol", side_effect=frozen),
                patch.object(q.quota, "validate_protocol", return_value=copy.deepcopy(self.kp)),
                patch.object(q, "parent_registration", return_value=(self.sr, q.owned.digest(self.sdk_registration))),
                patch.object(q.old, "continued_parent", return_value=self.osq),
                patch.object(q.old, "terminal_parent", side_effect=terminal),
                patch.object(q.owned, "process_identity", side_effect=lambda pid: identities.get(pid)),
                patch.object(q.owned, "assert_no_services"), patch.object(q.owned, "assert_vm_retired"),
                patch.object(q.owned, "assert_apps_idle"), patch.object(q.base, "ledger_terminal", return_value={"completed": 1}),
                patch.object(q.owned, "register_engine", return_value=self.engine),
                patch.object(q.sdk, "validate", side_effect=AssertionError("Historical SDK validator leaked into fresh handoff")),
                patch.object(q.owned.subprocess, "Popen", side_effect=AssertionError("Qualification must not launch")),
                patch.object(q.base if stage == "sdk" else q.image, "run", side_effect=invoke_sdk if stage == "sdk" else invoke_image),
            ]
            for item in patches:
                stack.enter_context(item)
            return getattr(q, "run_" + stage)(args, self.d, self.canonical, self.mr)

    def test_sdk_successfully_accepts_only_fresh_eight_and_new_digest(self):
        answer = self.execute("sdk")
        self.assertEqual([r["id"] for r in answer["kvmem"]], [r["id"] for r in self.kp["trials"]])
        self.assertEqual(answer["registered_engine"], self.engine)

    def test_image_successfully_accepts_fresh_eight_and_terminal_ten(self):
        answer = self.execute("image")
        self.assertEqual([r["id"] for r in answer["kvmem"]], [r["id"] for r in self.kp["trials"]])
        self.assertEqual(len(answer["sdk"]), 10)
        self.assertEqual(answer["new_gateways"], self.sq["new_gateways"])

    def reject_both(self):
        self.write_queues()
        for stage in ("sdk", "image"):
            with self.subTest(stage=stage), self.assertRaises(q.SafetyHalt):
                self.execute(stage)

    def test_old_inventory_cannot_be_counted_as_new_eight(self):
        self.mq["trials"][0]["id"] = self.old_kp["trials"][0]["id"]
        self.reject_both()

    def test_missing_or_duplicate_eighth_result_is_rejected(self):
        original = copy.deepcopy(self.mq["trials"])
        self.mq["trials"].pop()
        self.reject_both()
        self.mq["trials"] = original[:-1] + [original[0]]
        self.reject_both()

    def test_altered_result_bytes_and_row_digest_are_rejected(self):
        self.mq["trials"][0]["result_sha256"] = "changed"
        self.reject_both()

    def test_uncertain_action_never_admits_a_successor(self):
        row = self.mq["trials"][0]
        row["result"]["uncertain_actions"] = 1
        path = self.memory_protocol.parent / "attempts" / row["id"] / "result.json"
        q.save(path, row["result"])
        row["result_sha256"] = q.owned.digest(path)
        self.reject_both()

    def test_wrong_common_quota_or_protocol_digest_is_rejected(self):
        self.mq["common_gpu_quota_ratio"] = 0.5
        self.reject_both()
        self.mq["common_gpu_quota_ratio"] = 1.0
        self.mq["protocol_sha256"] = "old-digest"
        self.reject_both()

    def test_image_rejects_changed_memory_order(self):
        self.mq["trials"].reverse()
        self.write_queues()
        with self.assertRaises(q.SafetyHalt):
            self.execute("image")

    def test_completed_producer_with_reused_pid_is_rejected(self):
        reused = dict(self.memory_identity, start_ticks=999)
        for stage in ("sdk", "image"):
            with self.subTest(stage=stage), self.assertRaises(q.SafetyHalt):
                self.execute(stage, {120: reused})

    def test_live_completed_producer_still_requires_actual_retirement(self):
        for stage in ("sdk", "image"):
            with self.subTest(stage=stage):
                self.assertIsNone(self.execute(stage, {120: self.memory_identity}))

    def test_image_requires_exact_ten_and_confirmed_service_handoff(self):
        self.sq["trials"].pop()
        self.write_queues()
        with self.assertRaises(q.SafetyHalt):
            self.execute("image")
        self.sq["trials"] = self.results(self.sp["trials"], self.root / "replacement")
        self.sq["termination_confirmed"] = False
        self.write_queues()
        with self.assertRaises(q.SafetyHalt):
            self.execute("image")

    def test_image_rejects_lost_live_client_and_changed_registry(self):
        self.sq["model_client"] = self.identity(125)
        self.write_queues()
        with self.assertRaises(q.SafetyHalt):
            self.execute("image")
        self.sq["model_client"] = self.client
        q.save(self.registry, {"changed": True})
        self.write_queues()
        with self.assertRaises(q.SafetyHalt):
            self.execute("image")


if __name__ == "__main__":
    unittest.main()
