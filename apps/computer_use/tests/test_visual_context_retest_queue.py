"""Queue boundary tests with no model, VM or input executor."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import visual_context_retest_queue as queue
from visual_context_browser_run import pair_identity, admit_arguments
from visual_context_osworld_run import image_command


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qw3-image-queue-tests-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = argparse.Namespace(protocol=self.directory / "protocol.json", state_dir=self.directory / "state",
                                       wait_seconds=60, resource_wait_seconds=60, min_free_mib=56000)
        self.protocol = dict(model=copy.deepcopy(queue.MODEL), trials=queue.declared_trials(),
                             framework="/declared/framework", parent_app="http://parent",
                             acceptance="/declared/acceptance")
        queue.save(self.args.protocol, self.protocol)

    def test_plan_matches_frozen_interleaving_and_budgets(self):
        plan = queue.read(queue.ROOT / "docs/computer_use_visual_context_retest_plan_20261004.json")
        self.assertEqual(plan["trials"], queue.declared_trials())
        queue.validate_profile(plan)
        self.assertEqual(sum(r["kind"] == "original-osworld" for r in plan["trials"]), 8)

    def test_changed_model_or_advisor_is_rejected(self):
        for key, value in (("thinking_budget", 8192), ("max_output_tokens", 8192), ("advisor", True), ("kvmem", "on")):
            p = copy.deepcopy(self.protocol); p["model"][key] = value
            with self.subTest(key=key), self.assertRaises(queue.SafetyHalt): queue.validate_profile(p)

    def test_changed_order_missing_or_duplicate_trials_is_rejected(self):
        for rows in (self.protocol["trials"][::-1], self.protocol["trials"][:-1],
                     self.protocol["trials"][:-1] + [self.protocol["trials"][0]]):
            with self.assertRaises(queue.SafetyHalt): queue.validate_profile(dict(self.protocol, trials=rows))

    def test_image_flag_changes_only_model_client(self):
        gw = ["python", "-m", "qw3_computer.device_gateway", "--port", "18304"]
        self.assertNotIn("--image-history", image_command(gw, 16384, "full-retrievable"))
        app = ["python", "-m", "qw3_computer.app", "--port", "18305", "--desktop", "http://127.0.0.1:18304"]
        full = image_command(app, 16384, "full-retrievable")
        recent = image_command(app, 16384, "recent-observations")
        self.assertEqual(full[:-1], recent[:-1])
        self.assertEqual(full[full.index("--model-output-tokens") + 1], "16384")
        self.assertEqual(app[app.index("--port") + 1], "18305")

    def test_invalid_or_duplicate_policy_is_rejected(self):
        with self.assertRaises(ValueError): image_command([], 16384, "full")
        app = ["python", "-m", "qw3_computer.app", "--port", "18305", "--desktop", "http://old", "--image-history", "full-retrievable"]
        with self.assertRaises(ValueError): image_command(app, 16384, "recent-observations")

    def test_browser_pair_has_same_user_fields_and_unique_fixture_identity(self):
        import browser_pilot
        from platform_pilots import fixture_identity
        rows = [r for r in self.protocol["trials"] if r["kind"] == "normal-pilot"]
        identities = [fixture_identity(self.directory / r["id"]) for r in rows]
        self.assertEqual(len(set(identities)), 4)
        first = [browser_pilot.expected(pair_identity(r["repetition"])) for r in rows[:2]]
        self.assertEqual(first[0], first[1])
        self.assertNotEqual(first[0], browser_pilot.expected(pair_identity(2)))
        for rep in (True, "1", 3):
            with self.assertRaises(ValueError): pair_identity(rep)

    def test_cpu_manifest_rejects_empty_or_unpassed_suite(self):
        path = self.directory / "cpu.json"
        for tests, passed in ((0, True), (14, True), (20, False)):
            queue.save(path, dict(kind="visual_context_controller_cpu_boundary_qualification", passed=passed,
                                 tests_run=tests, real_model_calls=0, native_device_inputs=0))
            with self.assertRaises(queue.SafetyHalt): queue.validate_cpu(path)

    def test_browser_wrapper_rejects_other_runtime_budget_or_duplicate_args(self):
        row = next(r for r in self.protocol["trials"] if r["kind"] == "normal-pilot")
        controller = queue.Controller(self.args, self.protocol, {})
        cmd = controller.trial_command(row, self.directory / "attempts" / row["id"],
                                       "http://127.0.0.1:18217", self.directory / "clients" / row["id"])
        remaining = cmd[cmd.index("--case"):]
        self.assertEqual(admit_arguments(remaining, self.protocol, self.args.protocol, row["policy"]), row)
        for flag, changed in (("--seconds", "1800"), ("--runtime", "/existing/client"), ("--output", "/other/" + row["id"])):
            invalid = list(remaining); invalid[invalid.index(flag) + 1] = changed
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                admit_arguments(invalid, self.protocol, self.args.protocol, row["policy"])
        with self.assertRaises(ValueError):
            admit_arguments(remaining + ["--seconds", "900"], self.protocol, self.args.protocol, row["policy"])

    def test_frozen_helper_changed_is_rejected(self):
        path = self.directory / "helper.py"; path.write_text("original")
        sha = queue.owned.digest(path); path.write_text("changed")
        with self.assertRaises(queue.SafetyHalt): queue.pinned(path, sha)

    def execute_fake_campaign(self, fail_at=None):
        attempted = []
        def dispatch(controller, row):
            attempted.append(row["id"])
            if len(attempted) == fail_at:
                queue.base.terminal_trial(row, dict(finished_at=1, uncertain_actions=1))
            controller.state["trials"].append(dict(id=row["id"]))
            controller.checkpoint()
        with patch.object(queue, "validate", return_value=self.protocol), \
                patch.object(queue, "predecessors_ready", return_value={"registered_engine": {"pid": 123}}), \
                patch.object(queue.Controller, "dispatch", dispatch):
            if fail_at:
                with self.assertRaises(queue.SafetyHalt): queue.run(self.args)
            else: queue.run(self.args)
        return attempted, queue.read(self.args.state_dir / "queue.json")

    def test_full_twelve_control_sequence_completes_without_model_calls(self):
        attempted, state = self.execute_fake_campaign()
        self.assertEqual(attempted, [r["id"] for r in self.protocol["trials"]])
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["business_success_improvement_proven"])

    def test_unknown_input_stops_before_next_dispatch_and_preserves_evidence(self):
        attempted, state = self.execute_fake_campaign(fail_at=3)
        self.assertEqual(len(attempted), 3)
        self.assertEqual(len(state["trials"]), 2)
        self.assertEqual(state["status"], "halted-no-further-dispatch")
        self.assertTrue(state["live_children_and_evidence_preserved"])

    def test_existing_state_cannot_be_restarted(self):
        self.args.state_dir.mkdir()
        with patch.object(queue, "validate", return_value=self.protocol), self.assertRaises(FileExistsError):
            queue.run(self.args)

    def test_existing_attempt_is_rejected_before_retiring_engine(self):
        row = self.protocol["trials"][0]
        (self.directory / "attempts" / row["id"]).mkdir(parents=True)
        controller = queue.Controller(self.args, self.protocol, {"protocol_sha256": queue.owned.digest(self.args.protocol)})
        with patch.object(queue, "validate"), patch.object(controller, "parent_idle"), \
                patch.object(controller, "fresh_engine") as engine, self.assertRaises(queue.SafetyHalt):
            controller.dispatch(row)
        engine.assert_not_called()

    def trace(self, policy="full-retrievable"):
        inference = self.directory / "inference"
        trace = inference / "trace"; trace.mkdir(parents=True)
        blobs = inference / "full-history-blobs"; blobs.mkdir()
        body = {"messages": [{"role": "user", "content": "x" * 2048}]}
        data = json.dumps(body, ensure_ascii=False).encode()
        meta = dict(tree=queue.archive.encode(body, blobs), sha256=queue.archive.digest(data), bytes=len(data))
        queue.save(trace / "request.full.archive.json", meta)
        (trace / "request.json").write_bytes(data)
        queue.save(trace / "image-context.json", dict(policy=policy, source_sha256=meta["sha256"],
                   source_bytes=len(data), presented_sha256=meta["sha256"], presented_bytes=len(data)))
        queue.save(trace / "metrics.json", dict(usage={"input_tokens": 50, "output_tokens": 4}))
        return trace

    def test_actual_wire_verification_survives_lossless_archive(self):
        trace = self.trace(); first = queue.verify_trace(trace, "full-retrievable")
        self.assertEqual(queue.archive.archive_completed(trace.parent), 1)
        self.assertFalse((trace / "request.json").exists())
        self.assertEqual(queue.verify_trace(trace, "full-retrievable"), first)

    def test_source_blob_corruption_is_rejected(self):
        trace = self.trace()
        next((trace.parent / "full-history-blobs").glob("*.gz")).write_bytes(b"corrupt")
        with self.assertRaises((OSError, ValueError)): queue.verify_trace(trace, "full-retrievable")

    def test_wrong_policy_or_actual_wire_is_rejected(self):
        trace = self.trace()
        with self.assertRaises(queue.SafetyHalt): queue.verify_trace(trace, "recent-observations")
        (trace / "request.json").write_text('{"different":"wire"}')
        with self.assertRaises(queue.SafetyHalt): queue.verify_trace(trace, "full-retrievable")

    def test_missing_unknown_receipt_is_not_assumed_zero(self):
        for result in ({"finished_at": 1}, {"finished_at": 1, "uncertain_actions": False}):
            with self.assertRaises(queue.SafetyHalt): queue.base.terminal_trial(self.protocol["trials"][0], result)

    def test_predecessor_pid_reuse_is_rejected(self):
        expected = dict(pid=99, uid=1001, start_ticks=1, command=["old"], cwd=str(queue.ROOT), state="S")
        actual = dict(expected, start_ticks=2)
        with patch.object(queue.owned, "process_identity", return_value=actual), self.assertRaises(queue.SafetyHalt):
            queue.base.process_retired(expected)

    def predecessor_fixture(self):
        identities = [{"pid": n} for n in (11, 12, 13)]
        p = dict(osworld_queue="os", memory_queue="kv", sdk_queue="sdk", followup="follow",
                 memory_registration="reg", sdk_controller_identity=identities[2])
        values = {key: {"pid": identity["pid"], "status": "waiting"}
                  for key, identity in zip(("os", "kv", "sdk"), identities)}
        values.update(follow={"continuation_controller_identity": identities[0]}, reg={"identity": identities[1]})
        return p, values

    def test_incomplete_predecessors_wait_without_model_or_retirement(self):
        p, values = self.predecessor_fixture()
        with patch.object(queue, "read", side_effect=lambda key: values[key]), \
                patch.object(queue.base, "require_identity") as identity, \
                patch.object(queue.owned, "retire_registered") as retire, \
                patch.object(queue.owned, "gpu_snapshot") as gpu:
            self.assertIsNone(queue.predecessors_ready(p))
        self.assertEqual(identity.call_count, 3)
        retire.assert_not_called(); gpu.assert_not_called()

    def test_halted_or_changed_predecessor_cannot_start_successor(self):
        for update in ({"status": "halted-no-further-dispatch"}, {"pid": 999}):
            p, values = self.predecessor_fixture(); values["kv"].update(update)
            with patch.object(queue, "read", side_effect=lambda key: values[key]), \
                    patch.object(queue.base, "require_identity"), self.assertRaises(queue.SafetyHalt):
                queue.predecessors_ready(p)

    def test_completed_but_live_controller_is_still_waiting(self):
        p, values = self.predecessor_fixture()
        for key in ("os", "kv", "sdk"): values[key]["status"] = "completed"
        with patch.object(queue, "read", side_effect=lambda key: values[key]), \
                patch.object(queue.base, "process_retired", return_value=False):
            self.assertIsNone(queue.predecessors_ready(p))

    def test_sdk_missing_or_wrong_order_cannot_be_admitted(self):
        state = dict(status="completed", protocol_sha256="sha", trials=[{"id": "wrong"}])
        with self.assertRaises(queue.SafetyHalt):
            queue.mixed_terminal_rows([{"id": "expected"}], state, self.directory, "sha")

    def test_collection_preserves_original_business_grade(self):
        row = self.protocol["trials"][0]; output = self.directory / "attempts" / row["id"]
        (output / "app").mkdir(parents=True)
        grade = dict(finished_at=1, uncertain_actions=0, score=0, accepted=False)
        result_path = output / "result.json"; queue.save(result_path, grade)
        sha = queue.owned.digest(result_path)
        queue.save(output / "app/task.json", dict(image_history=row["policy"]))
        trace = self.trace()
        self.args.state_dir.mkdir()
        controller = queue.Controller(self.args, self.protocol, {"trials": []})
        with patch.object(queue.owned, "assert_no_services"), patch.object(queue.owned, "assert_vm_retired"):
            controller.collect(row, output, 0, trace.parent)
        self.assertEqual(queue.owned.digest(result_path), sha)
        self.assertEqual(controller.state["trials"][0]["result"], grade)
        self.assertTrue(queue.read(output / "visual-context-evidence.json")["business_grade_unchanged"])


if __name__ == "__main__": unittest.main()
