"""Exact candidate and real frozen queue boundaries, without model/device work."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bounded_continuation_retest_queue as queue


class BoundedQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / "plan.json"
        self.declaration = self.root / "declaration.json"
        for path in (self.plan, self.declaration): queue.save(path, {})
        self.protocol = dict(trials=queue.trials(), plan=str(self.plan), parent_declaration=str(self.declaration),
                             parent_queue=str(self.root / "parent.json"), runner_python=str(queue.q.quota.PYTHON),
                             frameworks=dict(control="/control", candidate="/candidate"),
                             acceptances=dict(control=dict(path="/control.json"), candidate=dict(path="/candidate.json")),
                             parent_app="fixture", engine_off_identity={"profile": "off"})
        self.args = argparse.Namespace(protocol=self.root / "protocol.json", state_dir=self.root / "state",
                                       wait_seconds=60, resource_wait_seconds=60, min_free_mib=56000)
        queue.save(self.args.protocol, self.protocol)
        queue.save(Path(self.protocol["parent_queue"]), dict(status="running"))

    def test_exact_original_plan_and_candidate_have_qualified_sources(self):
        path = queue.ROOT / "docs/computer_use_bounded_continuation_retest_plan_20261005.json"
        plan, build = queue.plan_proof(path)
        self.assertEqual(plan["trials"], queue.trials())
        self.assertEqual(build["changed_files"], ["app.py", "bounded_turn_continuation.py"])

    def test_original_native_batch_qualifies_exact_candidate_only(self):
        plan, _ = queue.plan_proof(queue.ROOT / "docs/computer_use_bounded_continuation_retest_plan_20261005.json")
        result = queue.native_proof(queue.RUNTIME / "bounded-turn-native-03/result.json", plan)
        self.assertEqual(result["runs"], 24)
        self.assertEqual(result["qw3_model_calls"], 0)

    def test_changed_or_unqualified_native_batch_is_rejected(self):
        plan = dict(candidate_manifest_sha256="exact")
        path = self.root / "native.json"
        for result in (dict(passed=False), dict(kind="actual_codex_bounded_turn_native_batch", passed=True, runs=23)):
            queue.save(path, result)
            with self.assertRaises(queue.SafetyHalt): queue.native_proof(path, plan)

    def test_frozen_plan_cannot_change_budget_or_order(self):
        path = self.root / "changed-plan.json"
        queue.save(path, dict(trials=queue.trials()[::-1]))
        with self.assertRaises(queue.SafetyHalt): queue.plan_proof(path)

    def test_eight_trials_have_balanced_arms_and_unchanged_budget(self):
        for task in ("051", "004"):
            for repeat in (1, 2):
                rows = [r for r in queue.trials() if r["task"] == task and r["repetition"] == repeat]
                self.assertEqual([r["arm"] for r in rows], ["control", "candidate"] if repeat == 1 else ["candidate", "control"])
                self.assertTrue(all((r["seconds"], r["tools"], r["execution_mode"]) == (1800, 250, "verified") for r in rows))

    def test_runner_uses_qualified_venv_and_gui_only_original_entry(self):
        controller = queue.Controller(self.args, self.protocol, dict(trials=[], engine_runtime="/engine"))
        for row in queue.trials():
            command = controller.trial_command(row, Path("/out"))
            self.assertEqual(command[:3], [str(queue.q.quota.PYTHON), "-m", "osworld_office_candidate_run"])
            self.assertEqual(command[command.index("--arm") + 1], "verified")
            self.assertEqual(command[command.index("--framework") + 1], self.protocol["frameworks"][row["arm"]])
            self.assertNotIn("--software", command)
            self.assertEqual(command[-2:], ["--model-output-tokens", "16384"])

    def loop(self, predecessor, dispatch=None):
        with patch.object(queue, "validate", return_value=self.protocol), \
                patch.object(queue, "terminal_predecessor", side_effect=predecessor), \
                patch.object(queue.Controller, "dispatch", side_effect=dispatch) as send, \
                patch.object(queue.owned.subprocess, "Popen", side_effect=AssertionError("No launch permitted")):
            return queue.run(self.args), send

    def test_real_frozen_eight_loop_completes_in_declared_order(self):
        sent = []
        result, send = self.loop(lambda *_: {"services": "fixture"}, lambda row: sent.append(copy.deepcopy(row)))
        self.assertEqual(sent, queue.trials())
        self.assertEqual(send.call_count, 8)
        self.assertEqual(result["kind"], "serial_matched_bounded_continuation_retest")
        self.assertTrue(result["termination_confirmed"])
        self.assertFalse(result["business_success_improvement_proven"])

    def test_missing_predecessor_stops_real_loop_without_dispatch(self):
        with patch.object(queue, "validate", return_value=self.protocol), \
                patch.object(queue, "terminal_predecessor", side_effect=queue.SafetyHalt("Actual parent absent")), \
                patch.object(queue.Controller, "dispatch") as send:
            with self.assertRaises(queue.SafetyHalt): queue.run(self.args)
            send.assert_not_called()
        self.assertEqual(queue.read(self.args.state_dir / "queue.json")["status"], "halted-no-further-dispatch")

    def test_new_failure_preserves_prior_attempts_and_stops_third_dispatch(self):
        sent = []
        def send(row):
            sent.append(row["id"])
            if len(sent) == 3: raise queue.SafetyHalt("new unknown")
        with self.assertRaises(queue.SafetyHalt): self.loop(lambda *_: {}, send)
        self.assertEqual(sent, [r["id"] for r in queue.trials()[:3]])
        state = queue.read(self.args.state_dir / "queue.json")
        self.assertTrue(state["live_children_and_evidence_preserved"])

    def test_existing_attempt_never_starts_new_engine_or_runner(self):
        controller = queue.Controller(self.args, self.protocol, dict(trials=[], protocol_sha256=queue.owned.digest(self.args.protocol)))
        out = self.args.protocol.parent / "attempts" / queue.trials()[0]["id"]
        out.mkdir(parents=True)
        with patch.object(queue, "validate", return_value=self.protocol), patch.object(controller, "parent_idle"), \
                patch.object(controller, "fresh_engine") as engine, patch.object(controller, "spawn") as launch:
            with self.assertRaises(queue.SafetyHalt): controller.dispatch(queue.trials()[0])
            engine.assert_not_called(); launch.assert_not_called()

    def task(self, arm, attempts=None):
        out = self.root / arm
        (out / "app/inference/trace").mkdir(parents=True)
        task = dict(thinking=True, model_output_tokens=16384, tool_call_limit=250, execution_mode="verified",
                    model_context_window=524288, thread_id="one-thread")
        if arm == "candidate":
            from bounded_turn_continuation import POLICY
            task["bounded_turn_continuation"] = dict(policy=copy.deepcopy(POLICY), attempts=attempts or [])
        queue.save(out / "app/task.json", task)
        queue.save(out / "app/inference/trace/request.json", dict(enable_thinking=True, max_tokens=16384))
        return out

    def test_actual_task_evidence_does_not_promote_continuation_to_success(self):
        for arm in ("control", "candidate"):
            evidence = queue.task_evidence(self.task(arm), arm)
            self.assertEqual(evidence["additional_turns"], 0)
            self.assertTrue(evidence["continuation_is_not_business_success"])
            self.assertFalse(evidence["task_budget_reset"])

    def test_continuation_limit_and_thread_changes_are_rejected(self):
        out = self.task("candidate", [dict(completed_turn_id=str(i), thread_id="one-thread", delivery="accepted") for i in range(3)])
        with self.assertRaises(queue.SafetyHalt): queue.task_evidence(out, "candidate")
        task = queue.read(out / "app/task.json")
        task["bounded_turn_continuation"]["attempts"] = [dict(completed_turn_id="first", thread_id="other")]
        queue.save(out / "app/task.json", task)
        with self.assertRaises(queue.SafetyHalt): queue.task_evidence(out, "candidate")

    def test_missing_actual_request_or_changed_output_mode_is_rejected(self):
        out = self.task("candidate")
        request = out / "app/inference/trace/request.json"
        queue.save(request, dict(enable_thinking=False, max_tokens=16384))
        with self.assertRaises(queue.SafetyHalt): queue.task_evidence(out, "candidate")
        request.unlink()
        with self.assertRaises(queue.SafetyHalt): queue.task_evidence(out, "candidate")

    def test_unknown_input_prevents_collecting_or_following_trial(self):
        controller = queue.Controller(self.args, self.protocol, dict(trials=[]))
        output = self.root / "out"
        output.mkdir()
        queue.save(output / "result.json", dict(uncertain_actions=1, finished_at=1))
        with self.assertRaises(queue.SafetyHalt): controller.collect(queue.trials()[0], output, 0)
        self.assertEqual(controller.state["trials"], [])

    def test_control_cannot_silently_use_candidate_lifecycle(self):
        out = self.task("control")
        task = queue.read(out / "app/task.json")
        task["bounded_turn_continuation"] = {"attempts": []}
        queue.save(out / "app/task.json", task)
        with self.assertRaises(queue.SafetyHalt): queue.task_evidence(out, "control")

    def test_actual_registered_wait_is_admitted_without_deployment(self):
        parent = dict(identity={"pid": 120})
        p = dict(self.protocol, parent_registration="/registration")
        state = dict(status="running")
        with patch.object(queue.q, "validate", return_value=({}, {}, {})), \
                patch.object(queue, "registration", return_value=parent), \
                patch.object(queue.q, "successor_queue", return_value=state) as actual, \
                patch.object(queue.owned.subprocess, "Popen", side_effect=AssertionError("No launch permitted")):
            self.assertIsNone(queue.terminal_predecessor(p, state))
            actual.assert_called_once()


if __name__ == "__main__": unittest.main()
