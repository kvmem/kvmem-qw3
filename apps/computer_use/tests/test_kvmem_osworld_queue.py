import json
import contextlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

import kvmem_osworld_queue as queue


class OwnedEngineTests(unittest.TestCase):
    def child(self):
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        p = subprocess.Popen(command)
        def cleanup():
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=5)
        self.addCleanup(cleanup)
        queue.register_child(p.pid, command)
        return p

    def test_changed_start_time_is_refused_without_signal(self):
        child = self.child()
        registered = queue.process_identity(child.pid)
        registered["start_ticks"] += 1
        with patch.object(queue.os, "kill") as kill:
            with self.assertRaises(queue.SafetyHalt):
                queue.retire_registered(registered)
            kill.assert_not_called()
        self.assertIsNone(child.poll())

    def test_changed_command_is_refused_without_signal(self):
        child = self.child()
        registered = queue.process_identity(child.pid)
        registered["command"] = ["another-user-job"]
        with patch.object(queue.os, "kill") as kill:
            with self.assertRaises(queue.SafetyHalt):
                queue.retire_registered(registered)
            kill.assert_not_called()
        self.assertIsNone(child.poll())

    def test_exact_owned_child_is_retired_and_reaped(self):
        child = self.child()
        receipt = queue.retire_registered(queue.process_identity(child.pid), timeout=5)
        self.assertTrue(receipt["retired"])
        self.assertEqual(child.wait(timeout=5), -15)


class QueueAdmissionTests(unittest.TestCase):
    def fixture(self, root):
        declarations = [{"id": f"trial-{i}"} for i in range(40)]
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({"trials": declarations}))
        p = {"parent_candidate_protocol": str(protocol),
             "parent_candidate_protocol_sha256": queue.digest(protocol)}
        rows = []
        for declaration in declarations:
            directory = root / "attempts" / declaration["id"]
            directory.mkdir(parents=True)
            result = {"finished_at": 123, "uncertain_actions": 0}
            (directory / "result.json").write_text(json.dumps(result))
            rows.append({"id": declaration["id"], "result": result})
        return p, {"status": "completed", "trials": rows,
                   "protocol_sha256": p["parent_candidate_protocol_sha256"]}

    def test_all_forty_actual_receipts_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            p, state = self.fixture(Path(temp))
            self.assertEqual(len(queue.terminal_parent(p, state)), 40)
            state["trials"].pop()
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_parent(p, state)

    def test_saved_result_change_is_not_hidden_by_completed_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p, state = self.fixture(root)
            (root / "attempts/trial-0/result.json").write_text(json.dumps({"finished_at": 456, "uncertain_actions": 0}))
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_parent(p, state)

    def test_any_uncertain_input_blocks_even_if_score_would_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p, state = self.fixture(root)
            state["trials"][0]["result"].update(uncertain_actions=1, score=1, accepted=True)
            (root / "attempts/trial-0/result.json").write_text(json.dumps(state["trials"][0]["result"]))
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_parent(p, state)

    def test_duplicate_trial_does_not_replace_a_missing_one(self):
        with tempfile.TemporaryDirectory() as temp:
            p, state = self.fixture(Path(temp))
            state["trials"][-1] = state["trials"][0]
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_parent(p, state)

    def test_missing_unknown_count_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p, state = self.fixture(root)
            del state["trials"][0]["result"]["uncertain_actions"]
            (root / "attempts/trial-0/result.json").write_text(json.dumps(state["trials"][0]["result"]))
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_parent(p, state)

    def test_paused_unknown_app_cannot_be_called_idle(self):
        with patch.object(queue, "request", return_value={"status": "stopped", "desktop_paused": True,
                                                          "uncertain_actions": [{"id": "unknown"}]}):
            with self.assertRaises(queue.SafetyHalt):
                queue.assert_idle("http://fixture")

    def test_optional_windows_connection_refused_is_distinct_from_timeout(self):
        idle = {"status": "idle", "desktop_paused": True, "uncertain_actions": []}
        with patch.object(queue, "request", side_effect=[idle, URLError(ConnectionRefusedError())]):
            queue.assert_apps_idle("http://fixture")
        with patch.object(queue, "request", side_effect=[idle, URLError(TimeoutError())]):
            with self.assertRaises(queue.SafetyHalt):
                queue.assert_apps_idle("http://fixture")

    def test_live_owned_vm_prevents_next_trial(self):
        body = [{"Config": {"Labels": {"qw3.goal": "cross-device-20261003"}},
                 "Name": "/qw3-linux-trial", "State": {"Running": True}}]
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(body), "")):
            with self.assertRaises(queue.SafetyHalt):
                queue.assert_vm_retired(Path("trial"))

    def test_docker_failure_is_not_evidence_that_vm_is_gone(self):
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "Cannot connect to Docker")):
            with self.assertRaises(queue.SafetyHalt):
                queue.assert_vm_retired(Path("trial"))
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "No such object: qw3-linux-trial")):
            queue.assert_vm_retired(Path("trial"))


class QueueLifecycleTests(unittest.TestCase):
    def exercise(self, root, uncertain=False):
        trials = [{"id": f"{task}-kvmem-{policy}-r{repeat}", "task": task, "kvmem": policy,
                   "arm": "verified", "repetition": repeat}
                  for repeat in (1, 2) for task in ("035", "080") for policy in ("off", "on")]
        parent = root / "parent"
        parent.mkdir()
        (parent / "protocol.json").write_text(json.dumps({"engine_runtime": str(root / "initial")}))
        (parent / "queue.json").write_text(json.dumps({"status": "completed", "pid": 100,
                                                     "trials": [{"id": f"parent-{i}"} for i in range(40)]}))
        protocol = root / "protocol.json"
        p = {"parent_candidate_protocol": str(parent / "protocol.json"), "trials": trials,
             "engine_identities": {"off": {"kvmem": "off"}, "on": {"kvmem": "on"}},
             "pilot_app": "http://fake-pilot", "engine_endpoint": "http://fake-engine",
             "framework": str(root / "framework"), "formal_campaign": str(root / "formal")}
        protocol.write_text(json.dumps(p))
        args = SimpleNamespace(protocol=protocol, state_dir=root / "state", wait_seconds=60,
                               resource_wait_seconds=60, min_free_mib=56000)
        commands, engines, retirements = [], {}, []
        class FakeChild:
            def __init__(self, command, **_):
                self.command = command
                self.pid = 1000 + len(commands)
                commands.append(command)
                if "--run-dir" in command:
                    engines[Path(command[command.index("--run-dir") + 1])] = self.pid
            def poll(self):
                return None
            def wait(self, timeout):
                if "--run-dir" in self.command:
                    return -15
                output = Path(self.command[self.command.index("--output") + 1])
                output.mkdir(parents=True)
                (output / "result.json").write_text(json.dumps({"finished_at": 123,
                    "uncertain_actions": 1 if uncertain else 0, "accepted": False}))
                return 0
        def register(runtime, _):
            return {"pid": engines.get(Path(runtime), 99), "command": ["registered-fixture"]}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(queue, "validate_protocol", return_value=p))
            stack.enter_context(patch.object(queue, "register_engine", side_effect=register))
            stack.enter_context(patch.object(queue, "process_identity", return_value=None))
            stack.enter_context(patch.object(queue, "terminal_parent", return_value=[{"id": "fixture"}]))
            stack.enter_context(patch.object(queue, "assert_apps_idle"))
            stack.enter_context(patch.object(queue, "assert_no_services"))
            stack.enter_context(patch.object(queue, "assert_vm_retired"))
            stack.enter_context(patch.object(queue, "model_lock", side_effect=contextlib.nullcontext))
            stack.enter_context(patch.object(queue, "retire_registered", side_effect=lambda evidence:
                retirements.append(evidence["pid"]) or {"retired": True}))
            stack.enter_context(patch.object(queue, "gpu_snapshot", return_value={"free_mib": 60000}))
            stack.enter_context(patch.object(queue, "request", return_value={"data": [{"context_window": 524288}]}))
            stack.enter_context(patch.object(queue, "register_child", side_effect=lambda pid, command: {"pid": pid}))
            stack.enter_context(patch.object(queue.subprocess, "Popen", side_effect=FakeChild))
            if uncertain:
                with self.assertRaises(queue.SafetyHalt):
                    queue.run(args)
                result = json.loads((args.state_dir / "queue.json").read_text())
            else:
                result = queue.run(args)
        return result, commands, retirements, trials

    def test_fresh_engine_per_trial_and_original_runner_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            result, commands, retirements, trials = self.exercise(Path(temp))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["trials"]), 8)
            self.assertEqual(len(retirements), 8)
            self.assertEqual(len(commands), 16)
            for index, trial in enumerate(trials):
                engine, runner = commands[2 * index:2 * index + 2]
                self.assertEqual(engine[engine.index("--kvmem") + 1], trial["kvmem"])
                self.assertEqual(engine[engine.index("--prefix-reuse") + 1], "off")
                self.assertIn("osworld_office_candidate_run", runner)
                self.assertEqual(runner[runner.index("--task") + 1], trial["task"])
                self.assertEqual(runner[runner.index("--model-output-tokens") + 1], "16384")

    def test_uncertain_first_trial_preserves_engine_and_never_dispatches_second(self):
        with tempfile.TemporaryDirectory() as temp:
            result, commands, retirements, _ = self.exercise(Path(temp), uncertain=True)
            self.assertEqual(result["status"], "halted-no-further-dispatch")
            self.assertTrue(result["engine_preserved"])
            self.assertEqual(len(commands), 2)
            self.assertEqual(len(retirements), 1)
            self.assertEqual(result["trials"], [])


if __name__ == "__main__":
    unittest.main()
