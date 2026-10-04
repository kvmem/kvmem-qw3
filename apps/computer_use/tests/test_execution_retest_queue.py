import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import execution_retest_queue as queue


class AdmissionTests(unittest.TestCase):
    def test_client_scan_skips_init_and_requires_runtime_port_and_owner(self):
        runtime = Path("/tmp/expected-app")
        identity = {"pid": 2, "uid": queue.os.getuid(), "cwd": str(queue.ROOT), "state": "S",
                    "command": ["python", "-m", "qw3_computer.app", "--runtime", str(runtime), "--port", "18213"]}
        entries = [SimpleNamespace(name="1"), SimpleNamespace(name="other"), SimpleNamespace(name="2")]
        with patch.object(queue.owned, "process_identity", return_value=identity) as inspect:
            self.assertEqual(queue.find_client(runtime, 18213, entries), identity)
            inspect.assert_called_once_with(2)
            with self.assertRaises(queue.SafetyHalt):
                queue.find_client(runtime, 9999, entries)

    def receipts(self, directory):
        declared = [{"id": "trial-" + str(i)} for i in range(8)]
        rows = []
        for item in declared:
            path = directory / "attempts" / item["id"] / "result.json"
            path.parent.mkdir(parents=True)
            result = {"finished_at": 12, "uncertain_actions": 0, "score": 0}
            path.write_text(json.dumps(result))
            rows.append(dict(item, result=result, result_sha256=queue.owned.digest(path)))
        return declared, {"status": "completed", "protocol_sha256": "frozen", "trials": rows}

    def test_eight_original_failed_but_terminal_receipts_are_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            declared, state = self.receipts(Path(temp))
            self.assertEqual(len(queue.exact_rows(declared, state, Path(temp), "frozen")), 8)

    def test_missing_or_duplicate_kvmem_attempt_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            declared, state = self.receipts(Path(temp))
            state["trials"][-1] = state["trials"][0]
            with self.assertRaises(queue.SafetyHalt):
                queue.exact_rows(declared, state, Path(temp), "frozen")

    def test_receipt_hash_tampering_is_refused_even_with_matching_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declared, state = self.receipts(root)
            state["trials"][0]["result"]["score"] = 1
            (root / "attempts/trial-0/result.json").write_text(json.dumps(state["trials"][0]["result"]))
            with self.assertRaises(queue.SafetyHalt):
                queue.exact_rows(declared, state, root, "frozen")

    def test_pid_reuse_is_preserved_and_not_treated_as_retirement(self):
        expected = dict(pid=123, uid=1001, start_ticks=1, command=["old"], cwd="/old")
        actual = dict(expected, start_ticks=2, command=["new"], state="S")
        with patch.object(queue.owned, "process_identity", return_value=actual), \
                patch.object(queue.owned, "retire_registered") as retire:
            with self.assertRaises(queue.SafetyHalt):
                queue.process_retired(expected)
            retire.assert_not_called()

    def test_readonly_ledger_refuses_pending_and_keeps_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.closing(sqlite3.connect(root / "receipts.sqlite")) as db:
                with db:
                    db.execute("CREATE TABLE actions(id TEXT,status TEXT)")
                    db.executemany("INSERT INTO actions VALUES (?,?)", [("done", "completed"), ("lost", "uncertain")])
            before = (root / "receipts.sqlite").read_bytes()
            with self.assertRaises(queue.SafetyHalt):
                queue.ledger_terminal(root)
            self.assertEqual((root / "receipts.sqlite").read_bytes(), before)

    def test_completed_and_rejected_ledger_rows_allow_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.closing(sqlite3.connect(root / "receipts.sqlite")) as db:
                with db:
                    db.execute("CREATE TABLE actions(id TEXT,status TEXT)")
                    db.executemany("INSERT INTO actions VALUES (?,?)", [("done", "completed"), ("not-sent", "rejected")])
            self.assertEqual(queue.ledger_terminal(root), {"completed": 1, "rejected": 1})

    def test_pending_gateway_blocks_deployment_before_any_retirement(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(state_dir=Path(temp), protocol=Path(temp) / "protocol.json")
            p = {"previous_app": "http://fixture", "previous_catalogue_sha256": "old",
                 "previous_app_identity": {"pid": 1},
                 "previous_gateways": {"linux": {"identity": {"pid": 2}, "item": {"token_file": "/tmp/token"}}}}
            controller = queue.Controller(args, p, {})
            with patch.object(queue.owned, "model_lock", side_effect=contextlib.nullcontext), \
                    patch.object(queue.owned, "assert_apps_idle"), \
                    patch.object(queue.owned, "digest", return_value="old"), \
                    patch.object(queue, "require_identity"), \
                    patch.object(queue, "ledger_terminal", side_effect=queue.SafetyHalt("unknown")), \
                    patch.object(queue.owned, "retire_registered") as retire, \
                    patch.object(queue, "gateway_api") as api:
                with self.assertRaises(queue.SafetyHalt):
                    controller.deploy()
                retire.assert_not_called()
                api.assert_not_called()


class TrialTests(unittest.TestCase):
    def test_missing_unknown_count_and_boolean_zero_are_refused(self):
        row = {"kind": "normal-pilot"}
        for result in ({"owned_task_retired": True},
                       {"owned_task_retired": True, "uncertain_actions": False}):
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_trial(row, result)

    def test_business_artifact_cannot_override_unknown_or_live_task(self):
        for result in ({"accepted": True, "uncertain_actions": 1, "owned_task_retired": True},
                       {"accepted": True, "uncertain_actions": 0, "owned_task_retired": False}):
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_trial({"kind": "normal-pilot"}, result)

    def test_failed_business_result_can_be_retired_without_becoming_pass(self):
        result = {"accepted": False, "uncertain_actions": 0, "owned_task_retired": True}
        queue.terminal_trial({"kind": "normal-pilot"}, result)
        self.assertFalse(result["accepted"])

    def test_admin_recovery_cannot_be_relabelled_unassisted_or_business(self):
        result = {"accepted": False, "uncertain_actions": 0, "owned_task_retired": True,
                  "administrative_recovery": True, "unassisted_recovery_claimed": False}
        queue.terminal_trial({"kind": "receipt-loss"}, result)
        for change in ({"accepted": True}, {"unassisted_recovery_claimed": True}, {"administrative_recovery": False}):
            with self.assertRaises(queue.SafetyHalt):
                queue.terminal_trial({"kind": "receipt-loss"}, dict(result, **change))

    def simulation(self, unknown_at=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root / "protocol.json"
            protocol.write_text("frozen")
            registry = root / "registry.json"
            registry.write_text("{}")
            kinds = ["receipt-loss"] * 2 + ["normal-pilot"] * 4 + ["receipt-loss"] * 2 + ["original-osworld"] * 2
            rows = [{"id": "trial-" + str(i), "kind": k, "device": "linux-vm", "case": "linux_text",
                     "seconds": 360, "task": "004", "arm": "verified"} for i, k in enumerate(kinds)]
            p = {"trials": rows, "app": "http://fixture", "app_runtime": str(root / "app"),
                 "framework": str(root / "framework"), "engine_off_identity": {},
                 "bundle": str(root / "bundle"), "acceptance": str(root / "acceptance")}
            args = SimpleNamespace(protocol=protocol, state_dir=root / "state", wait_seconds=60,
                                   resource_wait_seconds=60, min_free_mib=56000)
            dispatched = []
            def restore(controller, _):
                controller.checkpoint(engine_runtime=str(root / "engine"), registered_engine={})
            def deploy(controller):
                controller.checkpoint(registry=str(registry), registry_sha256=queue.owned.digest(registry), model_client={})
            def spawn(controller, command, name, framework=None):
                dispatched.append(name)
                row = rows[len(dispatched) - 1]
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                result = {"accepted": False, "finished_at": 12, "uncertain_actions": int(name == unknown_at),
                          "owned_task_retired": True, "administrative_recovery": True,
                          "unassisted_recovery_claimed": False}
                filename = "result.json" if row["kind"] == "original-osworld" else "integration-result.json"
                (output / filename).write_text(json.dumps(result))
                return SimpleNamespace(wait=lambda **_: 0), {}
            with patch.object(queue, "validate", return_value=(p, {})), \
                    patch.object(queue, "predecessors_ready", return_value={"registered_engine": {}}), \
                    patch.object(queue.Controller, "restore_engine", restore), \
                    patch.object(queue.Controller, "deploy", deploy), \
                    patch.object(queue.Controller, "spawn", spawn), \
                    patch.object(queue.owned, "assert_apps_idle"), \
                    patch.object(queue.owned, "assert_idle"), \
                    patch.object(queue.owned, "assert_no_services"), \
                    patch.object(queue.owned, "assert_vm_retired"), \
                    patch.object(queue.owned, "register_engine"), \
                    patch.object(queue, "require_identity"), \
                    patch.object(queue.owned, "model_lock", side_effect=contextlib.nullcontext):
                if unknown_at:
                    with self.assertRaises(queue.SafetyHalt):
                        queue.run(args)
                else:
                    queue.run(args)
            state = queue.read(args.state_dir / "queue.json")
            return dispatched, state

    def test_mock_ten_trial_queue_preserves_every_failed_result(self):
        dispatched, state = self.simulation()
        self.assertEqual(dispatched, ["trial-" + str(i) for i in range(10)])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["trials"]), 10)
        self.assertTrue(all(row["result"]["accepted"] is False for row in state["trials"]))

    def test_unknown_in_third_trial_stops_rest_of_queue(self):
        dispatched, state = self.simulation("trial-2")
        self.assertEqual(dispatched, ["trial-0", "trial-1", "trial-2"])
        self.assertEqual(state["status"], "halted-no-further-dispatch")
        self.assertEqual(len(state["trials"]), 2)
        self.assertTrue(state["live_children_and_evidence_preserved"])


if __name__ == "__main__":
    unittest.main()
