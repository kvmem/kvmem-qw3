"""Strict retired-VM admission without model requests or VM commands."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import retired_051_continuation as q


class ScopedTests(unittest.TestCase):
    def setUp(self):
        ids = [f"trial-{i}" for i in range(40)]
        ids[11], ids[18] = "022-legacy-r1", "051-verified-r1"
        self.p = dict(diagnostic_first_trials=[], trials=[{"id": id} for id in ids])
        self.results = {id: dict(finished_at=10, uncertain_actions=1 if id in q.CASES else 0, accepted=False) for id in ids[:19]}
        self.state = dict(status="halted-no-further-model-dispatch", pid=123, current_trial="051-verified-r1",
                          trials=[dict(id=id, result=self.results[id]) for id in ids[:19]])
        self.initial = copy.deepcopy(self.state["trials"][:12])

    def admit(self): return q.halted_rows(self.p, self.state, self.initial, self.results, 123)

    def test_exact_nineteen_rows_keep_both_unknowns_without_replay(self):
        result = self.admit()
        self.assertEqual(result, self.state["trials"])
        self.assertEqual(sum(r["result"]["uncertain_actions"] for r in result), 2)
        self.assertEqual(len(q.original.ordered_trials(self.p)[len(result):]), 21)
        result[0]["result"]["accepted"] = True
        self.assertFalse(self.state["trials"][0]["result"]["accepted"])

    def test_new_third_unknown_is_rejected(self):
        self.results["trial-4"]["uncertain_actions"] = 1
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_unknown_cannot_be_cleared(self):
        self.results["051-verified-r1"]["uncertain_actions"] = 0
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_unknown_cannot_be_regraded_as_success(self):
        self.results["051-verified-r1"]["accepted"] = True
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_boolean_unknown_count_is_not_zero(self):
        self.results["trial-17"]["uncertain_actions"] = False
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_changed_or_missing_terminal_result_is_rejected(self):
        self.results["trial-17"] = dict(self.results["trial-17"], score=1)
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_order_or_duplicate_trial_is_rejected(self):
        self.state["trials"][16] = self.state["trials"][17]
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_original_twelve_rows_are_immutable(self):
        self.state["trials"][3]["result"]["score"] = 0.5
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_only_exact_halted_pid_is_admitted(self):
        self.state["pid"] = 124
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_still_running_checkpoint_is_rejected(self):
        self.state["status"] = "running"
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_unconfirmed_task_retirement_is_rejected(self):
        self.results["trial-17"]["owned_task_stop_unconfirmed"] = True
        with self.assertRaises(q.SafetyHalt): self.admit()

    def unknown_fixture(self, directory):
        (directory / "gateway").mkdir()
        request = dict(desktop_id="retired-test", action={"kind": "click", "x": 10, "y": 10})
        with sqlite3.connect(directory / "gateway/receipts.sqlite") as db:
            db.execute("CREATE TABLE actions(id TEXT,status TEXT,request TEXT,result TEXT)")
            db.execute("INSERT INTO actions VALUES (?,?,?,?)", ("retained", "uncertain", json.dumps(request), None))
        q.save(directory / "final_agent_state.json", dict(status="paused", desktop_paused=True,
               uncertain_actions=[dict(receipt_id="retained", status="uncertain", action_id="local-id")]))
        q.save(directory / "vm.json", dict(container_id="retired-container"))

    def test_readonly_unknown_inspection_preserves_receipt_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); self.unknown_fixture(directory)
            path = directory / "gateway/receipts.sqlite"; data = path.read_bytes()
            result = q.unknown(directory, "retained")
            self.assertEqual(result["receipt_id"], "retained")
            self.assertEqual(path.read_bytes(), data)

    def test_completed_unknown_row_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); self.unknown_fixture(directory)
            with sqlite3.connect(directory / "gateway/receipts.sqlite") as db:
                db.execute('UPDATE actions SET status="completed"')
            with self.assertRaises(q.SafetyHalt): q.unknown(directory, "retained")

    def execute_runner(self, new_unknown=False):
        with tempfile.TemporaryDirectory() as folder:
            campaign = Path(folder); parent = campaign / "continued_queue.json"
            q.save(parent, self.state); q.save(campaign / "queue.json", {"trials": self.initial})
            declaration = campaign / "declaration.json"
            d = dict(campaign=str(campaign), parent_queue=str(parent), continued_queue=str(campaign / "continued_queue_02.json"))
            q.save(declaration, d)
            p = dict(self.p, frameworks={"unchanged": "/private/framework"}, engine_runtime="/owned/engine",
                     formal_campaign="/frozen/formal", pilot_app="http://idle")
            p["trials"] = [dict(r, framework_name="unchanged", task="004", arm="verified") for r in p["trials"]]
            q.save(campaign / "protocol.json", p)
            dispatched = []
            class FakeChild:
                pid = 77
                def wait(self, **_): return 0
            def launch(command, **_):
                output = Path(command[command.index("--output") + 1]); output.mkdir(parents=True)
                dispatched.append(output.name)
                q.save(output / "result.json", dict(finished_at=20, uncertain_actions=int(new_unknown), accepted=False))
                return FakeChild()
            with patch.object(q, "_validate", return_value=(d, p, copy.deepcopy(self.state["trials"]))), \
                    patch.object(q.original.subprocess, "Popen", launch), \
                    patch.object(q.owned, "register_child", return_value={"pid": 77}), \
                    patch.object(q.owned, "assert_no_services"), patch.object(q.owned, "assert_vm_retired"):
                if new_unknown:
                    with self.assertRaises(q.SafetyHalt): q.run(declaration)
                else: q.run(declaration)
            state = q.read(campaign / "continued_queue_02.json")
            self.assertEqual(q.read(parent), self.state)
            self.assertIs(q.original.validate, q.ORIGINAL_VALIDATE)
            return dispatched, state

    def test_actual_frozen_loop_executes_only_remaining_twenty_one(self):
        dispatched, state = self.execute_runner()
        self.assertEqual(dispatched, [r["id"] for r in self.p["trials"][19:]])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["trials"][:19], self.state["trials"])
        self.assertEqual(state["historical_unknowns_preserved"], 2)

    def test_actual_frozen_loop_stops_on_any_new_unknown(self):
        dispatched, state = self.execute_runner(new_unknown=True)
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(len(state["trials"]), 20)
        self.assertEqual(state["status"], "halted-no-further-model-dispatch")
        self.assertEqual(state["historical_unknowns_preserved"], 2)


if __name__ == "__main__": unittest.main()
