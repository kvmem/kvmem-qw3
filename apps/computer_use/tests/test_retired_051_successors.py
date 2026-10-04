"""Admission and real frozen-loop waiting tests; no model/device operations."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retired_051_successors as q


class TerminalTests(unittest.TestCase):
    def setUp(self):
        ids = [f"trial-{i}" for i in range(40)]
        ids[11], ids[18] = list(q.parent.CASES)
        self.p = dict(diagnostic_first_trials=[], trials=[dict(id=id) for id in ids])
        self.results = {id: dict(finished_at=10, uncertain_actions=int(id in q.parent.CASES), accepted=False) for id in ids}
        self.state = dict(status="completed", historical_unknowns_preserved=2,
                          trials=[dict(id=id, result=self.results[id]) for id in ids])
        self.initial = copy.deepcopy(self.state["trials"][:19])

    def admit(self): return q.terminal_rows(self.p, self.state, self.initial, self.results)

    def test_all_forty_keep_two_unknowns_and_nineteen_original_results(self):
        rows = self.admit(); self.assertEqual(len(rows), 40)
        self.assertEqual(sum(r["result"]["uncertain_actions"] for r in rows), 2)
        rows[0]["result"]["accepted"] = True
        self.assertFalse(self.state["trials"][0]["result"]["accepted"])

    def test_new_positive_business_result_is_allowed(self):
        self.results["trial-25"]["accepted"] = True
        self.assertTrue(self.admit()[25]["result"]["accepted"])

    def test_new_unknown_stops_all_successors(self):
        self.results["trial-25"]["uncertain_actions"] = 1
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_old_unknown_cannot_be_cleared(self):
        self.results["051-verified-r1"]["uncertain_actions"] = 0
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_old_failed_result_cannot_be_regraded(self):
        self.results["051-verified-r1"]["accepted"] = True
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_boolean_unknown_count_rejected(self):
        self.results["trial-25"]["uncertain_actions"] = False
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_missing_actual_result_rejected(self):
        del self.results["trial-25"]
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_unfinished_result_rejected(self):
        self.results["trial-25"].pop("finished_at")
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_unconfirmed_retirement_rejected(self):
        self.results["trial-25"]["owned_task_stop_unconfirmed"] = True
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_duplicate_or_reordered_trial_rejected(self):
        self.state["trials"][25], self.state["trials"][26] = self.state["trials"][26], self.state["trials"][25]
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_partial_inventory_rejected(self):
        self.state["trials"].pop()
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_running_forty_rows_not_terminal(self):
        self.state["status"] = "running"
        with self.assertRaises(q.SafetyHalt): self.admit()


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(pid=120, uid=1001, start_ticks=123, cwd=str(q.owned.ROOT), command=["python", "owned"], state="S")
        self.state = dict(pid=120, status="running", declaration="sha", trials=[])

    def guard(self, actual):
        with patch.object(q.owned, "process_identity", return_value=actual):
            return q.guard_queue(self.state, self.identity, "sha", "declaration")

    def test_actual_live_full_identity_is_required(self): self.assertEqual(self.guard(self.identity), self.state)

    def test_running_state_without_process_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.guard(None)

    def test_pid_reuse_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.guard(dict(self.identity, start_ticks=124))

    def test_running_zombie_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.guard(dict(self.identity, state="Z"))

    def test_halted_parent_rejected_even_with_live_pid(self):
        self.state["status"] = "halted-no-dispatch"
        with self.assertRaises(q.SafetyHalt): self.guard(self.identity)

    def test_wrong_declaration_or_queue_pid_rejected(self):
        for key, value in (("declaration", "changed"), ("pid", 121)):
            with self.subTest(key=key):
                state = dict(self.state, **{key: value})
                with patch.object(q.owned, "process_identity", return_value=self.identity):
                    with self.assertRaises(q.SafetyHalt): q.guard_queue(state, self.identity, "sha", "declaration")

    def test_completed_producer_may_be_retired(self):
        self.state["status"] = "completed"; self.assertEqual(self.guard(None), self.state)

    def test_former_zero_trial_waiter_is_read_only_admitted(self):
        state = dict(pid=120, status="halted-no-dispatch", trials=[])
        with patch.object(q.owned, "process_identity", return_value=None): q.former_waiter(state, self.identity)

    def test_former_waiter_with_trial_child_or_pid_reuse_rejected(self):
        for change in (dict(trials=[{}]), dict(child={"pid": 9}), dict(engine_launcher_pid=9)):
            with patch.object(q.owned, "process_identity", return_value=None):
                with self.assertRaises(q.SafetyHalt):
                    q.former_waiter(dict(dict(pid=120, status="halted-no-dispatch", trials=[]), **change), self.identity)
        with patch.object(q.owned, "process_identity", return_value=self.identity):
            with self.assertRaises(q.SafetyHalt): q.former_waiter(dict(pid=120, status="halted-no-dispatch", trials=[]), self.identity)


class FrozenRunnerTests(unittest.TestCase):
    def test_real_sdk_wait_loop_never_deploys_when_memory_producer_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); followup = root / "followup.json"; mr_path = root / "mr.json"; protocol = root / "sdk.json"
            declaration = root / "quarantine.json"
            for path in (followup, mr_path, protocol, declaration): q.save(path, {})
            memory_dir = root / "memory"; memory_dir.mkdir()
            q.save(memory_dir / "queue.json", dict(pid=120, status="waiting", continued_successor_declaration_sha256=q.owned.digest(followup)))
            identity = dict(pid=120, uid=1001, start_ticks=123, cwd=str(q.owned.ROOT), command=["python", "owned"], state="S")
            d = dict(sdk_protocol=str(protocol), sdk_state_dir=str(root / "state"), memory_state_dir=str(memory_dir),
                     continued_osworld_queue=str(root / "continued.json"), continuation_controller_identity=identity)
            args = argparse.Namespace(followup=followup, memory_controller=mr_path, wait_seconds=60,
                                      resource_wait_seconds=60, min_free_mib=56000)
            with patch.object(q, "validate_followup", return_value=(d, None, None, None)), \
                    patch.object(q, "registration", return_value={"identity": identity}), \
                    patch.object(q.sdk, "validate", return_value=({"quarantine_declaration": str(declaration),
                        "successor_wrapper_sha256": "original-wrapper", "quarantine_declaration_sha256": "original-quarantine"}, {})), \
                    patch.object(q, "continued_parent", return_value={"status": "running"}), \
                    patch.object(q.owned, "process_identity", return_value=None), \
                    patch.object(q.sdk.Controller, "deploy") as deploy, patch.object(q.sdk.Controller, "restore_engine") as restore, \
                    patch.object(q.owned.subprocess, "Popen") as launch:
                with self.assertRaises(q.SafetyHalt): q.run_sdk(args, d)
                deploy.assert_not_called(); restore.assert_not_called(); launch.assert_not_called()
            state = q.read(root / "state/queue.json")
            self.assertEqual(state["trials"], []); self.assertEqual(state["planned_trials"], 10)
            self.assertEqual(state["status"], "halted-no-further-dispatch")

    def test_real_image_loop_preserves_all_services_on_predecessor_halt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); followup = root / "followup.json"; mr_path = root / "mr.json"; sr_path = root / "sr.json"
            protocol = root / "image.json"
            for path in (followup, mr_path, sr_path, protocol): q.save(path, {})
            identity = dict(pid=120, cwd=str(q.owned.ROOT), command=["python", q.MODULE, "--memory-controller", str(mr_path)])
            d = dict(image_protocol=str(protocol), image_state_dir=str(root / "state"), continued_osworld_queue=str(root / "os.json"),
                     memory_state_dir=str(root / "memory"), sdk_state_dir=str(root / "sdk"))
            args = argparse.Namespace(followup=followup, memory_controller=mr_path, sdk_controller=sr_path,
                                      wait_seconds=60, resource_wait_seconds=60, min_free_mib=56000)
            with patch.object(q, "validate_followup", return_value=(d, None, None, None)), \
                    patch.object(q, "registration", return_value={"identity": identity}), \
                    patch.object(q, "IMAGE_VALIDATE", return_value={"trials": []}), \
                    patch.object(q, "image_ready", side_effect=q.SafetyHalt("predecessor halted")), \
                    patch.object(q.image.Controller, "dispatch") as dispatch, \
                    patch.object(q.image.Controller, "fresh_engine") as engine, patch.object(q.owned.subprocess, "Popen") as launch:
                with self.assertRaises(q.SafetyHalt): q.run_image(args, d)
                dispatch.assert_not_called(); engine.assert_not_called(); launch.assert_not_called()
            state = q.read(root / "state/queue.json")
            self.assertEqual(state["trials"], []); self.assertEqual(state["planned_trials"], 12)
            self.assertEqual(state["status"], "halted-no-further-dispatch")

    def test_memory_reuses_exact_source_except_reader_and_metadata(self):
        original = inspect.getsource(q.owned.run); changed = q.patched_memory_source(original)
        restored = changed.replace('previous = continued_parent(d, followup_sha, args.followup)',
            'previous = json.loads((parent_dir / "queue.json").read_text())').replace(
            '"initial_registered_engine": initial, "planned_trials": 8, "continued_successor_wrapper_sha256": wrapper_sha, '
            '"continued_successor_declaration_sha256": followup_sha}', '"initial_registered_engine": initial, "planned_trials": 8}')
        self.assertEqual(restored, original)
        compile(changed, "cpu-frozen-loop", "exec")

    def test_changed_or_duplicate_patch_anchor_rejected(self):
        original = inspect.getsource(q.owned.run)
        for changed in (original.replace('parent_dir / "queue.json"', 'parent_dir / "other.json"'), original + original):
            with self.assertRaises(q.SafetyHalt): q.patched_memory_source(changed)

    def test_real_memory_wait_loop_sends_no_inputs_when_parent_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); followup = root / "followup.json"; declaration = root / "decl.json"
            parent_protocol = root / "parent.json"; parent_queue = root / "continued_queue_02.json"
            memory_protocol = root / "memory.json"
            q.save(followup, {}); q.save(declaration, {}); q.save(parent_protocol, dict(engine_runtime=str(root / "engine")))
            q.save(memory_protocol, {})
            identity = dict(pid=120, uid=1001, start_ticks=123, cwd=str(q.owned.ROOT), command=["python", "owned"], state="S")
            q.save(parent_queue, dict(pid=120, status="running", historical_unknowns_preserved=2,
                                     continuation_declaration_sha256=q.owned.digest(declaration)))
            d = dict(memory_protocol=str(memory_protocol), memory_state_dir=str(root / "new-state"),
                     continued_osworld_queue=str(parent_queue), continuation_declaration=str(declaration),
                     continuation_controller_identity=identity)
            p = dict(parent_candidate_protocol=str(parent_protocol), engine_identities={"off": {}}, trials=[])
            args = argparse.Namespace(followup=followup, wait_seconds=60, resource_wait_seconds=60, min_free_mib=56000)
            with patch.object(q, "validate_followup", return_value=(d, None, None, None)), \
                    patch.object(q, "MEMORY_VALIDATE", return_value=p), \
                    patch.object(q.owned, "register_engine", return_value=identity), \
                    patch.object(q.owned, "process_identity", return_value=None), \
                    patch.object(q.owned.subprocess, "Popen") as launch, \
                    patch.object(q.owned, "retire_registered") as retire:
                with self.assertRaises(q.SafetyHalt): q.run_memory(args, d)
                launch.assert_not_called(); retire.assert_not_called()
            state = q.read(root / "new-state/queue.json")
            self.assertEqual(state["trials"], [])
            self.assertEqual(state["status"], "halted-no-further-dispatch")
            self.assertEqual(state["planned_trials"], 8)

    def test_image_wait_checks_all_three_producers_without_model_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); followup = root / "followup.json"; q.save(followup, {})
            identity = dict(pid=120, uid=1001, start_ticks=123, cwd=str(q.owned.ROOT), command=["python", "owned"], state="S")
            p = dict(memory_queue=str(root / "memory.json"), sdk_queue=str(root / "sdk.json"))
            for name in ("memory", "sdk"):
                q.save(root / (name + ".json"), dict(pid=120, status="waiting", continued_successor_declaration_sha256=q.owned.digest(followup)))
            with patch.object(q, "continued_parent", return_value=dict(status="running")), \
                    patch.object(q.owned, "process_identity", return_value=identity), \
                    patch.object(q, "terminal_parent") as terminal, patch.object(q.owned, "register_engine") as engine:
                self.assertIsNone(q.image_ready(p, {}, followup, {"identity": identity}, {"identity": identity}))
                terminal.assert_not_called(); engine.assert_not_called()

    def test_image_halted_sdk_prevents_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); followup = root / "followup.json"; q.save(followup, {})
            identity = dict(pid=120, uid=1001, start_ticks=123, cwd=str(q.owned.ROOT), command=["python", "owned"], state="S")
            p = dict(memory_queue=str(root / "memory.json"), sdk_queue=str(root / "sdk.json"))
            for name, status in (("memory", "completed"), ("sdk", "halted-no-dispatch")):
                q.save(root / (name + ".json"), dict(pid=120, status=status, continued_successor_declaration_sha256=q.owned.digest(followup)))
            with patch.object(q, "continued_parent", return_value=dict(status="completed")), \
                    patch.object(q.owned, "process_identity", return_value=None), patch.object(q, "terminal_parent") as terminal:
                with self.assertRaises(q.SafetyHalt): q.image_ready(p, {}, followup, {"identity": identity}, {"identity": identity})
                terminal.assert_not_called()


if __name__ == "__main__": unittest.main()
