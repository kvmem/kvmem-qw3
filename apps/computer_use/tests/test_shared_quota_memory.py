import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import shared_quota_memory as q


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "declaration.json"; q.owned.save(self.path, {})
        self.prefix = [dict(id="035-kvmem-off-r1", result_sha256="unchanged", result=dict(accepted=False))]
        previous = self.root / "previous.json"; q.owned.save(previous, dict(completed_prefix=self.prefix))
        self.d = dict(previous_continuation=str(previous), previous_registrations={})
        self.receipt = dict(kind="exact_idle_quota_transition_retirement", passed=True,
                            declaration_sha256=q.owned.digest(self.path), old_memory_completed_prefix=1,
                            model_calls=0, device_inputs=0, retired=[])
        for index, stage in enumerate(q.STAGES):
            reg = self.root / (stage + "-registration.json"); queue = self.root / (stage + "-queue.json")
            identity = dict(pid=100 + index, uid=os.getuid(), start_ticks=120, cwd=str(q.ROOT), command=["owned", stage])
            q.owned.save(reg, dict(identity=identity, queue=str(queue)))
            q.owned.save(queue, dict(pid=identity["pid"], status="halted-no-further-dispatch",
                                    trials=self.prefix if stage == "memory" else []))
            self.d["previous_registrations"][stage] = str(reg)
            self.receipt["retired"].append(dict(stage=stage, identity=identity, registration=str(reg),
                registration_sha256=q.owned.digest(reg), queue=str(queue), queue_sha256=q.owned.digest(queue)))

    def check(self, actual=None):
        with patch.object(q.owned, "process_identity", return_value=actual):
            q.retirement_evidence(self.d, self.path, self.receipt)

    def change_queue(self, stage, **changes):
        row = next(r for r in self.receipt["retired"] if r["stage"] == stage)
        p = Path(row["queue"]); state = q.read(p); state.update(changes); q.owned.save(p, state)
        row["queue_sha256"] = q.owned.digest(p)

    def test_exact_retirement_preserves_unaccepted_original_prefix(self):
        self.check(); self.assertFalse(q.read(self.receipt["retired"][0]["queue"])["trials"][0]["result"]["accepted"])

    def test_duplicate_stage_cannot_replace_another_predecessor(self):
        self.receipt["retired"][4] = copy.deepcopy(self.receipt["retired"][3])
        with self.assertRaises(q.SafetyHalt): self.check()

    def test_different_process_with_same_pid_is_not_receipted(self):
        self.receipt["retired"][0]["identity"]["start_ticks"] += 1
        with self.assertRaises(q.SafetyHalt): self.check()

    def test_live_or_zombie_predecessor_prevents_admission(self):
        for state in ("S", "Z"):
            with self.subTest(state=state), self.assertRaises(q.SafetyHalt): self.check(dict(state=state))

    def test_a_successor_that_dispatched_work_cannot_be_called_idle(self):
        self.change_queue("sdk", trials=[dict(id="dispatched")])
        with self.assertRaises(q.SafetyHalt): self.check()

    def test_a_changed_old_result_cannot_be_reused_as_prefix(self):
        self.change_queue("memory", trials=[dict(id="035-kvmem-off-r1", result=dict(accepted=True))])
        with self.assertRaises(q.SafetyHalt): self.check()

    def test_active_child_or_engine_field_prevents_admission(self):
        for key in ("child", "current_child", "registered_engine", "engine_launcher_pid"):
            self.change_queue("memory", **{key: 123})
            with self.subTest(key=key), self.assertRaises(q.SafetyHalt): self.check()
            self.change_queue("memory", **{key: None})

    def test_queue_changes_after_receipt_are_detected(self):
        p = Path(self.receipt["retired"][0]["queue"]); p.write_text(p.read_text() + " ")
        with self.assertRaises(q.SafetyHalt): self.check()


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.profile_path = self.root / "profile.json"
        profile = dict(engine_identities=dict(on=dict(command=["frozen", "--quota", "1.0"], binary_sha256="fixed", qw3_env={})))
        q.owned.save(self.profile_path, profile); self.profile = profile
        runtime = self.root / "engine"; runtime.mkdir()
        self.identity = dict(pid=120, uid=os.getuid(), start_ticks=123, cwd=str(q.ROOT), command=profile["engine_identities"]["on"]["command"])
        manifest = dict(pid=120, owns_process_only=True, shared_quota_profile_sha256=q.owned.digest(self.profile_path),
                        shared_quota_wrapper_sha256=q.owned.digest(q.engine.__file__))
        q.owned.save(runtime / "manifest.json", manifest)
        self.d = dict(profile=str(self.profile_path))
        self.receipt = dict(kind="actual_common_quota_startup_qualification", passed=True,
            profile_sha256=q.owned.digest(self.profile_path), owned_engine_actual_retired=True, task_trials_dispatched=0,
            device_inputs=0, health_model_calls=1, health_response="READY", advertised_context_tokens=524288,
            wrapper_sha256=q.owned.digest(q.engine.__file__), engine_profile=profile["engine_identities"]["on"],
            engine_identity=self.identity, engine_runtime=str(runtime), manifest_sha256=q.owned.digest(runtime / "manifest.json"),
            startup_gpu=dict(free_mib=56000))

    def check(self, actual=None):
        with patch.object(q.engine, "qualify_profile", return_value=self.profile), \
                patch.object(q.owned, "engine_identity", return_value=self.profile["engine_identities"]["on"]), \
                patch.object(q.owned, "process_identity", return_value=actual):
            q.startup_evidence(self.d, self.receipt)

    def test_owned_startup_health_and_retirement_are_required(self):
        self.check()
        with self.assertRaises(q.SafetyHalt): self.check(self.identity)

    def test_health_context_and_quota_are_not_inferred_from_process_alive(self):
        for key, value in (("health_response", ""), ("health_model_calls", 0), ("advertised_context_tokens", 131072),
                           ("engine_profile", {}), ("task_trials_dispatched", 1)):
            before = self.receipt[key]; self.receipt[key] = value
            with self.subTest(key=key), self.assertRaises(q.SafetyHalt): self.check()
            self.receipt[key] = before

    def test_low_headroom_or_changed_manifest_is_rejected(self):
        self.receipt["startup_gpu"]["free_mib"] = 55999
        with self.assertRaises(q.SafetyHalt): self.check()
        self.receipt["startup_gpu"]["free_mib"] = 56000
        self.receipt["manifest_sha256"] = "different"
        with self.assertRaises(q.SafetyHalt): self.check()


class NaturalCapacityFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        protocol=self.root/"protocol.json";q.owned.save(protocol,dict(engine_identities=dict(on={})))
        previous=self.root/"previous.json";q.owned.save(previous,dict(continued_protocol=str(protocol)))
        self.d=dict(previous_continuation=str(previous))
        runtime=self.root/"engines/035-kvmem-on-r1";runtime.mkdir(parents=True)
        q.owned.save(runtime/"manifest.json",dict(pid=120,owns_process_only=True,min_free_mib=93000,existing_compute_pids=[]))
        (runtime/"engine.log").write_text(q.previous.CAPACITY)
        self.evidence=dict(runtime=str(runtime),manifest_sha256=q.owned.digest(runtime/"manifest.json"),log_sha256=q.owned.digest(runtime/"engine.log"))
        self.state=dict(error="SafetyHalt: Fresh engine exited; no model trial dispatched",current_trial="035-kvmem-on-r1",
                        current_engine_runtime=str(runtime),engine_launcher_pid=120,last_gpu_snapshot=dict(free_mib=97234,compute_pids=[]))

    def check(self, actual=None):
        with patch.object(q.owned,"engine_identity",return_value={}),patch.object(q.owned,"process_identity",return_value=actual), \
                patch.object(q.owned,"assert_no_services"),patch.object(q.owned,"assert_vm_retired"):
            q.capacity_failure_evidence(self.d,self.state,self.evidence)

    def test_exact_natural_gate_failure_is_read_only_and_retired(self): self.check()

    def test_live_failed_engine_or_changed_ceiling_log_is_rejected(self):
        with self.assertRaises(q.SafetyHalt):self.check(dict(pid=120,state="S"))
        self.evidence["log_sha256"]="changed"
        with self.assertRaises(q.SafetyHalt):self.check()

    def test_an_actual_task_directory_disallows_no_task_failure_claim(self):
        (self.root/"attempts/035-kvmem-on-r1").mkdir(parents=True)
        with self.assertRaises(q.SafetyHalt):self.check()

    def test_failure_with_a_different_gate_or_other_gpu_owner_is_not_this_scope(self):
        self.state["last_gpu_snapshot"]["free_mib"]=56000
        with self.assertRaises(q.SafetyHalt):self.check()
        self.state["last_gpu_snapshot"]["free_mib"]=97234;self.state["last_gpu_snapshot"]["compute_pids"]=["other"]
        with self.assertRaises(q.SafetyHalt):self.check()


class FrozenLoopTests(unittest.TestCase):
    def loop(self, halt_at=None):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); protocol = root / "protocol.json"
        parent = root / "parent.json"; q.owned.save(parent, dict(engine_runtime=str(root / "unused")))
        rows = [dict(id=f"quota1-{task}-kvmem-{mode}-r{rep}", task=task, kvmem=mode, arm="verified", repetition=rep)
                for rep in (1, 2) for task in ("035", "080") for mode in ("off", "on")]
        q.owned.save(protocol, {})
        p = dict(parent_candidate_protocol=str(parent), trials=rows, pilot_app="synthetic",
                 engine_identities=dict(off={}, on={}), engine_endpoint="synthetic", framework="fixed", formal_campaign="fixed")
        d = dict(continued_protocol=str(protocol), state_dir=str(root / "state"), profile="qualified-profile", original_followup=str(parent), python=str(q.PYTHON))
        declaration = root / "declaration.json"; q.owned.save(declaration, d)
        args = argparse.Namespace(declaration=declaration, wait_seconds=60, resource_wait_seconds=60)
        commands = []; runner_count = [0]
        def spawn(command, **kwargs):
            commands.append(command)
            obj = types.SimpleNamespace(pid=120 + len(commands), poll=lambda: None)
            if "osworld_office_candidate_run" in command:
                runner_count[0] += 1
                output = Path(command[command.index("--output") + 1]); output.mkdir(parents=True)
                q.owned.save(output / "result.json", dict(finished_at=1, uncertain_actions=1 if halt_at == runner_count[0] else 0,
                                                          accepted=False, original_evaluation=0))
            obj.wait = lambda **kwargs: 0
            return obj
        with patch.object(q, "validate", return_value=(d, {})), patch.object(q, "validate_protocol", return_value=p), \
                patch.object(sys, "executable", str(q.PYTHON)), patch.object(q.old, "continued_parent", return_value=dict(status="completed", pid=300, trials=[])), \
                patch.object(q.old, "terminal_parent", return_value=[]), patch.object(q.owned, "process_identity", return_value=None), \
                patch.object(q.owned, "assert_apps_idle"), patch.object(q.owned, "assert_no_services"), patch.object(q.owned, "assert_vm_retired"), \
                patch.object(q.owned, "model_lock"), patch.object(q.owned, "gpu_snapshot", return_value=dict(free_mib=56000)), \
                patch.object(q.owned, "request", return_value=dict(data=[dict(context_window=524288)])), \
                patch.object(q.owned, "register_engine", side_effect=lambda *a: dict(pid=120+len(commands))), \
                patch.object(q.owned, "retire_registered", return_value=dict(retired=True)), \
                patch.object(q.owned, "register_child", side_effect=lambda pid, command: dict(pid=pid, command=command)), \
                patch.object(q.owned.subprocess, "Popen", side_effect=spawn):
            try: q.run(args)
            except q.SafetyHalt:
                if halt_at is None: raise
        return q.read(root / "state/queue.json"), commands

    def test_all_eight_actual_frozen_loop_rows_use_new_profile_without_old_prefix_reuse(self):
        state, commands = self.loop()
        self.assertEqual(state["status"], "completed"); self.assertEqual(len(state["trials"]), 8)
        self.assertEqual(len(commands), 16)
        for c in commands:
            if "--profile" in c:
                self.assertIn("shared_quota_engine.py", c[1]); self.assertEqual(c[c.index("--min-free-mib")+1], "56000")
            else:
                self.assertEqual(c[0], str(q.PYTHON)); self.assertEqual(c[c.index("--model-output-tokens")+1], "16384")

    def test_unknown_input_stops_before_any_next_dispatch(self):
        state, commands = self.loop(halt_at=3)
        self.assertEqual(state["status"], "halted-no-further-dispatch")
        self.assertEqual(len(state["trials"]), 2); self.assertEqual(len(commands), 6)

    def test_inventory_patch_preserves_other_guards_and_refuses_source_drift(self):
        source = inspect.getsource(q.owned.validate_protocol)
        changed = q.patch_validator(source); compile(changed, "inventory", "exec")
        self.assertIn("source_identity(framework)", changed)
        self.assertIn("quota1-", changed)
        with self.assertRaises(q.SafetyHalt): q.patch_validator("changed")

    def test_frozen_loop_patch_refuses_shape_change(self):
        with self.assertRaises(q.SafetyHalt): q.patch_loop("changed")


if __name__ == "__main__": unittest.main()
