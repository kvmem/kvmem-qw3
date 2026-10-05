"""Real frozen wait loops and producer admission, without model/device input."""
import argparse
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner_runtime_successors as q


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.declaration = self.root / "declaration.json"
        self.protocol = self.root / "protocol.json"; self.queue = self.root / "queue.json"
        for p in (self.declaration, self.protocol): q.save(p, {})
        self.identity = dict(pid=120, uid=os.getuid(), start_ticks=123, cwd=str(q.ROOT), command=["python", "owned"], state="S")
        self.reg = dict(identity=self.identity, queue=str(self.queue), protocol_sha256=q.owned.digest(self.protocol))
        self.state = dict(pid=120, status="waiting", trials=[], protocol_sha256=self.reg["protocol_sha256"],
                          runtime_successor_declaration_sha256=q.owned.digest(self.declaration),
                          runtime_successor_wrapper_sha256=q.owned.digest(Path(q.__file__)))

    def admit(self, actual):
        q.save(self.queue, self.state)
        with patch.object(q.owned, "process_identity", return_value=actual):
            return q.successor_queue(self.reg, self.declaration)

    def test_live_actual_producer_is_admitted_without_mutating_state(self):
        before = copy.deepcopy(self.state); self.assertEqual(self.admit(self.identity), before)

    def test_state_alone_without_actual_process_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.admit(None)

    def test_zombie_is_not_a_live_producer(self):
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, state="Z"))

    def test_start_time_reuse_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, start_ticks=124))

    def test_other_user_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, uid=self.identity["uid"]+1))

    def test_other_command_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, command=["python", "other"]))

    def test_other_directory_is_rejected(self):
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, cwd="/tmp"))

    def test_halted_actual_live_producer_is_rejected(self):
        self.state["status"] = "halted-no-further-dispatch"
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_completed_retired_producer_is_admitted_for_terminal_checks(self):
        self.state["status"] = "completed"; self.assertEqual(self.admit(None), self.state)

    def test_completed_state_with_pid_reuse_is_rejected(self):
        self.state["status"] = "completed"
        with self.assertRaises(q.SafetyHalt): self.admit(dict(self.identity, start_ticks=124))

    def test_changed_protocol_is_rejected(self):
        self.state["protocol_sha256"] = "changed"
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_changed_declaration_is_rejected(self):
        self.state["runtime_successor_declaration_sha256"] = "changed"
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_changed_wrapper_source_is_rejected(self):
        self.state["runtime_successor_wrapper_sha256"] = "changed"
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)

    def test_wrong_queue_pid_is_rejected(self):
        self.state["pid"] = 121
        with self.assertRaises(q.SafetyHalt): self.admit(self.identity)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name); self.path = root / "registration.json"
        self.declaration, self.protocol = root / "declaration.json", root / "protocol.json"
        for p in (self.declaration, self.protocol): q.save(p, {})
        self.d = dict(controller_python=str(q.CONTROLLER_PYTHON), sdk_protocol=str(self.protocol), sdk_state_dir=str(root / "state"))
        command = [self.d["controller_python"], "-m", q.MODULE, "--declaration", str(self.declaration), "--stage", "sdk"]
        self.r = dict(kind="actual_runner_runtime_successor_identity", stage="sdk", declaration=str(self.declaration),
                      declaration_sha256=q.owned.digest(self.declaration), wrapper_sha256=q.owned.digest(Path(q.__file__)),
                      protocol=str(self.protocol), protocol_sha256=q.owned.digest(self.protocol), queue=str(root / "state/queue.json"),
                      identity=dict(pid=120,uid=os.getuid(),start_ticks=123,cwd=str(q.ROOT),command=command,state="S"))

    def admit(self):
        q.save(self.path, self.r); return q.successor_registration(self.path, "sdk", self.declaration, self.d)

    def test_exact_stage_registration_is_read_only_admitted(self):
        self.assertEqual(self.admit(), self.r)

    def test_different_stage_is_rejected(self):
        self.r["stage"] = "image"
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_old_halted_controller_kind_is_rejected(self):
        self.r["kind"] = "actual_retired_051_successor_identity"
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_registration_source_change_is_rejected(self):
        self.r["wrapper_sha256"] = "changed"
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_old_queue_routing_is_rejected(self):
        self.r["queue"] = str(self.protocol.parent / "old/queue.json")
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_interpreter_symlink_must_keep_venv_path(self):
        self.r["identity"]["command"][0] = str(q.CONTROLLER_PYTHON.resolve())
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_other_declaration_in_process_command_is_rejected(self):
        self.r["identity"]["command"][4] = "/tmp/other.json"
        with self.assertRaises(q.SafetyHalt): self.admit()

    def test_extra_command_options_are_not_silently_admitted(self):
        self.r["identity"]["command"] += ["--min-free-mib", "1"]
        with self.assertRaises(q.SafetyHalt): self.admit()


class CapacityProducerTests(ProducerTests):
    def admit(self, actual):
        self.reg["kind"] = "actual_kvmem_capacity_continuation_identity"
        self.state.pop("runtime_successor_declaration_sha256",None)
        self.state.pop("runtime_successor_wrapper_sha256",None)
        self.state["capacity_continuation_sha256"] = q.owned.digest(self.declaration)
        self.state["capacity_continuation_wrapper_sha256"] = q.owned.digest(Path(q.capacity.__file__))
        self.state["qualified_runner_python"] = str(q.memory.PYTHON)
        q.save(self.queue,self.state)
        with patch.object(q.owned,"process_identity",return_value=actual):
            return q.memory_queue(self.reg,self.declaration)

    # These inherited tests mutate the other producer's keys; use the actual
    # capacity keys so a corruption must still fail its own admission.
    def test_changed_declaration_is_rejected(self):
        self.admit(self.identity);self.state["capacity_continuation_sha256"]="changed";q.save(self.queue,self.state)
        with patch.object(q.owned,"process_identity",return_value=self.identity):
            with self.assertRaises(q.SafetyHalt):q.memory_queue(self.reg,self.declaration)

    def test_changed_wrapper_source_is_rejected(self):
        self.admit(self.identity);self.state["capacity_continuation_wrapper_sha256"]="changed";q.save(self.queue,self.state)
        with patch.object(q.owned,"process_identity",return_value=self.identity):
            with self.assertRaises(q.SafetyHalt):q.memory_queue(self.reg,self.declaration)

    def test_live_actual_producer_is_admitted_without_mutating_state(self):
        state=self.admit(self.identity);self.assertEqual(state,self.state)

    def test_completed_retired_producer_is_admitted_for_terminal_checks(self):
        self.state["status"]="completed";state=self.admit(None);self.assertEqual(state,self.state)


class FrozenLoopTests(unittest.TestCase):
    def fixture(self, root, stage):
        declaration, protocol = root / "declaration.json", root / "protocol.json"
        quarantine = root / "quarantine.json"
        for p in (declaration, protocol, quarantine): q.save(p, {})
        args = argparse.Namespace(declaration=declaration, protocol=protocol,state_dir=root / "state",
                                  wait_seconds=60,resource_wait_seconds=60,min_free_mib=56000)
        d = dict(controller_python=str(q.CONTROLLER_PYTHON),runner_python=str(q.memory.PYTHON),
                 memory_continuation=str(declaration),original_followup=str(declaration),memory_queue=str(root / "memory/queue.json"),
                 memory_protocol=str(root / "memory/protocol.json"),**{stage+"_protocol":str(protocol),stage+"_state_dir":str(args.state_dir)})
        canonical = dict(continued_osworld_queue=str(root / "original/queue.json"),continuation_controller_identity={"pid":119})
        mr = dict(identity={"pid":120},queue=d["memory_queue"])
        p = dict(trials=[],quarantine_declaration=str(quarantine),successor_wrapper_sha256="original-wrapper",
                 quarantine_declaration_sha256="original-quarantine")
        return args,d,canonical,mr,p

    def test_sdk_real_wait_loop_does_not_deploy_when_memory_process_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            args,d,canonical,mr,p = self.fixture(Path(folder),"sdk")
            with patch.object(q,"validate",return_value=(d,canonical,mr)), \
                    patch.object(q,"frozen_protocol",return_value=(p,{})), \
                    patch.object(q,"memory_queue",side_effect=q.SafetyHalt("Actual memory producer missing")), \
                    patch.object(q.sdk.Controller,"deploy") as deploy,patch.object(q.sdk.Controller,"restore_engine") as engine, \
                    patch.object(q.owned.subprocess,"Popen") as launch:
                with self.assertRaises(q.SafetyHalt): q.run_sdk(args,d,canonical,mr)
                deploy.assert_not_called(); engine.assert_not_called(); launch.assert_not_called()
            state=q.read(args.state_dir/"queue.json")
            self.assertEqual(state["trials"],[]);self.assertEqual(state["planned_trials"],10)
            self.assertEqual(state["status"],"halted-no-further-dispatch")

    def test_sdk_trial_command_changes_only_runner_interpreter(self):
        with tempfile.TemporaryDirectory() as folder:
            args,d,canonical,mr,p = self.fixture(Path(folder),"sdk")
            p.update(app="http://127.0.0.1:1",app_runtime="/runtime",bundle="/bundle",framework="/framework",acceptance="/acceptance")
            captured={}
            def run(a):
                controller=q.base.Controller(a,p,{"registry":"/registry","engine_runtime":"/engine"})
                rows=[dict(kind="receipt-loss",device="linux-vm"),dict(kind="normal-pilot",case="linux_text",seconds=360),
                      dict(kind="original-osworld",task="004",arm="verified")]
                for row in rows:
                    original=q.sdk.Controller.trial_command(controller,row,Path("/output"))
                    actual=controller.trial_command(row,Path("/output"))
                    self.assertEqual(actual,[d["runner_python"],*original[1:]])
                captured.update(controller.state); return captured
            with patch.object(q,"validate",return_value=(d,canonical,mr)), \
                    patch.object(q,"frozen_protocol",return_value=(p,{})),patch.object(q.base,"run",side_effect=run):
                q.run_sdk(args,d,canonical,mr)
            self.assertEqual(captured["qualified_runner_python"],str(q.memory.PYTHON))

    def test_image_real_wait_loop_preserves_services_on_sdk_halt(self):
        self._halted_loop("image")

    def test_thinking_real_wait_loop_preserves_services_on_image_halt(self):
        self._halted_loop("thinking")

    def test_feedback_real_wait_loop_preserves_services_on_thinking_halt(self):
        self._halted_loop("feedback")

    def _halted_loop(self,stage):
        with tempfile.TemporaryDirectory() as folder:
            args,d,canonical,mr,p = self.fixture(Path(folder),stage)
            previous={"image":"sdk","thinking":"image","feedback":"thinking"}[stage]
            regpath=Path(folder)/"registration.json";queue=Path(folder)/"previous.json"
            q.save(regpath,{});q.save(queue,dict(pid=120,status="halted-no-further-dispatch",trials=[]))
            d[previous+"_registration"]=str(regpath)
            reg=dict(identity={"pid":120},queue=str(queue))
            controller=q.feedback.Controller if stage=="feedback" else q.image.Controller
            with patch.object(q,"validate",return_value=(d,canonical,mr)), \
                    patch.object(q,"frozen_protocol",return_value=p), \
                    patch.object(q,"parent_registration",return_value=(reg,q.owned.digest(regpath))), \
                    patch.object(q,"memory_queue",return_value={"status":"running"}), \
                    patch.object(q,"successor_queue",side_effect=q.SafetyHalt("Actual predecessor halted")), \
                    patch.object(controller,"dispatch") as dispatch,patch.object(controller,"fresh_engine") as engine, \
                    patch.object(q.owned.subprocess,"Popen") as launch:
                with self.assertRaises(q.SafetyHalt): getattr(q,"run_"+stage)(args,d,canonical,mr)
                dispatch.assert_not_called();engine.assert_not_called();launch.assert_not_called()
            state=q.read(args.state_dir/"queue.json")
            self.assertEqual(state["trials"],[]);self.assertEqual(state["status"],"halted-no-further-dispatch")

    def test_sdk_physical_receipts_route_to_new_memory_directory_without_profile_change(self):
        original=dict(parent_osworld_queue="old-os",parent_kvmem_queue="old-memory",parent_kvmem_protocol="old-protocol",
                      predecessor_processes={"osworld":{},"memory":{}},trials=[{"id":"original-id"}],
                      model_output_tokens=16384,model_context_tokens=524288,thinking=True,dflash2=True,kvmem="off")
        d=dict(memory_queue="new-memory/queue.json",memory_protocol="new-memory/protocol.json")
        canonical=dict(continued_osworld_queue="continued-os",continuation_controller_identity={"pid":119})
        before=copy.deepcopy(original);actual=q.sdk_protocol(original,d,canonical,{"identity":{"pid":120}})
        changed={key for key in actual if actual[key]!=original[key]}
        self.assertEqual(changed,{"parent_osworld_queue","parent_kvmem_queue","parent_kvmem_protocol","predecessor_processes"})
        self.assertEqual(original,before)

    def test_saved_nested_validator_is_used_while_live_runner_callbacks_are_patched(self):
        callback=lambda p: {"original":str(p)}
        with patch.dict(q.FROZEN_VALIDATORS,{"thinking":callback}), \
                patch.object(q.thinking,"validate",side_effect=AssertionError("Live routing callback recursively invoked")):
            self.assertEqual(q.frozen_protocol("thinking",Path("/protocol")),{"original":"/protocol"})


if __name__=="__main__":unittest.main()
