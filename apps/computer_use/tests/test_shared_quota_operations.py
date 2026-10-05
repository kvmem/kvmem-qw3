import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import shared_quota_operations as q


class NativeProcessTests(unittest.TestCase):
    def owned_child(self):
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        child = subprocess.Popen(command, cwd=q.ROOT)
        identity = q.owned.register_child(child.pid, command)
        def cleanup():
            if child.poll() is None: q.owned.retire_registered(identity, timeout=2)
            child.wait(timeout=2)
        self.addCleanup(cleanup)
        return child, identity

    def test_actual_pidfd_freeze_and_resume_the_same_owned_handle(self):
        child, identity = self.owned_child()
        with q.frozen_controls([dict(identity=identity)]) as handles:
            self.assertEqual(len(handles), 1)
            self.assertIn(q.owned.process_identity(child.pid)["state"], {"T", "t"})
        self.assertIsNone(child.poll())
        self.assertNotIn(q.owned.process_identity(child.pid)["state"], {"T", "t"})

    def test_failed_idle_check_resumes_without_terminating(self):
        child, identity = self.owned_child()
        with self.assertRaises(q.SafetyHalt):
            with q.frozen_controls([dict(identity=identity)]): raise q.SafetyHalt("active input")
        self.assertIsNone(child.poll())
        self.assertNotIn(q.owned.process_identity(child.pid)["state"], {"T", "t"})

    def test_changed_start_identity_is_rejected_before_any_signal(self):
        child, identity = self.owned_child(); wrong = dict(identity, start_ticks=identity["start_ticks"]+1)
        with self.assertRaises(q.SafetyHalt):
            with q.frozen_controls([dict(identity=wrong)]): pass
        self.assertIsNone(child.poll())
        self.assertNotIn(q.owned.process_identity(child.pid)["state"], {"T", "t"})

    def test_unlisted_process_remains_live_during_a_freeze(self):
        child, identity = self.owned_child(); other, other_identity = self.owned_child()
        with q.frozen_controls([dict(identity=identity)]):
            self.assertIsNone(other.poll())
            self.assertNotIn(q.owned.process_identity(other_identity["pid"])["state"], {"T", "t"})


class IdleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); original = self.root / "protocol.json"
        q.owned.save(original, dict(pilot_app="synthetic"))
        previous = self.root / "previous.json"; self.prefix = [dict(id="old-off", result=dict(accepted=False))]
        q.owned.save(previous, dict(completed_prefix=self.prefix, continued_protocol=str(original)))
        self.d = dict(previous_continuation=str(previous), continued_protocol=str(original))
        self.rows = []; self.states = {}
        for index, stage in enumerate(q.memory.STAGES):
            identity = dict(pid=100+index,uid=os.getuid(),start_ticks=12,command=["owned",stage],cwd=str(q.ROOT),state="T")
            queue = self.root / (stage + ".json")
            self.states[stage] = dict(pid=identity["pid"], status=q.WAITING[stage], trials=self.prefix if stage=="memory" else [])
            if stage == "memory": self.states[stage].update(current_trial="035-kvmem-on-r1",shared_admission_min_free_mib=93000)
            self.rows.append(dict(stage=stage, identity=identity, queue=str(queue)))

    def check(self, descendants=()):
        for row in self.rows: q.owned.save(Path(row["queue"]), self.states[row["stage"]])
        with patch.object(q, "checked_instance", side_effect=lambda i:i), patch.object(q,"children",return_value=list(descendants)), \
                patch.object(q.owned,"assert_apps_idle"), patch.object(q.owned,"assert_no_services"), patch.object(q,"port_free"):
            return q.idle_states(self.d,self.rows,stopped=True)

    def test_exact_five_idle_states_retain_only_the_old_prefix(self):
        snapshots = self.check(); self.assertEqual(len(snapshots),5)
        self.assertEqual(snapshots[0]["state"]["trials"],self.prefix)

    def test_any_child_prevents_retirement_even_when_queue_says_waiting(self):
        with self.assertRaises(q.SafetyHalt): self.check([333])

    def test_engine_or_input_worker_fields_prevent_retirement(self):
        self.states["memory"]["engine_launcher_pid"] = 123
        with self.assertRaises(q.SafetyHalt): self.check()

    def test_successor_progress_or_changed_memory_gate_prevents_retirement(self):
        self.states["sdk"]["trials"]=[dict(id="started")]
        with self.assertRaises(q.SafetyHalt): self.check()
        self.states["sdk"]["trials"]=[];self.states["memory"]["shared_admission_min_free_mib"]=56000
        with self.assertRaises(q.SafetyHalt): self.check()


class StartupLifecycleTests(unittest.TestCase):
    def exercise(self, *, other_backend=False, headroom=56000, http_failure=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        declaration=root/"declaration.json";profile_path=root/"profile.json";protocol=root/"protocol.json"
        q.owned.save(profile_path,{});q.owned.save(protocol,dict(pilot_app="synthetic"));q.owned.save(declaration,{})
        expected=dict(command=["synthetic-qw3","serve"],binary_sha256="fixed",qw3_env={})
        profile=dict(engine_identities=dict(on=expected))
        d=dict(profile=str(profile_path),continued_protocol=str(protocol),startup_qualification=str(root/"startup-qualification.json"))
        live=[True];captured=[];command=[]
        def spawn(argv,**kwargs):
            command[:]=argv
            runtime=Path(argv[argv.index("--run-dir")+1]);runtime.mkdir()
            q.owned.save(runtime/"manifest.json",{})
            return types.SimpleNamespace(pid=120,poll=lambda:None if live[0] else 0,wait=lambda **kw:0)
        def actual(pid):
            return dict(pid=120,uid=os.getuid(),start_ticks=10,cwd=str(q.ROOT),command=command,state="S") if live[0] else None
        def retire(identity,**kwargs): captured.append(identity);live[0]=False;return dict(retired=True)
        def response(request,**kwargs):
            payload=json.loads(request.data)
            self.assertEqual([m["role"] for m in payload["messages"]],["system","user","assistant","user"])
            if http_failure: raise HTTPError(request.full_url,400,"Bad Request",{},io.BytesIO(b'{"error":"request rejected before generation"}'))
            return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content="READY"))])).encode())
        opener=types.SimpleNamespace(open=response)
        with patch.object(q,"qualification"),patch.object(q.memory,"validate",return_value=(d,{})), \
                patch.object(q.memory.engine,"qualify_profile",return_value=profile),patch.object(q,"port_free"), \
                patch.object(q.owned,"assert_apps_idle"),patch.object(q.owned,"model_lock"), \
                patch.object(q.owned,"gpu_snapshot",return_value=dict(free_mib=headroom,compute_pids=["other-user"])), \
                patch.object(q.subprocess,"Popen",side_effect=spawn),patch.object(q.owned,"process_identity",side_effect=actual), \
                patch.object(q.owned,"request",return_value=dict(data=[dict(id="synthetic",context_window=524288)])), \
                patch.object(q.owned,"register_engine",return_value=dict(pid=121 if other_backend else 120)), \
                patch.object(q,"build_opener",return_value=opener),patch.object(q.owned,"retire_registered",side_effect=retire), \
                patch.object(q.memory,"startup_evidence"):
            state=q.startup(declaration,root/"cpu.json")
        return state,captured,command

    def test_one_synthetic_health_call_and_only_own_engine_retired(self):
        state,retired,command=self.exercise()
        self.assertTrue(state["passed"]);self.assertEqual(state["health_model_calls"],1)
        self.assertEqual(state["device_inputs"],0);self.assertEqual(state["task_trials_dispatched"],0)
        self.assertEqual(len(retired),1);self.assertEqual(retired[0]["command"],command)

    def test_another_backend_cannot_supply_the_health_result(self):
        state,retired,_=self.exercise(other_backend=True)
        self.assertFalse(state["passed"]);self.assertEqual(state["health_model_calls"],0)
        self.assertEqual(retired[0]["pid"],120)

    def test_low_headroom_causes_no_process_launch_or_model_call(self):
        state,retired,command=self.exercise(headroom=55999)
        self.assertFalse(state["passed"]);self.assertEqual(retired,[]);self.assertEqual(command,[])
        self.assertEqual(state["health_model_calls"],0)

    def test_http_failure_retains_body_and_retires_only_own_engine(self):
        state,retired,_=self.exercise(http_failure=True)
        self.assertFalse(state["passed"]);self.assertEqual(len(retired),1)
        self.assertIn("request rejected before generation",state["health_http_error_body"])


if __name__=="__main__":unittest.main()
