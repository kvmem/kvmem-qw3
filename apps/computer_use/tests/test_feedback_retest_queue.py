"""Safety boundaries for real-task feedback ablation; no model or device."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import feedback_retest_queue as queue


class FeedbackQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qw3-feedback-queue-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.identity = dict(pid=123,uid=1001,start_ticks=456,command=["python","-m","thinking_budget_replay_queue"],cwd="/root")
        self.protocol = dict(kind="declared_matched_literal_feedback_osworld_retest",trials=queue.trials(),
                             model=copy.deepcopy(queue.image.MODEL),model_concurrency=1,overwrite_old_attempts=False,
                             advisor_enabled=False,parent_controller=self.identity,parent_protocol_sha256="declared",
                             parent_queue=str(self.directory / "parent/queue.json"),
                             parent_protocol=str(self.directory / "parent/protocol.json"),parent_app="http://parent",
                             engine_off_identity={"profile":"off"},frameworks={"off":"/private/off","on":"/private/on"},
                             acceptances={"off":{"path":"/private/off/acceptance.json"},"on":{"path":"/private/on/acceptance.json"}})
        self.args = argparse.Namespace(protocol=self.directory / "protocol.json",state_dir=self.directory / "state",
                                       wait_seconds=60,resource_wait_seconds=60,min_free_mib=56000)
        queue.save(self.args.protocol,self.protocol)

    def test_tasks_have_two_balanced_pairs_and_original_budget(self):
        queue.validate_profile(self.protocol)
        for task in ("013","004"):
            for repetition in (1,2):
                pair = [r for r in queue.trials() if r["task"]==task and r["repetition"]==repetition]
                self.assertEqual({r["feedback"] for r in pair},{"off","on"})
                self.assertTrue(all((r["seconds"],r["tools"],r["arm"])==(1800,250,"verified") for r in pair))
        first = [r["feedback"] for r in queue.trials() if r["task"]=="013" and r["repetition"]==1]
        second = [r["feedback"] for r in queue.trials() if r["task"]=="013" and r["repetition"]==2]
        self.assertEqual(first,second[::-1])

    def test_profile_cannot_add_advisor_or_change_thinking_kv_or_budget(self):
        for key,value in (("advisor",True),("thinking",False),("thinking_budget",8192),("kvmem","on"),("max_output_tokens",8192)):
            p = copy.deepcopy(self.protocol); p["model"][key]=value
            with self.subTest(key=key),self.assertRaises(queue.SafetyHalt): queue.validate_profile(p)

    def test_trial_inventory_cannot_be_changed_or_reordered(self):
        for rows in (queue.trials()[:-1],queue.trials()[::-1],queue.trials()[:-1]+queue.trials()[:1]):
            with self.assertRaises(queue.SafetyHalt): queue.validate_profile(dict(self.protocol,trials=rows))

    def test_live_wait_requires_actual_registered_handle(self):
        state=dict(pid=123,protocol_sha256="declared",status="waiting-for-image-twelve")
        with patch.object(queue.base,"require_identity",return_value=self.identity) as actual:
            self.assertIsNone(queue.terminal_predecessor(self.protocol,state));actual.assert_called_once_with(self.identity)
        with patch.object(queue.base,"require_identity",side_effect=queue.SafetyHalt("missing")):
            with self.assertRaises(queue.SafetyHalt): queue.terminal_predecessor(self.protocol,state)

    def test_halted_or_changed_predecessor_never_dispatches(self):
        for fields in ({"status":"halted-no-further-dispatch"},{"pid":999},{"protocol_sha256":"changed"}):
            state=dict(pid=123,protocol_sha256="declared",status="waiting-for-image-twelve");state.update(fields)
            with self.assertRaises(queue.SafetyHalt): queue.terminal_predecessor(self.protocol,state)

    def test_terminal_file_does_not_replace_process_retirement(self):
        state=dict(pid=123,protocol_sha256="declared",status="completed")
        with patch.object(queue.base,"process_retired",return_value=False):
            self.assertIsNone(queue.terminal_predecessor(self.protocol,state))
        with patch.object(queue.base,"process_retired",side_effect=queue.SafetyHalt("PID reused")):
            with self.assertRaises(queue.SafetyHalt): queue.terminal_predecessor(self.protocol,state)

    def parent_state(self):
        rows=[]
        for spec in queue.predecessor.trials():
            path=self.directory / "parent/attempts" / spec["id"] / "result.json"
            path.parent.mkdir(parents=True)
            result=dict(termination_confirmed=True,tool_calls_executed=0,business_success_claimed=False)
            queue.save(path,result);rows.append(dict(id=spec["id"],result=result,result_sha256=queue.owned.digest(path)))
        return dict(pid=123,protocol_sha256="declared",status="completed",termination_confirmed=True,
                    parent_services_preserved=True,final_engine_policy="off",trials=rows,
                    parent_services=dict(model_client=self.identity,registry="/registry",registry_sha256="sha",
                                         new_gateways=[dict(process_identity=self.identity,command=["python","--runtime","/gateway"])],
                                         registered_engine={"pid":88}),engine_runtime="/last-thinking-engine",registered_engine={"pid":99})

    def test_missing_or_duplicate_terminal_replays_fail(self):
        original=self.parent_state()
        for rows in (original["trials"][:-1],original["trials"][:-1]+original["trials"][:1]):
            state=dict(original,trials=rows)
            with patch.object(queue.base,"process_retired",return_value=True),self.assertRaises(queue.SafetyHalt):
                queue.terminal_predecessor(self.protocol,state)

    def test_changed_durable_replay_result_fails_before_engine_or_input(self):
        state=self.parent_state();path=self.directory / "parent/attempts" / state["trials"][0]["id"] / "result.json"
        queue.save(path,dict(state["trials"][0]["result"],tool_calls_executed=1))
        with patch.object(queue.base,"process_retired",return_value=True),patch.object(queue.owned,"register_engine") as engine:
            with self.assertRaises(queue.SafetyHalt):queue.terminal_predecessor(self.protocol,state)
            engine.assert_not_called()

    def test_handoff_admits_last_thinking_engine_and_checks_gateway_unknowns(self):
        state=self.parent_state()
        with patch.object(queue.base,"process_retired",return_value=True),patch.object(queue.owned,"assert_no_services"), \
                patch.object(queue.base,"require_identity"),patch.object(queue.owned,"assert_apps_idle"), \
                patch.object(queue.image,"pinned"),patch.object(queue.base,"ledger_terminal") as ledger, \
                patch.object(queue.owned,"register_engine",return_value={"pid":99}) as engine, \
                patch.object(queue.owned,"same_process",side_effect=lambda a,b:a==b):
            services=queue.terminal_predecessor(self.protocol,state)
            self.assertEqual(services["registered_engine"],{"pid":99})
            engine.assert_called_once_with(Path("/last-thinking-engine"),self.protocol["engine_off_identity"])
            ledger.assert_called_once_with(Path("/gateway"))
            ledger.side_effect=queue.SafetyHalt("unknown input")
            with self.assertRaises(queue.SafetyHalt):queue.terminal_predecessor(self.protocol,state)

    def request(self,mode="off",inferred=False):
        directory=self.directory / (mode+str(inferred)) / "trace";directory.mkdir(parents=True)
        feedback=dict(condition={"kind":"control_text","text":"Prize Items"},passed=False,
                      comparison_evidence=dict(input_delivery_inferred=inferred,independent_business_result_verified=False))
        body=dict(enable_thinking=True,max_tokens=16384,messages=[dict(role="tool",content=json.dumps(feedback))] if mode=="on" else [])
        queue.save(directory / "request.json",body)
        return directory.parent

    def test_trace_keeps_failed_criterion_and_never_promotes_input_delivery(self):
        rows=queue.trace_evidence(self.request("on"),"on")
        self.assertEqual(rows[0]["failed_conditions_on_wire"],1)
        with self.assertRaises(queue.SafetyHalt):queue.trace_evidence(self.request("on",True),"on")

    def test_no_feedback_in_off_arm_and_no_empty_model_trial(self):
        self.assertEqual(queue.trace_evidence(self.request(),"off")[0]["conditions_on_wire"],0)
        with self.assertRaises(queue.SafetyHalt):queue.trace_evidence(self.request("on"),"off")
        with self.assertRaises(queue.SafetyHalt):queue.trace_evidence(self.directory / "empty","on")

    def test_trial_command_selects_only_its_private_feedback_framework(self):
        controller=queue.Controller(self.args,self.protocol,dict(engine_runtime="/engine"))
        off,on=queue.trials()[:2];left=controller.trial_command(off,Path("/out/off"));right=controller.trial_command(on,Path("/out/on"))
        self.assertEqual(left[left.index("--framework")+1],"/private/off")
        self.assertEqual(right[right.index("--framework")+1],"/private/on")
        for key in ("--task","--arm","--model-output-tokens","--formal-campaign","--pilot-app"):
            self.assertEqual(left[left.index(key)+1],right[right.index(key)+1])

    def fake_campaign(self,fail_at=None):
        attempted=[]
        def dispatch(controller,row):
            attempted.append(row["id"])
            if len(attempted)==fail_at:raise queue.SafetyHalt("unconfirmed input")
            controller.state["trials"].append(dict(row));controller.checkpoint()
        with patch.object(queue,"validate",return_value=self.protocol),patch.object(queue,"terminal_predecessor",return_value={}), \
                patch.object(queue,"read",return_value={}),patch.object(queue.Controller,"dispatch",dispatch):
            if fail_at:
                with self.assertRaises(queue.SafetyHalt):queue.run(self.args)
            else:queue.run(self.args)
        state=json.loads((self.args.state_dir / "queue.json").read_text())
        return attempted,state

    def test_full_mock_sequence_retains_declared_order(self):
        attempted,state=self.fake_campaign()
        self.assertEqual(attempted,[r["id"] for r in queue.trials()]);self.assertEqual(state["status"],"completed")

    def test_unknown_at_third_trial_preserves_state_and_stops_dispatch(self):
        attempted,state=self.fake_campaign(3)
        self.assertEqual(len(attempted),3);self.assertEqual(len(state["trials"]),2)
        self.assertEqual(state["status"],"halted-no-further-dispatch");self.assertTrue(state["live_children_and_evidence_preserved"])

    def test_observation_deadline_preserves_waiting_services(self):
        with patch.object(queue,"validate",return_value=self.protocol),patch.object(queue,"read",return_value={}), \
                patch.object(queue,"terminal_predecessor",return_value=None),patch.object(queue.time,"monotonic",side_effect=[0,61]), \
                patch.object(queue.Controller,"dispatch") as dispatch:
            with self.assertRaises(queue.SafetyHalt):queue.run(self.args)
            dispatch.assert_not_called()

    def test_existing_attempt_is_never_replayed(self):
        row=queue.trials()[0];output=self.directory / "attempts" / row["id"];output.mkdir(parents=True)
        controller=queue.Controller(self.args,self.protocol,{"protocol_sha256":"sha"})
        with patch.object(queue,"validate"),patch.object(queue.image,"pinned"),patch.object(controller,"parent_idle"), \
                patch.object(controller,"fresh_engine") as engine:
            with self.assertRaises(queue.SafetyHalt):controller.dispatch(row)
            engine.assert_not_called()


if __name__=="__main__":unittest.main()
