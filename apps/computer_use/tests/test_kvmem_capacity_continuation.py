import copy
import inspect
import unittest

import kvmem_capacity_continuation as q


class CapacityAdmissionTests(unittest.TestCase):
    def fixture(self):
        result = dict(finished_at=1, uncertain_actions=0, status="evaluator-unavailable", accepted=False)
        state = dict(status="halted-no-further-dispatch", error=q.ERROR, current_trial="035-kvmem-on-r1",
                     trials=[dict(id="035-kvmem-off-r1", kvmem="off", result=result, result_sha256="original")])
        return [state, result, q.CAPACITY, []]

    def test_exact_startup_failure_preserves_independent_unavailable_grade(self):
        values = self.fixture(); original = copy.deepcopy(values); prefix = q.failure_scope(*values)
        self.assertEqual(values, original); self.assertEqual(prefix, original[0]["trials"])
        self.assertFalse(prefix[0]["result"]["accepted"])

    def test_live_queue_is_not_restarted(self):
        v=self.fixture();v[0]["status"]="resource-wait"
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_other_runtime_failure_is_not_capacity_admitted(self):
        v=self.fixture();v[0]["error"]="other"
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_other_trial_is_not_admitted(self):
        v=self.fixture();v[0]["current_trial"]="080-kvmem-on-r1"
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_any_task_directory_requires_reconciliation(self):
        v=self.fixture();v[3]=["attempts/035-kvmem-on-r1"]
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_incomplete_prefix_is_rejected(self):
        v=self.fixture();v[1].pop("finished_at")
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_unknown_prefix_input_is_not_cleared(self):
        v=self.fixture();v[1]["uncertain_actions"]=1
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_boolean_unknown_count_is_not_integer_zero(self):
        v=self.fixture();v[1]["uncertain_actions"]=False
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_unconfirmed_stop_is_rejected(self):
        v=self.fixture();v[1]["owned_task_stop_unconfirmed"]=True
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_changed_prefix_result_is_rejected(self):
        v=self.fixture();v[0]["trials"][0]["result"]=dict(v[1],accepted=True)
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_additional_completed_trial_is_not_silently_replayed(self):
        v=self.fixture();v[0]["trials"].append(copy.deepcopy(v[0]["trials"][0]))
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_other_capacity_message_requires_a_new_scope(self):
        v=self.fixture();v[2]=q.CAPACITY.replace("1304","1500")
        with self.assertRaises(q.SafetyHalt):q.failure_scope(*v)

    def test_frozen_body_keeps_original_profile_commands_and_remaining_slice(self):
        source=inspect.getsource(q.owned.run);changed=q.patch_source(source);compile(changed,"cpu-capacity-body","exec")
        for anchor in ('"--context-tokens", "524288"','"--kvmem-cpu-gib", "48"','"--model-output-tokens", "16384"',
                       'child.wait(timeout=5100)','for row in p["trials"][len(completed_prefix):]:','assert_vm_retired(output)'):
            self.assertIn(anchor,changed)
        self.assertNotIn('"--kvmem-gpu-memory-ratio"',changed)

    def test_source_shape_change_prevents_compilation_dispatch(self):
        source=inspect.getsource(q.owned.run).replace('for row in p["trials"]:','changed_loop:',1)
        with self.assertRaises(q.SafetyHalt):q.patch_source(source)


if __name__=="__main__":unittest.main()
