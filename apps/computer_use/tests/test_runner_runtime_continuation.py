import copy
import inspect
import unittest

import runner_runtime_continuation as continuation


class RuntimeContinuationTests(unittest.TestCase):
    def fixture(self):
        return [dict(status='halted-no-further-dispatch',trials=[],current_trial='035-kvmem-off-r1'),
                dict(status='infrastructure-failed',score=None,finished_at=1,accepted=False,
                     error='ImportError: cssselect does not seem to be installed. retained'),
                {},"ModuleNotFoundError: No module named 'cssselect'\n task_035.py",[]]

    def test_exact_pre_vm_failure_has_narrow_admission(self):
        values=self.fixture();before=copy.deepcopy(values);continuation.failure_scope(*values);self.assertEqual(values,before)

    def test_live_queue_cannot_be_restarted(self):
        v=self.fixture();v[0]['status']='resource-wait'
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_any_completed_trial_is_preserved_instead_of_replayed(self):
        v=self.fixture();v[0]['trials']=[dict(id='035-kvmem-off-r1')]
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_other_task_or_runtime_error_is_not_admitted(self):
        v=self.fixture();v[0]['current_trial']='080-kvmem-on-r1'
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_initialized_vm_is_not_declared_input_free(self):
        v=self.fixture();v[2]['initialized_at']=1
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_any_vm_gateway_or_task_evidence_requires_separate_reconciliation(self):
        for name in ('vm.json','initial.png','task_instruction.json','app','gateway'):
            with self.subTest(name=name):
                v=self.fixture();v[4]=[name]
                with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_unknown_or_unfinished_result_is_not_zero_input(self):
        v=self.fixture();v[1].pop('finished_at')
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_scored_or_accepted_result_is_not_infrastructure_only(self):
        v=self.fixture();v[1]['score']=0
        with self.assertRaises(continuation.SafetyHalt):continuation.failure_scope(*v)

    def test_frozen_trial_body_budgets_and_commands_remain_present(self):
        source=inspect.getsource(continuation.owned.run);patched=continuation.patch_memory_source(source)
        compile(patched,'qualified-runtime-body','exec')
        for text in ('"--model-output-tokens", "16384"','child.wait(timeout=5100)','"--context-tokens", "524288"',
                     'Existing attempt cannot be replayed or overwritten','Trial is not durably terminal'):
            self.assertIn(text,patched)
        self.assertIn('runtime_declaration["preserved_engine_runtime"]',patched)

    def test_source_shape_change_is_refused(self):
        source=inspect.getsource(continuation.owned.run).replace('initial_runtime =','changed_runtime =',1)
        with self.assertRaises(continuation.SafetyHalt):continuation.patch_memory_source(source)


if __name__=='__main__':unittest.main()
