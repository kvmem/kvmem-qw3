import copy
import inspect
import unittest

import kvmem_osworld_queue as base
import osworld_retired_trial_successors as successor
import resumed_kvmem_queue
from test_osworld_queue_continuation import ContinuationTests


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        seed=ContinuationTests();seed.setUp()
        self.protocol=seed.protocol;self.original=seed.state['trials'];self.results=seed.results
        for row in self.protocol['trials'][12:]:self.results[row['id']]={'finished_at':2,'uncertain_actions':0,'accepted':False}
        self.state={'status':'completed','historical_unknowns_preserved':1,
            'trials':[{'id':row['id'],'result':copy.deepcopy(self.results[row['id']])} for row in self.protocol['trials']]}

    def reject(self):
        with self.assertRaises(base.SafetyHalt):successor.terminal_rows(self.protocol,self.state,self.original,self.results)

    def test_all_forty_keep_original_unknown_and_failed_business_results(self):
        before=copy.deepcopy((self.state,self.original,self.results))
        result=successor.terminal_rows(self.protocol,self.state,self.original,self.results)
        self.assertEqual(len(result),40);self.assertEqual(result[11]['result']['uncertain_actions'],1)
        self.assertFalse(result[11]['result']['accepted']);self.assertEqual((self.state,self.original,self.results),before)

    def test_twelve_or_thirty_nine_are_not_completion(self):
        self.state['trials']=self.state['trials'][:39];self.reject()
        self.state['trials']=self.state['trials'][:12];self.reject()

    def test_duplicate_trial_cannot_replace_an_unexecuted_trial(self):
        self.state['trials'][-1]=copy.deepcopy(self.state['trials'][-2]);self.reject()

    def test_reordered_trial_blocks(self):
        self.state['trials'][20],self.state['trials'][21]=self.state['trials'][21],self.state['trials'][20];self.reject()

    def test_running_or_halted_parent_is_not_admitted(self):
        for status in ('running','halted-no-further-model-dispatch'):
            self.state['status']=status;self.reject()

    def test_new_unknown_blocks_all_successors(self):
        self.results['trial-20']['uncertain_actions']=1
        self.state['trials'][20]['result']=copy.deepcopy(self.results['trial-20']);self.reject()

    def test_original_unknown_not_backfilled_as_delivered(self):
        self.results['022-legacy-r1']['uncertain_actions']=0
        self.state['trials'][11]['result']=copy.deepcopy(self.results['022-legacy-r1']);self.reject()

    def test_original_result_cannot_be_regraded(self):
        self.results['022-legacy-r1']['accepted']=True
        self.state['trials'][11]['result']=copy.deepcopy(self.results['022-legacy-r1']);self.reject()

    def test_new_changed_result_blocks(self):
        self.results['trial-20']['score']=1;self.reject()

    def test_nonterminal_or_unconfirmed_result_blocks(self):
        self.results['trial-20']['finished_at']=None
        self.state['trials'][20]['result']=copy.deepcopy(self.results['trial-20']);self.reject()

    def test_bool_unknown_count_is_not_zero(self):
        self.results['trial-20']['uncertain_actions']=False
        self.state['trials'][20]['result']=copy.deepcopy(self.results['trial-20']);self.reject()

    def test_original_quarantine_record_must_remain_explicit(self):
        self.state['historical_unknowns_preserved']=0;self.reject()

    def test_frozen_eight_run_changes_only_parent_queue_and_wrapper_attribution(self):
        source=inspect.getsource(base.run);patched=resumed_kvmem_queue.patched_run_source(source)
        compile(patched,'isolated-continued-memory-run','exec')
        changed=[(a,b) for a,b in zip(source.splitlines(),patched.splitlines()) if a!=b]
        self.assertEqual(len(changed),2)
        for critical in ('register_engine(initial_runtime','result["uncertain_actions"] != 0','retire_registered(registered)',
                         'assert_apps_idle(p["pilot_app"])','assert_vm_retired(parent_dir / "attempts" / row["id"])'):
            self.assertIn(critical,patched)

    def test_changed_frozen_run_shape_is_not_silently_patched(self):
        source=inspect.getsource(base.run).replace('parent_dir / "queue.json"','parent_dir / "other.json"')
        with self.assertRaises(base.SafetyHalt):resumed_kvmem_queue.patched_run_source(source)

    def test_relative_command_path_is_resolved_against_registered_cwd(self):
        identity={'cwd':'/tmp/own-test','command':['python','-m','module','--followup','input.json']}
        self.assertTrue(successor.command_path(identity,'--followup','/tmp/own-test/input.json'))
        self.assertFalse(successor.command_path(identity,'--followup','/tmp/other-test/input.json'))

    def test_absent_duplicate_or_missing_command_argument_is_rejected(self):
        for command in (['python'],['python','--followup'],['python','--followup','a','--followup','b']):
            self.assertFalse(successor.command_path({'cwd':'/tmp','command':command},'--followup','/tmp/a'))


if __name__=='__main__':unittest.main()
