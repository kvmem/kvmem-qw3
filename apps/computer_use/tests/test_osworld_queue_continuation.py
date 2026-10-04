import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import osworld_queue_continuation as continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        trials=[{'id':'trial-'+str(i)} for i in range(40)];trials[11]['id']=continuation.CASE
        self.protocol={'trials':trials,'diagnostic_first_trials':[r['id'] for r in trials[:12]]}
        self.results={r['id']:{'finished_at':1,'uncertain_actions':int(i==11),'accepted':False} for i,r in enumerate(trials[:12])}
        self.state={'status':'halted-no-further-model-dispatch','pid':continuation.CONTROLLER['pid'],
                    'current_trial':continuation.CASE,'trials':[{'id':r['id'],'result':copy.deepcopy(self.results[r['id']])} for r in trials[:12]]}

    def reject(self):
        with self.assertRaises(continuation.owned.SafetyHalt):continuation.original_rows(self.protocol,self.state,self.results)

    def test_exact_retired_failure_is_carried_without_regrading_or_mutating_source(self):
        before=copy.deepcopy((self.state,self.results))
        rows=continuation.original_rows(self.protocol,self.state,self.results)
        self.assertEqual((self.state,self.results),before);self.assertEqual(rows[-1]['result']['uncertain_actions'],1)
        self.assertFalse(rows[-1]['result']['accepted']);rows[-1]['result']['accepted']=True
        self.assertFalse(self.state['trials'][-1]['result']['accepted'])

    def test_second_unknown_blocks_continuation(self):
        self.results['trial-1']['uncertain_actions']=1
        self.state['trials'][1]['result']=copy.deepcopy(self.results['trial-1']);self.reject()

    def test_original_unknown_cannot_be_cleared_or_reclassified(self):
        self.results[continuation.CASE]['uncertain_actions']=0
        self.state['trials'][11]['result']=copy.deepcopy(self.results[continuation.CASE]);self.reject()

    def test_missing_terminal_result_blocks(self):
        self.results['trial-1']=None;self.reject()

    def test_mutated_result_blocks(self):
        self.results['trial-1']['score']=1;self.reject()

    def test_missing_finished_at_blocks(self):
        self.results['trial-1']['finished_at']=None
        self.state['trials'][1]['result']=copy.deepcopy(self.results['trial-1']);self.reject()

    def test_unconfirmed_retirement_blocks(self):
        self.results['trial-1']['owned_task_stop_unconfirmed']=True
        self.state['trials'][1]['result']=copy.deepcopy(self.results['trial-1']);self.reject()

    def test_success_cannot_be_backfilled(self):
        self.results[continuation.CASE]['accepted']=True
        self.state['trials'][11]['result']=copy.deepcopy(self.results[continuation.CASE]);self.reject()

    def test_duplicate_or_missing_trial_blocks(self):
        self.state['trials'][1]=copy.deepcopy(self.state['trials'][0]);self.reject()

    def test_completed_or_wrong_controller_checkpoint_blocks(self):
        self.state['status']='completed';self.reject()
        self.state['status']='halted-no-further-model-dispatch';self.state['pid']+=1;self.reject()

    def test_order_or_named_failure_change_blocks(self):
        self.state['trials'][0],self.state['trials'][1]=self.state['trials'][1],self.state['trials'][0];self.reject()

    def test_larger_or_smaller_prefix_cannot_hide_unexecuted_or_replayed_trials(self):
        self.state['trials'].pop();self.reject()

    def test_original_fourty_order_and_all_remaining_trial_identities_are_preserved(self):
        rows=continuation.ordered_trials(self.protocol)
        self.assertEqual(rows,self.protocol['trials']);self.assertEqual(len(rows[12:]),28)

    def test_bad_declaration_order_rejected(self):
        self.protocol['diagnostic_first_trials'].append('unknown')
        with self.assertRaises(continuation.owned.SafetyHalt):continuation.ordered_trials(self.protocol)

    def test_retired_pid_is_safe_but_live_or_reused_pid_blocks(self):
        identity=continuation.CONTROLLER
        with patch.object(continuation.owned,'process_identity',return_value=None):continuation.retired_controller(identity)
        for actual in (identity|{'state':'S'},identity|{'state':'Z','start_ticks':identity['start_ticks']+1}):
            with patch.object(continuation.owned,'process_identity',return_value=actual):
                with self.assertRaises(continuation.owned.SafetyHalt):continuation.retired_controller(identity)


if __name__=='__main__':unittest.main()
