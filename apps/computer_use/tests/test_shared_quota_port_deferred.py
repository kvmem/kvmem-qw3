"""Frozen deferred loops/full handoff plus strict routing-copy boundaries."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_quota_port_deferred as adapter


def retained_tests(name, queue):
    spec = importlib.util.spec_from_file_location("port_deferred_retained_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.queue = queue
    return module


bounded_tests = retained_tests("test_bounded_continuation_retest_queue", adapter.BOUNDED)
query_tests = retained_tests("test_query_refresh_replay_queue", adapter.QUERY)


class BoundedQueueTests(bounded_tests.BoundedQueueTests):
    pass


class QueryQueueTests(query_tests.QueueTests):
    pass


class FullHandoffTests(query_tests.FullHandoffTests):
    pass


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {name:self.root / (name + '.json') for name in ('old-reg','old-protocol','old-declaration','new-reg','new-protocol','new-declaration','static')}
        for path in self.paths.values(): adapter.save(path, {})
        self.original = dict(parent_registration=str(self.paths['old-reg']), parent_protocol=str(self.paths['old-protocol']),
            parent_queue='old-queue', parent_controller={'pid':120}, parent_declaration=str(self.paths['old-declaration']),
            trials=copy.deepcopy(adapter.BOUNDED.trials()), model={'thinking':True,'max_tokens':16384},
            frameworks={'control':'unchanged-control','candidate':'unchanged-candidate'},
            sources={'seed':{'source_request':'unchanged-private-input'}},
            files=[dict(path=str(path),sha256=adapter.owned.digest(path)) for name,path in self.paths.items() if name.startswith('old-') or name=='static'])
        self.parent=dict(registration_path=str(self.paths['new-reg']),protocol=str(self.paths['new-protocol']),queue='new-queue',identity={'pid':121})

    def routed(self, stage='bounded'):
        return adapter.routed_protocol(stage,self.original,self.parent,self.paths['new-declaration'])

    def test_only_exact_parent_fields_and_parent_pins_change(self):
        before=copy.deepcopy(self.original);actual=self.routed();adapter.route_proof('bounded',self.original,actual)
        self.assertEqual(self.original,before)
        allowed={*adapter.ROUTING,'parent_declaration','files'}
        self.assertTrue({key for key in actual if actual[key]!=before[key]}<=allowed)
        self.assertEqual(actual['trials'],before['trials'])

    def test_query_keeps_saved_inputs_generation_and_model_profile_unchanged(self):
        actual=self.routed('query');adapter.route_proof('query',self.original,actual)
        for key in ('sources','model','frameworks','trials','parent_declaration'):self.assertEqual(actual[key],self.original[key])

    def test_changed_trial_or_budget_cannot_be_hidden_as_routing(self):
        for key in ('trials','model'):
            actual=self.routed();actual[key]={}
            with self.subTest(key=key),self.assertRaises(adapter.SafetyHalt):adapter.route_proof('bounded',self.original,actual)

    def test_changed_saved_input_or_framework_is_rejected(self):
        for key in ('sources','frameworks'):
            actual=self.routed('query');actual[key]={}
            with self.subTest(key=key),self.assertRaises(adapter.SafetyHalt):adapter.route_proof('query',self.original,actual)

    def test_nonparent_source_pin_cannot_be_removed(self):
        actual=self.routed();actual['files']=[item for item in actual['files'] if item['path']!=str(self.paths['static'])]
        with self.assertRaises(adapter.SafetyHalt):adapter.route_proof('bounded',self.original,actual)

    def test_new_parent_pin_must_match_actual_bytes(self):
        actual=self.routed();adapter.save(self.paths['new-reg'],{'changed':True})
        with self.assertRaises(adapter.SafetyHalt):adapter.route_proof('bounded',self.original,actual)

    def test_frozen_sources_and_terminal_handoff_inspection_are_exact(self):
        for stage,module in (('bounded',adapter.original_bounded),('query',adapter.original_query)):
            self.assertEqual(adapter.owned.digest(module.__file__),adapter.SOURCE_SHA[stage])
        self.assertEqual(adapter.original_query.inspect.getsource(adapter.BOUNDED.terminal_predecessor),
                         adapter.original_query.inspect.getsource(adapter.original_bounded.terminal_predecessor))

    def test_bounded_namespace_uses_restored_producer_without_mutating_original_namespace(self):
        self.assertIs(adapter.BOUNDED.q,adapter.q)
        self.assertIsNot(adapter.original_bounded.q,adapter.q)
        self.assertIs(adapter.QUERY.bounded,adapter.BOUNDED)

    def test_missing_declaration_cannot_enter_a_model_loop(self):
        with patch.object(adapter.BOUNDED,'DECLARATION',None),self.assertRaises(adapter.SafetyHalt):adapter.bounded_validate(Path('/absent'))
        with patch.object(adapter.QUERY,'DECLARATION',None),self.assertRaises(adapter.SafetyHalt):adapter.query_validate(Path('/absent'))

    def original_fixture(self):
        reg=dict(protocol=str(self.paths['old-protocol']),protocol_sha256='exact',controller_sha256='source',planned_trials=8,
                 queue=str(self.paths['static']),identity=dict(pid=120,uid=os.getuid(),cwd=str(adapter.ROOT)))
        state=dict(status='halted-no-further-dispatch',trials=[],controller_sha256='source',protocol_sha256='exact')
        adapter.save(self.paths['old-reg'],reg)
        adapter.save(self.paths['static'],state)
        return reg,state

    def check_original(self,actual=None):
        with patch.dict(adapter.SOURCE_PROTOCOLS,bounded=self.paths['old-protocol']),patch.dict(adapter.SOURCE_REGISTRATIONS,bounded=self.paths['old-reg']),\
                patch.dict(adapter.PROTOCOL_SHA,bounded='exact'),patch.dict(adapter.SOURCE_SHA,bounded='source'),\
                patch.object(adapter.q.image,'pinned'),patch.object(adapter.owned,'process_identity',return_value=actual):
            return adapter.original_scope('bounded')

    def test_original_halted_zero_trial_scope_is_preserved(self):
        reg,state=self.original_fixture();before=self.paths['static'].read_bytes();self.check_original()
        self.assertEqual(self.paths['static'].read_bytes(),before)

    def test_live_or_nonempty_original_cannot_be_rebound(self):
        reg,state=self.original_fixture()
        with self.assertRaises(adapter.SafetyHalt):self.check_original(reg['identity'])
        state['trials']=[{'id':'already-dispatched'}];adapter.save(self.paths['static'],state)
        with self.assertRaises(adapter.SafetyHalt):self.check_original()

    def test_original_failure_status_cannot_be_changed_to_completed(self):
        reg,state=self.original_fixture();state['status']='completed';adapter.save(self.paths['static'],state)
        with self.assertRaises(adapter.SafetyHalt):self.check_original()


if __name__ == '__main__':unittest.main()
