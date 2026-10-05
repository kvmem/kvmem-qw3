import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import condition_schema_model_queue as q


class SeparateQueueTests(unittest.TestCase):
    def test_waiting_parent_requires_actual_same_process_and_never_dispatches(self):
        reg={'identity':{'pid':123}}
        with patch.object(q,'parent_state',return_value=(reg,{'status':'waiting'})),patch.object(q.base,'require_identity') as identity,\
             patch.object(q,'original_receipts') as receipts:
            self.assertIsNone(q.predecessor_ready({}))
        identity.assert_called_once_with(reg['identity']);receipts.assert_not_called()

    def test_completed_parent_must_actually_retire_before_receipt_audit(self):
        reg={'identity':{'pid':123}}
        with patch.object(q,'parent_state',return_value=(reg,{'status':'completed'})),patch.object(q.base,'process_retired',return_value=False),\
             patch.object(q,'original_receipts') as receipts:
            self.assertIsNone(q.predecessor_ready({}))
        receipts.assert_not_called()

    def test_incomplete_native_inventory_cannot_skip_to_new_cohort(self):
        parent={'trials':[{'id':str(i)} for i in range(8)]}
        state={'termination_confirmed':True,'parent_services_preserved':True,'final_engine_policy':'off',
            'trials':[{'id':str(i)} for i in range(7)]}
        with patch.object(q.native.Controller,'collect') as collector,self.assertRaises(q.SafetyHalt):q.original_receipts(parent,state,Path('/tmp/fixture'))
        collector.assert_not_called()

    def test_complete_inventory_reuses_all_eight_original_collectors_in_memory(self):
        specs=[{'id':str(i)} for i in range(8)];records=[dict(r,exit_code=0,result={'accepted':False}) for r in specs]
        parent={'trials':specs};state={'termination_confirmed':True,'parent_services_preserved':True,'final_engine_policy':'off','trials':records}
        before=copy.deepcopy(state)
        def collect(instance,spec,output,code):instance.state['trials'].append(copy.deepcopy(records[int(spec['id'])]))
        with patch.object(q.native.Controller,'collect',side_effect=collect) as collector:
            self.assertEqual(q.original_receipts(parent,state,Path('/tmp/fixture')),records)
        self.assertEqual(collector.call_count,8);self.assertEqual(state,before)

    def test_changed_original_grade_or_collector_failure_is_not_hidden(self):
        specs=[{'id':str(i)} for i in range(8)];state={'termination_confirmed':True,'parent_services_preserved':True,'final_engine_policy':'off',
            'trials':[dict(s,exit_code=0) for s in specs]}
        def changed(instance,*_):instance.state['trials'].append({'changed':True})
        for behavior in [changed,q.SafetyHalt('unknown original delivery')]:
            with patch.object(q.native.Controller,'collect',side_effect=behavior),self.assertRaises(q.SafetyHalt):q.original_receipts({'trials':specs},state,Path('/tmp/fixture'))

    def test_dispatch_changes_only_entry_module_and_private_ports(self):
        function=q.dispatch_function()
        self.assertIn('condition_schema_model_trial',repr(function.__code__.co_consts))
        self.assertIn('19060',repr(function.__code__.co_consts))
        self.assertNotIn('native_control_model_trial',repr(function.__code__.co_consts))
        self.assertEqual(function.__globals__['BOUNDS'],q.native.BOUNDS)
        self.assertIs(function.__globals__['validate'],q.validate)

    def test_dispatch_observation_timeout_polls_same_child_without_restart(self):
        from types import SimpleNamespace
        row={'id':'one','seconds':360};child=Mock();child.wait.side_effect=[q.native.subprocess.TimeoutExpired('same',10),0]
        obj=object.__new__(q.Controller);obj.args=SimpleNamespace(protocol=Path('/tmp/fixture/declaration.json'),state_dir=Path('/tmp/fixture/state'))
        obj.p={'prepared_study':{'path':'/tmp/fixture/study.json'}}
        obj.state={'protocol_sha256':'same','engine_runtime':'/tmp/fixture/engine','registered_engine':{'pid':456}}
        obj.fresh_engine=Mock();obj.parent_idle=Mock();obj.checkpoint=Mock();obj.spawn=Mock(return_value=(child,{'pid':123}));obj.collect=Mock()
        globals_=obj.dispatch.__globals__
        with patch.dict(globals_,validate=lambda _:None),patch.object(q.owned,'digest',return_value='same'),\
             patch.object(q.Path,'exists',return_value=False),patch.object(q.owned,'model_lock'),patch.object(q.native,'save'),\
             patch.dict(globals_,save=lambda *_:None),patch.object(globals_['stack'].adapter,'validate_study',return_value=(None,None,Path('/tmp/fixture/package'),None)),\
             patch.object(q.base,'require_identity') as identity:
            obj.dispatch(row)
        self.assertEqual(obj.spawn.call_count,1);self.assertEqual(child.wait.call_count,2);identity.assert_called_once_with({'pid':123})
        obj.collect.assert_called_once();obj.fresh_engine.assert_called_once()

    def test_complete_handoff_uses_native_final_engine_and_retains_original_query_receipts(self):
        native_state={'status':'completed','engine_runtime':'/tmp/native-final','registered_engine':{'pid':456}}
        query_state={'engine_runtime':'/tmp/query-old','registered_engine':{'pid':321}}
        reg={'identity':{'pid':123},'declaration':'/tmp/native/declaration.json'}
        query_reg={'identity':{'pid':122},'protocol':'/tmp/query/protocol.json'}
        before=copy.deepcopy(query_state);p={'engine_off_identity':{'profile':'original'}}
        with patch.object(q,'parent_state',return_value=(reg,native_state)),patch.object(q.base,'process_retired',return_value=True),\
             patch.object(q.native,'validate',return_value={'trials':'eight'}),patch.object(q,'original_receipts',return_value=['eight-original-receipts']) as collect,\
             patch.object(q.native,'parent_state',return_value=(query_reg,query_state)),patch.object(q.native,'parent_protocol',return_value={'query':'original'}),\
             patch.object(q.native,'query_receipts',return_value=['four-query-receipts']) as query,\
             patch.object(q.native,'earlier_handoff',return_value={}) as earlier,patch.object(q.owned,'register_engine',return_value={'pid':456}) as engine,\
             patch.object(q.owned,'same_process',return_value=True):
            services=q.predecessor_ready(p)
        collect.assert_called_once();query.assert_called_once()
        self.assertEqual(earlier.call_args.args[1]['engine_runtime'],'/tmp/native-final')
        self.assertEqual(earlier.call_args.args[1]['registered_engine'],{'pid':456})
        engine.assert_called_once_with(Path('/tmp/native-final'),p['engine_off_identity'])
        self.assertEqual(services['original_native_predecessor_receipts'],['eight-original-receipts'])
        self.assertEqual(query_state,before)

    def test_transient_observation_error_preserves_same_handle_and_missing_artifact_is_not_ignored(self):
        controller=Mock();reg={'identity':{'pid':123}}
        with patch.object(q,'predecessor_ready',side_effect=TimeoutError()),patch.object(q,'parent_state',return_value=(reg,{'status':'waiting'})),\
             patch.object(q.base,'require_identity') as identity:
            self.assertIsNone(q.observe_predecessor(controller,{}))
        identity.assert_called_once_with(reg['identity']);controller.checkpoint.assert_called_once()
        with patch.object(q,'predecessor_ready',side_effect=FileNotFoundError('original missing')),self.assertRaises(FileNotFoundError):
            q.observe_predecessor(controller,{})


if __name__=='__main__':unittest.main()
