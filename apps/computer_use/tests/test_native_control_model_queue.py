"""Run the real native queue/receipt logic with explicit host/model fixtures."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import native_control_model_queue as q
import native_control_video_probe as video


class QueryReceiptsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        sources = {}
        for name in ('seed','pressure'):
            path = self.root/(name+'.json');path.write_bytes(json.dumps({'messages':[{'role':'user','content':name}]}).encode())
            sources[name] = dict(source_request=str(path),request_sha256=q.owned.digest(path))
        self.protocol = dict(trials=q.deferred.QUERY.trials(),sources=sources)
        self.state = dict(termination_confirmed=True,parent_services_preserved=True,final_engine_policy='off',trials=[])
        for spec in self.protocol['trials']:
            out = self.root/'attempts'/spec['id'];out.mkdir(parents=True)
            leaves = []
            for name in ('seed','pressure'):
                leaf = out/name;leaf.mkdir();(leaf/'request.json').write_bytes(Path(sources[name]['source_request']).read_bytes())
                result = dict(status='completed',termination_confirmed=True,original_generation_profile_unchanged=True,
                              tool_calls_executed=0,device_inputs=0,business_success_claimed=False,actual_request_sha256=sources[name]['request_sha256'])
                q.save(leaf/'result.json',result);leaves.append(dict(name=name,result=result,result_sha256=q.owned.digest(leaf/'result.json')))
            result = dict(status='completed',termination_confirmed=True,original_generation_profile_unchanged=True,
                          tool_calls_executed=0,device_inputs=0,business_success_claimed=False,request_order=['seed','pressure'],rows=leaves)
            q.save(out/'sequence-result.json',result)
            row = dict(id=spec['id'],result=result,result_sha256=q.owned.digest(out/'sequence-result.json'))
            q.save(out/'retrieval-evidence.json',dict(mode=spec['mode'],result_sha256=row['result_sha256'],tool_calls_executed=0,
                                                     mechanism_qualified=False,business_or_latency_improvement_proven=False))
            row['evidence_sha256'] = q.owned.digest(out/'retrieval-evidence.json');self.state['trials'].append(row)

    def invoke(self):
        with patch.object(q.owned,'assert_no_services') as retired:
            result = q.query_receipts(self.protocol,self.state,self.root)
        self.assertEqual(retired.call_count,4)
        return result

    def rewrite(self,row):
        out = self.root/'attempts'/row['id'];q.save(out/'sequence-result.json',row['result'])
        row['result_sha256'] = q.owned.digest(out/'sequence-result.json')
        diagnostic = q.read(out/'retrieval-evidence.json');diagnostic['result_sha256'] = row['result_sha256']
        q.save(out/'retrieval-evidence.json',diagnostic);row['evidence_sha256'] = q.owned.digest(out/'retrieval-evidence.json')

    def test_all_four_actual_wire_receipts_are_required_even_without_retrieval_gain(self):
        self.assertEqual([r['id'] for r in self.invoke()],[r['id'] for r in self.protocol['trials']])

    def test_missing_or_reordered_query_rows_cannot_handoff(self):
        self.state['trials'].reverse()
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_unknown_sequence_retirement_cannot_handoff(self):
        row = self.state['trials'][0];row['result']['termination_confirmed'] = False;self.rewrite(row)
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_leaf_model_wire_cannot_change_even_if_own_receipts_still_match(self):
        out = self.root/'attempts'/self.protocol['trials'][0]['id']/'pressure'
        (out/'request.json').write_text('{"messages":[]}')
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_read_only_leaf_cannot_add_device_inputs(self):
        row = self.state['trials'][0];leaf = row['result']['rows'][0]
        leaf['result']['device_inputs'] = 1
        path = self.root/'attempts'/row['id']/leaf['name']/'result.json';q.save(path,leaf['result']);leaf['result_sha256'] = q.owned.digest(path)
        self.rewrite(row)
        with self.assertRaises(q.SafetyHalt):self.invoke()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(protocol=self.root/'declaration.json',state_dir=self.root/'state',min_free_mib=56000,resource_wait_seconds=21600)
        q.save(self.args.protocol,{})
        self.protocol = dict(trials=[dict(id='trial-'+str(i),case='linux_text',arm='native-keyboard-control',repetition=1,seconds=360) for i in range(8)],
                             registration=str(self.root/'registration.json'),prepared_study=dict(path='fixture-study'))

    def loop(self,send,parent=None):
        with patch.object(q,'validate',return_value=self.protocol),patch.object(q,'predecessor_ready',return_value={} if parent is None else parent),\
             patch.object(q.owned,'process_identity',return_value={'pid':os.getpid()}),patch.object(q.Controller,'dispatch',send):
            return q.run(self.args)

    def test_original_eight_trial_order_and_failed_grade_are_preserved(self):
        seen = []
        def send(controller,row):seen.append(copy.deepcopy(row));controller.state['trials'].append(dict(row,accepted=False))
        result = self.loop(send)
        self.assertEqual(seen,self.protocol['trials']);self.assertTrue(result['termination_confirmed'])
        self.assertEqual(result['final_engine_policy'],'off');self.assertTrue(all(not r['accepted'] for r in result['trials']))

    def test_second_error_stops_without_overwriting_first_or_replaying(self):
        seen = []
        def send(controller,row):
            seen.append(row['id'])
            if len(seen)==2:raise q.SafetyHalt('Unconfirmed owned action')
            controller.state['trials'].append(dict(row,result_sha256='original-preserved'))
        with self.assertRaises(q.SafetyHalt):self.loop(send)
        state = q.read(self.args.state_dir/'queue.json')
        self.assertEqual(seen,['trial-0','trial-1']);self.assertEqual(state['trials'][0]['result_sha256'],'original-preserved')
        self.assertTrue(state['live_children_and_evidence_preserved'])

    def test_existing_attempt_refuses_before_engine_or_child_launch(self):
        row = self.protocol['trials'][0];(self.root/'attempts'/row['id']).mkdir(parents=True)
        controller = q.Controller(self.args,self.protocol,{'protocol_sha256':q.owned.digest(self.args.protocol)})
        with patch.object(q,'validate'),patch.object(controller,'fresh_engine') as engine,patch.object(controller,'spawn') as spawn:
            with self.assertRaises(q.SafetyHalt):controller.dispatch(row)
            engine.assert_not_called();spawn.assert_not_called()

    def test_unfinished_actual_parent_handle_is_verified_and_waited(self):
        reg = dict(identity={'pid':100});state = dict(status='waiting')
        with patch.object(q,'parent_state',return_value=(reg,state)),patch.object(q.base,'require_identity') as verify,patch.object(q,'parent_protocol') as admit:
            self.assertIsNone(q.predecessor_ready({}));verify.assert_called_once_with(reg['identity']);admit.assert_not_called()

    def test_completed_but_live_parent_cannot_handoff(self):
        with patch.object(q,'parent_state',return_value=({'identity':{'pid':100}},{'status':'completed'})),\
             patch.object(q.base,'process_retired',return_value=False),patch.object(q,'query_receipts') as receipts:
            self.assertIsNone(q.predecessor_ready({}));receipts.assert_not_called()

    def test_changed_parent_generation_refuses_instead_of_waiting_or_restarting(self):
        with patch.object(q,'parent_state',return_value=({'identity':{'pid':100}},{'status':'waiting'})),\
             patch.object(q.base,'require_identity',side_effect=q.SafetyHalt('Generation changed')):
            with self.assertRaises(q.SafetyHalt):q.predecessor_ready({})

    def test_observation_timeout_polls_same_child_without_relaunch_or_new_budget(self):
        row = self.protocol['trials'][0]
        state = dict(protocol_sha256=q.owned.digest(self.args.protocol),engine_runtime='original-engine',registered_engine={'pid':200})
        controller = q.Controller(self.args,self.protocol,state);self.args.state_dir.mkdir()
        child = Mock();identity = {'pid':201};child.wait.side_effect = [subprocess.TimeoutExpired('same',10),subprocess.TimeoutExpired('same',10),0]
        with ExitStack() as patches:
            for owner,name,kwargs in [(q,'validate',{}),(controller,'fresh_engine',{}),(controller,'parent_idle',{}),
                 (q.owned,'model_lock',{}),(q.stack.adapter,'validate_study',{'return_value':({},row,Path('/selected-framework'),{})}),
                 (controller,'checkpoint',{}),(controller,'collect',{})]:patches.enter_context(patch.object(owner,name,**kwargs))
            spawn = patches.enter_context(patch.object(controller,'spawn',return_value=(child,identity)))
            verify = patches.enter_context(patch.object(q.base,'require_identity'))
            collect = controller.collect
            controller.dispatch(row)
        self.assertEqual(spawn.call_count,1);self.assertEqual(child.wait.call_count,3);self.assertEqual(verify.call_count,2)
        collect.assert_called_once_with(row,self.root/'attempts'/row['id'],0)
        self.assertEqual(spawn.call_args.args[2],Path('/selected-framework'))
        self.assertIn('--trial-id',spawn.call_args.args[0])

    def test_scoped_full_handoff_source_change_is_rejected(self):
        with patch.object(q.inspect,'getsource',return_value='def predecessor_ready(p): pass'),self.assertRaises(q.SafetyHalt):q.earlier_handoff({}, {})


spec = importlib.util.spec_from_file_location('native_retained_full_handoff',Path(__file__).with_name('test_query_refresh_replay_queue.py'))
retained = importlib.util.module_from_spec(spec);spec.loader.exec_module(retained);retained.queue = q.deferred.QUERY


class FullPriorHandoffTests(retained.FullHandoffTests):
    """Execute the original full inventory checks plus bounded8 with final query engine."""
    def setUp(self):
        super().setUp()
        self.bounded = dict(trials=q.deferred.BOUNDED.trials())
        self.bound_path = self.root/'bounded/protocol.json';self.bound_path.parent.mkdir();q.save(self.bound_path,self.bounded)
        rows = []
        for row in self.bounded['trials']:
            out = self.bound_path.parent/'attempts'/row['id'];(out/'app').mkdir(parents=True);q.save(out/'app/task.json',{})
            result = dict(finished_at=1,uncertain_actions=0,owned_task_retired=True);q.save(out/'result.json',result)
            digest = q.owned.digest(out/'result.json')
            q.save(out/'bounded-continuation-evidence.json',dict(original_result_sha256=digest,arm=row['arm'],task_sha256=q.owned.digest(out/'app/task.json')))
            rows.append(dict(id=row['id'],result=result,result_sha256=digest,evidence_sha256=q.owned.digest(out/'bounded-continuation-evidence.json')))
        self.bstate = dict(status='completed',trials=rows,termination_confirmed=True,parent_services_preserved=True,final_engine_policy='off')
        self.breg = dict(identity=self.identity(500),queue='fixture-bounded-queue',protocol=str(self.bound_path))
        self.current = dict(parent_registration='fixture-bounded-reg',parent_controller=self.breg['identity'],parent_queue=self.breg['queue'])

    def invoke(self,retired=lambda identity:True,registered=None):
        namespace = q.deferred.QUERY;chain = namespace.q
        with ExitStack() as patches:
            fixtures = [(chain,'validate',lambda path:(self.d,dict(continuation_controller_identity=self.identity(150)),dict(identity=self.identity(151)))),
                (chain,'successor_registration',lambda path,stage,*args:self.regs[stage]),
                (chain,'successor_queue',lambda reg,path:next(self.states[s] for s in chain.STAGES if reg==self.regs[s])),
                (chain,'frozen_protocol',lambda stage,path:(self.protocols[stage],{}) if stage=='sdk' else self.protocols[stage]),
                (chain,'memory_queue',lambda *args:self.memory),(chain.quota,'validate_protocol',lambda path:self.kp),
                (chain.old,'continued_parent',lambda *args:dict(status='completed')),
                (chain.old,'terminal_parent',lambda *args:[dict(id='explicit-historical40-fixture')]),
                (q.base,'process_retired',retired),(q.base,'require_identity',lambda identity:identity),(q.base,'ledger_terminal',lambda runtime:{}),
                (q.owned,'assert_no_services',lambda path:None),(q.owned,'assert_vm_retired',lambda path:None),(q.owned,'assert_apps_idle',lambda app:None),
                (q.owned,'register_engine',lambda runtime,expected:registered or self.engine),
                (namespace,'registration',lambda path:(self.breg,self.bstate)),(namespace.bounded,'validate',lambda path:self.parent|self.bounded)]
            for owner,name,value in fixtures:patches.enter_context(patch.object(owner,name,side_effect=value))
            return q.earlier_handoff(self.current,self.last)

    def test_all_eight_bounded_receipts_survive_final_query_engine_handoff(self):
        result = self.invoke();self.assertEqual(len(result['query_refresh_predecessor_receipts']),8)
        self.assertEqual(result['registered_engine'],self.engine)

    def test_missing_bounded_trial_is_not_hidden_by_completed_query(self):
        self.bstate['trials'].pop()
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_unknown_bounded_action_refuses_handoff(self):
        row = self.bstate['trials'][0];row['result']['uncertain_actions'] = 1
        path = self.bound_path.parent/'attempts'/row['id']/'result.json';q.save(path,row['result']);row['result_sha256'] = q.owned.digest(path)
        with self.assertRaises(q.SafetyHalt):self.invoke()


class VideoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root = Path(self.tmp.name)
        movie = self.root/'continuous.mp4';movie.write_bytes(b'fixed-video-fixture');self.digest = q.owned.digest(movie)
        q.save(self.root/'private-recorder-identity.json',{'pid':100})
        (self.root/'video').mkdir()
        (self.root/'video/screen.mp4').write_bytes(movie.read_bytes())
        q.save(self.root/'video/recording.json',dict(complete=True,status='complete',input_events=0,executor_observe_calls=0,
            video_frames=3,width=20,height=10,artifacts={'screen.mp4':dict(sha256=self.digest)}))

    def invoke(self,frames=3):
        decoded = dict(path=str(self.root/'continuous.mp4'),sha256=self.digest,frames=frames,width=20,height=10)
        with patch.object(q.base,'process_retired',return_value=True),\
             patch.object(q.subprocess,'run',return_value=Mock(stdout=json.dumps(decoded))):return q.video_evidence(self.root)

    def test_independent_decode_and_recorder_retirement_are_required(self):
        self.assertEqual(self.invoke()['independently_decoded']['frames'],3)

    def test_partial_video_decode_is_not_success(self):
        with self.assertRaises(q.SafetyHalt):self.invoke(2)

    def test_modified_original_video_is_rejected_before_decoder(self):
        (self.root/'continuous.mp4').write_bytes(b'changed')
        with patch.object(q.base,'process_retired',return_value=True),patch.object(q.subprocess,'run') as decoder:
            with self.assertRaises(q.SafetyHalt):q.video_evidence(self.root)
            decoder.assert_not_called()

    def test_live_recorder_is_preserved_without_accepting_video(self):
        with patch.object(q.base,'process_retired',return_value=False),patch.object(q.subprocess,'run') as decoder:
            with self.assertRaises(q.SafetyHalt):q.video_evidence(self.root)
            decoder.assert_not_called()


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=q.RUNTIME,prefix='native-queue-cpu-fixture-');self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name);self.path = self.root/'declaration.json'
        self.study = dict(trials=[dict(id='preserved-'+str(i)) for i in range(8)])
        self.p = dict(kind='declared_serial_native_control_original_pilots',bounds=copy.deepcopy(q.BOUNDS),model_concurrency=1,
            advisor_enabled=False,overwrite_old_attempts=False,state_dir=str(self.root/'queue-runtime-01'),
            registration=str(self.root/'registration.json'),trials=self.study['trials'],engine_off_identity={'profile':'original'})
        for key in ('qualification','stack_audit','prepared_study','parent_registration','parent_declaration','parent_protocol'):
            path = self.root/(key+'.json');q.save(path,self.study if key=='prepared_study' else {})
            self.p[key] = q.ref(path)
        refs = [self.p[key] for key in ('qualification','stack_audit','prepared_study','parent_registration','parent_declaration','parent_protocol')]
        refs += [q.ref(q.ROOT/'apps/computer_use'/name) for name in q.HELPERS]
        self.p['files'] = refs

    def invoke(self):
        q.save(self.path,self.p)
        with patch.object(q.stack,'STUDY_SHA',self.p['prepared_study']['sha256']),\
             patch.object(q.stack.adapter,'validate_study',return_value=(self.study,None,None,{'engine_off_identity':{'profile':'original'}})),\
             patch.object(q,'qualification'),patch.object(q,'qualified_stacks'),patch.object(q,'parent_state'):
            return q.validate(self.path)

    def test_exact_declared_inventory_and_paths_are_accepted(self):self.assertEqual(self.invoke(),self.p)

    def test_removed_required_source_pin_is_rejected(self):
        self.p['files'].pop()
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_changed_trial_order_is_not_a_route_change(self):
        self.p['trials'] = list(reversed(self.p['trials']))
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_changed_task_resource_bound_is_rejected(self):
        self.p['bounds']['model_ready_seconds'] += 1
        with self.assertRaises(q.SafetyHalt):self.invoke()

    def test_other_queue_registration_cannot_be_overwritten(self):
        self.p['registration'] = str(self.root.parent/'other-registration.json')
        with self.assertRaises(q.SafetyHalt):self.invoke()


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root = Path(self.tmp.name)
        self.row = dict(id='original-text-control-r1',case='linux_text',arm='native-keyboard-control',repetition=1,seconds=360)
        self.out = self.root/'attempts'/self.row['id'];self.pilot = self.out/'linux_text-r1';self.pilot.mkdir(parents=True)
        self.outcome = dict(uncertain_actions=0,owned_task_retired=True,app_status='paused',artifact_passed=False,
                            unassisted_passed=False,accepted=False,video_collection_passed=True)
        q.save(self.pilot/'integration-result.json',self.outcome)
        task = dict(thinking=True,model_output_tokens=16384,tool_call_limit=100,execution_mode='verified',
                    model_context_window=524288,device_platform='linux',tool_calls_used=9,software_root=None)
        q.save(self.pilot/'task.json',task)
        self.framework = self.root/'qualified-framework';self.original = dict(helper_sha256={'platform_pilots.py':'original-harness'})
        manifest = dict(prompt='same original task',harness_sha256='original-harness',deadline_seconds=360,model_output_tokens=16384,
                        no_model_access_to_initializer_or_grader=True,container_id='original-private-vm')
        q.save(self.pilot/'pilot-manifest.json',manifest)
        (self.out/'stack').mkdir()
        service = dict(status='retired',vm_retired=True,producer={'pid':100},services={'client':{'identity':{'pid':101},'runtime':str(self.root/'client')}},
                       vm={'container_id':'original-private-vm'},framework=str(self.framework))
        q.save(self.out/'stack/stack.json',service)
        result = dict(kind='one_frozen_private_native_control_qw3_trial',status='completed',termination_confirmed=True,
                      trial_id=self.row['id'],case=self.row['case'],arm=self.row['arm'],repetition=1,controller_sha256=q.TRIAL_SHA,
                      study_sha256='original-study',outcome=self.outcome,original_result_sha256=q.owned.digest(self.pilot/'integration-result.json'),
                      actual_task_sha256=q.owned.digest(self.pilot/'task.json'),prompt_sha256=q.trial.hashlib.sha256(manifest['prompt'].encode()).hexdigest(),
                      stack_sha256=q.owned.digest(self.out/'stack/stack.json'),actual_model_requests=[dict(trace='actual-request-fixture')])
        q.save(self.out/'result.json',result)
        self.controller = q.Controller(argparse.Namespace(protocol=self.root/'declaration.json',state_dir=self.root/'state'),
                                       {'prepared_study':{'path':'fixture-study','sha256':'original-study'}},{'trials':[]})

    def invoke(self):
        with patch.object(q.stack.adapter,'validate_study',return_value=({},self.row,self.framework,self.original)),\
             patch.object(q.base,'process_retired',return_value=True),\
             patch.object(q.subprocess,'run',return_value=Mock(returncode=1,stderr='No such object')),\
             patch.object(q.trial,'wire_evidence',return_value=[dict(trace='actual-request-fixture')]),\
             patch.object(q,'video_evidence',return_value={'independently_decoded':{'frames':3}}),\
             patch.object(q.owned,'assert_no_services'),patch.object(self.controller,'checkpoint'):
            self.controller.collect(self.row,self.out,0)

    def test_original_failed_business_grade_is_collected_without_rewriting_it(self):
        before = (self.pilot/'integration-result.json').read_bytes();self.invoke()
        self.assertEqual(len(self.controller.state['trials']),1)
        self.assertFalse(self.controller.state['trials'][0]['result']['outcome']['accepted'])
        self.assertEqual((self.pilot/'integration-result.json').read_bytes(),before)

    def test_grade_tamper_is_rejected_before_counting_trial(self):
        q.save(self.pilot/'integration-result.json',self.outcome|{'accepted':True})
        with self.assertRaises(q.SafetyHalt):self.invoke()
        self.assertEqual(self.controller.state['trials'],[])

    def test_unknown_action_is_not_counted_as_a_completed_trial(self):
        result = q.read(self.out/'result.json');result['outcome']['uncertain_actions'] = 1
        q.save(self.pilot/'integration-result.json',result['outcome']);result['original_result_sha256'] = q.owned.digest(self.pilot/'integration-result.json')
        q.save(self.out/'result.json',result)
        with self.assertRaises(q.SafetyHalt):self.invoke()
        self.assertEqual(self.controller.state['trials'],[])


if __name__ == '__main__':unittest.main()
