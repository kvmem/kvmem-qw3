"""Isolated image-history boundaries; no live device or model requests."""
import base64
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock

from PIL import Image
import jsonschema

from visual_working_context_candidate import build_candidate, HISTORY_TOOLS


ROOT = Path(__file__).resolve().parents[1]
TEMP = tempfile.TemporaryDirectory(prefix='qw3-visual-working-tests-')
OUTPUT = Path(TEMP.name)/'candidate'
BUILD = build_candidate(ROOT/'.runtime/completion-comparison-feedback-candidate-01/framework/qw3_computer',
                        ROOT/'.runtime/completion-comparison-feedback-candidate-01/build.json', OUTPUT)
PACKAGE = OUTPUT/'framework/qw3_computer'
package = types.ModuleType('qw3_visual_test'); package.__path__=[str(PACKAGE)]
sys.modules[package.__name__]=package
history_module=importlib.import_module(package.__name__+'.observation_history')
visual=importlib.import_module(package.__name__+'.visual_working_context')
archive=importlib.import_module(package.__name__+'.visual_request_archive')
codec=importlib.import_module(package.__name__+'.request_archive')
desktop_module=importlib.import_module(package.__name__+'.desktop')
app=importlib.import_module(package.__name__+'.app')
codex=importlib.import_module(package.__name__+'.codex')


def observed(n, **overrides):
    stream=io.BytesIO(); Image.new('RGB',(4,3),(n % 256, 70, 90)).save(stream,format='PNG')
    data=stream.getvalue()
    result={'observation_id':f'obs{n}', 'desktop_id':'desktop1','device_id':'device1',
            'session_id':'session1','display_id':'primary','width':4,'height':3,
            'sha256':hashlib.sha256(data).hexdigest(),'image':base64.b64encode(data).decode(),
            'captured_at':n, 'foreground':{'handle':'window1','process':'test-app','title':f'View {n}'},
            'capabilities':{'platform':'linux'},'accessibility':{'status':'ok','truncated':False,'controls':[]}}
    return result | overrides


def context(n=6, name='mcp__computer__observe'):
    messages=[{'role':'system','content':'original instruction'},
              {'role':'user','content':[{'type':'text','text':'User reference'},
                  {'type':'image_url','image_url':{'url':'data:image/png;base64,'+observed(99)['image']}}]}]
    for i in range(n):
        obs=observed(i)
        messages += [{'role':'assistant','content':None,'reasoning_content':f'thought{i}',
                     'tool_calls':[{'id':f'call{i}','type':'function',
                                   'function':{'name':name,'arguments':'{}'}}]},
                    {'role':'tool','tool_call_id':f'call{i}','content':[
                        {'type':'text','text':'Wall time: 0.1 seconds\nOutput:'},
                        {'type':'text','text':json.dumps({'observation':{k:v for k,v in obs.items() if k!='image'}})},
                        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+obs['image']}}]}]
    return {'messages':messages,'tools':[{'type':'function','function':{
                'name':'mcp__computer__history','parameters':HISTORY_TOOLS[0]['inputSchema']}}],
            'max_tokens':16384,'enable_thinking':True,'kvmem_enabled':False}


class WorkingContextTests(unittest.TestCase):
    def test_only_older_known_images_are_replaced_source_and_reasoning_are_immutable(self):
        body=context(); original=copy.deepcopy(body)
        view, report=visual.working_context(body)
        self.assertEqual(body,original);self.assertEqual(report['source_image_count'],7)
        self.assertEqual(report['dropped_images'],4);self.assertEqual(report['retained_image_count'],3)
        self.assertEqual(view['messages'][2],body['messages'][1])
        for key in ('tools','max_tokens','enable_thinking','kvmem_enabled'):self.assertEqual(view[key],body[key])
        for i,message in enumerate(body['messages']):
            presented=view['messages'][i+1]
            if message['role']!='tool':self.assertEqual(presented,message)
            else:
                self.assertEqual(presented['content'][:2],message['content'][:2])
                if i<10:
                    ref=json.loads(presented['content'][2]['text'])
                    self.assertFalse(ref['input_target_eligible']);self.assertTrue(ref['historical_read_only'])

    def test_no_history_tool_keeps_full_input_with_accurate_image_count(self):
        body=context();body['tools']=[]
        view,report=visual.working_context(body)
        self.assertIs(view,body);self.assertFalse(report['applied']);self.assertEqual(report['retained_image_count'],7)

    def test_explicit_reference_pin_keeps_original_image(self):
        body=context();pin=history_module.observation_reference(observed(0))
        view,report=visual.working_context(body,[pin])
        self.assertEqual(report['dropped_images'],3);self.assertEqual(report['retained_image_count'],4)
        self.assertEqual(view['messages'][4],body['messages'][3])

    def test_missing_pin_does_not_silently_discard_reference(self):
        body=context();pin=history_module.observation_reference(observed(88))
        view,report=visual.working_context(body,[pin])
        self.assertIs(view,body);self.assertFalse(report['applied']);self.assertEqual(report['missing_pins'],[pin])
        self.assertEqual(report['retained_image_count'],7)

    def test_incomplete_identity_and_corrupt_image_are_protected(self):
        body=context();tool=body['messages'][3]
        payload=json.loads(tool['content'][1]['text']);payload['observation'].pop('session_id')
        tool['content'][1]['text']=json.dumps(payload)
        body['messages'][5]['content'][2]['image_url']['url']='data:image/png;base64,broken'
        view,report=visual.working_context(body)
        self.assertEqual(report['dropped_images'],2)
        self.assertEqual(view['messages'][4],body['messages'][3])
        self.assertEqual(view['messages'][6],body['messages'][5])

    def test_foreign_tool_images_are_not_selected(self):
        body=context(name='mcp__other__observe')
        view,report=visual.working_context(body)
        self.assertEqual(report['eligible_observation_images'],0);self.assertEqual(report['retained_image_count'],7)

    def test_ambiguous_multi_image_or_multi_metadata_is_protected(self):
        body=context();body['messages'][3]['content'].append(copy.deepcopy(body['messages'][3]['content'][2]))
        body['messages'][5]['content'].append(copy.deepcopy(body['messages'][5]['content'][1]))
        view,report=visual.working_context(body)
        self.assertEqual(report['eligible_observation_images'],4)
        for i in (3,5):self.assertEqual(view['messages'][i+1],body['messages'][i])

    def test_malformed_image_url_is_protected(self):
        body=context();body['messages'][3]['content'][2]['image_url']=None
        view,report=visual.working_context(body)
        self.assertEqual(view['messages'][4],body['messages'][3]);self.assertEqual(report['eligible_observation_images'],5)

    def test_each_device_session_display_retains_its_recent_observations(self):
        body=context()
        for i in (0,1,2):
            payload=json.loads(body['messages'][3+i*2]['content'][1]['text'])
            payload['observation']['session_id']='old-session'
            body['messages'][3+i*2]['content'][1]['text']=json.dumps(payload)
        _,report=visual.working_context(body)
        self.assertEqual(report['dropped_images'],2);self.assertEqual(report['retained_image_count'],5)

    def test_historical_read_is_separate_from_latest_live_images(self):
        body=context()
        for i in (4,5):
            payload=json.loads(body['messages'][3+i*2]['content'][1]['text'])
            payload['observation']['historical_read_only']=True
            body['messages'][3+i*2]['content'][1]['text']=json.dumps(payload)
        _,report=visual.working_context(body)
        self.assertEqual(report['dropped_images'],3)
        self.assertEqual([p['observation_id'] for p in report['selected']],['obs2','obs3','obs5'])

    def test_unvalidated_or_excessive_pins_fail_without_changing_source(self):
        body=context();old=copy.deepcopy(body)
        for pins in ([{'observation_id':'obs0'}],[history_module.observation_reference(observed(i)) for i in range(5)]):
            with self.assertRaises(ValueError):visual.working_context(body,pins)
        self.assertEqual(body,old)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='qw3-own-history-');self.runtime=Path(self.tmp.name)
        self.desktop=desktop_module.Desktop('http://127.0.0.1:9','fixture',self.runtime/'device')
        self.desktop.task_id='task1'
        for i in range(6):self.desktop.remember(observed(i))
        self.history=history_module.ObservationHistory(self.desktop,self.runtime)

    def tearDown(self):
        self.history.close();self.desktop.db.close();self.tmp.cleanup()

    def test_read_original_bytes_does_not_promote_latest_or_dispatch_input(self):
        latest=copy.deepcopy(self.desktop.latest)
        self.desktop.request=AsyncMock(side_effect=AssertionError('No device request allowed'))
        result=self.history.invoke({'operation':'read','observation_id':'obs0'})
        self.assertEqual(result['observation']['image'],observed(0)['image'])
        self.assertEqual(self.desktop.latest,latest);self.desktop.request.assert_not_called()
        self.assertFalse(result['input_dispatched']);self.assertFalse(result['history']['promoted_to_latest'])
        self.assertTrue(result['observation']['historical_read_only'])

    def test_task_scope_denies_other_task_even_if_observation_id_known(self):
        self.desktop.task_id='task2';self.desktop.remember(observed(50));self.desktop.task_id='task1'
        with self.assertRaises(ValueError):self.history.invoke({'operation':'read','observation_id':'obs50'})
        self.assertEqual(len(self.history.invoke({'operation':'list'})['observations']),6)

    def test_model_paths_and_extra_fields_are_denied(self):
        for args in ({'operation':'read','observation_id':'../screenshots'},
                     {'operation':'read','observation_id':'obs0','task_id':'task1'},
                     {'operation':'list','before_rowid':True}, {'operation':'list','before_rowid':2**63}, {'operation':'list','path':'device'},
                     {'operation':'read','observation_id':'obs0','before_rowid':1}, {'operation':'other'}):
            with self.assertRaises(ValueError):self.history.invoke(args)

    def test_cursor_pages_are_bounded_and_do_not_skip_observations(self):
        for i in range(6,45):self.desktop.remember(observed(i))
        collected=[];args={'operation':'list'}
        while True:
            result=self.history.invoke(args);self.assertLessEqual(len(result['observations']),20)
            collected += [o['observation_id'] for o in result['observations']]
            if result['next_before_rowid'] is None:break
            args={'operation':'list','before_rowid':result['next_before_rowid']}
        self.assertEqual(collected,[f'obs{i}' for i in range(44,-1,-1)])

    def test_pin_limit_unpin_and_restart_persistence(self):
        for i in range(4):self.history.invoke({'operation':'pin','observation_id':f'obs{i}'})
        with self.assertRaises(ValueError):self.history.invoke({'operation':'pin','observation_id':'obs4'})
        self.history.invoke({'operation':'pin','observation_id':'obs0'})
        self.history.close();self.history=history_module.ObservationHistory(self.desktop,self.runtime)
        self.assertEqual(len(self.history.reference_pins()),4)
        self.history.invoke({'operation':'unpin','observation_id':'obs1'})
        self.history.invoke({'operation':'pin','observation_id':'obs4'})
        self.assertEqual(len(self.history.reference_pins()),4)

    def test_pins_are_task_scoped(self):
        self.history.invoke({'operation':'pin','observation_id':'obs0'})
        self.desktop.task_id='task2';self.assertEqual(self.history.reference_pins(),[])
        self.desktop.task_id='task1';self.assertEqual(len(self.history.reference_pins()),1)

    def test_selected_device_owner_isolated_even_with_same_observation_id(self):
        self.history.invoke({'operation':'pin','observation_id':'obs0'})
        other=desktop_module.Desktop('http://127.0.0.1:8','fixture',self.runtime/'other-device')
        other.task_id='task1';other.remember(observed(0,device_id='other-device'))
        self.history.desktop=other
        try:
            self.assertEqual(self.history.reference_pins(),[])
            result=self.history.invoke({'operation':'read','observation_id':'obs0'})
            self.assertEqual(result['observation']['device_id'],'other-device')
        finally:other.db.close();self.history.desktop=self.desktop

    def test_changed_bytes_or_metadata_cannot_be_pinned_or_read(self):
        obs=observed(0);path=self.desktop.runtime/'screenshots'/(obs['sha256']+'.png')
        path.write_bytes(b'corrupt')
        for operation in ('read','pin'):
            with self.assertRaises(ValueError):self.history.invoke({'operation':operation,'observation_id':'obs0'})
        self.assertEqual(self.history.reference_pins(),[])

    def test_symlink_image_is_denied(self):
        obs=observed(0);path=self.desktop.runtime/'screenshots'/(obs['sha256']+'.png')
        copy_path=self.runtime/'image.png';copy_path.write_bytes(path.read_bytes());path.unlink();path.symlink_to(copy_path)
        with self.assertRaises(ValueError):self.history.invoke({'operation':'read','observation_id':'obs0'})

    def test_dimension_mismatch_is_denied(self):
        row=json.loads(self.desktop.db.execute('SELECT metadata FROM observations WHERE id="obs0"').fetchone()[0])
        row['width']=100;self.desktop.db.execute('UPDATE observations SET metadata=? WHERE id="obs0"',(json.dumps(row),));self.desktop.db.commit()
        with self.assertRaises(ValueError):self.history.invoke({'operation':'read','observation_id':'obs0'})


class ArchiveTests(unittest.TestCase):
    def test_full_source_roundtrip_and_actual_presented_identity(self):
        body=context();original=copy.deepcopy(body)
        with tempfile.TemporaryDirectory() as directory:
            trace=Path(directory)/'trace';trace.mkdir()
            view,report=archive.present_request(body,[],trace)
            meta=json.loads((trace/'request.full.archive.json').read_text())
            restored=codec.decode(meta['tree'],trace.parent/'full-history-blobs')
            self.assertEqual(restored,body);self.assertEqual(body,original)
            data=json.dumps(restored,ensure_ascii=False).encode()
            self.assertEqual(meta['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(report['presented_sha256'],hashlib.sha256(json.dumps(view,ensure_ascii=False).encode()).hexdigest())
            self.assertEqual((trace/'request.full.archive.json').stat().st_mode & 0o777,0o600)
            self.assertFalse(report['business_success_improvement_measured'])

    def test_missing_archive_cannot_dispatch_selected_input(self):
        with self.assertRaises(ValueError):archive.present_request(context(),[],None)

    def test_existing_full_source_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            trace=Path(directory)/'trace';trace.mkdir()
            archive.present_request(context(),[],trace)
            with self.assertRaises(FileExistsError):archive.present_request(context(),[],trace)


class ControllerHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.desktop=desktop_module.Desktop('http://127.0.0.1:9','fixture',self.root/'device')
        self.desktop.task_id='task1';self.desktop.remember(observed(0));self.desktop.remember(observed(1))
        self.controller=app.Controller(self.root/'app',self.desktop,'http://127.0.0.1:9/v1',0,
            workflow=True,image_history='recent-observations',tool_call_limit=4)
        self.controller.state.update(status='running',id='task1',execution_mode='verified',image_history='recent-observations',device_platform='windows')
        self.controller.note_gui_observation=AsyncMock(side_effect=AssertionError('Historical image is not live progress'))

    async def asyncTearDown(self):
        self.controller.state['status']='paused';await self.controller.close();self.tmp.cleanup()

    async def test_historical_read_is_counted_and_cannot_reset_progress_or_latest(self):
        latest=copy.deepcopy(self.desktop.latest)
        result=await self.controller.tool('history',{'operation':'read','observation_id':'obs0'})
        self.assertEqual(self.desktop.latest,latest);self.assertEqual(self.controller.state['tool_calls_used'],1)
        self.controller.note_gui_observation.assert_not_called()
        payload=json.loads(result['content'][0]['text']);self.assertTrue(payload['observation']['historical_read_only'])
        self.assertFalse(payload['history']['input_target_eligible'])

    async def test_bad_history_request_still_consumes_lifetime_budget(self):
        with self.assertRaises(desktop_module.DesktopError):await self.controller.tool('history',{'operation':'read','observation_id':'missing'})
        self.assertEqual(self.controller.state['tool_calls_used'],1)

    async def test_archived_observation_cannot_be_used_for_input(self):
        await self.controller.tool('history',{'operation':'read','observation_id':'obs0'})
        self.desktop.paused=False
        self.desktop.request=AsyncMock(side_effect=AssertionError('No input or recapture allowed'))
        with self.assertRaises(desktop_module.DesktopValidationError) as error:
            await self.desktop.act({'desktop_id':'desktop1','observation_id':'obs0','action_id':'history-stale-01',
                'action':{'kind':'click','x':500,'y':500,'button':'left','clicks':1},'coordinate_space':'normalized_1000'})
        self.assertEqual(error.exception.problem['code'],'stale_observation')
        self.desktop.request.assert_not_called()
        self.assertEqual(self.desktop.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0],0)

    async def test_paused_history_is_denied_without_resetting_budget(self):
        self.controller.state['status']='paused'
        with self.assertRaises(desktop_module.DesktopError):await self.controller.tool('history',{'operation':'list'})
        self.assertEqual(self.controller.state['tool_calls_used'],0)

    async def test_checkpoint_mode_change_is_denied(self):
        self.controller.state['image_history']='full'
        with self.assertRaises(desktop_module.DesktopError):self.controller.require_saved_inference_mode()

    async def test_default_mode_keeps_old_tool_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=app.Controller(directory,self.desktop,'http://127.0.0.1:9/v1',0,workflow=True)
            try:
                self.assertIsNone(controller.history)
                with self.assertRaises(desktop_module.DesktopError):await controller.tool('history',{'operation':'list'})
            finally:controller.workflow.close()


class AdmissionTests(unittest.TestCase):
    def test_recent_policy_requires_workflow(self):
        with self.assertRaises(ValueError):app.Controller('/tmp/never-created-visual-candidate',None,'unused',0,image_history='recent-observations')

    def test_schema_accepts_only_documented_operation_fields(self):
        schema=HISTORY_TOOLS[0]['inputSchema'];jsonschema.Draft202012Validator.check_schema(schema)
        for args in ({'operation':'list'},{'operation':'list','before_rowid':20},
                     {'operation':'read','observation_id':'obs0'},{'operation':'pin','observation_id':'obs0'},
                     {'operation':'unpin','observation_id':'obs0'}):jsonschema.validate(args,schema)
        for args in ({'operation':'read'},{'operation':'list','observation_id':'obs0'},
                     {'operation':'pin','observation_id':'obs0','before_rowid':20},
                     {'operation':'list','before_rowid':0}):
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(args,schema)

    def test_history_tool_enabled_only_by_explicit_config(self):
        import tomllib
        with tempfile.TemporaryDirectory() as directory:
            codex.write_config(directory,'http://127.0.0.1:9/v1',sys.executable,['fixture'],workflow=True,history=True)
            config=tomllib.loads((Path(directory)/'config.toml').read_text())
            self.assertEqual(config['mcp_servers']['computer']['enabled_tools'],['observe','act','workflow','history'])

    def test_build_preserves_delivery_and_execution_sources(self):
        for name in ('desktop.py','execution.py','device_gateway.py','device_protocol.py','coordinates.py','action_validation.py'):
            self.assertEqual(BUILD['source_files_sha256'][name],BUILD['candidate_files_sha256'][name])
        self.assertFalse(BUILD['real_model_retest']);self.assertEqual(BUILD['default_policy'],'full')


if __name__=='__main__':unittest.main()
