import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import browser_runtime_metadata as metadata
from browser_runtime_candidate import build_candidate, patch_file
from qw3_computer.linux_driver import foreground_tree_script
import test_desktop as fixtures


SDK = Path(__file__).resolve().parents[1] / '.runtime/candidate51-retest-20261004/framework/qw3_computer'
# Import only this isolated source; the running package and its bindings stay intact.
sys.modules['qw3_computer.browser_runtime_metadata'] = metadata
candidate = types.ModuleType('qw3_computer._browser_runtime_candidate_test')
candidate.__package__ = 'qw3_computer'
exec(compile(patch_file('desktop.py', (SDK/'desktop.py').read_bytes()), 'browser-runtime-test', 'exec'), candidate.__dict__)


def native_tab(resource='19.2 MB', title='TeamChat'):
    provenance = dict(provider='native-atspi-chrome-selected-tab', browser_version='Google Chrome 130.0.6723.58',
        bus_name=':1.12', object_path='/org/a11y/atspi/accessible/85', window_path='/org/a11y/atspi/accessible/1',
        owner_pid=1081, owner_uid=1000, owner_start_ticks=1769, foreground_handle='0x12',
        frame_title=title+' - Google Chrome', page_title=title, selected=True, visible=True, showing=True,
        enabled=True, native_class='Tab', parent_roles=['frame','panel','page tab list','panel'], resource=resource,
        focus_objects=[dict(object_path='/org/a11y/atspi/accessible/99', role='document web')])
    return dict(name=title+' - Memory usage - '+resource, kind='page-tab', value='\ufffc',
        x=1,y=1,width=2,height=1,center_x=2,center_y=1,browser_runtime_metadata=provenance)


def observation(resource='19.2 MB', title='TeamChat'):
    return dict(desktop_id='desktop',device_id='linux',session_id='session',display_id='primary',width=8,height=8,rotation=0,
        capabilities=dict(platform='linux'), foreground=dict(handle='0x12',process='Google-chrome',title=title+' - Google Chrome'),
        accessibility=dict(status='ok',provider=metadata.PROVIDER,truncated=True,
            controls=[native_tab(resource,title),dict(name='Task field',kind='entry',value='Original')]))


class MetadataTests(unittest.TestCase):
    def test_source_resource_only_preserves_raw_labels_and_task_field(self):
        old,new=observation(),observation('45.0 MB');before=copy.deepcopy((old,new))
        proof=metadata.runtime_only_changes(old,new)
        self.assertEqual(proof['changes'][0]['old_raw_name'],'TeamChat - Memory usage - 19.2 MB')
        self.assertTrue(proof['positive_focus_object_unchanged'])
        self.assertEqual((old,new),before)

    def test_user_title_containing_resource_words_is_not_stripped(self):
        title='Policy - Memory usage - 12 MB'
        proof=metadata.runtime_only_changes(observation(title=title),observation('45.0 MB',title))
        self.assertEqual(proof['changes'][0]['source_identity']['page_title'],title)

    def test_title_field_value_layout_and_focus_changes_are_not_runtime_only(self):
        mutations=[lambda x:x['foreground'].update(title='Different - Google Chrome'),
                   lambda x:x['accessibility']['controls'][1].update(value='Changed'),
                   lambda x:x['accessibility']['controls'][0].update(x=2),
                   lambda x:x['accessibility']['controls'][0]['browser_runtime_metadata'].update(focus_objects=[]),
                   lambda x:x['accessibility']['controls'][0].update(value='Actual task text'),
                   lambda x:x['accessibility']['controls'][1].update(native_focused='true')]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                old,new=observation(),observation('45.0 MB');mutate(new)
                self.assertIsNone(metadata.runtime_only_changes(old,new))

    def test_native_generation_bus_object_window_and_selected_state_are_required(self):
        changes=dict(owner_pid=1082,owner_uid=1001,owner_start_ticks=1770,bus_name=':1.13',
                     object_path='/org/a11y/atspi/accessible/86',window_path='/org/a11y/atspi/accessible/2',
                     selected=False,showing=False,enabled=False,native_class='AriaTab',
                     foreground_handle='0x99',parent_roles=['frame','document web','page tab list'])
        for key,value in changes.items():
            with self.subTest(key=key):
                old,new=observation(),observation('45.0 MB')
                new['accessibility']['controls'][0]['browser_runtime_metadata'][key]=value
                self.assertIsNone(metadata.runtime_only_changes(old,new))

    def test_missing_old_provenance_duplicate_tab_and_provider_failure_refuse(self):
        for failure in ('missing','duplicate','status','provider','truncated','platform','session'):
            with self.subTest(failure=failure):
                old,new=observation(),observation('45.0 MB')
                if failure=='missing':old['accessibility']['controls'][0].pop('browser_runtime_metadata')
                elif failure=='duplicate':
                    for x in (old,new):x['accessibility']['controls'].append(copy.deepcopy(x['accessibility']['controls'][0]))
                elif failure=='status':new['accessibility']['status']='unavailable'
                elif failure=='provider':new['accessibility']['provider']='html'
                elif failure=='truncated':new['accessibility']['truncated']=False
                elif failure=='platform':new['capabilities']['platform']='windows'
                else:new['session_id']='new-session'
                self.assertIsNone(metadata.runtime_only_changes(old,new))

    def test_invalid_units_ambiguous_name_and_non_native_tab_refuse(self):
        for value in ('19.2 %','-4 MB','1e6 MB','019 MB','19.2 MB - Approved','',None):
            with self.subTest(value=value):
                old,new=observation(),observation()
                new['accessibility']['controls'][0]['browser_runtime_metadata']['resource']=value
                self.assertIsNone(metadata.runtime_only_changes(old,new))
        old,new=observation(),observation('45.0 MB')
        for x in (old,new):x['accessibility']['controls'][0]['kind']='label'
        self.assertIsNone(metadata.runtime_only_changes(old,new))

    def test_no_change_does_not_mint_an_extra_exception(self):
        self.assertIsNone(metadata.runtime_only_changes(observation(),observation()))

    def test_native_extension_keeps_official_reader_and_handles_title_as_literal(self):
        front=observation()['foreground']|dict(title="Own title'; raise RuntimeError('untrusted') # - Google Chrome")
        source=metadata.native_browser_tree_script(foreground_tree_script(front['title']),front)
        compile(source,'qualified-native-extension','exec')
        self.assertIn('browser_original=a.create_atspi_node',source)
        self.assertIn("role.startswith('document')",source)
        self.assertIn(repr(front),source)
        with self.assertRaises(ValueError):metadata.native_browser_tree_script('unknown source',front)


class DesktopBoundaryTests(unittest.IsolatedAsyncioTestCase):
    observation=fixtures.DesktopTests.observation
    action=fixtures.DesktopTests.action
    change_pixel=fixtures.DesktopTests.change_pixel
    enable_expired_step=fixtures.DesktopTests.enable_expired_step

    async def asyncSetUp(self):
        with patch('test_desktop.Desktop',candidate.Desktop):
            await fixtures.DesktopTests.asyncSetUp(self)

    async def asyncTearDown(self):
        if hasattr(self,'execution'):self.execution.close()
        await fixtures.DesktopTests.asyncTearDown(self)

    async def prepared_action(self,kind='click',focus=True):
        model=observation()
        self.observation_extra={k:v for k,v in model.items() if k not in {'accessibility','width','height','desktop_id'}}
        self.accessibility_extra=dict(provider=metadata.PROVIDER,truncated=True)
        self.controls=copy.deepcopy(model['accessibility']['controls'])
        self.controls[1].update(x=4,y=4,width=4,height=4,center_x=6,center_y=6)
        self.controls.append(dict(name=model['foreground']['title'],kind='frame',x=0,y=1,width=8,height=7,center_x=4,center_y=4))
        if not focus:self.controls[0]['browser_runtime_metadata']['focus_objects']=[]
        await self.desktop.observe();await self.enable_expired_step()
        step=dict(goal='Perform intended input',conditions=[dict(kind='control_text',text='Saved')])
        action=dict(kind='click',x=750,y=1000,button='left',clicks=1)
        if kind=='key':action=dict(kind='key',key='ENTER')
        elif kind=='type':action=dict(kind='type',text='Requested text')
        elif kind=='click_control':action=dict(kind=kind,control_id=self.desktop.latest['accessibility']['controls'][1]['control_id'])
        elif kind=='click_native_tab':action=dict(kind='click_control',control_id=self.desktop.latest['accessibility']['controls'][0]['control_id'])
        elif kind=='drag':action=dict(kind='drag',from_x=750,from_y=1000,to_x=500,to_y=1000,duration_ms=300,modifier='none')
        request=self.action(step=step,action=action,coordinate_space='normalized_1000' if kind in {'click','drag'} else 'pixels')
        self.controls[0]['name']='TeamChat - Memory usage - 45.0 MB'
        self.controls[0]['browser_runtime_metadata']['resource']='45.0 MB'
        self.change_pixel((7,0))
        return request

    async def refused(self,request):
        with self.assertRaises(candidate.DesktopRejected):await self.desktop.act(request)
        self.assertEqual(self.effects,0);self.assertEqual(self.desktop.unresolved(),[])

    async def test_visual_runtime_delta_dispatches_once_without_marking_step_complete(self):
        request=await self.prepared_action();result=await self.desktop.act(request)
        self.assertTrue(result['observation_revalidation']['browser_runtime_metadata_context_unchanged'])
        self.assertFalse(result['step_check']['verified'])
        self.assertTrue((await self.desktop.act(request))['duplicate']);self.assertEqual(self.effects,1)
        self.assertEqual(self.desktop.latest['accessibility']['controls'][0]['name'],'TeamChat - Memory usage - 45.0 MB')

    async def test_full_window_change_away_from_target_refuses(self):
        request=await self.prepared_action();self.change_pixel((0,7));await self.refused(request)

    async def test_task_value_change_refuses_unchanged_pixels(self):
        request=await self.prepared_action();self.controls[1]['value']='Changed requirement';await self.refused(request)

    async def test_keyboard_requires_positive_unchanged_native_focus(self):
        request=await self.prepared_action('key',focus=False);await self.refused(request)

    async def test_keyboard_with_same_positive_focus_dispatches_once(self):
        request=await self.prepared_action('type');result=await self.desktop.act(request)
        self.assertTrue(result['observation_revalidation']['browser_runtime_metadata_context_unchanged'])
        self.assertEqual(self.effects,1)

    async def test_focus_change_refuses_even_when_window_pixels_are_identical(self):
        request=await self.prepared_action('key')
        self.controls[0]['browser_runtime_metadata']['focus_objects'][0]['object_path']='/org/a11y/atspi/accessible/100'
        await self.refused(request)

    async def test_drag_endpoint_outside_window_refuses(self):
        request=await self.prepared_action('drag');request['action']['to_y']=0;await self.refused(request)

    async def test_control_target_must_still_have_same_name_value_and_bounds(self):
        request=await self.prepared_action('click_control');self.controls[1]['name']='Different target';await self.refused(request)

    async def test_same_native_selected_tab_target_accepts_only_resource_name_change(self):
        request=await self.prepared_action('click_native_tab');result=await self.desktop.act(request)
        self.assertTrue(result['observation_revalidation']['target_control_valid'])
        self.assertTrue(result['observation_revalidation']['browser_runtime_metadata_context_unchanged'])
        self.assertEqual(self.effects,1)
        self.assertEqual((self.last_dispatched['action']['x'],self.last_dispatched['action']['y']),(2,1))
        self.assertTrue((await self.desktop.act(request))['duplicate']);self.assertEqual(self.effects,1)

    async def test_changed_session_refuses_without_receipt_reservation(self):
        request=await self.prepared_action();self.observation_extra['session_id']='new-session';await self.refused(request)


class BuildTests(unittest.TestCase):
    def test_actual_framework_inventory_preserves_runner_and_grader_sources(self):
        acceptance=SDK.parents[1]/'acceptance.json'
        declared=json.loads(acceptance.read_text())['framework_files']
        with tempfile.TemporaryDirectory() as temporary:
            result=build_candidate(SDK,acceptance,Path(temporary)/'candidate',auxiliary_framework=Path(__file__).resolve().parents[1])
            self.assertEqual(len(result['source_files_sha256'])+len(result['unchanged_auxiliary_framework_files_sha256']),len(declared))
            for name,digest in result['unchanged_auxiliary_framework_files_sha256'].items():
                self.assertEqual(digest,declared[name])
                self.assertEqual(hashlib.sha256((Path(temporary)/'candidate/source-archive/auxiliary'/name).read_bytes()).hexdigest(),digest)
            self.assertFalse((Path(temporary)/'candidate/framework/osworld_linux').exists())

    def test_build_keeps_frozen_source_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);source=p/'source';source.mkdir()
            for name in ('desktop.py','linux_driver.py','__init__.py'):
                (source/name).write_bytes((SDK/name).read_bytes())
            files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()}
            acceptance=p/'acceptance.json';acceptance.write_text(json.dumps(dict(framework_files=files)))
            result=build_candidate(source,acceptance,p/'output')
            self.assertEqual(set(result['changed_package_files']),{'desktop.py','linux_driver.py','browser_runtime_metadata.py'})
            self.assertFalse(result['real_model_retest']);self.assertFalse(result['native_qualified'])
            self.assertEqual({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()},files)
            with self.assertRaises(FileExistsError):build_candidate(source,acceptance,p/'output')

    def test_changed_source_or_inventory_refuses_before_build(self):
        with self.assertRaises(ValueError):patch_file('desktop.py',(SDK/'desktop.py').read_bytes()+b'\n')


if __name__=='__main__':unittest.main()
