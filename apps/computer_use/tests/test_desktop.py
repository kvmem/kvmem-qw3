import asyncio
import base64
import hashlib
import io
import json
import tempfile
import unittest
import time
from pathlib import Path

from aiohttp import web
from PIL import Image

from qw3_computer.desktop import Desktop, DesktopError, DesktopRejected, DesktopValidationError, model_observation
from test_bridge import serve


class ModelObservationTests(unittest.TestCase):
    def test_compact_view_preserves_targets_values_and_negative_selection_without_mutation(self):
        observation = {"desktop_id": "test", "observation_id": "fresh", "image": "full-screenshot",
            "accessibility": {"status": "ok", "truncated": False, "controls": [{
                "control_id": "toggle", "name": "Enabled", "value": "Off", "kind": "CheckBox",
                "x": 12, "y": 24, "width": 36, "height": 48, "center_x": 30, "center_y": 48,
                "states": {"checkable": True, "checked": False, "clickable": True, "scrollable": False}}]}}
        original = json.loads(json.dumps(observation))
        result = model_observation(observation)
        control = result["accessibility"]["controls"][0]
        self.assertEqual(control["control_id"], "toggle")
        self.assertEqual(control["value"], "Off")
        self.assertEqual(control["bounds"], [12, 24, 36, 48])
        self.assertEqual((control["center_x"], control["center_y"]), (30, 48))
        self.assertFalse(control["states"]["checked"])
        self.assertFalse(result["accessibility"]["truncated"])
        self.assertEqual(observation, original)
        self.assertNotIn("image", result)


class DesktopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sequence = self.effects = 0
        self.paused = True
        self.mode = "normal"
        self.controls = None
        self.accessibility_status = "ok"
        self.accessibility_extra = {}
        self.observation_extra = {}
        self.receipts = {}
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        self.png = buffer.getvalue()

        async def handle(request):
            if request.path == "/observe":
                return web.json_response(self.observation())
            if request.path == "/control":
                self.paused = (await request.json())["command"] != "resume"
                return web.json_response({"paused": self.paused})
            if request.path.startswith("/actions/"):
                return web.json_response(self.receipts.get(request.path.rsplit("/", 1)[1], {"status": "unknown"}))
            body = await request.json()
            self.last_dispatched = body
            if self.mode == "reject":
                return web.json_response({"delivery": "rejected", "error": "Foreground changed"}, status=409)
            self.effects += 1
            result = {"action_id": body["action_id"], "status": "completed", "observation": self.observation()}
            if self.mode != "unknown":
                self.receipts[body["action_id"]] = {"status": "completed", "result": result}
            self.entered.set()
            if self.mode == "hang":
                await self.release.wait()
            if self.mode in ("lost_response", "unknown"):
                request.transport.close()
            if self.mode == "no_progress":
                result["no_progress"] = True
            return web.json_response(result)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        self.runner, self.url = await serve(app)
        self.desktop = Desktop(self.url, "fixture-token", self.temp.name)
        await self.desktop.start()
        await self.desktop.control("resume")
        await self.desktop.observe()

    def observation(self):
        self.sequence += 1
        result = {"desktop_id": "fixture-desktop", "observation_id": str(self.sequence),
                "image": base64.b64encode(self.png).decode(), "sha256": hashlib.sha256(self.png).hexdigest(),
                "width": 8, "height": 8, "paused": self.paused}
        if self.controls is not None:
            result['accessibility'] = {'status': self.accessibility_status, 'controls': self.controls} | self.accessibility_extra
        return result | self.observation_extra

    async def observed_controls(self):
        self.controls = [
            {'name': 'Save', 'kind': 'Button', 'x': 0, 'y': 0, 'width': 4, 'height': 4, 'center_x': 2, 'center_y': 2},
            {'name': 'Save', 'kind': 'Button', 'x': 4, 'y': 4, 'width': 4, 'height': 4, 'center_x': 6, 'center_y': 6},
        ]
        return (await self.desktop.observe())['accessibility']['controls']

    def action(self, id="action_001", **changes):
        return {"desktop_id": "fixture-desktop", "observation_id": self.desktop.latest["observation_id"],
                "action_id": id, "action": {"kind": "type", "text": "hello"}} | changes

    async def asyncTearDown(self):
        self.release.set()
        await self.desktop.close()
        await self.runner.cleanup()
        self.temp.cleanup()

    async def test_duplicate_returns_receipt_without_second_effect(self):
        with self.assertRaises(DesktopValidationError):
            await self.desktop.act(self.action(''))
        self.assertEqual(self.effects,0)
        request = self.action('a1')
        first = await self.desktop.act(request)
        self.assertEqual(first['action_id'],'a1')
        self.assertRegex(self.last_dispatched['action_id'],r'^q3_[a-f0-9]{64}$')
        second = await self.desktop.act(request)
        self.assertEqual(first["observation"], second["observation"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(self.effects, 1)
        with self.assertRaisesRegex(DesktopError, "different arguments"):
            await self.desktop.act(request | {"action": {"kind": "key", "key": "ENTER"}})

    async def test_truncated_control_id_explains_copy_error_and_never_guesses_missing_characters(self):
        controls=await self.observed_controls()
        target=controls[0]['control_id']
        with self.assertRaises(DesktopValidationError) as caught:
            await self.desktop.act(self.action(action={'kind':'click_control','control_id':target[:-2]}))
        self.assertEqual(caught.exception.problem['code'],'invalid_control_id')
        self.assertIn('20 characters total',caught.exception.problem['hint'])
        self.assertEqual(self.effects,0)
        self.assertEqual(self.desktop.unresolved(),[])
        await self.desktop.act(self.action('correct_copy',action={'kind':'click_control','control_id':target}))
        self.assertEqual(self.effects,1)

    async def enable_expired_step(self):
        from qw3_computer.execution import Execution
        self.execution=Execution(self.desktop,Path(self.temp.name))
        self.desktop.execution=self.execution
        self.desktop.latest['captured_at']=int(time.time()*1000)-65000
        return {'goal':'Perform the requested action','conditions':[{'kind':'screen_changed'}]}

    async def test_expired_identical_view_refreshes_before_exactly_one_effect(self):
        step=await self.enable_expired_step()
        action=self.action(step=step)
        source=action['observation_id']
        result=await self.desktop.act(action)
        self.assertNotEqual(self.last_dispatched['observation_id'],source)
        self.assertTrue(result['observation_revalidation']['same_view'])
        self.assertTrue(result['observation_revalidation']['input_sent'])
        await self.desktop.act(action)
        self.assertEqual(self.effects,1)
        self.execution.close()

    async def test_expired_changed_view_returns_current_evidence_without_input(self):
        step=await self.enable_expired_step()
        action=self.action(step=step)
        pixels=io.BytesIO();Image.new('RGB',(8,8),'red').save(pixels,format='PNG')
        self.png=pixels.getvalue()
        with self.assertRaises(DesktopRejected) as caught:
            await self.desktop.act(action)
        self.assertEqual(caught.exception.observation['observation_id'],self.desktop.latest['observation_id'])
        self.assertTrue(caught.exception.evidence_refreshed)
        self.assertEqual(self.effects,0)
        self.assertEqual(self.desktop.unresolved(),[])
        self.execution.close()

    async def test_capture_timing_metadata_does_not_invalidate_identical_ui_evidence(self):
        self.accessibility_extra={'read_seconds':.5,'scope':'foreground_window'}
        await self.observed_controls()
        step=await self.enable_expired_step()
        action=self.action(step=step)
        self.accessibility_extra['read_seconds']=1.25
        result=await self.desktop.act(action)
        self.assertTrue(result['observation_revalidation']['same_view'])
        self.assertEqual(self.effects,1)
        self.execution.close()

    async def aged_context_control(self, region=True, target_index=0):
        self.observation_extra={'foreground':{'handle':'0x12','process':'editor','title':'Document'},
                                'capabilities':{'platform':'linux'}}
        if region:
            self.observation_extra['progress_region']={'bounds':[0,0,8,6],
                'coordinate_space':'pixels','scope':'linux_desktop_content_without_clock_and_dock'}
        controls=await self.observed_controls()
        step=await self.enable_expired_step()
        return self.action(step=step,action={'kind':'click_control','control_id':controls[target_index]['control_id']})

    def change_pixel(self, point):
        image=Image.open(io.BytesIO(self.png)).convert('RGB')
        image.putpixel(point,(255,0,0));buffer=io.BytesIO();image.save(buffer,format='PNG')
        self.png=buffer.getvalue()

    async def test_aged_control_ignores_clock_outside_proven_unchanged_context(self):
        action=await self.aged_context_control()
        self.change_pixel((7,7))
        result=await self.desktop.act(action)
        self.assertFalse(result['observation_revalidation']['same_view'])
        self.assertTrue(result['observation_revalidation']['target_context_unchanged'])
        self.assertTrue(result['observation_revalidation']['input_sent'])
        await self.desktop.act(action)
        self.assertEqual(self.effects,1)
        self.execution.close()

    async def test_changed_dialog_body_rejects_even_if_button_and_progress_hash_match(self):
        action=await self.aged_context_control()
        self.desktop.latest['progress_sha256']='unchanged-advertised-hash'
        self.observation_extra['progress_sha256']='unchanged-advertised-hash'
        self.change_pixel((5,3))  # The Save button is still blue and unchanged.
        with self.assertRaises(DesktopRejected):await self.desktop.act(action)
        record=json.loads(self.desktop.db.execute('SELECT evidence FROM revalidations').fetchone()[0])
        self.assertTrue(record['target_control_valid'])
        self.assertFalse(record['target_context_unchanged'])
        self.assertEqual(self.effects,0)
        self.assertEqual(self.desktop.unresolved(),[])
        self.execution.close()

    async def test_aged_control_without_declared_context_keeps_strict_view_requirement(self):
        action=await self.aged_context_control(region=False)
        self.change_pixel((7,7))
        with self.assertRaises(DesktopRejected):await self.desktop.act(action)
        self.assertEqual(self.effects,0)
        self.execution.close()

    async def test_aged_control_partly_outside_context_is_not_revalidated(self):
        action=await self.aged_context_control(target_index=1)
        self.change_pixel((7,0))  # Outside this button; inside the context.
        with self.assertRaises(DesktopRejected):await self.desktop.act(action)
        self.assertEqual(self.effects,0)
        self.execution.close()

    async def test_invalid_or_changed_context_is_not_trusted(self):
        action=await self.aged_context_control()
        self.change_pixel((7,7))
        self.observation_extra['progress_region']={'bounds':[0,0,8,99],
            'coordinate_space':'pixels','scope':'linux_desktop_content_without_clock_and_dock'}
        with self.assertRaises(DesktopRejected):await self.desktop.act(action)
        self.assertEqual(self.effects,0)
        self.execution.close()

    async def test_context_revalidation_never_overrides_changed_foreground(self):
        action=await self.aged_context_control()
        self.observation_extra['foreground']={'handle':'0x13','process':'editor','title':'Document'}
        with self.assertRaises(DesktopRejected):await self.desktop.act(action)
        self.assertEqual(self.effects,0)
        self.execution.close()

    async def test_control_target_is_rechecked_before_one_input_when_source_is_young(self):
        from qw3_computer.execution import Execution
        controls=await self.observed_controls()
        self.execution=Execution(self.desktop,Path(self.temp.name))
        self.desktop.execution=self.execution
        action=self.action(step={'goal':'Click Save','conditions':[{'kind':'screen_changed'}]},
                           action={'kind':'click_control','control_id':controls[0]['control_id']})
        source=action['observation_id']
        result=await self.desktop.act(action)
        self.assertNotEqual(self.last_dispatched['observation_id'],source)
        self.assertTrue(result['observation_revalidation']['target_control_valid'])
        await self.desktop.act(action)
        self.assertEqual(self.effects,1)
        self.execution.close()

    async def test_keyboard_covering_unchanged_control_bounds_rejects_without_input(self):
        from qw3_computer.execution import Execution
        controls=await self.observed_controls()
        self.execution=Execution(self.desktop,Path(self.temp.name))
        self.desktop.execution=self.execution
        action=self.action(step={'goal':'Click the intended field','conditions':[{'kind':'screen_changed'}]},
                           action={'kind':'click_control','control_id':controls[0]['control_id']})
        pixels=io.BytesIO()
        Image.new('RGB',(8,8),'red').save(pixels,format='PNG')
        self.png=pixels.getvalue()  # The tree still exposes the covered field.
        with self.assertRaises(DesktopRejected) as caught:
            await self.desktop.act(action)
        self.assertTrue(caught.exception.evidence_refreshed)
        self.assertEqual(self.effects,0)
        self.assertEqual(self.desktop.unresolved(),[])
        self.execution.close()

    async def test_expired_open_app_revalidates_explicit_app_intent_without_repeating(self):
        step=await self.enable_expired_step()
        pixels=io.BytesIO();Image.new('RGB',(8,8),'red').save(pixels,format='PNG')
        self.png=pixels.getvalue()
        action=self.action(step=step,action={'kind':'open_app','app':'edge'})
        result=await self.desktop.act(action)
        self.assertFalse(result['observation_revalidation']['same_view'])
        self.assertEqual(result['observation_revalidation']['scope'],'Explicit allowed app intent')
        await self.desktop.act(action)
        self.assertEqual(self.effects,1)
        self.execution.close()

    async def test_fresh_observation_verifies_pending_step_without_resending_input(self):
        from pathlib import Path
        from qw3_computer.execution import Execution
        execution=Execution(self.desktop,Path(self.temp.name))
        self.desktop.execution=execution
        try:
            request=self.action(step={'goal':'Wait for save','conditions':[{'kind':'control_text','text':'Saved'}]})
            self.assertFalse((await self.desktop.act(request))['step_check']['verified'])
            self.controls=[{'name':'Saved','kind':'Label','x':0,'y':0,'width':4,'height':4,'center_x':2,'center_y':2}]
            observation=await self.desktop.observe()
            self.assertTrue(observation['step_check']['verified'])
            self.assertEqual(execution.steps()[0]['status'],'verified')
            self.assertEqual(self.effects,1)
        finally:
            self.desktop.execution=None
            execution.close()

    async def test_verified_visual_pixel_estimate_is_rejected_before_input_then_normalized_dispatches(self):
        from pathlib import Path
        from qw3_computer.execution import Execution
        execution = Execution(self.desktop, Path(self.temp.name))
        self.desktop.execution = execution
        try:
            request = self.action(action={"kind": "click", "x": 4, "y": 4, "button": "left", "clicks": 1},
                                  coordinate_space="pixels", step={"goal": "Select target", "conditions": [{"kind": "screen_changed"}]})
            with self.assertRaises(DesktopValidationError) as rejected:
                await self.desktop.act(request)
            self.assertEqual(rejected.exception.problem["code"], "normalized_coordinates_required")
            self.assertEqual(self.effects, 0)
            normalized = request | {"coordinate_space": "normalized_1000",
                                    "action": request["action"] | {"x": 500, "y": 500}}
            result = await self.desktop.act(normalized)
            self.assertEqual(self.effects, 1)
            self.assertIn("coordinate_mapping", result)
            await self.desktop.act(normalized)
            self.assertEqual(self.effects, 1)
        finally:
            self.desktop.execution = None
            execution.close()

    async def test_normalized_click_converts_once_and_keeps_pixel_actions_unchanged(self):
        request = self.action(coordinate_space="normalized_1000",
            action={"kind": "click", "x": 500, "y": 1000, "button": "left", "clicks": 1})
        first = await self.desktop.act(request)
        self.assertEqual(self.last_dispatched["action"]["x"], 4)
        self.assertEqual(self.last_dispatched["action"]["y"], 7)
        self.assertNotIn("coordinate_space", self.last_dispatched)
        self.assertEqual(first["coordinate_mapping"]["input_action"]["x"], 500)
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects, 1)
        with self.assertRaisesRegex(DesktopError, "different arguments"):
            await self.desktop.act(request | {"coordinate_space": "pixels"})
        await self.desktop.act(self.action("pixel_action_02", coordinate_space="pixels",
            action={"kind": "click", "x": 3, "y": 5, "button": "left", "clicks": 1}))
        self.assertEqual(self.last_dispatched["action"]["x"], 3)
        self.assertEqual(self.last_dispatched["action"]["y"], 5)

    async def test_control_click_disambiguates_names_and_returns_new_observation_ids(self):
        controls = await self.observed_controls()
        self.assertNotEqual(controls[0]['control_id'], controls[1]['control_id'])
        request = self.action(action={'kind': 'click_control', 'control_id': controls[1]['control_id']})
        result = await self.desktop.act(request)
        self.assertEqual(self.last_dispatched['action'], {'kind': 'click', 'x': 6, 'y': 6, 'button': 'left', 'clicks': 1})
        self.assertEqual(result['coordinate_mapping']['source'], 'accessibility_control')
        fresh = result['observation']['accessibility']['controls']
        self.assertNotEqual(fresh[1]['control_id'], controls[1]['control_id'])
        self.assertEqual(fresh, self.desktop.latest['accessibility']['controls'])
        self.assertTrue((await self.desktop.act(request))['duplicate'])
        self.assertEqual(self.effects, 1)

    async def test_inline_field_kind_condition_passes_public_validation_and_preserves_duplicates_as_ambiguous(self):
        from qw3_computer.execution import Execution
        from qw3_computer.tool_schema import tools_for_platform
        self.controls=[{'name':'Name','kind':'entry','value':'Aster','x':0,'y':0,'width':3,'height':3,
                        'center_x':1,'center_y':1}]
        await self.desktop.observe()
        execution=Execution(self.desktop,Path(self.temp.name));self.desktop.execution=execution
        condition={'kind':'control_value','text':'Name','value':'Aster','control_kind':'entry'}
        try:
            first=await self.desktop.act(self.action('field_value_01',action={'kind':'type','text':'Aster'},
                step={'goal':'Enter the field value','conditions':[condition]}))
            self.assertTrue(first['step_check']['verified'])
            self.assertEqual(self.effects,1)
            self.controls.append(self.controls[0] | {'x':4,'center_x':5})
            await self.desktop.observe()
            second=await self.desktop.act(self.action('field_value_02',action={'kind':'type','text':'Aster'},
                step={'goal':'Enter the next field','conditions':[condition]}))
            self.assertFalse(second['step_check']['verified'])
            self.assertEqual(second['step_check']['conditions'][0]['matching_controls'],2)
            for platform in ['windows','linux','android']:
                self.assertNotIn('step',tools_for_platform(platform,True)[1]['inputSchema']['required'])
        finally:
            self.desktop.execution=None;execution.close()

    async def test_old_control_id_with_new_observation_is_rejected_before_dispatch(self):
        controls = await self.observed_controls()
        await self.desktop.observe()
        with self.assertRaises(DesktopValidationError) as caught:
            await self.desktop.act(self.action(action={'kind': 'click_control', 'control_id': controls[0]['control_id']}))
        self.assertEqual(caught.exception.problem['path'], '$.action.control_id')
        self.assertEqual(self.effects, 0)
        self.assertEqual(self.desktop.unresolved(), [])

    async def test_control_click_ignores_redundant_coordinate_space_without_rescaling(self):
        controls = await self.observed_controls()
        result = await self.desktop.act(self.action(coordinate_space='normalized_1000',
            action={'kind': 'click_control', 'control_id': controls[1]['control_id']}))
        self.assertEqual(self.last_dispatched['action']['x'], 6)
        self.assertEqual(self.last_dispatched['action']['y'], 6)
        self.assertEqual(result['coordinate_mapping']['coordinate_space'], 'pixels')
        self.assertEqual(result['coordinate_mapping']['ignored_coordinate_space'], 'normalized_1000')
        self.assertNotIn('coordinate_space', self.last_dispatched)
        self.assertEqual(self.effects, 1)

    async def test_control_click_still_rejects_unavailable_accessibility(self):
        controls = await self.observed_controls()
        self.accessibility_status = 'timeout'
        observation = await self.desktop.observe()
        self.assertTrue(all('control_id' not in c for c in observation['accessibility']['controls']))
        with self.assertRaises(DesktopValidationError):
            await self.desktop.act(self.action(action={'kind': 'click_control', 'control_id': controls[0]['control_id']}))
        self.assertEqual(self.effects, 0)

    async def test_invalid_control_center_does_not_expose_a_dispatchable_id(self):
        self.controls = [{'name': 'Wrong', 'kind': 'Button', 'x': 0, 'y': 0, 'width': 3, 'height': 3,
                         'center_x': 7, 'center_y': 7, 'control_id': 'ctl_1234567890abcdef'}]
        observation = await self.desktop.observe()
        self.assertNotIn('control_id', observation['accessibility']['controls'][0])
        with self.assertRaises(DesktopValidationError):
            await self.desktop.act(self.action(action={'kind': 'click_control', 'control_id': 'ctl_1234567890abcdef'}))
        self.assertEqual(self.effects, 0)

    async def test_control_lost_response_reconciles_after_restart_without_resolving_new_target(self):
        controls = await self.observed_controls()
        request = self.action(action={'kind': 'click_control', 'control_id': controls[0]['control_id']})
        self.mode = 'lost_response'
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        self.controls[0].update(x=4, y=4, center_x=6, center_y=6)
        await self.desktop.close()
        self.desktop = Desktop(self.url, 'fixture-token', self.temp.name)
        await self.desktop.start()
        await self.desktop.control('resume')
        result = await self.desktop.act(request)
        self.assertTrue(result['duplicate'])
        self.assertEqual(result['coordinate_mapping']['control_id'], controls[0]['control_id'])
        self.assertEqual(result['coordinate_mapping']['dispatched_action']['x'], 2)
        self.assertEqual(self.effects, 1)

    async def test_invalid_normalized_actions_never_dispatch(self):
        for value in (True, "500", 500.5, -1, 1001):
            with self.subTest(value=value), self.assertRaises(DesktopError):
                await self.desktop.act(self.action(coordinate_space="normalized_1000",
                    action={"kind": "click", "x": value, "y": 100, "button": "left", "clicks": 1}))
        for space in ("normalized_1000", "automatic"):
            with self.assertRaises(DesktopError):
                await self.desktop.act(self.action(coordinate_space=space))
        self.assertEqual(self.effects, 0)
        self.assertEqual(self.desktop.unresolved(), [])

    async def test_normalized_lost_response_reconciles_after_restart_without_reconversion(self):
        self.mode = "lost_response"
        request = self.action(coordinate_space="normalized_1000", action={"kind": "drag",
            "from_x": 0, "from_y": 500, "to_x": 1000, "to_y": 250, "duration_ms": 300, "modifier": "none"})
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        self.assertEqual(self.last_dispatched["action"] | {"kind": "drag"},
                         {"kind": "drag", "from_x": 0, "from_y": 4, "to_x": 7, "to_y": 2, "duration_ms": 300, "modifier": "none"})
        await self.desktop.close()
        self.desktop = Desktop(self.url, "fixture-token", self.temp.name)
        await self.desktop.start(); await self.desktop.control("resume")
        result = await self.desktop.act(request)
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["coordinate_mapping"]["width"], 8)
        self.assertEqual(result["coordinate_mapping"]["dispatched_action"], self.last_dispatched["action"])
        self.assertEqual(self.effects, 1)

    async def test_invalid_identity_stale_and_paused_never_dispatch(self):
        for change in ({"desktop_id": "wrong"}, {"observation_id": "old"}, {"action_id": "../../bad"}):
            with self.assertRaises(DesktopError):
                await self.desktop.act(self.action(**change))
        await self.desktop.control("pause")
        with self.assertRaises(DesktopError):
            await self.desktop.act(self.action())
        self.assertEqual(self.effects, 0)

    async def test_known_rejection_does_not_require_uncertain_ack(self):
        self.mode = "reject"
        request = self.action()
        with self.assertRaises(DesktopRejected):
            await self.desktop.act(request)
        self.assertEqual(self.desktop.unresolved(), [])
        self.assertFalse(self.desktop.paused)
        self.assertEqual(self.effects, 0)
        with self.assertRaisesRegex(DesktopError, "Previous action is rejected"):
            await self.desktop.act(request)
        self.assertEqual(self.effects, 0)

    async def test_lost_response_reconciles_after_app_restart_without_replay(self):
        self.mode = "lost_response"
        request = self.action()
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        self.assertTrue(self.desktop.paused)
        self.assertEqual(len(self.desktop.unresolved()), 1)
        await self.desktop.close()
        self.desktop = Desktop(self.url, "fixture-token", self.temp.name)
        await self.desktop.start()
        await self.desktop.control("resume")
        self.assertEqual(self.desktop.unresolved(), [])
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects, 1)

    async def test_unknown_delivery_requires_explicit_reconciliation(self):
        self.mode = "unknown"
        request = self.action()
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        with self.assertRaisesRegex(DesktopError, "尚未确认"):
            await self.desktop.control("resume")
        self.assertTrue(self.desktop.paused)
        await self.desktop.control("resume", acknowledge_uncertain=True)
        with self.assertRaisesRegex(DesktopError, "user_reconciled"):
            await self.desktop.act(request)
        self.assertEqual(self.effects, 1)

    async def test_cancel_after_effect_pauses_and_retains_action_id(self):
        self.mode = "hang"
        task = asyncio.create_task(self.desktop.act(self.action()))
        await asyncio.wait_for(self.entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.release.set()
        self.assertTrue(self.desktop.paused)
        self.assertTrue(self.paused)
        await self.desktop.control("resume")
        self.assertEqual(self.effects, 1)
        self.assertEqual(self.desktop.unresolved(), [])

    async def test_no_progress_prevents_next_dispatch(self):
        self.mode = "no_progress"
        await self.desktop.act(self.action())
        self.assertTrue(self.desktop.paused)
        with self.assertRaises(DesktopError):
            await self.desktop.act(self.action("action_002"))
        self.assertEqual(self.effects, 1)

    async def test_same_model_id_in_different_tasks_and_clients_has_distinct_receipts(self):
        self.desktop.task_id = 'first-task'
        first_request = self.action('click-name-field')
        first = await self.desktop.act(first_request)
        self.assertEqual(first['action_id'], 'click-name-field')
        first_wire = self.last_dispatched['action_id']
        self.desktop.task_id = 'second-task'
        second_request = self.action('click-name-field')
        await self.desktop.act(second_request)
        second_wire = self.last_dispatched['action_id']
        self.assertNotEqual(first_wire, second_wire)
        self.assertEqual(self.effects, 2)
        self.assertTrue((await self.desktop.act(second_request))['duplicate'])
        self.assertEqual(self.effects, 2)
        self.desktop.task_id = 'first-task'
        self.assertTrue((await self.desktop.act(first_request))['duplicate'])
        self.assertEqual(self.effects, 2)
        # A fresh application has a new local DB but shares the remote ledger.
        # Transfer the device lease; two applications must not own input together.
        await self.desktop.close()
        with tempfile.TemporaryDirectory() as directory:
            other = Desktop(self.url, 'fixture-token', directory)
            try:
                other.task_id = 'third-task'
                await other.start(); await other.control('resume')
                request = first_request | {'observation_id': other.latest['observation_id']}
                await other.act(request)
                self.assertNotIn(self.last_dispatched['action_id'], (first_wire, second_wire))
                self.assertEqual(self.effects, 3)
            finally:
                await other.close()

    async def test_second_client_cannot_own_same_device_input(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Desktop(self.url, 'fixture-token', directory)
            try:
                with self.assertRaisesRegex(RuntimeError, 'input owner'):
                    await other.start()
                self.assertEqual(self.effects, 0)
            finally:
                await other.close()

    async def test_legacy_completed_action_is_never_adopted_by_another_task(self):
        request = self.action('legacy-action-01')
        result = {'action_id': request['action_id'], 'status': 'completed', 'observation': self.desktop.latest}
        self.desktop.db.execute('INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)',
            (request['action_id'], 'old-task', json.dumps(request, sort_keys=True, ensure_ascii=False), 'completed', json.dumps(result), 1))
        self.desktop.db.commit()
        self.desktop.task_id = 'old-task'
        self.assertTrue((await self.desktop.act(request))['duplicate'])
        self.assertEqual(self.effects, 0)
        self.desktop.task_id = 'new-task'
        self.assertFalse((await self.desktop.act(request)).get('duplicate', False))
        self.assertEqual(self.effects, 1)
        self.assertNotEqual(self.last_dispatched['action_id'], request['action_id'])
        self.assertEqual(self.desktop.db.execute('SELECT task FROM actions WHERE id=?', (request['action_id'],)).fetchone()[0], 'old-task')

    async def test_legacy_uncertain_receipt_keeps_original_remote_identity(self):
        request = self.action('legacy-action-02')
        self.desktop.db.execute('INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)',
            (request['action_id'], 'old-task', json.dumps(request, sort_keys=True, ensure_ascii=False), 'uncertain', None, 1))
        self.desktop.db.commit()
        self.receipts[request['action_id']] = {'status':'completed', 'result':{
            'action_id':request['action_id'], 'status':'completed', 'observation':self.observation()}}
        self.desktop.task_id = 'old-task'
        await self.desktop.control('pause'); await self.desktop.control('resume')
        self.assertFalse(self.desktop.unresolved())
        self.assertTrue((await self.desktop.act(request))['duplicate'])
        self.assertEqual(self.effects, 0)

    async def test_reconciliation_rejects_receipt_for_a_different_action(self):
        request = self.action()
        self.mode = 'lost_response'
        with self.assertRaises(DesktopError): await self.desktop.act(request)
        key = self.desktop.unresolved()[0]['receipt_id']
        self.receipts[key]['result']['action_id'] = 'another-action'
        with self.assertRaisesRegex(DesktopError, 'does not match'):
            await self.desktop.control('resume')
        self.assertTrue(self.desktop.paused)
        self.assertEqual(len(self.desktop.unresolved()), 1)
        self.assertEqual(self.effects, 1)

    async def test_misplaced_action_id_reports_location_without_dispatch_or_reservation(self):
        from qw3_computer.desktop import DesktopValidationError
        request = self.action()
        request['action']['action_id'] = request.pop('action_id')
        with self.assertRaises(DesktopValidationError) as raised:
            await self.desktop.act(request)
        self.assertEqual(raised.exception.problem['path'], '$.action_id')
        self.assertIn('$.action.action_id', raised.exception.problem['hint'])
        self.assertEqual(self.effects, 0)
        self.assertEqual(self.desktop.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0], 0)
        request['action_id'] = request['action'].pop('action_id')
        await self.desktop.act(request)
        self.assertEqual(self.effects, 1)
