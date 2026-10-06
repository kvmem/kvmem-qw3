import asyncio
import json
from pathlib import Path
import unittest

from qw3_computer.desktop import DesktopError
from qw3_computer.execution import Execution
from qw3_computer.target_execution import TargetExecution
import test_desktop as desktop_fixture


class TargetExecutionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = desktop_fixture.DesktopTests.asyncSetUp
    asyncTearDown = desktop_fixture.DesktopTests.asyncTearDown

    def observation(self):
        result = desktop_fixture.DesktopTests.observation(self)
        result['capabilities'] = {'platform': 'linux', 'implemented': True, 'actions': ['open_app'],
                                 'open_apps': ['notepad'], 'app_processes': {'notepad': ['Gedit']},
                                 'available_now': self.mode != 'permission-after-input' or self.effects == 0}
        result['foreground'] = {'process': 'Gedit' if self.effects and self.mode == 'switch' else 'other', 'title': 'Note'}
        if self.effects and self.mode == 'outside':
            result['foreground']['process'] = 'Gedit'
            result['accessibility'] = {'status': 'outside_task_scope', 'controls': []}
        return result

    def request(self, **changes):
        return {'operation': 'ensure_app', 'app': 'notepad', 'transition_id': 'transition-one', 'timeout_seconds': 1} | changes

    async def test_one_launch_waits_for_actual_foreground_and_persists_no_image(self):
        self.mode = 'switch'
        target = TargetExecution(self.desktop)
        result = await target.invoke(self.request())
        self.assertTrue(result['verified']); self.assertTrue(result['input_sent'])
        self.assertEqual(self.effects, 1)
        raw = self.desktop.db.execute('SELECT record FROM target_transitions').fetchone()[0]
        self.assertNotIn('image', raw)
        self.assertIn('Foreground application', result['verification_scope'])

    async def test_launch_delivery_is_not_foreground_success_and_duplicate_never_relaunches(self):
        target = TargetExecution(self.desktop)
        result = await target.invoke(self.request())
        self.assertEqual(result['status'], 'timed_out'); self.assertFalse(result['verified'])
        self.assertGreater(result['polls'], 0)
        duplicate = await target.invoke(self.request())
        self.assertTrue(duplicate['duplicate']); self.assertEqual(self.effects, 1)

    async def test_restart_preserves_intent_and_does_not_resend(self):
        self.mode = 'switch'
        await TargetExecution(self.desktop).invoke(self.request())
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertTrue(result['duplicate']); self.assertTrue(result['verified'])
        self.assertEqual(self.effects, 1)

    async def test_application_already_foreground_needs_no_launch(self):
        self.mode = 'switch'; self.effects = 1
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertTrue(result['verified']); self.assertFalse(result['input_sent'])
        self.assertEqual(self.effects, 1)

    async def test_changed_intent_cannot_reuse_transition_identity(self):
        self.mode = 'switch'; target = TargetExecution(self.desktop)
        await target.invoke(self.request())
        with self.assertRaisesRegex(DesktopError, 'different intent'):
            await target.invoke(self.request(timeout_seconds=2))
        self.assertEqual(self.effects, 1)

    async def test_unadvertised_app_is_rejected_before_input(self):
        with self.assertRaisesRegex(DesktopError, 'unavailable'):
            await TargetExecution(self.desktop).invoke(self.request(app='calc'))
        self.assertEqual(self.effects, 0)

    async def test_unsupported_window_operation_is_rejected_before_input(self):
        with self.assertRaisesRegex(DesktopError, 'only operation'):
            await TargetExecution(self.desktop).invoke(self.request(operation='activate_window'))
        self.assertEqual(self.effects, 0)

    async def test_bad_wait_budget_is_rejected_before_input(self):
        for value in (True, '1', 0, 13):
            with self.assertRaises(DesktopError):
                await TargetExecution(self.desktop).invoke(self.request(timeout_seconds=value))
        self.assertEqual(self.effects, 0)

    async def test_permission_change_is_classified_without_relaunch(self):
        self.mode = 'permission-after-input'
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertEqual(result['status'], 'permission_required'); self.assertFalse(result['verified'])
        self.assertEqual(self.effects, 1)

    async def test_matching_app_outside_task_scope_cannot_verify(self):
        self.mode = 'outside'
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertFalse(result['verified']); self.assertEqual(self.effects, 1)

    async def test_uncertain_delivery_remains_paused_and_never_relaunches(self):
        self.mode = 'unknown'; target = TargetExecution(self.desktop)
        with self.assertRaises(DesktopError): await target.invoke(self.request())
        self.assertTrue(self.desktop.paused); self.assertEqual(len(self.desktop.unresolved()), 1)
        record = json.loads(self.desktop.db.execute('SELECT record FROM target_transitions').fetchone()[0])
        self.assertEqual(record['status'], 'delivery_uncertain'); self.assertIsNone(record['input_sent'])
        with self.assertRaisesRegex(DesktopError, 'paused'): await target.invoke(self.request())
        self.assertEqual(self.effects, 1)

    async def test_pause_interrupts_wait_without_replaying_input(self):
        target = TargetExecution(self.desktop)
        pending = asyncio.create_task(target.invoke(self.request()))
        await self.entered.wait()
        await self.desktop.control('pause')
        result = await pending
        self.assertEqual(result['status'], 'interrupted'); self.assertEqual(self.effects, 1)

    async def test_historical_success_does_not_claim_current_focus_after_change(self):
        self.mode = 'switch'; target = TargetExecution(self.desktop)
        await target.invoke(self.request())
        self.mode = 'normal'
        result = await target.invoke(self.request())
        self.assertTrue(result['historical_verified']); self.assertFalse(result['verified'])
        self.assertEqual(result['status'], 'state_changed'); self.assertEqual(self.effects, 1)

    async def test_existing_pending_business_step_is_preserved(self):
        execution = Execution(self.desktop, Path(self.temp.name)); self.desktop.execution = execution
        await execution.invoke({'operation':'begin_step','step_id':'business-step','goal':'Save the document',
                                'conditions':[{'kind':'foreground_title','text':'saved-document'}]})
        self.mode = 'switch'
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertTrue(result['verified'])
        step = execution.steps()[0]
        self.assertEqual(step['goal'], 'Save the document'); self.assertEqual(step['status'], 'pending')
        execution.close()

    async def test_new_verified_transition_uses_the_shared_step_ledger(self):
        execution = Execution(self.desktop, Path(self.temp.name)); self.desktop.execution = execution
        self.mode = 'switch'
        result = await TargetExecution(self.desktop).invoke(self.request())
        self.assertTrue(result['verified']); self.assertEqual(execution.steps()[0]['status'], 'verified')
        execution.close()
