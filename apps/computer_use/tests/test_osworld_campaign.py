import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from osworld_windows.campaign import execute, validate
from osworld_windows.native_run import terminal_status
from osworld_windows.scenarios import collection, verify_collection
from osworld_windows.source import sha256


def plan():
    return dict(formal_run=False, visual_history='pin-all', attempts=[
        dict(run='queue-check-' + task, task=task, kvmem=policy, timeout_seconds=60, tool_call_limit=20)
        for task, policy in [('004', 'on'), ('010', 'off')]])


def report(outcome='agent_completed', success=False):
    return dict(pipeline_passed=True, independent_grade={'strict_success': success},
                native_result=dict(desktop_stopped=True, infrastructure_ok=True,
                                   task=dict(outcome=outcome, queue_may_continue=True)))


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_and_unsuccessful_grading_continue(self):
        calls = []
        async def runner(args):
            calls.append(args.task)
            result = report('time_limit' if args.task == '004' else 'agent_completed')
            return result | terminal_status(result)
        with tempfile.TemporaryDirectory() as temporary:
            result = await execute(plan(), SimpleNamespace(output=Path(temporary) / 'queue'), runner=runner)
            self.assertEqual(calls, ['004', '010'])
            self.assertEqual([t['status'] for t in result['trials']], ['timed_out', 'task_failed'])
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['formal_run'])

    async def test_uncertain_stop_prevents_next_restore(self):
        for outcome in ['infrastructure_error', 'paused_uncertain', 'paused']:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                calls = []
                async def runner(args):
                    calls.append(args.task)
                    result = report(outcome)
                    return result | terminal_status(result)
                result = await execute(plan(), SimpleNamespace(output=Path(temporary) / 'queue'), runner=runner)
                self.assertEqual(calls, ['004'])
                self.assertEqual(result['status'], 'halted')
                self.assertEqual([t['task'] for t in result['unstarted']], ['010'])

    async def test_exception_is_retained_and_does_not_run_second_task(self):
        async def runner(args):
            raise RuntimeError('Lost export acknowledgement')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'queue'
            result = await execute(plan(), SimpleNamespace(output=output), runner=runner)
            self.assertEqual(result['status'], 'halted')
            self.assertEqual(result['trials'][0]['status'], 'infra_invalid')
            self.assertEqual(json.loads((output / 'result.json').read_text()), result)

    def test_requires_all_materials_and_rejects_formal_mislabel(self):
        for change in [dict(formal_run=True), dict(formal_run=None), dict(visual_history='auto')]:
            with self.assertRaises(ValueError):
                validate(plan() | change)
        for task in ['006', '999']:
            candidate = plan()
            candidate['attempts'][0]['task'] = task
            with self.assertRaises(ValueError):
                validate(candidate)

    def test_rejects_duplicate_and_unsafe_run_before_any_execution(self):
        for name in ['../other-run', 'a', 'queue-check-010']:
            candidate = plan()
            candidate['attempts'][0]['run'] = name
            with self.assertRaises(ValueError):
                validate(candidate)

    def test_success_requires_independent_grade_and_safe_terminal(self):
        self.assertEqual(terminal_status(report(success=True))['status'], 'passed')
        self.assertEqual(terminal_status(report())['status'], 'task_failed')
        for change in [dict(pipeline_passed=False), dict(independent_grade={}),
                       dict(independent_grade={'strict_success': None}), dict(grading_failure='unreadable receipt')]:
            self.assertFalse(terminal_status(report(success=True) | change)['queue_may_continue'])
        # A saved passing artifact at timeout is not an agent-completed run.
        self.assertEqual(terminal_status(report('time_limit', True))['status'], 'timed_out')


class CollectionBindingTests(unittest.TestCase):
    def test_missing_is_task_failure_but_lost_receipt_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                verify_collection('010', directory)
            receipt = dict(infrastructure_ok=True, complete=True,
                           files=[entry | {'status': 'missing'} for entry in collection('010')['files']])
            (directory / 'receipt.json').write_text(json.dumps(receipt))
            verify_collection('010', directory)
            (directory / 'checklist.docx').write_bytes(b'unaccounted bytes')
            with self.assertRaises(ValueError):
                verify_collection('010', directory)

    def test_tampered_artifact_or_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries = []
            for entry in collection('010')['files']:
                path = directory / entry['name']
                path.write_bytes(b'collected saved bytes')
                entries.append(entry | dict(status='copied', bytes=path.stat().st_size, sha256=sha256(path)))
            receipt = dict(infrastructure_ok=True, complete=True, files=entries)
            (directory / 'receipt.json').write_text(json.dumps(receipt))
            verify_collection('010', directory)
            for name in ['checklist.docx', 'guidelines.docx']:
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    verify_collection('010', directory)
                path.write_bytes(original)
            swapped = copy.deepcopy(receipt)
            swapped['files'].reverse()
            (directory / 'receipt.json').write_text(json.dumps(swapped))
            with self.assertRaises(ValueError):
                verify_collection('010', directory)


if __name__ == '__main__':
    unittest.main()
