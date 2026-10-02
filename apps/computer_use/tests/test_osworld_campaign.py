import copy
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from osworld_windows.campaign import execute, validate
from osworld_windows.native_run import DEFAULT_SNAPSHOT, SNAPSHOT_QUALIFICATIONS, require_evidence_space, snapshot_qualification, terminal_status
from osworld_windows.scenarios import collection, prepare, verify_collection
from osworld_windows.source import sha256
from osworld_windows.launch import connection_waits, prepare as prepare_launch, verify_wait_capability


def plan():
    return dict(formal_run=False, visual_history='pin-all', attempts=[
        dict(run='queue-check-' + task, task=task, kvmem=policy, timeout_seconds=60, tool_call_limit=20)
        for task, policy in [('004', 'on'), ('010', 'off')]])


def report(outcome='agent_completed', success=False):
    return dict(pipeline_passed=True, independent_grade={'strict_success': success},
                native_result=dict(desktop_stopped=True, infrastructure_ok=True,
                                   task=dict(outcome=outcome, queue_may_continue=True)))


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    def test_roomy_external_output_does_not_hide_full_shared_disk(self):
        with patch('osworld_windows.native_run.shutil.disk_usage', side_effect=[
                SimpleNamespace(free=100*1024**3), SimpleNamespace(free=2*1024**3)]):
            with self.assertRaisesRegex(ValueError, 'shared'):
                require_evidence_space(Path('/external/output'), Path('/workspace/shared'))

    def test_storage_filesystem_also_requires_the_recording_reserve(self):
        with patch('osworld_windows.native_run.shutil.disk_usage', return_value=SimpleNamespace(free=7*1024**3)):
            with self.assertRaisesRegex(ValueError, 'storage'):
                require_evidence_space(Path('/workspace/storage'))

    async def test_one_explicit_connection_profile_reaches_both_attempts(self):
        waits = []
        async def runner(args):
            waits.append((args.kvmem, args.backend_timeout, args.backend_read_timeout))
            r = report()
            return r | terminal_status(r)
        with tempfile.TemporaryDirectory() as temporary:
            selected = plan() | dict(backend_timeout_seconds=1800, backend_read_timeout_seconds=1500)
            result = await execute(selected, SimpleNamespace(output=Path(temporary) / 'queue'), runner=runner)
            self.assertEqual(waits, [('on', 1800, 1500), ('off', 1800, 1500)])
            self.assertEqual(result['plan'], selected)

    def test_invalid_or_partial_connection_profile_is_rejected(self):
        for change in [dict(backend_timeout_seconds=1800), dict(backend_read_timeout_seconds=1500),
                       dict(backend_timeout_seconds=600, backend_read_timeout_seconds=900),
                       dict(backend_timeout_seconds=True, backend_read_timeout_seconds=30)]:
            with self.assertRaises(ValueError):
                validate(plan() | change)

    async def test_cancellation_records_terminal_and_preserves_unstarted_task(self):
        started = asyncio.Event()
        async def runner(args):
            started.set()
            await asyncio.Event().wait()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'queue'
            queue = asyncio.create_task(execute(plan(), SimpleNamespace(output=output), runner=runner))
            await started.wait()
            queue.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queue
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['status'], 'interrupted')
            self.assertEqual(result['trials'][0]['status'], 'interrupted')
            self.assertFalse(result['trials'][0]['queue_may_continue'])
            self.assertEqual([t['task'] for t in result['unstarted']], ['010'])

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

    def test_restore_qualification_belongs_to_selected_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary)
            path = app / '.runtime' / SNAPSHOT_QUALIFICATIONS[DEFAULT_SNAPSHOT]
            path.parent.mkdir(parents=True)
            for result in [dict(passed=False, snapshot=DEFAULT_SNAPSHOT), dict(passed=True, snapshot='some-other-base')]:
                path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    snapshot_qualification(app, DEFAULT_SNAPSHOT)
            path.write_text(json.dumps(dict(passed=True, snapshot=DEFAULT_SNAPSHOT)))
            self.assertEqual(snapshot_qualification(app, DEFAULT_SNAPSHOT)['sha256'], sha256(path))
            with self.assertRaises(ValueError):
                snapshot_qualification(app, 'unqualified-snapshot')


class CollectionBindingTests(unittest.TestCase):
    def test_legacy_bundle_is_compatible_only_with_default_waits(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / 'manifest.json').write_text('{}')
            verify_wait_capability(bundle, connection_waits())
            with self.assertRaisesRegex(ValueError, 'configurable_backend_waits'):
                verify_wait_capability(bundle, connection_waits(1800, 1500))

    def test_native_configuration_binds_the_explicit_connection_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); bundle = root / 'bundle';bundle.mkdir()
            (bundle / 'archive.json').write_text(json.dumps({'sha256': 'a' * 64}))
            (bundle / 'manifest.json').write_text(json.dumps({'runtime_capabilities': ['configurable_backend_waits']}))
            staged = root / 'profile-check-001-inputs';staged.mkdir()
            config = prepare_launch('004', 'profile-check-001', 'Reviewed task prompt.', {'files': []},
                bundle, staged, root / 'launch', timeout=1800, tool_limit=300,
                backend='http://172.17.0.1:18084/v1', backend_timeout=1800, backend_read_timeout=1500)
            exported = json.loads((staged / 'native-task.json').read_text())
            self.assertEqual(exported, config)
            self.assertEqual(exported['backend_timeout_seconds'], 1800)
            self.assertEqual(exported['backend_read_timeout_seconds'], 1500)
            self.assertEqual(exported['public_sha256']['prompt.txt'], sha256(staged / 'prompt.txt'))

    def test_purchase_initializer_supplies_exact_review_prompt(self):
        from osworld_windows import prepare035
        class LocalSite:
            browser_url = 'http://teamchat.qw3.test:18090/?cookie=queue-prompt-035'
            def __init__(self, *args): pass
            def initialize(self, state): return state
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'apps/computer_use/.runtime/windows/shared').mkdir(parents=True)
            with patch.object(prepare035, 'ROOT', root), patch.object(prepare035, 'Site', LocalSite):
                manifest = prepare('035', 'queue-prompt-035', root / 'task', None)
            self.assertEqual(manifest['instruction'], (root / 'task/prompt.txt').read_text())
            self.assertIn('three update batches', manifest['instruction'])
            self.assertIn('December 2025', manifest['instruction'])

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
