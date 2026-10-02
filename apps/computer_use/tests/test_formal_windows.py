"""Host controls only. Fake runners never create formal VM/model results."""
import asyncio
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import formal_windows as formal


def profile():
    return dict(formal_run=True, v0_frozen=True, workflow_version='V0', kind='windows_formal_campaign_freeze',
        visual_history='pin-all', heartbeat_seconds=30, backend_timeout_seconds=1800,
        backend_read_timeout_seconds=1500, snapshot='qualified-clean', client_bundle_path='fixture-client',
        attempts=[dict(task=t, run='formal-control-' + t + '-' + m, kvmem=m,
                       timeout_seconds=1800, tool_call_limit=300) for t in formal.TASKS for m in ('on', 'off')])


def raw(args, *, status='task_failed', pending=False):
    return dict(task=args.task, run=args.output.name, kvmem=args.kvmem, status=status,
        pipeline_passed=True, queue_may_continue=True, formal_run=False,
        independent_grade=dict(strict_success=status == 'passed', artifact_checks_passed=pending,
            process_audit_status='not_audited' if pending else 'not_applicable'),
        native_result=dict(desktop_stopped=True, task=dict(outcome='time_limit' if status == 'timed_out' else 'agent_completed')))


class FormalMatrixTests(unittest.TestCase):
    def test_exact_original_scope_and_equal_pair_budgets(self):
        formal.validate_matrix(profile())
        for mutation in ('omit', 'replace', 'duplicate', 'budget', 'unsafe'):
            p = profile()
            if mutation == 'omit': p['attempts'].pop()
            elif mutation == 'replace': p['attempts'][4]['task'] = 'cap01'
            elif mutation == 'duplicate': p['attempts'][0]['run'] = p['attempts'][1]['run']
            elif mutation == 'budget': p['attempts'][1]['timeout_seconds'] += 1
            else: p['attempts'][0]['run'] = '../old-attempt'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                formal.validate_matrix(p)

    def test_current_incomplete_candidate_cannot_create_freeze(self):
        path = formal.ROOT / 'docs/computer_use_windows_v0_candidate_20261002.json'
        p = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'freeze.json'
            with self.assertRaisesRegex(ValueError, 'Preparation incomplete'):
                formal.freeze(p, output)
            self.assertFalse(output.exists())

    def test_preflight_and_unfrozen_profiles_cannot_start_formal_execution(self):
        for field in ('kind', 'formal_run', 'v0_frozen'):
            p = profile()
            p[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'formal freeze'):
                formal.validate_frozen(p)

    def test_references_bind_exact_json_pointer_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(formal, 'ROOT', Path(tmp)):
            path = Path(tmp) / 'evidence.json'
            path.write_text(json.dumps({'cases': [{'grade': {'strict_success': False}}]}))
            ref = dict(path='evidence.json', sha256=formal.sha(path), pointer='/cases/0/grade')
            self.assertEqual(formal.reference(ref), {'strict_success': False})
            path.write_text(json.dumps({'cases': [{'grade': {'strict_success': True}}]}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                formal.reference(ref)
            for bad in ('../outside.json', '/absolute.json', ''):
                with self.assertRaises(ValueError): formal.relative_path(bad)

    def test_pending_process_is_unknown_and_timeout_is_retained(self):
        args = SimpleNamespace(task='051', kvmem='off', output=Path('test-original-051'))
        p = raw(args, status='timed_out', pending=True)
        self.assertEqual(formal.grade_status(p), dict(status='process_review_pending', valid=False, review_pending=True))
        self.assertEqual(p['native_result']['task']['outcome'], 'time_limit')
        p['independent_grade']['artifact_checks_passed'] = False
        self.assertEqual(formal.grade_status(p)['status'], 'timed_out')
        self.assertTrue(formal.grade_status(p)['valid'])

    def test_missing_grade_or_unacknowledged_stop_not_valid(self):
        args = SimpleNamespace(task='004', kvmem='on', output=Path('test-original-004'))
        for condition in ('grade', 'stop', 'pipeline'):
            p = raw(args)
            if condition == 'grade': p['independent_grade'] = {}
            elif condition == 'stop': p['native_result']['desktop_stopped'] = False
            else: p['pipeline_passed'] = False
            self.assertFalse(formal.grade_status(p)['valid'])

    def test_backend_rejection_cannot_become_valid_model_zero(self):
        args = SimpleNamespace(task='051', kvmem='on', output=Path('test-original-051'))
        p = raw(args)
        p['native_result']['task']['outcome'] = 'agent_failed'
        p['inference_failures'] = [dict(status='http_error', http_status=413)]
        self.assertEqual(formal.grade_status(p), dict(status='backend_failed', valid=False, review_pending=False))


class ArchivedBindingControls(unittest.TestCase):
    def test_actual_native15_on_off_archives_bind_and_context_tampering_fails(self):
        # This checks real archived request receipts; neither archive is enrolled
        # as a formal original-task attempt.
        candidate = json.loads((formal.ROOT / 'docs/computer_use_windows_v0_candidate_20261002.json').read_text())
        if not all((formal.ROOT / ('apps/computer_use/.runtime/native-client-15-launch-' + m + '-02/result.json')).is_file() for m in ('on', 'off')):
            self.skipTest('Actual native15 qualification archives are required for this local binding control')
        for mode in ('on', 'off'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                archive = formal.ROOT / ('apps/computer_use/.runtime/native-client-15-launch-' + mode + '-02')
                r = json.loads((archive / 'result.json').read_text())
                p = copy.deepcopy(candidate); p['host_source_sha256'] = r['source_sha256']
                p['visual_history'] = r['visual_history']
                attempt = {k: r[k] for k in ('task', 'run', 'kvmem', 'timeout_seconds', 'tool_call_limit')}
                directory = Path(tmp)
                members = ['engine/manifest.json', 'launch/manifest.json', 'relay/requests.jsonl',
                    'guest/app/codex/config.toml', 'guest/app/codex/models.json']
                members += [str(path.relative_to(archive)) for path in (archive / 'guest/app/inference').glob('*/request.json')]
                for member in members:
                    target = directory / member; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(archive / member, target)
                formal.verify_execution(p, attempt, r, directory)
                catalog = directory / 'guest/app/codex/models.json'
                models = json.loads(catalog.read_text()); models['models'][0]['context_window'] = 131072
                catalog.write_text(json.dumps(models))
                with self.assertRaisesRegex(ValueError, 'context/compaction'):
                    formal.verify_execution(p, attempt, r, directory)
                shutil.copyfile(archive / 'guest/app/codex/models.json', catalog)
                receipts = directory / 'relay/requests.jsonl'
                receipts.write_text('')
                with self.assertRaisesRegex(ValueError, 'relay receipt'):
                    formal.verify_execution(p, attempt, r, directory)


class FormalQueueTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, path, runner, preparation=lambda p: dict(model_assets=[])):
        freeze = path.parent / 'frozen-input.json'
        freeze.write_text(json.dumps(profile()))
        args = SimpleNamespace(output=path, freeze=freeze, ffmpeg=Path('unused'), grader_python=Path('unused'))
        return await formal.execute(profile(), args, runner=runner, preparation=preparation, binding=lambda *a: None)

    async def test_enrollment_precedes_execution_and_failures_continue(self):
        calls = []
        async def runner(args):
            enrollment = args.output.parent / (args.output.name + '-enrollment.json')
            self.assertTrue(enrollment.exists())
            self.assertFalse(args.output.exists())
            self.assertTrue(json.loads(enrollment.read_text())['declared_before_execution'])
            calls.append(args.task)
            self.assertEqual((args.backend_timeout, args.backend_read_timeout), (1800, 1500))
            r = raw(args, status='timed_out' if args.kvmem == 'off' else 'task_failed')
            args.output.mkdir()
            (args.output / 'result.json').write_text(json.dumps(r))
            return r
        with tempfile.TemporaryDirectory() as tmp:
            r = await self.execute(Path(tmp) / 'round', runner)
            self.assertEqual(len(calls), 20)
            self.assertEqual(r['valid_results'], 20)
            self.assertEqual(r['status'], 'complete')
            self.assertFalse(any(t['independent_grade']['strict_success'] for t in r['trials']))

    async def test_process_review_pending_does_not_fake_zero_or_stop_next_task(self):
        async def runner(args):
            r = raw(args, pending=args.task == '051')
            args.output.mkdir()
            (args.output / 'result.json').write_text(json.dumps(r))
            return r
        with tempfile.TemporaryDirectory() as tmp:
            r = await self.execute(Path(tmp) / 'round', runner)
            self.assertEqual(r['completed_executions'], 20)
            self.assertEqual(r['valid_results'], 18)
            self.assertEqual(r['status'], 'execution_complete_review_pending')

    async def test_infrastructure_failure_halts_and_preserves_unstarted_cells(self):
        async def runner(args):
            r = raw(args)
            r.update(pipeline_passed=False, queue_may_continue=False, status='infra_invalid')
            args.output.mkdir()
            (args.output / 'result.json').write_text(json.dumps(r))
            return r
        with tempfile.TemporaryDirectory() as tmp:
            r = await self.execute(Path(tmp) / 'round', runner)
            self.assertEqual(r['status'], 'halted')
            self.assertEqual(r['valid_results'], 0)
            self.assertEqual(len(r['unstarted']), 19)

    async def test_evidence_changes_after_attempt_do_not_count_or_restore_next(self):
        checks = 0
        def preparation(p):
            nonlocal checks
            checks += 1
            if checks == 3: raise ValueError('Runner source changed')
            return dict(model_assets=[])
        async def runner(args):
            r = raw(args, status='passed')
            args.output.mkdir()
            (args.output / 'result.json').write_text(json.dumps(r))
            return r
        with tempfile.TemporaryDirectory() as tmp:
            r = await self.execute(Path(tmp) / 'round', runner, preparation)
            self.assertEqual(r['status'], 'halted')
            self.assertEqual(r['valid_results'], 0)
            self.assertEqual(len(r['trials']), 1)

    async def test_cancellation_records_interruption_and_does_not_enroll_next(self):
        started = asyncio.Event()
        async def runner(args):
            started.set()
            await asyncio.Event().wait()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'round'
            task = asyncio.create_task(self.execute(out, runner))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            r = json.loads((out / 'result.json').read_text())
            self.assertEqual(r['status'], 'interrupted')
            self.assertEqual(r['trials'][0]['status'], 'interrupted')
            self.assertEqual(len(r['unstarted']), 19)


if __name__ == '__main__':
    unittest.main()
