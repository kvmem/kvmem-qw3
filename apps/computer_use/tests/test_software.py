import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from qw3_computer.desktop import DesktopError
from qw3_computer.software import Software, atomic_json, process_environment, job_receipt


class SoftwareTests(unittest.IsolatedAsyncioTestCase):
    def test_completed_receipt_reports_verified_outputs_without_claiming_the_whole_task(self):
        job={'id':'job-test','status':'completed','project':'latex-demo','operation':'repair_latex',
             'needs_reconciliation':False,'verification':{'passed':True,'pdf':'C:/project/main.pdf'},
             'task_id':'private-owner','action_id':'private-action'}
        receipt=job_receipt(job,compact=True)
        result=receipt['verified_job_result']
        self.assertEqual(result['outputs']['pdf'],'C:/project/main.pdf')
        self.assertFalse(result['task_completion_claimed'])
        self.assertNotIn('private-owner',json.dumps(receipt))
        self.assertNotIn('next_poll',receipt)

    def test_failed_active_or_uncertain_job_never_recommends_reporting_success(self):
        base={'id':'job-test','status':'completed','needs_reconciliation':False,
              'verification':{'passed':True},'steps':[],'downloads':[]}
        for changes in [{'status':'running'},{'status':'failed'},{'needs_reconciliation':True},
                        {'verification':{'passed':False}},{'verification':{'passed':1}}]:
            self.assertNotIn('verified_job_result',job_receipt(base|changes,compact=True))

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.software = Software(Path(self.temp.name) / "workspace")
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        async def download(*args):
            self.entered.set()
            await self.gate.wait()
            raise DesktopError("Deliberate offline fixture; no installer is run")
        self.software.download = download
        self.software.registered_python = lambda: []

    async def asyncTearDown(self):
        await self.software.close()
        self.temp.cleanup()

    def start(self, **changes):
        args = {"operation": "setup_python", "action_id": "software-check-001", "project": "demo", "profile": "python-web"} | changes
        with patch("qw3_computer.software.sys.platform", "win32"):
            return self.software.start("task-1", args)

    async def test_duplicate_start_does_not_run_twice_and_other_task_cannot_adopt(self):
        first = self.start()
        await self.entered.wait()
        duplicate = self.start()
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(first["id"], duplicate["id"])
        self.assertEqual(len(self.software.jobs), 1)
        self.assertEqual(len(self.software.jobs[first["id"]]["steps"]), 1)
        with self.assertRaises(DesktopError):
            self.start(project="different")
        with self.assertRaises(DesktopError):
            await self.software.invoke("another-task", {"operation": "job_status", "job_id": first["id"]})

    async def test_poll_timeout_does_not_cancel_background_download(self):
        first = self.start(); await self.entered.wait()
        result = await self.software.invoke("task-1", {"operation": "job_status", "job_id": first["id"], "wait_seconds": 0})
        self.assertEqual(result["status"], "running")
        self.assertFalse(self.software.tasks[first["id"]].done())

    async def test_same_action_key_in_a_new_task_creates_a_distinct_owned_job(self):
        first = self.start(); await self.entered.wait()
        await self.software.pause("task-1")
        args = {"operation": "setup_python", "action_id": "software-check-001", "project": "demo", "profile": "python-web"}
        with patch("qw3_computer.software.sys.platform", "win32"):
            second = self.software.start("task-2", args)
        self.assertNotEqual(second["job_id"], first["job_id"])
        self.assertNotIn("action_id", second)
        with self.assertRaises(DesktopError):
            await self.software.invoke("task-2", {"operation": "job_status", "job_id": first["job_id"]})
        await self.software.pause("task-2")
        await self.software.close()
        self.software = Software(self.software.root)
        with patch("qw3_computer.software.sys.platform", "win32"):
            duplicate = self.software.start("task-2", args)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["job_id"], second["job_id"])
        self.assertEqual(len(self.software.jobs), 2)

    async def test_legacy_global_job_preserves_same_task_retry_after_upgrade(self):
        identity = "legacy-action-001"
        job = {"task_id": "legacy-task", "project": "demo", "profile": "python-web", "id": identity,
               "status": "completed", "attempt": 1, "steps": [], "needs_reconciliation": False}
        directory = self.software.path("jobs", identity); directory.mkdir()
        atomic_json(directory / "job.json", job)
        original = (directory / "job.json").read_bytes()
        await self.software.close(); self.software = Software(self.software.root)
        args = {"operation": "setup_python", "action_id": identity, "project": "demo", "profile": "python-web"}
        duplicate = self.software.start("legacy-task", args)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["job_id"], identity)
        self.assertEqual((directory / "job.json").read_bytes(), original)
        with self.assertRaises(DesktopError):
            self.software.start("legacy-task", args | {"project": "changed"})

    async def test_receipt_poll_targets_owned_job_without_exposing_task_identifier(self):
        first = self.start(); await self.entered.wait()
        self.assertNotIn("task_id", first)
        self.assertEqual(first["job_id"], first["id"])
        self.assertEqual(first["next_poll"]["job_id"], first["id"])
        poll = first["next_poll"] | {"wait_seconds": 0}
        result = await self.software.invoke("task-1", poll)
        self.assertEqual(result["job_id"], first["id"])
        self.assertEqual(self.software.jobs[first["id"]]["task_id"], "task-1")
        self.assertNotIn("task_id", self.software.inspect("task-1")["jobs"][0])

    async def test_cancelled_poll_propagates_without_cancelling_install_job(self):
        first = self.start(); await self.entered.wait()
        poll = asyncio.create_task(self.software.invoke("task-1", {
            "operation": "job_status", "job_id": first["id"], "wait_seconds": 10}))
        await asyncio.sleep(0); poll.cancel()
        with self.assertRaises(asyncio.CancelledError): await poll
        self.assertFalse(self.software.tasks[first["id"]].done())

    async def test_pause_records_checkpoint_and_explicit_resume_retains_job_identity(self):
        first = self.start(); await self.entered.wait()
        await self.software.pause("task-1")
        job = self.software.jobs[first["id"]]
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(job["needs_reconciliation"])
        self.assertEqual(job["steps"][0]["status"], "cancelled")
        await self.software.invoke("task-1", {"operation": "resume_job", "job_id": first["id"]})
        self.assertEqual(job["attempt"], 2)
        self.gate.set()
        await self.software.tasks[first["id"]]
        self.assertEqual(job["status"], "failed")
        self.assertNotIn("verification", job)
        self.assertEqual([s["attempt"] for s in job["steps"]], [1, 2])

    async def test_immediate_pause_before_worker_starts_is_durably_cancelled(self):
        first = self.start()
        await self.software.pause("task-1")
        job = self.software.jobs[first["id"]]
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(self.entered.is_set())
        stored = json.loads(self.software.path("jobs", first["id"], "job.json").read_text())
        self.assertEqual(stored["status"], "cancelled")
        await self.software.invoke("task-1", {"operation": "resume_job", "job_id": first["id"]})
        await self.entered.wait()
        self.assertEqual(job["attempt"], 2)

    async def test_restart_marks_inflight_job_uncertain_and_forbids_new_dispatch(self):
        first = self.start(); await self.entered.wait()
        await self.software.pause("task-1")
        job = self.software.jobs[first["id"]]
        job["status"] = "running"; self.software.save(job)
        await self.software.close()
        self.software = Software(self.software.root)
        self.assertEqual(self.software.jobs[first["id"]]["status"], "uncertain")
        with self.assertRaises(DesktopError): self.start(action_id="software-new-002")
        with self.assertRaises(DesktopError):
            await self.software.invoke("task-1", {"operation": "resume_job", "job_id": first["id"]})

    async def test_invalid_names_and_command_injection_are_rejected_before_job_creation(self):
        for name in ("../outside", "C:\\outside", "demo;calc", "demo\n", "CON", "nul", "lpt1", "a/b"):
            with self.subTest(name=name), self.assertRaises(DesktopError): self.start(project=name)
        with self.assertRaises(DesktopError):
            await self.software.invoke("task-1", {"operation": "setup_python", "command": "anything"})
        self.assertEqual(self.software.jobs, {})

    async def test_managed_project_does_not_overwrite_an_existing_foreign_venv(self):
        root = self.software.path("projects", "demo"); root.mkdir(); (root / ".venv").mkdir()
        marker = root / ".venv/user-data.txt"; marker.write_text("preserve")
        first = self.start(); await self.software.tasks[first["id"]]
        job = self.software.jobs[first["id"]]
        self.assertEqual(job["status"], "failed")
        self.assertIn("unmanaged", job["error"])
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse(self.entered.is_set())

    async def test_conflicting_project_files_and_corrupt_cache_are_retained(self):
        path = self.software.path("projects", "user-note.txt"); path.write_text("original")
        with self.assertRaises(DesktopError): self.software.put(path, "replacement")
        self.assertEqual(path.read_text(), "original")
        asset = self.software.catalog["python"]
        cache = self.software.path("downloads", asset["filename"]); cache.write_bytes(b"corrupt")
        with self.assertRaisesRegex(DesktopError, "hash mismatch"):
            await Software.download(self.software, {"id": "cache-check"}, asset)
        self.assertEqual(cache.read_bytes(), b"corrupt")

    async def test_workspace_cannot_be_opened_by_two_owners(self):
        with self.assertRaises(DesktopError): Software(self.software.root)

    def repair_fixture(self):
        project = self.software.path("projects", "demo"); project.mkdir()
        python = project / ".venv/Scripts/python.exe"
        python.parent.mkdir(parents=True); python.touch()
        atomic_json(project / ".qw3-project.json", {"version": 1, "profile": "python-web", "python": self.software.catalog["python"]["version"]})
        packages = self.software.catalog["profiles"]["python-web"]["packages"]
        (project / "requirements.lock").write_text("".join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}\n" for a in packages))
        (project / "main.py").write_text("# Keep the user's source\n")
        (project / "open-terminal.cmd").write_text("wrong launcher\n")
        facts = {"version": self.software.catalog["python"]["version"], "executable": str(python),
                 "prefix": str(project / ".venv"), "base_prefix": str(self.software.path("python", self.software.catalog["python"]["version"])),
                 "packages": {a["name"]: a["version"] for a in packages},
                 "imports": {n: {"ok": True, "file": str(project / ".venv/Lib/site-packages" / n / "__init__.py")} for n in ("requests", "pytest")}}
        return project, facts

    async def test_repair_missing_module_despite_present_metadata_and_preserves_source_and_backup(self):
        project, facts = self.repair_fixture()
        calls = []
        async def command(job, tag, argv, **kwargs):
            calls.append((tag, list(map(str, argv))))
            if tag == "diagnose-project":
                value = json.loads(json.dumps(facts)); value["imports"]["requests"] = {"ok": False, "error": "ModuleNotFoundError"}
                return json.dumps(value)
            return json.dumps(facts)
        self.software.command = command
        self.software.download = AsyncMock(return_value={"cache_hit": True})
        first = self.start(operation="repair_python"); await self.software.tasks[first["id"]]
        result = self.software.jobs[first["id"]]
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertFalse(result["diagnosis"]["dependencies_ok"])
        self.assertTrue(result["verification"]["passed"])
        reinstall = next(argv for tag, argv in calls if tag == "repair-dependencies")
        self.assertIn("--force-reinstall", reinstall)
        self.assertIn("--require-hashes", reinstall)
        self.assertEqual(Path(result["launcher_backup"]["backup"]).read_text(), "wrong launcher\n")
        self.assertEqual((project / "main.py").read_text(), "# Keep the user's source\n")
        self.assertFalse(any(tag in {"install-python", "create-venv"} for tag, argv in calls))
        with self.assertRaises(DesktopError): self.start(operation="setup_python")

    async def test_repair_foreign_interpreter_refuses_mutation(self):
        project, facts = self.repair_fixture()
        facts["base_prefix"] = str(self.software.path("python", "other"))
        self.software.command = AsyncMock(return_value=json.dumps(facts))
        first = self.start(operation="repair_python"); await self.software.tasks[first["id"]]
        result = self.software.jobs[first["id"]]
        self.assertEqual(result["status"], "failed")
        self.assertIn("identity is inconsistent", result["error"])
        self.assertEqual(self.software.command.await_count, 1)
        self.assertFalse(self.entered.is_set())
        self.assertEqual((project / "open-terminal.cmd").read_text(), "wrong launcher\n")

    async def test_repair_unknown_dependency_lock_is_preserved(self):
        project, _ = self.repair_fixture()
        (project / "requirements.lock").write_text("user-owned requirements\n")
        self.software.command = AsyncMock()
        first = self.start(operation="repair_python"); await self.software.tasks[first["id"]]
        self.assertEqual(self.software.jobs[first["id"]]["status"], "failed")
        self.software.command.assert_not_called()
        self.assertEqual((project / "requirements.lock").read_text(), "user-owned requirements\n")

    @unittest.skipUnless(sys.platform == "win32", "requires the real Windows known-folder API")
    async def test_installer_environment_can_resolve_common_appdata(self):
        code = ("import ctypes,json; b=ctypes.create_unicode_buffer(32768);"
                "h=ctypes.windll.shell32.SHGetFolderPathW(None,35,None,0,b);"
                "print(json.dumps(dict(hresult=h,path=b.value)))")
        result = subprocess.run([sys.executable, "-c", code], env=process_environment(),
                                capture_output=True, text=True, timeout=15, check=True)
        facts = json.loads(result.stdout)
        self.assertEqual(facts["hresult"], 0)
        self.assertEqual(Path(facts["path"]), Path(os.environ["PROGRAMDATA"]))


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import hashlib
        self.temp = tempfile.TemporaryDirectory()
        self.software = Software(Path(self.temp.name) / "workspace")
        self.job = {"id": "download-fixture"}
        self.software.path("jobs", self.job["id"]).mkdir()
        self.payload = b"reviewed fixture bytes"
        self.asset = {"filename": "fixture.bin", "url": "https://mirrors.jlu.edu.cn/fixture.bin",
                      "mirrors": ["https://mirrors.hit.edu.cn/fixture.bin"], "max_bytes": 100,
                      "sha256": hashlib.sha256(self.payload).hexdigest()}

    async def asyncTearDown(self):
        await self.software.close()
        self.temp.cleanup()

    async def download_with(self, responses):
        from unittest.mock import MagicMock
        class Response:
            def __init__(inner, status, payload):
                inner.status = status; inner.payload = payload; inner.content = inner
            async def __aenter__(inner): return inner
            async def __aexit__(inner, *args): pass
            async def iter_chunked(inner, size):
                yield inner.payload
        client = MagicMock()
        client.get.side_effect = [Response(*r) for r in responses]
        session = MagicMock()
        session.__aenter__.return_value = client
        self.client = client
        with patch('qw3_computer.software.aiohttp.TCPConnector'), patch(
                'qw3_computer.software.aiohttp.ClientSession', return_value=session) as factory:
            try:
                return await self.software.download(self.job, self.asset)
            finally:
                if factory.called: self.assertFalse(factory.call_args.kwargs['trust_env'])

    async def test_denied_source_falls_back_once_and_audits_verified_bytes(self):
        result = await self.download_with([(403, b''), (200, self.payload)])
        self.assertEqual(result['source_host'], 'mirrors.hit.edu.cn')
        self.assertEqual(self.client.get.call_count, 2)
        saved = json.loads(self.software.path('jobs', self.job['id'], 'job.json').read_text())
        self.assertEqual([r['status'] for r in saved['downloads']], ['unavailable', 'verified'])
        self.assertEqual(saved['downloads'][0]['http_status'], 403)
        self.assertEqual(self.software.path('downloads', 'fixture.bin').read_bytes(), self.payload)
        self.assertFalse(list(self.software.path('downloads').glob('*.partial-*')))

    async def test_integrity_failure_does_not_fall_back_or_publish_asset(self):
        with self.assertRaisesRegex(DesktopError, 'hash mismatch'):
            await self.download_with([(200, b'changed bytes'), (200, self.payload)])
        self.assertEqual(self.client.get.call_count, 1)
        self.assertEqual(self.job['downloads'][0]['status'], 'failed')
        self.assertFalse(list(self.software.path('downloads').iterdir()))

    async def test_all_sources_unavailable_names_asset_hosts_and_statuses(self):
        with self.assertRaises(DesktopError) as context:
            await self.download_with([(403, b''), (503, b'')])
        message = str(context.exception)
        for part in ('fixture.bin', 'mirrors.jlu.edu.cn: HTTP 403', 'mirrors.hit.edu.cn: HTTP 503'):
            self.assertIn(part, message)
        self.assertEqual(self.client.get.call_count, 2)
        self.assertFalse(list(self.software.path('downloads').iterdir()))

    async def test_unreviewed_fallback_is_rejected_before_any_network_request(self):
        self.asset['mirrors'] = ['https://unreviewed.invalid/fixture.bin']
        with self.assertRaisesRegex(DesktopError, 'Unreviewed'):
            await self.download_with([])
        self.client.get.assert_not_called()


class ReceiptTests(unittest.TestCase):
    def test_model_poll_is_bounded_while_complete_evidence_remains_available(self):
        downloads = [{'file': f'asset-{i}.bin', 'host': 'mirrors.hit.edu.cn', 'status': 'verified',
                      'http_status': 200, 'bytes': 1024, 'sha256': 'a' * 64, 'url': 'https://mirrors.hit.edu.cn/' + 'x' * 200}
                     for i in range(200)]
        job = {'id': 'job-test', 'task_id': 'private-task', 'action_id': 'private-action', 'status': 'running',
               'stage': 'install_latex', 'needs_reconciliation': False, 'downloads': downloads,
               'steps': [{'status': 'completed', 'result': {'assets': downloads}}]}
        original = json.dumps(job)
        compact = job_receipt(job, compact=True)
        self.assertLess(len(json.dumps(compact)), 1500)
        self.assertEqual(compact['progress']['assets_downloaded'], 200)
        self.assertEqual(compact['next_poll']['job_id'], job['id'])
        self.assertNotIn('task_id', compact)
        self.assertEqual(job_receipt(job)['downloads'], downloads)
        self.assertEqual(json.dumps(job), original)

    def test_terminal_receipt_retains_error_diagnosis_and_actual_delivery(self):
        job = {'id': 'job-test', 'status': 'failed', 'error': 'ctexart.cls absent',
               'needs_reconciliation': True, 'diagnosis': {'missing_file': 'ctexart.cls'}}
        receipt = job_receipt(job, compact=True)
        self.assertEqual(receipt['error'], job['error'])
        self.assertEqual(receipt['diagnosis'], job['diagnosis'])
        self.assertTrue(receipt['needs_reconciliation'])
        self.assertNotIn('next_poll', receipt)
        job.update(status='completed', needs_reconciliation=False,
                   verification={'passed': True, 'pdf': 'C:/project/build/actual.pdf'})
        self.assertEqual(job_receipt(job, compact=True)['verification'], job['verification'])


if __name__ == "__main__":
    unittest.main()
