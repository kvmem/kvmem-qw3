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
from qw3_computer.software import Software, atomic_json, process_environment


class SoftwareTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
