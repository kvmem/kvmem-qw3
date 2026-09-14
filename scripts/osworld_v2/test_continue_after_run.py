#!/usr/bin/env python3
"""Exercise deferred launching and partial-result accounting without the GPU."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from continue_after_run import process_identity, progress, write_json


SCRIPT = Path(__file__).with_name("continue_after_run.py")


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("Timed out waiting for test process")


class ContinuationTest(unittest.TestCase):
    def test_prediction_error_stops_batch_without_creating_zero_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts/osworld_v2").mkdir(parents=True)
            worker = root / "fake_worker.py"
            worker.write_text("""import json, os, pathlib, time
root = pathlib.Path(os.environ['QW3_OSWORLD_RESULT_ROOT'])
run = root / (os.environ['QW3_OSWORLD_RUN_LABEL'] + '-' + os.environ['QW3_OSWORLD_RUN_ID'])
task = run / 'osworld/pyautogui/screenshot/gpt-qw3/tasks/052'
task.mkdir(parents=True)
(task / 'step_timing.jsonl').write_text(json.dumps({'status': 'predict_error', 'error': 'HTTP 413'}) + '\\n')
time.sleep(30)
""")
            (root / "scripts/osworld_v2/run_qw3_full.sh").write_text(
                f"#!/bin/bash\nexec {sys.executable} {worker}\n")
            metadata = root / "previous.json"
            write_json(metadata, {"status": "paused_for_repair"})
            selection = root / "tasks.json"
            write_json(selection, {"tasks": ["052", "004"]})
            state = root / "state.json"
            plan = root / "plan.json"
            write_json(plan, {
                "root": str(root), "state_path": str(state), "poll_seconds": 0.05,
                "predecessor_run": str(root / "previous"), "predecessor_metadata": str(metadata),
                "wait_for_processes": [], "task_ids": ["052", "004"],
                "stop_on_prediction_failure": True,
                "settings": {"OSWORLD_TEST_META": str(selection), "QW3_OSWORLD_PORT": "0",
                             "QW3_OSWORLD_RUN_LABEL": "test", "QW3_OSWORLD_RESULT_ROOT": str(root / "results")}})
            supervisor = subprocess.Popen([sys.executable, str(SCRIPT), "--plan", str(plan)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                _, stderr = supervisor.communicate(timeout=8)
                self.assertEqual(supervisor.returncode, 0, stderr.decode())
                final = json.loads(state.read_text())
                self.assertEqual(final["status"], "stopped_prediction_failure")
                self.assertEqual(final["prediction_failure_tasks"], ["052"])
                self.assertEqual(final["task_scores"], {})
                self.assertEqual(final["completed_tasks"], 0)
                self.assertFalse((Path(final["run_dir"]) / 'osworld/pyautogui/screenshot/gpt-qw3/tasks/004').exists())
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                supervisor.wait(timeout=3)
                supervisor.stderr.close()

    def test_waits_for_both_processes_and_keeps_missing_score_unset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts/osworld_v2").mkdir(parents=True)
            worker = root / "fake_worker.py"
            worker.write_text("""import json, os, pathlib
root = pathlib.Path(os.environ['QW3_OSWORLD_RESULT_ROOT'])
run = root / (os.environ['QW3_OSWORLD_RUN_LABEL'] + '-' + os.environ['QW3_OSWORLD_RUN_ID'])
task = run / 'osworld/pyautogui/screenshot/gpt-qw3/tasks/004'
task.mkdir(parents=True)
(task / 'result.txt').write_text('0.75')
""")
            (root / "scripts/osworld_v2/run_qw3_full.sh").write_text(
                f"#!/bin/bash\nexec {sys.executable} {worker}\n")
            previous = root / "previous"
            task = previous / "osworld/pyautogui/screenshot/gpt-qw3/tasks/007"
            task.mkdir(parents=True)
            (task / "result.txt").write_text("0.5")
            metadata = root / "previous.json"
            write_json(metadata, {"status": "completed"})
            selection = root / "tasks.json"
            write_json(selection, {"tasks": ["004", "005"]})
            state = root / "state.json"
            plan_path = root / "plan.json"
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            processes = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                         for _ in range(2)]
            supervisor = None
            try:
                write_json(plan_path, {
                    "root": str(root), "state_path": str(state), "poll_seconds": 0.05,
                    "predecessor_run": str(previous), "predecessor_metadata": str(metadata),
                    "predecessor_task_id": "007",
                    "wait_for_processes": [{"pid": p.pid, "start_ticks": process_identity(p.pid)} for p in processes],
                    "task_ids": ["004", "005"], "settings": {
                        "OSWORLD_TEST_META": str(selection), "QW3_OSWORLD_PORT": str(port),
                        "QW3_OSWORLD_RUN_LABEL": "test", "QW3_OSWORLD_RESULT_ROOT": str(root / "results")}})
                supervisor = subprocess.Popen([sys.executable, str(SCRIPT), "--plan", str(plan_path)],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                wait_for(state.exists)
                self.assertEqual(json.loads(state.read_text())["status"], "waiting_for_predecessor")
                processes[0].terminate()
                processes[0].wait()
                time.sleep(0.15)
                self.assertFalse((root / "results").exists(), "Server still alive: must not launch")
                processes[1].terminate()
                processes[1].wait()
                _, stderr = supervisor.communicate(timeout=8)
                self.assertEqual(supervisor.returncode, 0, stderr.decode())
                final = json.loads(state.read_text())
                self.assertEqual(final["predecessor_score"], 0.5)
                self.assertEqual(final["status"], "completed_with_infrastructure_failures")
                self.assertEqual(final["task_scores"], {"004": 0.75})
                self.assertEqual(final["infrastructure_failure_tasks"], ["005"])
                self.assertEqual(final["mean_score_completed"], 0.75)
                again = subprocess.run([sys.executable, str(SCRIPT), "--plan", str(plan_path)],
                                       capture_output=True, timeout=3)
                self.assertNotEqual(again.returncode, 0, "Must reject duplicate scheduling")
                self.assertEqual(len(list((root / "results").iterdir())), 1)
            finally:
                for p in processes + ([supervisor] if supervisor else []):
                    if p.poll() is None:
                        p.terminate()
                    p.wait(timeout=3)
                    if p.stderr is not None:
                        p.stderr.close()

    def test_missing_or_invalid_results_are_not_zero_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "osworld/pyautogui/screenshot/gpt-qw3/tasks"
            for task, value in [("001", "nan"), ("002", ""), ("003", "0")]:
                directory = task_root / task
                directory.mkdir(parents=True)
                (directory / "result.txt").write_text(value)
            value = progress(root, ["001", "002", "003", "004"])
            self.assertEqual(value["task_scores"], {"003": 0.0})
            self.assertEqual(value["tasks_without_result"], ["001", "002", "004"])


if __name__ == "__main__":
    unittest.main()
