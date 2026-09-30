"""Crash and restart the host application during a real Windows model task.

Requires a clean VM snapshot and a running dedicated QW3 engine. This Linux
test driver kills only its own application process group, then checks that
restart is paused and explicit resume completes the original saved-file task.
"""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

import aiohttp

from windows_task import CASES


async def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.run_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    runtime = output / "app"
    result = {"kind": "real_model_application_crash_resume", "passed": False, "case": args.case}
    command = [sys.executable, "-m", "qw3_computer.app", "--runtime", str(runtime),
               "--desktop-token", str(args.desktop_token.resolve())]
    env = dict(os.environ, PYTHONPATH=str(root / "apps/computer_use"))
    app = None
    started = time.monotonic()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=45)) as client:
        async def api(route, body=None):
            async with client.request("GET" if body is None else "POST", "http://127.0.0.1:18083" + route, json=body) as response:
                data = await response.json(); response.raise_for_status(); return data
        async def launch(label):
            nonlocal app
            with (output / (label + ".log")).open("wb") as log:
                app = await asyncio.create_subprocess_exec(*command, cwd=root, env=env, stdout=log, stderr=log,
                                                          start_new_session=True)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if app.returncode is not None:
                    raise RuntimeError("Application exited during startup")
                try:
                    state = await api("/api/state")
                    if state.get("ready"):
                        return state
                except aiohttp.ClientConnectionError:
                    pass
                await asyncio.sleep(0.25)
            raise TimeoutError("Application startup")
        def actions():
            with sqlite3.connect(runtime / "desktop/desktop.sqlite") as db:
                return [list(row) for row in db.execute("SELECT id,status FROM actions ORDER BY created")]
        try:
            try:
                await api("/api/state")
            except aiohttp.ClientConnectionError:
                pass
            else:
                raise RuntimeError("An application already owns the test port")
            initial = await launch("first-start")
            assert initial["status"] == "idle"
            token = args.desktop_token.read_text().strip()
            async with client.get("http://127.0.0.1:18082/files", headers={"Authorization": "Bearer " + token}) as response:
                files = await response.json(); response.raise_for_status()
            assert files["files"] == [], "Restore the clean VM snapshot before this test"
            task = await api("/api/task", {"prompt": CASES[args.case]["prompt"]})
            result["task_id"] = task["id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = await api("/api/state")
                completed = [row for row in actions() if row[1] == "completed"]
                if state["status"] == "running" and completed and not state.get("current_action"):
                    result["before_crash"] = state
                    result["actions_before_crash"] = actions()
                    os.killpg(app.pid, signal.SIGKILL)
                    await app.wait()
                    break
                if state["status"] in ("completed", "failed", "paused", "stopped"):
                    raise RuntimeError("Task ended before the crash injection")
                await asyncio.sleep(0.05)
            else:
                raise TimeoutError("No completed action before crash deadline")
            await asyncio.sleep(1)
            restored = await launch("restart")
            result["restored"] = restored
            assert restored["status"] == "paused" and restored["desktop_paused"]
            assert restored["id"] == task["id"]
            before = actions(); await asyncio.sleep(2)
            assert actions() == before, "Restart dispatched actions without explicit resume"
            await api("/api/control", {"command": "resume"})
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                state = await api("/api/state")
                if state["status"] in ("completed", "failed", "stopped", "paused"):
                    result["final"] = state
                    assert state["status"] == "completed", "Resumed task did not complete"
                    actual = next(f for f in state["files"] if f["name"] == CASES[args.case]["filename"])
                    assert actual["content"] == CASES[args.case]["content"]
                    result["verified_file"] = actual
                    result["passed"] = True
                    break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError("Resumed task deadline")
        except BaseException as exc:
            result["error"] = repr(exc)
            raise
        finally:
            if app and app.returncode is None:
                app.terminate()
                try:
                    await asyncio.wait_for(app.wait(), 15)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(app.pid, signal.SIGKILL)
                    await app.wait()
            result["elapsed_seconds"] = time.monotonic() - started
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", choices=["notepad", "cross_app"], default="cross_app")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--desktop-token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    asyncio.run(run(parser.parse_args()))
