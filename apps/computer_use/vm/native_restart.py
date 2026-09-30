"""Crash only the owned Windows app tree and resume its real model task.

Setup-only qualification driver, never an agent tool. Run with the offline
client's embedded Python after restoring a clean guest and starting the relay.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import aiohttp

from windows_task import CASES


async def run(args):
    if sys.platform != "win32":
        raise RuntimeError("This qualification requires native Windows")
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "app"
    result = {"kind": "windows_native_application_crash_resume", "passed": False, "case": args.case}
    command = [sys.executable, "-X", "utf8", "-m", "qw3_computer.app", "--runtime", str(runtime),
               "--desktop", "http://127.0.0.1:5000", "--desktop-token", str(args.desktop_token),
               "--backend", args.backend, "--backend-token", str(args.backend_token)]
    app = None
    started = time.monotonic()
    token = args.desktop_token.read_text(encoding="utf-8-sig").strip()
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=45)) as client:
        async def api(route, body=None):
            async with client.request("GET" if body is None else "POST", "http://127.0.0.1:18083" + route, json=body) as response:
                data = await response.json()
                response.raise_for_status()
                return data

        async def guest(route):
            async with client.get("http://127.0.0.1:5000" + route, headers={"Authorization": "Bearer " + token}) as response:
                data = await response.json()
                response.raise_for_status()
                return data

        async def kill_owned_tree(label):
            if app is None or app.returncode is not None:
                return
            proc = await asyncio.create_subprocess_exec(
                "taskkill.exe", "/PID", str(app.pid), "/T", "/F",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            log, _ = await asyncio.wait_for(proc.communicate(), 15)
            (output / f"{label}-taskkill.log").write_bytes(log)
            await asyncio.wait_for(app.wait(), 15)
            result[label + "_exit_code"] = app.returncode
            if proc.returncode != 0:
                raise RuntimeError(f"Owned process-tree termination failed: {proc.returncode}")

        async def launch(label):
            nonlocal app
            with (output / f"{label}.log").open("wb") as log:
                app = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log,
                                                          creationflags=subprocess.CREATE_NO_WINDOW)
            result[label + "_pid"] = app.pid
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if app.returncode is not None:
                    raise RuntimeError("Native application exited during startup")
                try:
                    state = await api("/api/state")
                    if state.get("ready"):
                        return state
                except aiohttp.ClientConnectionError:
                    pass
                await asyncio.sleep(0.25)
            raise TimeoutError("Native application startup")

        def actions():
            with sqlite3.connect((runtime / "desktop/desktop.sqlite").as_uri() + "?mode=ro", uri=True) as db:
                return [list(row) for row in db.execute("SELECT id,status FROM actions ORDER BY created")]

        try:
            try:
                await api("/api/state")
            except aiohttp.ClientConnectionError:
                pass
            else:
                raise RuntimeError("Another application owns the test port")
            assert (await guest("/files"))["files"] == [], "Restore a clean VM snapshot first"
            assert (await launch("first-start"))["status"] == "idle"
            task = await api("/api/task", {"prompt": CASES[args.case]["prompt"]})
            result["task_id"] = task["id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = await api("/api/state")
                completed = [row for row in actions() if row[1] == "completed"]
                if state["status"] == "running" and completed and not state.get("current_action"):
                    result["before_crash"] = state
                    result["actions_before_crash"] = actions()
                    await kill_owned_tree("crash")
                    break
                if state["status"] in ("completed", "failed", "paused", "stopped"):
                    raise RuntimeError("Task ended before crash injection")
                await asyncio.sleep(0.05)
            else:
                raise TimeoutError("No completed action before crash deadline")
            await asyncio.sleep(1)
            restored = await launch("restart")
            result["restored"] = restored
            assert restored["status"] == "paused" and restored["desktop_paused"]
            assert restored["id"] == task["id"]
            before = actions()
            await asyncio.sleep(2)
            assert actions() == before, "Restart dispatched an action without explicit resume"
            result["paused_without_actions_seconds"] = 2
            await api("/api/control", {"command": "resume"})
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                state = await api("/api/state")
                if state["status"] in ("completed", "failed", "stopped", "paused"):
                    result["state"] = state
                    assert state["status"] == "completed", "Resumed task did not complete"
                    files = (await guest("/files"))["files"]
                    assert len(files) == 1, "Unexpected additional task output"
                    actual = files[0]
                    assert actual["name"] == CASES[args.case]["filename"]
                    assert actual["content"] == CASES[args.case]["content"]
                    assert actual in state["files"], "Application and guest evidence differ"
                    result.update(passed=True, verified_file=actual)
                    break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError("Resumed native task deadline")
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            try:
                if app and app.returncode is None:
                    try:
                        state = await api("/api/state")
                        if state["status"] in ("starting", "running", "verifying"):
                            await api("/api/control", {"command": "stop"})
                    finally:
                        await kill_owned_tree("cleanup")
            except Exception as exc:
                result.update(passed=False, cleanup_error=repr(exc))
            result["elapsed_seconds"] = time.monotonic() - started
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error", "cleanup_error")}), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", choices=["notepad", "cross_app"], default="cross_app")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-token", type=Path, required=True)
    parser.add_argument("--desktop-token", type=Path, default=Path(r"C:\Qw3Computer\executor-token.txt"))
    parser.add_argument("--timeout", type=float, default=300)
    sys.exit(0 if asyncio.run(run(parser.parse_args())) else 1)
