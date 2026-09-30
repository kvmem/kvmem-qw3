"""Repeat live Windows tasks from an identical disk snapshot, retaining failures."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import aiohttp

from compare_replay import monitor
from windows_task import CASES, run as task_run


async def stop(proc):
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 30)
        except TimeoutError:
            proc.kill(); await proc.wait()


async def command(*args):
    proc = await asyncio.create_subprocess_exec(*map(str, args), stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT)
    stdout, _ = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {args[0]}: {stdout.decode(errors='replace')[-2000:]}")
    return stdout.decode()


async def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.run_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
    assert info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") == "codex-goal-20260930"
    shared = next(Path(m["Source"]) for m in info["Mounts"] if m["Destination"] == "/shared")
    token_file = shared / "executor-token.txt"
    token = token_file.read_text().strip()
    snapshot = shared.parent / "snapshots" / args.snapshot / "manifest.json"
    summary = {"kind": "live_model_windows_snapshot_trials", "snapshot": json.loads(snapshot.read_text()),
               "task_definitions": {case: CASES[case] for case in args.cases}, "method": args.method,
               "source_commit": (await command("git", "rev-parse", "HEAD")).strip(),
               "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (root / "apps/computer_use").rglob("*.py") if ".runtime" not in p.parts},
               "trials": [], "status": "running"}
    app_url, desktop_url = "http://127.0.0.1:18083", "http://127.0.0.1:18082"
    env = dict(os.environ, PYTHONPATH=str(root / "apps/computer_use"))
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=4)) as client:
        async def get(url, desktop=False):
            headers = {"Authorization": "Bearer " + token} if desktop else {}
            async with client.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        async def ready(url, proc=None, desktop=False, predicate=lambda data: True):
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc and proc.returncode is not None:
                    raise RuntimeError("Owned service exited during startup")
                try:
                    data = await get(url, desktop)
                    if predicate(data):
                        return data
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(1)
            raise TimeoutError("Service did not become ready: " + url)
        # Check before any VM reset, including a model job owned by another task.
        for url in (app_url + "/api/state", "http://127.0.0.1:18080/health"):
            try:
                await get(url)
            except (aiohttp.ClientConnectionError, TimeoutError):
                pass
            else:
                raise RuntimeError("A service port is occupied; stop the existing app/engine before running the suite")
        for kvmem in args.kvmem:
            for repetition in range(args.repetitions):
                for case in args.cases:
                    label = f"{kvmem}-{case}-{repetition}"
                    directory = output / label; directory.mkdir()
                    trial = {"case": case, "kvmem": kvmem, "repetition": repetition,
                             "passed": False, "directory": str(directory), "started_at": time.time()}
                    summary["trials"].append(trial)
                    app = engine = sampler = None
                    stopped = asyncio.Event()
                    print("TRIAL", label, flush=True)
                    try:
                        await command("docker", "stop", "-t", "120", "qw3-computer-windows")
                        await command(sys.executable, root / "apps/computer_use/vm/snapshot.py", "restore", "--name", args.snapshot)
                        await command("docker", "start", "qw3-computer-windows")
                        trial["initial_health"] = await ready(desktop_url + "/health", desktop=True)
                        files = await get(desktop_url + "/files", desktop=True)
                        assert files["files"] == [], "Snapshot output directory is not empty"
                        engine_command = [sys.executable, str(root / "apps/computer_use/run_engine.py"), "--method", args.method,
                                          "--kvmem", kvmem, "--lock-timeout", "30", "--run-dir", str(directory / "engine")]
                        with (directory / "engine-launch.log").open("wb") as log:
                            engine = await asyncio.create_subprocess_exec(*engine_command, cwd=root, env=env, stdout=log, stderr=log)
                        sampler = asyncio.create_task(monitor(engine.pid, directory / "resources.jsonl", stopped))
                        await ready("http://127.0.0.1:18080/health", engine)
                        app_command = [sys.executable, "-m", "qw3_computer.app", "--runtime", str(directory / "app"),
                                       "--desktop-token", str(token_file)]
                        with (directory / "app-launch.log").open("wb") as log:
                            app = await asyncio.create_subprocess_exec(*app_command, cwd=root, env=env, stdout=log, stderr=log)
                        initial = await ready(app_url + "/api/state", app, predicate=lambda state: state.get("ready"))
                        (directory / "initial-state.json").write_text(json.dumps(initial, indent=2))
                        await task_run(SimpleNamespace(run_dir=directory / "task", case=case, app=app_url,
                                                       timeout=args.task_timeout, pause_after_seconds=0))
                        task_result = json.loads((directory / "task/result.json").read_text())
                        # Verify through the guest again, independent of model/UI messages.
                        actual = await get(desktop_url + "/files", desktop=True)
                        delivered = next(f for f in actual["files"] if f["name"] == CASES[case]["filename"])
                        assert delivered == task_result["verified_file"]
                        trial.update(passed=task_result["passed"], elapsed_seconds=task_result["elapsed_seconds"],
                                     verified_file=delivered, final_health=await get(desktop_url + "/health", desktop=True))
                        if case == "cross_app":
                            import sqlite3
                            with sqlite3.connect(directory / "app/desktop/desktop.sqlite") as db:
                                actions = [json.loads(row[0])["action"] for row in db.execute("SELECT request FROM actions WHERE status='completed' ORDER BY created")]
                            assert {"kind": "key", "key": "CTRL+C"} in actions and {"kind": "key", "key": "CTRL+V"} in actions
                            assert not any(action.get("kind") == "type" and CASES[case]["content"] in action.get("text", "") for action in actions)
                            trial["clipboard_path_verified"] = True
                    except Exception as exc:
                        trial.update(passed=False, error=repr(exc))
                        task_result = directory / "task/result.json"
                        if task_result.exists():
                            trial["elapsed_seconds"] = json.loads(task_result.read_text()).get("elapsed_seconds")
                        print("FAILED", label, repr(exc), flush=True)
                    finally:
                        await stop(app)
                        await stop(engine)
                        stopped.set()
                        if sampler:
                            await sampler
                        trial["finished_at"] = time.time()
                        (directory / "result.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2))
                        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                    print("RESULT", label, trial["passed"], flush=True)
                    if args.stop_on_failure and not trial["passed"]:
                        summary["status"] = "stopped_on_failure"
                        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                        return
    summary["status"] = "completed"
    summary["passed"] = all(t["passed"] for t in summary["trials"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot", default="qw3-type-20261001")
    parser.add_argument("--method", choices=["none", "mtp", "dflash2"], default="dflash2")
    parser.add_argument("--kvmem", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--stop-on-failure", action="store_true")
    asyncio.run(run(parser.parse_args()))
