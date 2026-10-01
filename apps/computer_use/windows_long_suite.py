"""Run the real Windows archive workflow from a clean snapshot and fresh model.

Only run after every other VM/model suite is terminal. The app runs on Linux;
all desktop actions run in the Windows guest. This is separate from frozen
replay and from native Windows client qualification.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import aiohttp

from compare_replay import monitor
from windows_long_task import run as run_task
from windows_suite import command, stop


async def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
    assert info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") == "codex-goal-20260930"
    shared = next(Path(m["Source"]) for m in info["Mounts"] if m["Destination"] == "/shared")
    token_file = shared / "executor-token.txt"
    token = token_file.read_text().strip()
    summary = {"kind": "live_windows_archive_snapshot_trial", "status": "running", "passed": False,
               "kvmem": args.kvmem, "task_timeout": args.task_timeout, "started_at": time.time(),
               "client_platform": "linux", "desktop_platform": "windows",
               "source_commit": (await command("git", "rev-parse", "HEAD")).strip(),
               "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (root / "apps/computer_use").rglob("*.py") if ".runtime" not in p.parts},
               "snapshot": json.loads((shared.parent / "snapshots" / args.snapshot / "manifest.json").read_text())}
    engine = app = sampler = None
    stopped = asyncio.Event()
    env = dict(os.environ, PYTHONPATH=str(root / "apps/computer_use"))
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=4)) as client:
        async def get(url, guest=False):
            async with client.get(url, headers={"Authorization": "Bearer " + token} if guest else {}) as response:
                response.raise_for_status()
                return await response.json()

        async def ready(url, process=None, guest=False, app_ready=False):
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process and process.returncode is not None:
                    raise RuntimeError("Owned service exited before startup")
                try:
                    result = await get(url, guest)
                    if not app_ready or result.get("ready"):
                        return result
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(1)
            raise TimeoutError("Service startup: " + url)

        try:
            for url in ("http://127.0.0.1:18083/api/state", "http://127.0.0.1:18080/health"):
                try:
                    await get(url)
                except (aiohttp.ClientConnectionError, TimeoutError):
                    pass
                else:
                    raise RuntimeError("Another app or engine is running")
            await command("docker", "stop", "-t", "120", "qw3-computer-windows")
            await command(sys.executable, root / "apps/computer_use/vm/snapshot.py", "restore", "--name", args.snapshot)
            await command("docker", "start", "qw3-computer-windows")
            summary["initial_health"] = await ready("http://127.0.0.1:18082/health", guest=True)
            assert (await get("http://127.0.0.1:18082/files", True))["files"] == []
            fixture = root / "apps/computer_use/vm/long_fixture.html"
            shutil.copyfile(fixture, shared / "long_fixture.html")
            shutil.copyfile(fixture, output / "long_fixture.html")
            fixture_ready = shared / "long-task-ready.json"
            if fixture_ready.exists():
                shutil.move(fixture_ready, output / "prior-long-task-ready.json")
            await command(sys.executable, root / "apps/computer_use/vm/provision.py",
                          root / "apps/computer_use/vm/prepare_long_task.ps1")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if fixture_ready.exists():
                    try:
                        staged = json.loads(fixture_ready.read_text(encoding="utf-8-sig"))
                        assert staged["sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
                        break
                    except (ValueError, OSError):
                        pass
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError("Archive fixture was not staged")
            shutil.copyfile(fixture_ready, output / "long-task-ready.json")
            with (output / "engine-launch.log").open("wb") as log:
                engine = await asyncio.create_subprocess_exec(sys.executable, root / "apps/computer_use/run_engine.py",
                    "--method", "dflash2", "--kvmem", args.kvmem, "--lock-timeout", "30",
                    "--guided-refresh", args.guided_refresh,
                    "--visual-history", "retrievable", "--trace-retrieval", "--run-dir", output / "engine",
                    cwd=root, env=env, stdout=log, stderr=log)
            sampler = asyncio.create_task(monitor(engine.pid, output / "resources.jsonl", stopped))
            await ready("http://127.0.0.1:18080/health", engine)
            with (output / "app-launch.log").open("wb") as log:
                app = await asyncio.create_subprocess_exec(sys.executable, "-m", "qw3_computer.app",
                    "--runtime", output / "app", "--desktop-token", token_file, "--trace-backend",
                    cwd=root, env=env, stdout=log, stderr=log)
            await ready("http://127.0.0.1:18083/api/state", app, app_ready=True)
            summary["passed"] = await run_task(SimpleNamespace(
                run_dir=output / "task", app_runtime=output / "app", engine_dir=output / "engine",
                app="http://127.0.0.1:18083", desktop="http://127.0.0.1:18082", desktop_token=token_file,
                fixture_ready=output / "long-task-ready.json", timeout=args.task_timeout))
            summary["final_health"] = await get("http://127.0.0.1:18082/health", True)
            assert summary["final_health"]["paused"], "Desktop did not finish paused"
            assert summary["initial_health"]["desktop_id"] == summary["final_health"]["desktop_id"]
            summary["status"] = "completed"
        except Exception as exc:
            summary.update(status="failed", passed=False, error=repr(exc))
        finally:
            await stop(app)
            await stop(engine)
            stopped.set()
            if sampler:
                await sampler
            summary["finished_at"] = time.time()
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps({k: summary.get(k) for k in ("status", "passed", "error")}), flush=True)
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot", default="qw3-type-20261001")
    parser.add_argument("--kvmem", choices=["on", "off"], default="on")
    parser.add_argument("--guided-refresh", choices=["off", "both"], default="off")
    parser.add_argument("--task-timeout", type=float, default=900)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
