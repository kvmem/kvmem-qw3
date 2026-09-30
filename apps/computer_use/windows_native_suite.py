"""Qualify the Windows-hosted app/Codex/MCP client against the host GPU.

Requires the staged, hashed offline client ZIP. Every trial restores the same
VM snapshot, qualifies its native dependencies, runs one real task, and retains
the guest's complete runtime. Run only after other VM/model suites have ended.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import aiohttp

from compare_replay import monitor
from windows_suite import command, stop
from windows_task import CASES


async def run(args):
    root = Path(__file__).resolve().parents[2]
    if args.restart and "browser" in args.cases:
        raise ValueError("Crash/resume qualification supports --cases notepad cross_app")
    output = args.run_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
    assert info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") == "codex-goal-20260930"
    shared = next(Path(m["Source"]) for m in info["Mounts"] if m["Destination"] == "/shared")
    token = (shared / "executor-token.txt").read_text().strip()
    bundle_hash = (shared / "client-candidate.sha256").read_text().strip()
    assert hashlib.sha256((shared / "client-candidate.zip").read_bytes()).hexdigest() == bundle_hash
    bundle = json.loads((args.bundle_dir / "manifest.json").read_text())
    assert json.loads((args.bundle_dir / "archive.json").read_text())["sha256"] == bundle_hash
    assert bundle["source_files"]["windows_task.py"] == hashlib.sha256((root / "apps/computer_use/windows_task.py").read_bytes()).hexdigest()
    case_helper = root / "apps/computer_use/vm" / ("run_native_restart.ps1" if args.restart else "run_native_case.ps1")
    helpers = [Path(__file__), case_helper,
               root / "apps/computer_use/vm/test_native_client.ps1",
               root / "apps/computer_use/vm/model_relay.py"]
    if args.restart:
        helpers.append(root / "apps/computer_use/vm/native_restart.py")
        shutil.copyfile(root / "apps/computer_use/vm/native_restart.py", shared / "native_restart.py")
    helper_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    for helper in helpers:
        shutil.copyfile(helper, output / helper.name)
    summary = {"kind": "windows_native_real_model_snapshot_trials", "status": "running", "trials": [],
               "bundle_sha256": bundle_hash, "bundle": bundle, "method": args.method, "restart": args.restart,
               "helper_sha256": helper_hashes,
               "source_commit": (await command("git", "rev-parse", "HEAD")).strip(),
               "task_definitions": {c: CASES[c] for c in args.cases},
               "snapshot": json.loads((shared.parent / "snapshots" / args.snapshot / "manifest.json").read_text())}
    env = dict(os.environ, PYTHONPATH=str(root / "apps/computer_use"))
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=4)) as client:
        async def get(url, guest=False):
            async with client.get(url, headers={"Authorization": "Bearer " + token} if guest else {}) as response:
                response.raise_for_status(); return await response.json()
        async def ready(url, proc=None, guest=False):
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc and proc.returncode is not None:
                    raise RuntimeError("Owned service exited during startup")
                try:
                    return await get(url, guest)
                except (aiohttp.ClientError, TimeoutError):
                    await asyncio.sleep(1)
            raise TimeoutError("Service startup: " + url)
        async def result_file(path, predicate, deadline_seconds):
            deadline = time.monotonic() + deadline_seconds
            while time.monotonic() < deadline:
                if path.exists():
                    try:
                        result = json.loads(path.read_text(encoding="utf-8-sig"))
                        if predicate(result):
                            return result
                    except (ValueError, OSError):
                        pass  # The guest replaces this small result at completion.
                await asyncio.sleep(1)
            raise TimeoutError("Guest result deadline: " + path.name)

        # This checks the current services; operators must also wait for any
        # existing VM suite process itself to be terminal before starting here.
        for url in ("http://127.0.0.1:18083/api/state", "http://127.0.0.1:18080/health"):
            try:
                await get(url)
            except (aiohttp.ClientConnectionError, TimeoutError):
                pass
            else:
                raise RuntimeError("An app or engine is already running")
        for policy in args.kvmem:
            for repetition in range(args.repetitions):
                for case in args.cases:
                    label = f"{policy}-{case}-{repetition}"
                    run_id = output.name + "-" + label
                    directory = output / label; directory.mkdir()
                    trial = {"case": case, "kvmem": policy, "repetition": repetition, "passed": False,
                             "run_id": run_id, "directory": str(directory), "started_at": time.time()}
                    summary["trials"].append(trial)
                    engine = relay = sampler = None
                    stopped = asyncio.Event()
                    print("NATIVE TRIAL", label, flush=True)
                    try:
                        await command("docker", "stop", "-t", "120", "qw3-computer-windows")
                        await command(sys.executable, root / "apps/computer_use/vm/snapshot.py", "restore", "--name", args.snapshot)
                        await command("docker", "start", "qw3-computer-windows")
                        trial["initial_health"] = await ready("http://127.0.0.1:18082/health", guest=True)
                        assert (await get("http://127.0.0.1:18082/files", True))["files"] == []
                        # Archive old shared results before the setup helper writes
                        # another result; never accept a previous trial's status.
                        prior = directory / "prior-shared-results"; prior.mkdir()
                        for p in shared.glob("native-client-tests*"):
                            if p.is_file():
                                shutil.move(p, prior / p.name)
                        await command(sys.executable, root / "apps/computer_use/vm/provision.py",
                                      root / "apps/computer_use/vm/test_native_client.ps1")
                        qualification = await result_file(shared / "native-client-tests.json", lambda r: True, 180)
                        trial["native_contracts"] = qualification
                        assert qualification["passed"] and qualification["process_exit_code"] == 0, "Native contract tests failed"
                        assert qualification["bundle_sha256"] == bundle_hash
                        for p in shared.glob("native-client-tests*"):
                            if p.is_file():
                                shutil.copyfile(p, directory / p.name)
                        with (directory / "engine-launch.log").open("wb") as log:
                            engine = await asyncio.create_subprocess_exec(sys.executable, str(root / "apps/computer_use/run_engine.py"),
                                "--method", args.method, "--kvmem", policy, "--lock-timeout", "30", "--run-dir", str(directory / "engine"),
                                cwd=root, env=env, stdout=log, stderr=log)
                        sampler = asyncio.create_task(monitor(engine.pid, directory / "resources.jsonl", stopped))
                        await ready("http://127.0.0.1:18080/health", engine)
                        with (directory / "relay-launch.log").open("wb") as log:
                            relay = await asyncio.create_subprocess_exec(sys.executable, str(root / "apps/computer_use/vm/model_relay.py"),
                                "--run-dir", str(directory / "relay"), cwd=root, env=env, stdout=log, stderr=log)
                        relay_manifest = await result_file(directory / "relay/manifest.json", lambda r: True, 15)
                        await asyncio.sleep(0.5)
                        if relay.returncode is not None:
                            raise RuntimeError("VM model relay failed to bind")
                        shutil.copyfile(directory / "relay/model-token.txt", shared / "native-model-token.txt")
                        (shared / "native-model-token.txt").chmod(0o600)
                        for name in ("native-case-result.json", "native-case-progress.json"):
                            if (shared / name).exists():
                                shutil.move(shared / name, prior / name)
                        (shared / "native-case.json").write_text(json.dumps({"case": case, "run_id": run_id,
                            "backend": f"http://{relay_manifest['bind']}:18084/v1"}))
                        await command(sys.executable, root / "apps/computer_use/vm/provision.py", case_helper)
                        result = await result_file(shared / "native-case-result.json", lambda r: True, 600 if args.restart else 420)
                        assert result.get("run_id") == run_id, "Native result belongs to another trial or setup failed before reading its run ID"
                        trial["native_result"] = result
                        export = shared / "native-results" / run_id
                        if export.exists():
                            shutil.copytree(export, directory / "guest")
                        assert result["passed"] and result["driver_exit_code"] == 0, result.get("error", "Native task failed")
                        expected = result["task"]["verified_file"]
                        files = (await get("http://127.0.0.1:18082/files", True))["files"]
                        assert len(files) == 1, "The task produced an unexpected additional output file"
                        actual = next(f for f in files
                                      if f["name"] == CASES[case]["filename"])
                        assert actual == expected, "Native client and guest output evidence differ"
                        if case == "cross_app":
                            app_relative = "restart/app" if args.restart else "app"
                            database = directory / "guest" / app_relative / "desktop/desktop.sqlite"
                            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
                                actions = [json.loads(row[0])["action"] for row in db.execute(
                                    "SELECT request FROM actions WHERE status='completed' ORDER BY created")]
                            assert {"kind": "key", "key": "CTRL+C"} in actions and {"kind": "key", "key": "CTRL+V"} in actions
                            assert not any(a.get("kind") == "type" and CASES[case]["content"] in a.get("text", "") for a in actions)
                            trial["clipboard_path_verified"] = True
                        trial.update(passed=True, elapsed_seconds=result["task"]["elapsed_seconds"], verified_file=actual)
                    except Exception as exc:
                        trial.update(passed=False, error=repr(exc))
                        print("NATIVE FAILED", label, repr(exc), flush=True)
                    finally:
                        await stop(relay); await stop(engine)
                        stopped.set()
                        if sampler:
                            await sampler
                        trial["finished_at"] = time.time()
                        (directory / "result.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2))
                        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                    print("NATIVE RESULT", label, trial["passed"], flush=True)
                    if not trial["passed"]:
                        summary["status"] = "stopped_on_failure"
                        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                        return
    summary.update(status="completed", passed=all(t["passed"] for t in summary["trials"]))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("apps/computer_use/.runtime/windows-native-client-04"))
    parser.add_argument("--snapshot", default="qw3-scroll-20261001")
    parser.add_argument("--method", choices=["none", "mtp", "dflash2"], default="dflash2")
    parser.add_argument("--kvmem", nargs="+", choices=["on", "off"], default=["on"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--restart", action="store_true", help="Inject a native app crash and require explicit resume")
    asyncio.run(run(parser.parse_args()))
