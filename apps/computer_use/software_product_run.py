"""Run one real local-model software product trial from the owned Windows baseline."""
import argparse
import asyncio
import base64
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import time

import aiohttp
from aiohttp import web

from compare_replay import monitor
from osworld_windows.bundle import prepare as prepare_bundle
from osworld_windows.native_run import require_evidence_space, snapshot_qualification
from osworld_windows.source import ROOT, sha256
from vm.model_relay import create_app as relay_app
from windows_suite import command, stop

APP = ROOT / "apps/computer_use"
SHARE = APP / ".runtime/windows/shared"


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def wait_json(path, timeout):
    async with asyncio.timeout(timeout):
        while not path.is_file():
            await asyncio.sleep(.5)
        return json.loads(path.read_text(encoding="utf-8-sig"))


async def run(args):
    output = args.output.resolve()
    identity = output.name
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,40}", identity):
        raise ValueError("Use a new short attempt identity")
    output.mkdir(parents=True, exist_ok=False)
    lock = open("/tmp/qw3-osworld-vm.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {"kind": "local_model_software_product_trial", "status": "preparing", "passed": False,
              "task": "product-python-repair" if args.repair_fixture else "product-python-new", "model_used": True, "advisor_used": False,
              "execution_mode": "local_model_with_controlled_software_tools", "started_at": time.time()}
    recorder = None; relay = None; sampler = None
    sampled = asyncio.Event()

    def event(stage):
        report["stage"] = stage
        save(output / "result.json", report)
        print(stage, flush=True)

    try:
        if args.acceptance_manifest:
            frozen = json.loads(args.acceptance_manifest.read_text())
            if frozen["task"] != report["task"] or identity not in frozen["trial_ids"] or args.timeout != frozen["timeout_seconds"]:
                raise ValueError("Run identity, task or budget differs from frozen acceptance")
            if args.thinking != frozen.get("thinking", False):
                raise ValueError("Thinking mode differs from frozen acceptance")
            for relative, expected in frozen["source_sha256"].items():
                if sha256(ROOT / relative) != expected:
                    raise ValueError("Frozen acceptance source changed: " + relative)
            if json.loads((args.bundle / "archive.json").read_text())["sha256"] != frozen["bundle_sha256"]:
                raise ValueError("Native bundle differs from frozen acceptance")
            report["acceptance"] = {"manifest_sha256": sha256(args.acceptance_manifest), "definition": frozen}
        info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
        if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
            raise ValueError("VM ownership mismatch")
        storage = Path(next(m["Source"] for m in info["Mounts"] if m["Destination"] == "/storage"))
        report["space"] = require_evidence_space(output, SHARE, storage)
        if not args.repair_fixture:
            report["snapshot_qualification"] = snapshot_qualification(APP, args.snapshot)
        baseline = args.baseline_sources.resolve()
        if (baseline / "snapshot.py").read_bytes() != (APP / "vm/snapshot.py").read_bytes():
            raise ValueError("Baseline restore helper differs from reviewed code")
        for name in ("fixture.html", "bootstrap.ps1"):
            if (baseline / name).read_bytes() != (APP / "vm" / name).read_bytes():
                raise ValueError("Baseline fixture/bootstrap changed")
        report["restore_source_sha256"] = {name: sha256(baseline / name) for name in ("snapshot.py", "executor.ps1", "fixture.html", "bootstrap.ps1")}
        engine = json.loads((args.engine / "manifest.json").read_text())
        actual = Path(f"/proc/{engine['pid']}/cmdline").read_bytes().split(b"\0")
        if not actual or actual[0].decode() != engine["command"][0] or engine["kvmem"] != "off":
            raise ValueError("Selected local baseline engine is not live or is not KVMem-off")
        report["engine"] = engine
        if args.acceptance_manifest:
            if engine["binary_sha256"] != frozen["engine_binary_sha256"] or engine["command"] != frozen["engine_command"]:
                raise ValueError("Engine differs from frozen acceptance profile")
        sources = [Path(__file__), APP / "software_product_guest.py", APP / "vm/executor.ps1",
                   *sorted((APP / "qw3_computer").glob("*.py")), APP / "qw3_computer/software_catalog.json"]
        report["source_commit"] = (await command("git", "rev-parse", "HEAD")).strip()
        report["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
        shutil.copyfile(__file__, output / "executed-host.py")
        shutil.copyfile(APP / "software_product_guest.py", output / "executed-guest.py")
        shutil.copyfile(APP / "vm/executor.ps1", output / "executed-executor.ps1")
        token = (SHARE / "executor-token.txt").read_text().strip()
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=5)) as client:
            async def desktop(route):
                async with client.get("http://127.0.0.1:18082" + route, headers={"Authorization": "Bearer " + token}) as response:
                    response.raise_for_status()
                    return await response.json()
            health = await desktop("/health")
            if not health["paused"]:
                raise ValueError("Desktop is active; refusing snapshot restore")
            previous = await desktop("/observe")
            (output / "before-restore.png").write_bytes(base64.b64decode(previous.pop("image")))
            save(output / "before-restore.json", previous)
            if args.repair_fixture:
                report["fixture_preserved_without_restore"] = True
                report["initial_health"] = health
            else:
                event("restoring_clean_baseline")
                await command("docker", "stop", "-t", "120", "qw3-computer-windows")
                (output / "prior-share-archive.json").write_text(await command(sys.executable, APP / "vm/archive_share.py", "shared-archive-" + identity))
                await command(sys.executable, baseline / "snapshot.py", "restore", "--name", args.snapshot)
                shutil.copyfile(output / "executed-executor.ps1", SHARE / "executor.ps1")
                await command("docker", "start", "qw3-computer-windows")
                async with asyncio.timeout(180):
                    while True:
                        try:
                            health = await desktop("/health")
                            break
                        except (aiohttp.ClientError, TimeoutError):
                            await asyncio.sleep(1)
                if not health["paused"] or (await desktop("/files"))["files"]:
                    raise ValueError("Restored executor is not clean and paused")
                report["restored_health"] = health
        event("installing_qualified_native_client")
        install = prepare_bundle(args.bundle, identity, SHARE, output / "bundle-install")
        report["native_bundle"] = install
        # Allow the restored Windows shell to finish initializing before its
        # first administrative setup command, as required by P0 evidence.
        await asyncio.sleep(8)
        await command(sys.executable, APP / "vm/provision.py", output / "bundle-install/install.ps1")
        installed = await wait_json(SHARE / (identity + "-bundle-install.json"), 120)
        if installed.get("passed") is not True or installed.get("bundle_sha256") != install["archive"]["sha256"]:
            raise ValueError("Native bundle installation failed")
        report["native_installation"] = installed
        info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
        network = info["NetworkSettings"]["Networks"]["bridge"]
        model_token = secrets.token_hex(32)
        credential = SHARE / (identity + "-model-token.txt")
        credential.write_text(model_token); credential.chmod(0o600)
        relay = web.AppRunner(relay_app("http://127.0.0.1:18080", network["IPAddress"], model_token,
                                       receipt_path=output / "model-requests.jsonl"), handler_cancellation=True)
        await relay.setup()
        await web.TCPSite(relay, network["Gateway"], 18084).start()
        sampler = asyncio.create_task(monitor(engine["pid"], output / "resources.jsonl", sampled))
        env = dict(os.environ, PYTHONPATH=str(APP))
        with (output / "recorder.log").open("wb") as log:
            recorder = await asyncio.create_subprocess_exec(sys.executable, "-m", "vm.record_console", "--output", str(output / "video"),
                "--ffmpeg", str(args.ffmpeg.resolve()), cwd=ROOT, env=env, stdout=log, stderr=log)
        await wait_json(output / "video/recording.json", 30)
        guest_path = SHARE / (identity + "-guest.py")
        shutil.copyfile(output / "executed-guest.py", guest_path)
        configuration = {"run": identity, "bundle_sha256": install["archive"]["sha256"],
                         "backend": f"http://{network['Gateway']}:18084/v1", "timeout_seconds": args.timeout,
                         "task": report["task"], "thinking": args.thinking,
                         "guest_script_sha256": sha256(guest_path), "executor_sha256": sha256(output / "executed-executor.ps1")}
        if args.repair_fixture:
            fixture = json.loads(args.repair_fixture.read_text())
            if fixture.get("status") != "ready_fault_injected":
                raise ValueError("Repair fixture is not ready")
            if args.acceptance_manifest:
                definition = frozen["repair_fixture"]
                if (fixture.get("reset_from_sha256") != definition["sha256"]
                        or fixture.get("fixture_source_sha256") != definition["reset_source_sha256"]
                        or fixture.get("workspace") != definition["workspace"]):
                    raise ValueError("Repair reset differs from frozen acceptance inputs")
            shutil.copyfile(args.repair_fixture, SHARE / (identity + "-fixture.json"))
            shutil.copyfile(args.repair_fixture, output / "fixture.json")
            configuration["repair_fixture_sha256"] = sha256(args.repair_fixture)
        save(SHARE / (identity + "-configuration.json"), configuration)
        save(output / "configuration.json", configuration)
        script = f'''$ErrorActionPreference='Stop'
$python='C:\\Qw3Computer\\clients\\{install['archive']['sha256'][:12]}\\python\\python.exe'
$guest='\\\\host.lan\\Data\\{identity}-guest.py'
$configuration='\\\\host.lan\\Data\\{identity}-configuration.json'
if ((Get-FileHash $guest -Algorithm SHA256).Hash.ToLowerInvariant() -cne '{sha256(guest_path)}') {{throw 'Guest source changed'}}
$arguments='-X utf8 "'+$guest+'" --configuration "'+$configuration+'"'
$worker=Start-Process $python -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput '\\\\host.lan\\Data\\{identity}-guest.stdout.log' -RedirectStandardError '\\\\host.lan\\Data\\{identity}-guest.stderr.log'
@{{pid=$worker.Id;started=(Get-Date -Format o)}} | ConvertTo-Json | Set-Content '\\\\host.lan\\Data\\{identity}-worker.json' -Encoding UTF8
'''
        (output / "launch.ps1").write_text(script, encoding="utf-8-sig")
        event("running_native_product_trial")
        await command(sys.executable, APP / "vm/provision.py", output / "launch.ps1")
        report["guest_worker"] = await wait_json(SHARE / (identity + "-worker.json"), 60)
        guest = await wait_json(SHARE / (identity + "-result.json"), args.timeout + 300)
        report["guest_result"] = guest
        shutil.copytree(SHARE / (identity + "-evidence"), output / "guest")
        for suffix in ("stdout.log", "stderr.log"):
            source = SHARE / (identity + "-guest." + suffix)
            if source.exists(): shutil.copyfile(source, output / ("guest." + suffix))
        report.update(status="complete", passed=guest.get("passed") is True and guest.get("desktop_stopped") is True)
        credential.unlink()
    except Exception as exc:
        report.update(status="halted", error=f"{type(exc).__name__}: {exc}")
    finally:
        await stop(recorder)
        sampled.set()
        if sampler:
            await sampler
        if relay:
            await relay.cleanup()
        if (output / "video/recording.json").is_file():
            report["recording"] = json.loads((output / "video/recording.json").read_text())
        report["finished_at"] = time.time()
        receipts = output / "model-requests.jsonl"
        report["model_used"] = receipts.exists() and receipts.stat().st_size > 0
        save(output / "result.json", report)
        lock.close()
    print(json.dumps({k: report.get(k) for k in ("status", "passed", "error")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--snapshot", default="qw3-osworld-client15-20261002")
    parser.add_argument("--baseline-sources", type=Path, default=APP / ".runtime/local-diagnostic-baseline-vm-daaaf6c")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repair-fixture", type=Path, help="Use an independently prepared broken project, preserving the current VM")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--acceptance-manifest", type=Path, help="Require a predeclared frozen formal-acceptance trial")
    result = asyncio.run(run(parser.parse_args()))
    raise SystemExit(0 if result["status"] == "complete" else 1)
