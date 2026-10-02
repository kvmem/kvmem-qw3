"""Attach to an owned QW3 engine for a readonly native Windows boundary probe."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys

import aiohttp

from compare_replay import monitor
from windows_suite import command, stop
from .source import ROOT, sha256

UNIT = ("Archived application observation: the status panel showed an idle local workspace. "
        "No document edit, navigation, download, clipboard update or form submission occurred during this interval.\n")


async def run(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,60}", args.output.name):
        raise ValueError("Invalid diagnostic identity")
    if not 1 <= args.segment_tokens <= 65000 or not 1 <= args.observations <= 25 or not 1 <= args.timeout <= 14400:
        raise ValueError("Invalid diagnostic budget")
    if not 30 <= args.backend_read_timeout <= args.backend_timeout <= args.timeout:
        raise ValueError("Require 30 <= backend read timeout <= backend timeout <= task timeout")
    app = ROOT / "apps/computer_use"
    share = app / ".runtime/windows/shared"
    output = args.output.resolve()
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise ValueError("Long-context qualification requires at least 2 GiB free for evidence")
    output.mkdir(parents=True, exist_ok=False)
    lock = open("/tmp/qw3-osworld-vm.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = output.name
    engine = json.loads((args.engine_dir / "manifest.json").read_text())
    if Path(f"/proc/{engine['pid']}/exe").resolve() != (ROOT / "build-dflash-integration/qw3").resolve():
        raise ValueError("Attached engine identity does not match the owned binary")
    report = dict(kind="readonly_native_long_context_qualification", formal_run=False, passed=False,
                  engine=engine, engine_owned_by_caller=True, synthetic_padding=True, desktop_actions_available=False,
                  source_commit=(await command("git", "rev-parse", "HEAD")).strip())
    report["engine_directory"] = str(args.engine_dir.resolve())
    helper = app / "osworld_windows/qualify_long_context_guest.py"
    report["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in (Path(__file__), helper)}
    stopped = asyncio.Event()
    sampler = recorder = relay = None
    native = share / (identity + "-output")
    staged = share / (identity + "-inputs")
    token = (share / "executor-token.txt").read_text().strip()
    environment = dict(os.environ, PYTHONPATH=str(app))

    async def launch(name, argv):
        with (output / (name + ".log")).open("wb") as log:
            return await asyncio.create_subprocess_exec(*map(str, argv), cwd=ROOT, env=environment, stdout=log, stderr=log)

    async def wait_json(path, seconds, predicate=lambda value: True):
        async with asyncio.timeout(seconds):
            while True:
                if path.exists():
                    try:
                        value = json.loads(path.read_text(encoding="utf-8-sig"))
                        if predicate(value):
                            return value
                    except ValueError:
                        pass
                await asyncio.sleep(.5)

    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=30)) as session:
        async def get(url, guest=False):
            async with session.get(url, headers={"Authorization": "Bearer " + token} if guest else {}) as response:
                response.raise_for_status()
                return await response.json()
        try:
            info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
            if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
                raise ValueError("Not the task-owned VM")
            report["initial_health"] = await get("http://127.0.0.1:18082/health", True)
            if not report["initial_health"]["paused"]:
                raise ValueError("Desktop must already be paused")
            await get("http://127.0.0.1:18080/health")
            staged.mkdir(exist_ok=False)
            shutil.copyfile(helper, staged / "probe.py")
            shutil.copyfile(Path(__file__), output / "host-runner-source.py")
            async with session.post("http://127.0.0.1:18080/tokenize", json={"content": UNIT * 100}) as response:
                response.raise_for_status()
                per_hundred = (await response.json())["count"]
            padding = UNIT * int(args.segment_tokens * 100 / per_hundred)
            if len(padding) > 390000:
                raise ValueError("Native tool-output guard exceeded")
            (staged / "padding.txt").write_text(padding, encoding="utf-8")
            async with session.post("http://127.0.0.1:18080/tokenize", json={"content": padding}) as response:
                response.raise_for_status()
                report["measured_segment_tokens"] = (await response.json())["count"]
            report.update(segment_bytes=len(padding.encode()), observations_requested=args.observations,
                          timeout_seconds=args.timeout, auto_compact_token_limit=950000,
                          backend_timeout_seconds=args.backend_timeout,
                          backend_read_timeout_seconds=args.backend_read_timeout,
                          padding_sha256=sha256(staged / "padding.txt"))
            sampler = asyncio.create_task(monitor(engine["pid"], output / "resources.jsonl", stopped))
            relay = await launch("relay", [sys.executable, app / "vm/model_relay.py", "--run-dir", output / "relay",
                "--backend-timeout", args.backend_timeout, "--backend-read-timeout", args.backend_read_timeout])
            await wait_json(output / "relay/manifest.json", 20)
            shutil.copyfile(output / "relay/model-token.txt", staged / "model-token.txt")
            (staged / "model-token.txt").chmod(0o600)
            network = info["NetworkSettings"]["Networks"]["bridge"]
            archive = json.loads((args.bundle / "archive.json").read_text())
            client_id = archive["sha256"][:12]
            script = r'''$ErrorActionPreference='Stop'
try {
    $client='C:\Qw3Computer\clients\CLIENT_ID'
    $manifest=Get-Content "$client\manifest.json" -Raw | ConvertFrom-Json
    if ((Get-FileHash "$client\manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'MANIFEST_SHA') {throw 'Native client manifest mismatch'}
    foreach ($entry in $manifest.source_files.PSObject.Properties) {
        if ((Get-FileHash (Join-Path "$client\app" $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw 'Native client file changed'}
    }
    $staged='\\host.lan\Data\IDENTITY-inputs'
    if ((Get-FileHash "$staged\probe.py" -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'HELPER_SHA') {throw 'Diagnostic helper changed'}
    if ((Get-FileHash "$staged\padding.txt" -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'PADDING_SHA') {throw 'Diagnostic padding changed'}
    $env:PATH="$client\codex\bin;$env:PATH"
    $env:PYTHONPATH="$client\app"
    $env:PYTHONUTF8='1'
    $arguments='-X utf8 "'+$staged+'\probe.py" --identity IDENTITY --output "\\host.lan\Data\IDENTITY-output" --padding "'+$staged+'\padding.txt" --token-file "'+$staged+'\model-token.txt" --backend http://GATEWAY:18084/v1 --observations OBSERVATIONS --timeout TIMEOUT --backend-timeout MODEL_WAIT --backend-read-timeout READ_WAIT'
    $probe=Start-Process "$client\python\python.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$staged\stdout.log" -RedirectStandardError "$staged\stderr.log"
    @{started=$true;pid=$probe.Id;client_sha256='BUNDLE_SHA'} | ConvertTo-Json | Set-Content "$staged\launch-result.json" -Encoding UTF8
} catch {
    @{started=$false;error=$_.Exception.Message} | ConvertTo-Json | Set-Content '\\host.lan\Data\IDENTITY-inputs\launch-result.json' -Encoding UTF8
}
'''
            for key, value in dict(CLIENT_ID=client_id, BUNDLE_SHA=archive["sha256"], MANIFEST_SHA=sha256(args.bundle / "manifest.json"), IDENTITY=identity,
                HELPER_SHA=sha256(helper), PADDING_SHA=report["padding_sha256"], GATEWAY=network["Gateway"],
                MODEL_WAIT=str(args.backend_timeout), READ_WAIT=str(args.backend_read_timeout),
                OBSERVATIONS=str(args.observations), TIMEOUT=str(args.timeout)).items():
                script = script.replace(key, value)
            (output / "launch.ps1").write_text(script)
            recorder = await launch("recorder", [sys.executable, app / "vm/record_console.py", "--output", output / "video", "--ffmpeg", args.ffmpeg])
            await wait_json(output / "video/recording.json", 20, lambda value: value.get("status") == "recording")
            (output / "progress.json").write_text(json.dumps(report, indent=2))
            await command(sys.executable, app / "vm/provision.py", output / "launch.ps1")
            launch_receipt = await wait_json(staged / "launch-result.json", 60)
            report["launch_receipt"] = launch_receipt
            if launch_receipt.get("started") is not True:
                raise RuntimeError("Native launch failed: " + launch_receipt.get("error", "unknown"))
            report["native_result"] = await wait_json(native / "result.json", args.timeout + 120)
            shutil.copytree(native, output / "native")
            report["passed"] = report["native_result"].get("passed") is True
        except Exception as exc:
            report["error"] = repr(exc)
        finally:
            # If the native helper did not publish its terminal receipt, stop
            # only its exact staged command, never a process found by name.
            receipt = report.get("launch_receipt", {})
            if receipt.get("started") is True and not (native / "result.json").exists():
                cleanup = r'''$ErrorActionPreference='Stop'
$result=@{kind='owned_readonly_diagnostic_cleanup';passed=$false}
try {
    $process=Get-CimInstance Win32_Process -Filter 'ProcessId = DRIVER_PID'
    if ($process) {
        if (!$process.CommandLine.Contains('\\host.lan\Data\IDENTITY-inputs\probe.py')) {throw 'Driver command identity changed'}
        & taskkill.exe /PID DRIVER_PID /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) {throw 'Owned diagnostic tree did not stop'}
    }
    $result.passed=$true
} catch {$result.error=$_.Exception.Message}
$result | ConvertTo-Json | Set-Content '\\host.lan\Data\IDENTITY-inputs\cleanup-result.json' -Encoding UTF8
'''.replace('DRIVER_PID', str(int(receipt["pid"]))).replace('IDENTITY', identity)
                (output / "cleanup.ps1").write_text(cleanup)
                try:
                    await command(sys.executable, app / "vm/provision.py", output / "cleanup.ps1")
                    report["native_emergency_cleanup"] = await wait_json(staged / "cleanup-result.json", 30)
                except Exception as exc:
                    report["native_cleanup_error"] = repr(exc)
            await stop(recorder)
            if (output / "video/recording.json").exists():
                report["recording"] = json.loads((output / "video/recording.json").read_text())
                report["passed"] &= report["recording"].get("complete") is True
            await stop(relay)
            stopped.set()
            if sampler:
                try:
                    await sampler
                except Exception as exc:
                    report["sampler_error"] = repr(exc)
                    report["passed"] = False
            try:
                report["final_health"] = await get("http://127.0.0.1:18082/health", True)
                report["passed"] &= report["final_health"]["paused"] is True
            except Exception as exc:
                report["final_health_error"] = repr(exc)
                report["passed"] = False
            (output / "result.json").write_text(json.dumps(report, indent=2))
            lock.close()
    print(json.dumps({key: report[key] for key in ("passed", "formal_run")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "apps/computer_use/.runtime/windows-native-client-13")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--segment-tokens", type=int, default=60000)
    parser.add_argument("--observations", type=int, default=17)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--backend-timeout", type=int, default=600)
    parser.add_argument("--backend-read-timeout", type=int, default=180)
    arguments = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(arguments))["passed"] else 1)
