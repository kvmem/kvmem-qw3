"""Replay immutable native Windows inputs without dispatching desktop tools.

Isolates prefix state and engine fixes. A completed replay is not task success,
native Codex compaction qualification, or a matched live-trajectory comparison.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import aiohttp

from compare_replay import monitor, replay
from windows_suite import stop
from .source import ROOT, sha256


async def run(args):
    source = args.source.resolve()
    requests = sorted((source / "native/guest-runtime/inference").glob("*/request.json"),
        key=lambda path: json.loads(path.with_name("metrics.json").read_text())["started_at"])
    counts = [int(value) for value in re.findall(r"request plan stage=draft[^\n]*prompt_tokens=(\d+)",
        (source.parent / "engine/engine.log").read_text())]
    if len(counts) != len(requests) or not 1 <= args.first <= args.last <= len(requests):
        raise ValueError("Captured request sequence is incomplete or range is invalid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind="immutable_native_input_engine_control", formal_run=False, tools_dispatched=False,
        source=str(source), original_engine=json.loads((source.parent / "engine/manifest.json").read_text()),
        kvmem=args.kvmem, prefix_reuse=args.prefix_reuse, requests=[], passed=False, status="starting",
        source_commit=None, script_sha256=sha256(Path(__file__)))
    proc = sampler = None
    stopped = asyncio.Event()
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2))
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "apps/computer_use"))
    try:
        argv = [sys.executable, ROOT / "apps/computer_use/run_engine.py", "--kvmem", args.kvmem,
                "--guided-refresh", "both" if args.kvmem == "on" else "off", "--visual-history", "pin-all",
                "--prefix-reuse", args.prefix_reuse, "--run-dir", output / "engine"]
        if args.kvmem == "on":
            argv.append("--trace-retrieval")
        report["command"] = list(map(str, argv))
        inputs = []
        for number in range(args.first, args.last + 1):
            path = requests[number - 1]
            body = path.read_bytes()
            target = output / f"request-{number:02}.json"
            target.write_bytes(body)
            inputs.append(dict(number=number, path=str(target), original_path=str(path), bytes=len(body),
                               sha256=hashlib.sha256(body).hexdigest(), expected_prompt_tokens=counts[number - 1]))
        report["immutable_inputs"] = inputs
        save()
        with (output / "launch.log").open("wb") as log:
            proc = await asyncio.create_subprocess_exec(*map(str, argv), cwd=ROOT, env=environment, stdout=log, stderr=log)
        sampler = asyncio.create_task(monitor(proc.pid, output / "resources.jsonl", stopped))
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as client:
            async with asyncio.timeout(240):
                while True:
                    if proc.returncode is not None:
                        raise RuntimeError("Owned engine exited during startup")
                    try:
                        async with client.get("http://127.0.0.1:18080/health", timeout=aiohttp.ClientTimeout(total=3)) as response:
                            response.raise_for_status()
                            break
                    except (aiohttp.ClientError, TimeoutError):
                        await asyncio.sleep(.5)
            report["engine"] = json.loads((output / "engine/manifest.json").read_text())
            report["same_binary_as_original"] = report["engine"]["binary_sha256"] == report["original_engine"]["binary_sha256"]
            report["status"] = "running"; save()
            for entry in inputs:
                raw = Path(entry["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("Replay input changed")
                target = output / f"response-{entry['number']:02}"
                target.mkdir()
                log_path = output / "engine/engine.log"
                offset = log_path.stat().st_size
                print("request", entry["number"], "expected_prompt_tokens", entry["expected_prompt_tokens"], flush=True)
                result = await replay(client, "http://127.0.0.1:18080", json.loads(raw), proc.pid, target)
                with log_path.open("rb") as log:
                    log.seek(offset); segment = log.read().decode()
                (target / "engine-segment.log").write_text(segment)
                result.update(number=entry["number"], expected_prompt_tokens=entry["expected_prompt_tokens"],
                              usage_matches_capture=result.get("usage", {}).get("prompt_tokens") == entry["expected_prompt_tokens"])
                report["requests"].append(result); save()
                print(json.dumps({key: result.get(key) for key in ("number", "status", "elapsed_seconds", "usage_matches_capture", "error")}), flush=True)
                if result["status"] != "completed" or not result["usage_matches_capture"]:
                    break
            report["passed"] = (len(report["requests"]) == len(inputs) and
                all(row["status"] == "completed" and row["usage_matches_capture"] for row in report["requests"]))
            report["status"] = "completed" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
    finally:
        await stop(proc)
        stopped.set()
        if sampler:
            await sampler
        report["finished_at"] = time.time()
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kvmem", choices=("on", "off"), default="on")
    parser.add_argument("--prefix-reuse", choices=("auto", "off"), default="off")
    parser.add_argument("--first", type=int, default=6)
    parser.add_argument("--last", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=1200)
    arguments = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(arguments))["passed"] else 1)
