"""Paired KVMem on/off × decoder replay of frozen local desktop requests.

No tools are dispatched. Prefix reuse is disabled on both sides. Every profile
gets the same ordered requests; repetitions retain the profile's vision cache.
Startup/capacity failures remain in the result rather than changing one input.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import aiohttp

from qw3_computer.protocol import read_sse


def process_resources(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        return {"cpu_seconds": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
                "rss_bytes": int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                "peak_rss_bytes": int(status.get("VmHWM", "0 kB").split()[0]) * 1024}
    except (FileNotFoundError, ProcessLookupError):
        return {}


async def monitor(pid, output, stopped):
    with output.open("w") as log:
        while not stopped.is_set():
            proc = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
                                                        "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            row = {"time": time.time(), **process_resources(pid)}
            if proc.returncode == 0:
                values = stdout.decode().strip().splitlines()[0].split(",")
                row.update(gpu_memory_bytes=int(values[0]) * 1024 * 1024,
                           gpu_utilization_percent=float(values[1]), gpu_power_watts=float(values[2]))
            log.write(json.dumps(row) + "\n"); log.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stopped.wait(), 1)


async def replay(session, endpoint, body, pid, output):
    started = time.monotonic()
    result = {"started_at": time.time(), "status": "pending"}
    before = process_resources(pid)
    tools = {}
    text = ""
    try:
        async with session.post(endpoint + "/v1/chat/completions", json=body) as response:
            result["http_status"] = response.status
            if response.status != 200:
                result.update(status="http_error", error=(await response.text())[:4000])
                return result
            done = False
            with (output / "chunks.jsonl").open("w") as log:
                async for data in read_sse(response.content):
                    if data == "[DONE]":
                        done = True
                        break
                    log.write(data + "\n")
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        result["usage"] = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content") or delta.get("tool_calls"):
                            result.setdefault("first_output_seconds", time.monotonic() - started)
                        text += delta.get("content") or ""
                        for call in delta.get("tool_calls", []):
                            merged = tools.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                            merged["id"] = call.get("id") or merged["id"]
                            for key in ("name", "arguments"):
                                merged[key] += call.get("function", {}).get(key, "")
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
            result.update(status="completed" if done and result.get("finish_reason") in ("stop", "tool_calls") else "incomplete",
                          text=text, tool_calls=list(tools.values()))
            for tool in tools.values():
                json.loads(tool["arguments"])
    except (aiohttp.ClientError, ValueError, TimeoutError) as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        after = process_resources(pid)
        result["process"] = after
        if before and after:
            result["cpu_seconds"] = after["cpu_seconds"] - before["cpu_seconds"]
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


async def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((args.corpus / "manifest.json").read_text())
    for case in cases:
        path = args.corpus / case["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"], "Frozen input changed"
    summary = {"kind": "fixed_input_replay_no_tool_dispatch", "prefix_reuse": False,
               "corpus": str(args.corpus.resolve()), "cases": cases, "profiles": [],
               "vision_cache": "fresh per profile, retained across ordered requests/repetitions"}
    endpoint = f"http://127.0.0.1:{args.port}"
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=600)) as session:
        for method in args.methods:
            for kvmem in args.kvmem:
                label = f"{method}-kvmem-{kvmem}"
                directory = output / label
                directory.mkdir()
                profile = {"method": method, "kvmem": kvmem, "runs": [], "status": "starting"}
                summary["profiles"].append(profile)
                command = [sys.executable, str(root / "apps/computer_use/run_engine.py"), "--method", method,
                           "--kvmem", kvmem, "--prefix-reuse", "off", "--port", str(args.port),
                           "--kvmem-cpu-gib", str(args.kvmem_cpu_gib),
                           "--lock-timeout", "30",
                           "--visual-history", args.visual_history,
                           "--run-dir", str(directory / "engine")]
                if args.trace_retrieval:
                    command.append("--trace-retrieval")
                print("starting", label, flush=True)
                with (directory / "launcher.log").open("wb") as log:
                    proc = await asyncio.create_subprocess_exec(*command, cwd=root, stdout=log, stderr=log)
                stopped = asyncio.Event()
                sampler = asyncio.create_task(monitor(proc.pid, directory / "resources.jsonl", stopped))
                start = time.monotonic()
                try:
                    while True:
                        if proc.returncode is not None:
                            raise RuntimeError(f"Engine exited during startup ({proc.returncode})")
                        try:
                            async with session.get(endpoint + "/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                                if response.status == 200:
                                    break
                        except (aiohttp.ClientError, TimeoutError):
                            pass
                        if time.monotonic() - start > 240:
                            raise TimeoutError("Engine startup timed out")
                        await asyncio.sleep(1)
                    profile["startup_seconds"] = time.monotonic() - start
                    for repetition in range(args.repetitions):
                        for case in cases:
                            trial = directory / f"{case['name']}-{repetition}"
                            trial.mkdir()
                            body = json.loads((args.corpus / case["path"]).read_text())
                            result = await replay(session, endpoint, body, proc.pid, trial)
                            profile["runs"].append({"case": case["name"], "repetition": repetition,
                                                    "result_path": str(trial / "result.json"), **result})
                            print(label, case["name"], repetition, result["status"], round(result["elapsed_seconds"], 3), flush=True)
                    profile["status"] = "completed"
                except Exception as exc:
                    profile.update(status="failed", error=repr(exc))
                    print(label, "failed", repr(exc), flush=True)
                finally:
                    if proc.returncode is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(proc.wait(), 30)
                        except TimeoutError:
                            proc.kill(); await proc.wait()
                    stopped.set()
                    await sampler
                    profile["exit_code"] = proc.returncode
                    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["none", "mtp", "dflash2"], default=["none", "mtp", "dflash2"])
    parser.add_argument("--kvmem", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--kvmem-cpu-gib", type=int, default=48)
    parser.add_argument("--visual-history", choices=["pin-all", "retrievable"], default="pin-all")
    parser.add_argument("--trace-retrieval", action="store_true")
    parser.add_argument("--port", type=int, default=18080)
    asyncio.run(run(parser.parse_args()))
