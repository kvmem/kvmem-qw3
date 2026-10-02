"""Actual Windows Codex/QW3 compaction probe with synthetic observation padding.

Only live executor observations are available. No actions, fixture answers,
shells or evaluators are exposed to the model. Padding is explicitly synthetic;
this is a component qualification, not an OSWorld task-success measurement.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config


def readonly_fixture(args):
    padding = args.padding.read_text(encoding="utf-8")
    if len(padding) > 390000:
        raise ValueError("Padding exceeds the native tool-output byte guard")
    token = Path(r"C:\Qw3Computer\executor-token.txt").read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    count = 0
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        if method == "initialize":
            value = dict(protocolVersion="2024-11-05", capabilities={"tools": {}},
                         serverInfo=dict(name="readonly-long-context-diagnostic", version="1"))
        elif method == "tools/list":
            value = {"tools": [dict(name="observe", description="Read the live Windows screen and synthetic archive segment. No actions.",
                inputSchema=dict(type="object", properties={}, additionalProperties=False))]}
        elif method == "tools/call" and request.get("params", {}).get("name") == "observe":
            if count >= args.observations + 1:
                value = dict(isError=True, content=[dict(type="text", text="Diagnostic observation budget exhausted. Finish now.")])
            else:
                req = urllib.request.Request("http://127.0.0.1:5000/observe", headers={"Authorization": "Bearer " + token})
                observation = json.load(opener.open(req, timeout=20))
                image = observation.pop("image")
                count += 1
                marker = hashlib.sha256((args.identity + ":" + str(count)).encode()).hexdigest()[:16]
                metadata = dict(observation=observation, sequence=count, marker=marker,
                                padding_kind="synthetic neutral archive, not desktop evidence")
                with args.ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metadata | dict(guest_monotonic_ns=time.monotonic_ns())) + "\n")
                text = json.dumps(metadata) + "\n" + padding
                value = dict(content=[dict(type="text", text=text), dict(type="image", data=image, mimeType="image/png")])
        elif method == "ping":
            value = {}
        else:
            print(json.dumps(dict(jsonrpc="2.0", id=request["id"], error=dict(code=-32601, message="Unsupported readonly diagnostic method"))), flush=True)
            continue
        print(json.dumps(dict(jsonrpc="2.0", id=request["id"], result=value)), flush=True)


async def run(args):
    if os.name != "nt" or os.environ.get("USERNAME") != "Qw3Test":
        raise ValueError("Only the dedicated Windows test user may run this probe")
    from aiohttp import ClientSession, ClientTimeout, web
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Path(r"C:\Qw3Computer\LongContextQualification") / args.identity
    runtime.mkdir(parents=True, exist_ok=False)
    ledger = args.output / "observations.jsonl"
    report = dict(kind="native_actual_model_compaction_diagnostic", formal_run=False,
                  real_long_context_inference=True, synthetic_padding=True,
                  desktop_actions_available=False, observations_requested=args.observations,
                  backend_timeout_seconds=args.backend_timeout,
                  backend_read_timeout_seconds=args.backend_read_timeout,
                  auto_compact_token_limit=950000, passed=False, started_unix_ns=time.time_ns(),
                  guest_driver_pid=os.getpid())
    (args.output / "ownership.json").write_text(json.dumps(report))
    client = None
    runner = None
    events = []
    thread = turn = None
    try:
        token = Path(r"C:\Qw3Computer\executor-token.txt").read_text().strip()
        async with ClientSession(trust_env=False, timeout=ClientTimeout(total=10)) as session:
            async with session.get("http://127.0.0.1:5000/health", headers={"Authorization": "Bearer " + token}) as response:
                response.raise_for_status()
                report["initial_health"] = await response.json()
                if not report["initial_health"]["paused"]:
                    raise ValueError("Readonly diagnostic requires an already paused desktop")
        backend_waits = {}
        if (args.backend_timeout, args.backend_read_timeout) != (600, 180):
            backend_waits = dict(backend_timeout_seconds=args.backend_timeout,
                                 backend_read_timeout_seconds=args.backend_read_timeout)
        runner = web.AppRunner(create_app(args.backend, backend_token=args.token_file.read_text().strip(),
            trace_dir=runtime / "inference", trace_backend=True, **backend_waits), handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        write_config(runtime / "codex", f"http://127.0.0.1:{port}/v1", sys.executable,
                     [str(Path(__file__).resolve()), "--mcp", "--padding", str(args.padding),
                      "--ledger", str(ledger), "--identity", args.identity, "--observations", str(args.observations)])
        client = CodexClient(runtime / "codex")
        await client.start()
        thread = await client.new_thread()
        prompt = (f"Readonly component diagnostic. Call observe exactly {args.observations} times sequentially. "
                  "Every response contains the live screen plus explicitly synthetic archive padding; do not act on the desktop. "
                  "Track sequence and marker fields, ignoring repeated archive padding. Retain the first marker and the latest marker, "
                  "including across any context compaction. After the final observation, return only JSON with keys "
                  "observations, first_marker, latest_marker. Use the observed values. Finish without more tool calls.")
        (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")
        turn = await client.turn(thread, prompt)
        report.update(thread_id=thread, turn_id=turn, codex_pid=client.proc.pid)
        (args.output / "ownership.json").write_text(json.dumps(report))
        with (args.output / "events.jsonl").open("w", encoding="utf-8") as log:
            async with asyncio.timeout(args.timeout):
                while True:
                    event = await client.events.get()
                    event["diagnostic_received_unix_ns"] = time.time_ns()
                    events.append(event)
                    log.write(json.dumps(event) + "\n"); log.flush()
                    if event.get("method") == "turn/completed" and event["params"]["turn"]["id"] == turn:
                        report["turn"] = event["params"]["turn"]
                        break
                    if event.get("method") == "app/disconnected":
                        raise RuntimeError("Owned native Codex disconnected")
        report["completed_compactions"] = sum(event.get("method") == "item/completed" and
            event.get("params", {}).get("item", {}).get("type") == "contextCompaction" for event in events)
        if report["turn"]["status"] != "completed":
            raise RuntimeError(report["turn"].get("error", {}).get("message", "Native turn did not complete"))
        replies = [event["params"]["item"]["text"] for event in events if event.get("method") == "item/completed"
                   and event.get("params", {}).get("item", {}).get("type") == "agentMessage"]
        text = replies[-1].strip() if replies else ""
        report["final_text"] = text
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        report["answer"] = json.loads(text)
        report["control_completed"] = report["turn"]["status"] == "completed"
    except Exception as exc:
        report["error"] = repr(exc)
        if client and thread and turn and not report.get("turn"):
            try:
                await client.interrupt(thread, turn)
            except Exception as interrupt:
                report["interrupt_error"] = repr(interrupt)
    finally:
        if client:
            if client.proc and client.proc.returncode is None:
                # Terminate exactly the child tree created by this helper.
                killed = await asyncio.create_subprocess_exec("taskkill.exe", "/PID", str(client.proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await killed.communicate()
                await client.proc.wait()
            await client.close()
        if runner:
            await runner.cleanup()
        if ledger.exists():
            observations = [json.loads(line) for line in ledger.read_text().splitlines()]
            report["observations_received"] = len(observations)
            if observations:
                expected = dict(observations=args.observations, first_marker=observations[0]["marker"],
                                latest_marker=observations[-1]["marker"])
                report["expected"] = expected
                report["answer_correct"] = report.get("answer") == expected and len(observations) == args.observations
        metrics = [json.loads(path.read_text()) for path in (runtime / "inference").glob("*/metrics.json")]
        report["request_count"] = len(metrics)
        report["maximum_measured_prompt_tokens"] = max((metric.get("usage", {}).get("input_tokens", 0) for metric in metrics), default=0)
        report["request_failures"] = [metric for metric in metrics if metric.get("status") != "completed"]
        report["passed"] = (report.get("control_completed") is True and report.get("answer_correct") is True and
                            report.get("completed_compactions", 0) >= 1 and report["maximum_measured_prompt_tokens"] >= 900000)
        report["ended_unix_ns"] = time.time_ns()
        shutil.copytree(runtime, args.output / "guest-runtime")
        (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--padding", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--observations", type=int, default=17)
    parser.add_argument("--backend")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--backend-timeout", type=int, default=600)
    parser.add_argument("--backend-read-timeout", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.mcp:
        readonly_fixture(arguments)
    else:
        result = asyncio.run(run(arguments))
        raise SystemExit(0 if result["passed"] else 1)
