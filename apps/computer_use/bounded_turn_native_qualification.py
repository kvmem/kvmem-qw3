"""Actual pinned Codex lifecycle qualification using a synthetic local backend.

One isolated process per control/candidate, platform catalogue and scenario.
No QW3 request, cloud advisor or real desktop input is used.
"""
import argparse
import asyncio
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time

from aiohttp import web
from PIL import Image


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])


async def qualify(candidate, output, arm, platform, scenario):
    import kvmem_osworld_queue as owned

    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    build = json.loads((candidate / "build.json").read_text())
    if build.get("kind") != "isolated_bounded_zero_tool_turn_candidate":
        raise ValueError("Use a declared bounded-turn candidate")
    package = candidate / arm / "qw3_computer"
    for name, expected in build["framework_files"][arm].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Declared private package changed")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(package.parent))
    app = importlib.import_module("qw3_computer.app")
    desktop_module = importlib.import_module("qw3_computer.desktop")
    calls, checks = [], []

    def require(value, name):
        checks.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    class SyntheticDesktop(desktop_module.Desktop):
        def __init__(self):
            super().__init__("http://127.0.0.1:9", "synthetic-unused", output / "device")
            self.sequence = 0

        async def start(self):
            return await self.observe()

        async def observe(self):
            self.sequence += 1
            image = io.BytesIO()
            Image.new("RGB", (128, 96), (40, 70, 90)).save(image, "PNG")
            pixels = image.getvalue()
            observation = {"observation_id": "obs" + str(self.sequence), "desktop_id": "synthetic-" + platform,
                           "device_id": "fixture", "session_id": "fixture-session", "display_id": "primary",
                           "width": 128, "height": 96, "sha256": hashlib.sha256(pixels).hexdigest(),
                           "image": base64.b64encode(pixels).decode(), "captured_at": time.time(),
                           "foreground": {"handle": "one", "process": "fixture", "title": "Fixture visible"},
                           "capabilities": {"platform": platform},
                           "accessibility": {"status": "ok", "truncated": False, "controls": []}}
            return self.remember(observation) | {"image": observation["image"]}

        async def control(self, command, acknowledge_uncertain=False):
            if command not in {"pause", "resume", "stop"}:
                raise AssertionError("Unexpected synthetic control")
            self.paused = command != "resume"
            return {"paused": self.paused}

        async def request(self, method, route, body=None):
            if method == "GET" and route == "/files":
                return {"files": []}
            raise AssertionError("Native input and external device requests are forbidden")

    conditions = [{"kind": "foreground_title", "text": "Fixture visible"}]
    operations = ([{"operation": "request_takeover", "reason": "Qualification needs user input"}]
                  if scenario == "takeover" else [
                      {"operation": "begin_step", "step_id": "visible", "goal": "Verify the synthetic fixture title", "conditions": conditions},
                      {"operation": "wait", "step_id": "visible", "timeout_seconds": 0},
                      {"operation": "finish", "conditions": conditions, "timeout_seconds": 0}])

    async def backend(request):
        body = await request.json()
        calls.append(body)
        index = len(calls) - 2
        if scenario not in {"bounded", "lost-reply"} and 0 <= index < len(operations):
            name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"] == "mcp__computer__workflow")
            delta = {"tool_calls": [{"index": 0, "id": "fixture-" + str(len(calls)), "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(operations[index])}}]}
            finish = "tool_calls"
        else:
            delta = {"content": "I'll first observe the desktop state before proceeding."}
            finish = "stop"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for content, reason in ((delta, None), ({}, finish)):
            frame = {"choices": [{"index": 0, "delta": content, "finish_reason": reason}]}
            await response.write(("data: " + json.dumps(frame) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    backend_app = web.Application()
    backend_app.router.add_post("/v1/chat/completions", backend)
    backend_runner, url = await serve(backend_app)
    desktop = SyntheticDesktop()
    controller = app.Controller(output / "app", desktop, url + "/v1", 0, workflow=True,
                                device_platform=platform, thinking=True, model_output_tokens=16384, tool_call_limit=250,
                                output_policy="task")
    app_runner, app_url = await serve(app.web_app(controller))
    controller.port = int(app_url.rsplit(":", 1)[1])
    result = {"kind": "actual_codex_bounded_turn_transport_qualification", "arm": arm, "platform_catalogue": platform,
              "scenario": scenario, "checks": checks, "passed": False, "qw3_model_calls": 0, "native_device_inputs": 0,
              "native_device_tested": False, "business_success_improvement_proven": False, "advisor_enabled": False,
              "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidate_build_sha256": hashlib.sha256((candidate / "build.json").read_bytes()).hexdigest()}
    identity = None
    try:
        await controller.start()
        identity = owned.process_identity(controller.client.proc.pid)
        require(identity is not None, "actual owned Codex process identity recorded")
        result["codex_identity"] = identity
        if scenario == "lost-reply" and arm == "candidate":
            original_turn = controller.client.turn
            turns = 0
            async def lose_second_reply(*args, **kwargs):
                nonlocal turns
                acknowledged = await original_turn(*args, **kwargs)
                turns += 1
                if turns == 2:
                    result["faulted_acknowledged_new_turn_id"] = acknowledged
                    raise TimeoutError("Synthetic lost acknowledgement after the native turn started")
                return acknowledged
            controller.client.turn = lose_second_reply
        await controller.start_task("Exercise synthetic visible checks only; this is lifecycle transport qualification.")
        task_id = controller.state["id"]
        async with asyncio.timeout(45):
            while controller.state["status"] in {"starting", "running", "verifying"}:
                await asyncio.sleep(.02)
        if scenario == "lost-reply" and arm == "candidate":
            # Device pause is observed before the owned runtime finishes
            # closing. Verify the same lifecycle handle through actual exit;
            # do not treat its first paused snapshot as model retirement.
            async with asyncio.timeout(10):
                while not controller.consume_task.done():
                    await asyncio.sleep(.01)
                await controller.consume_task
        state = controller.snapshot()
        require(state["id"] == task_id, "same task retained")
        require(state["tool_call_limit"] == 250, "cumulative tool budget not reset")
        require(all(c["enable_thinking"] and c["max_tokens"] == 16384 for c in calls), "thinking and output budget retained")
        require(desktop.db.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0, "zero actual device input ledger entries")
        if arm == "control":
            require(len(calls) == 1 and state["tool_calls_used"] == 0, "original controller stops after first zero-tool turn")
            require(state["status"] == "paused", "original unverified task is not falsely completed")
        elif scenario == "lost-reply":
            attempts = state["bounded_turn_continuation"]["attempts"]
            require(len(attempts) == 1 and attempts[0]["delivery"] == "uncertain", "lost native acknowledgement stays uncertain without retry")
            require(state["status"] == "paused" and not state["ready"], "lost acknowledgement pauses input and disables the disconnected runtime")
            require(controller.client.proc.returncode is not None, "unidentified actual Codex turn stopped by retiring its owned runtime")
            require(result["faulted_acknowledged_new_turn_id"] != attempts[0]["completed_turn_id"], "failure injected after a new native turn really started")
            require(1 <= len(calls) <= 2 and state["tool_calls_used"] == 0, "lost acknowledgement grants no repeated model turn or extra input")
        else:
            attempts = state["bounded_turn_continuation"]["attempts"]
            require(all(a["delivery"] == "accepted" for a in attempts), "new turns acknowledged with durable receipts")
            require(all(a["thread_id"] == state["thread_id"] for a in attempts), "actual continuation reuses the Codex thread")
            require(len({a["turn_id"] for a in attempts}) == len(attempts), "actual acknowledged turn IDs are unique")
            require(len(calls) >= 2, "continuation crosses actual Codex backend transport")
            require(any("Durable checked/pending steps" in str(c["messages"]) for c in calls[1:]), "recovery checkpoint reaches actual backend")
            if scenario == "bounded":
                require(len(calls) == 3 and len(attempts) == 2 and state["tool_calls_used"] == 0,
                        "exactly two additional native turns then bound reached")
                require(state["status"] == "paused", "continued announcements never count as completion")
            elif scenario == "takeover":
                require(len(attempts) == 1 and state["tool_calls_used"] == 1, "one continuation reaches the actual workflow MCP tool")
                require(state["status"] == "paused" and state.get("takeover", {}).get("reason") == "Qualification needs user input",
                        "explicit native MCP takeover pauses without another automatic turn")
            else:
                require(len(attempts) == 1 and state["tool_calls_used"] == 3, "one continuation preserves the three checked workflow calls")
                require(state["status"] == "completed" and state.get("completion_origin") == "checked_workflow",
                        "only checked synthetic visible conditions complete the task")
        require(desktop.paused, "input remains paused at terminal state")
        events = [json.loads(line) for line in (output / "app/codex-events.jsonl").read_text().splitlines()]
        threads = {e["params"]["threadId"] for e in events if e.get("method") in {"turn/started", "turn/completed"}}
        require(threads == {state["thread_id"]}, "actual native event history belongs to one thread")
        result.update(passed=True, synthetic_backend_requests=len(calls), terminal_state=state,
                      actual_codex_version="0.159.2", actual_codex_transport=True)
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        try:
            await controller.close()
            require(identity is not None and owned.process_identity(identity["pid"]) is None,
                    "owned native Codex process actually retired")
        finally:
            await app_runner.cleanup()
            await backend_runner.cleanup()
            result["passed"] = result["passed"] and all(c["passed"] for c in checks)
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--platform", choices=("windows", "linux", "android"), required=True)
    parser.add_argument("--scenario", choices=("bounded", "takeover", "finish", "lost-reply"), required=True)
    args = parser.parse_args()
    result = asyncio.run(qualify(args.candidate, args.output, args.arm, args.platform, args.scenario))
    print(json.dumps({k: result[k] for k in ("arm", "platform_catalogue", "scenario", "passed", "synthetic_backend_requests")}))


if __name__ == "__main__":
    main()
