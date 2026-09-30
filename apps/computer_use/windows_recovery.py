"""Exercise real VM no-progress handling and a lost action reply.

Run only while the model application is stopped. The transport fault is
injected after the real Windows executor durably completes the action.
"""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import time
import uuid
import sys

import aiohttp
from aiohttp import web

from qw3_computer.desktop import Desktop, DesktopError


async def run(args):
    output = args.run_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    token = args.desktop_token.read_text().strip()
    result = {"kind": "real_windows_executor_injected_transport_fault", "passed": False}
    headers = {"Authorization": "Bearer " + token}
    intercept = {"id": None, "dispatches": 0}
    delivered, release = asyncio.Event(), asyncio.Event()
    async with aiohttp.ClientSession(headers=headers, trust_env=False) as upstream:
        # Do not interfere with an active application even when it is paused.
        try:
            async with upstream.get("http://127.0.0.1:18083/api/state"):
                raise RuntimeError("Stop the model application before testing the executor")
        except aiohttp.ClientConnectionError:
            pass

        async def forward(request):
            raw = await request.read()
            selected = request.path == "/act" and json.loads(raw).get("action_id") == intercept["id"]
            if selected:
                intercept["dispatches"] += 1
            async with upstream.request(request.method, args.desktop + request.path,
                                        data=raw, headers={"Content-Type": "application/json"}) as response:
                data = await response.read()
                status = response.status
            if selected:
                assert status == 200
                result["guest_receipt"] = json.loads(data)
                delivered.set()
                await release.wait()
            return web.Response(status=status, body=data, content_type="application/json")

        proxy = web.Application()
        proxy.router.add_route("*", "/{path:.*}", forward)
        runner = web.AppRunner(proxy, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        desktop = Desktop(f"http://127.0.0.1:{port}", token, output / "desktop")
        async def act(action, action_id=None):
            return await desktop.act({"desktop_id": desktop.latest["desktop_id"],
                                      "observation_id": desktop.latest["observation_id"],
                                      "action_id": action_id or uuid.uuid4().hex, "action": action})
        started = time.monotonic()
        try:
            await desktop.start(); await desktop.control("resume")
            await act({"kind": "open_app", "app": "explorer"})
            await asyncio.sleep(2)
            # No caret or typing: repeated Escape should leave Explorer unchanged.
            observations = []
            for _ in range(10):
                await desktop.observe()
                before = desktop.latest["sha256"]
                response = await act({"kind": "key", "key": "ESC"})
                observations.append({"before": before, "after": desktop.latest["sha256"],
                                     "no_progress": response["no_progress"]})
                if response["no_progress"]:
                    break
            assert desktop.paused and observations[-1]["no_progress"], "No-progress guard did not pause"
            result["no_progress"] = observations
            try:
                await act({"kind": "key", "key": "ESC"})
            except DesktopError:
                result["paused_action_rejected"] = True
            else:
                raise AssertionError("Action was allowed while paused")
            await desktop.control("resume"); await desktop.observe()
            action_id = uuid.uuid4().hex
            intercept["id"] = action_id
            arguments = {"desktop_id": desktop.latest["desktop_id"],
                         "observation_id": desktop.latest["observation_id"],
                         "action_id": action_id, "action": {"kind": "key", "key": "CTRL+L"}}
            task = asyncio.create_task(desktop.act(arguments))
            await asyncio.wait_for(delivered.wait(), 15)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert desktop.paused and [a["action_id"] for a in desktop.unresolved()] == [action_id]
            result["uncertain_before_reconciliation"] = desktop.unresolved()
            if args.restart_executor:
                before = await desktop.request("GET", "/health")
                helper = Path(__file__).resolve().parent / "vm/provision.py"
                process = await asyncio.create_subprocess_exec(sys.executable, str(helper),
                                                               str(helper.with_name("restart_executor.ps1")))
                assert await process.wait() == 0, "Executor restart setup command failed"
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        after = await desktop.request("GET", "/health")
                        if after["process_id"] != before["process_id"]:
                            result["executor_restart"] = {"before": before, "after": after}
                            break
                    except DesktopError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError("Executor did not restart")
            # Resume obtains GET /actions/id; it must not send POST /act again.
            await desktop.control("resume")
            assert not desktop.unresolved() and intercept["dispatches"] == 1
            duplicate = await desktop.act(arguments)
            assert duplicate["duplicate"] and intercept["dispatches"] == 1
            result.update(action_dispatches=intercept["dispatches"], duplicate_receipt=True, passed=True)
        except BaseException as exc:
            result["error"] = repr(exc)
            raise
        finally:
            release.set()
            with contextlib.suppress(DesktopError):
                await desktop.control("pause")
            await desktop.close(); await runner.cleanup()
            result["elapsed_seconds"] = time.monotonic() - started
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--desktop-token", type=Path, default=Path("apps/computer_use/.runtime/windows/shared/executor-token.txt"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--restart-executor", action="store_true")
    asyncio.run(run(parser.parse_args()))
