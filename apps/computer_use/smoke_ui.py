"""Exercise real UI buttons and pinned Codex with scripted local services.

Requires Playwright and a Chromium executable. No GPU or desktop VM is used;
this qualifies browser/application wiring, not model or OS task correctness.
"""
import argparse
import asyncio
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

from aiohttp import web
from PIL import Image
from playwright.async_api import async_playwright, expect

from qw3_computer.app import Controller, web_app
from qw3_computer.desktop import Desktop


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"kind": "browser_ui_real_codex_scripted_backend", "passed": False,
              "browser_executable": str(args.browser.resolve()), "no_gpu_or_vm": True}
    sequence, paused, actions, requests = 0, True, [], []
    held = {2: asyncio.Event(), 5: asyncio.Event()}
    cancelled = {2: asyncio.Event(), 5: asyncio.Event()}
    png = io.BytesIO()
    Image.new("RGB", (640, 400), "#dde7f1").save(png, format="PNG")
    image = png.getvalue()
    file_text = 'UI fixture <script>window.untrustedExecuted=true</script>\n'
    file = {"name": "ui-fixture.txt", "content": file_text,
            "sha256": hashlib.sha256(file_text.encode()).hexdigest(), "size": len(file_text.encode())}

    def observation():
        nonlocal sequence
        sequence += 1
        return {"desktop_id": "ui-fixture", "observation_id": str(sequence),
                "image": base64.b64encode(image).decode(), "sha256": hashlib.sha256(image).hexdigest(),
                "width": 640, "height": 400, "paused": paused,
                "foreground": {"title": "Scripted desktop fixture", "process": "fixture"}}

    async def desktop(request):
        nonlocal paused
        assert request.headers.get("Authorization") == "Bearer ui-fixture-token"
        if request.path == "/observe":
            return web.json_response(observation())
        if request.path == "/files":
            return web.json_response({"files": [file] if len(actions) == 2 else []})
        body = await request.json()
        if request.path == "/control":
            paused = body["command"] != "resume"
            return web.json_response({"paused": paused})
        assert request.path == "/act" and not paused
        assert body["observation_id"] == str(sequence)
        actions.append(body)
        return web.json_response({"action_id": body["action_id"], "status": "completed", "observation": observation()})

    async def backend(request):
        body = await request.json()
        requests.append(body)
        number = len(requests)
        if number in held:
            held[number].set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled[number].set()
                raise
        assert number in (1, 3, 4), f"Unexpected inference request {number}"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def chunk(delta=None, finish=None):
            data = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
            await response.write(("data: " + json.dumps(data) + "\n\n").encode())

        if number in (1, 3):
            tool = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("act"))
            arguments = {"desktop_id": "ui-fixture", "observation_id": str(sequence),
                         "action_id": f"ui-action-{number}", "action": {"kind": "type", "text": "fixture"}}
            await chunk({"tool_calls": [{"index": 0, "id": f"call_ui_{number}", "type": "function",
                                         "function": {"name": tool, "arguments": json.dumps(arguments)}}]})
            await chunk(finish="tool_calls")
        else:
            await chunk({"content": "UI fixture complete. <script>window.untrustedExecuted=true</script>"})
            await chunk(finish="stop")
        await response.write(b"data: [DONE]\n\n")
        return response

    desktop_app = web.Application()
    desktop_app.router.add_route("*", "/{path:.*}", desktop)
    backend_app = web.Application()
    backend_app.router.add_post("/v1/chat/completions", backend)
    desktop_runner, desktop_url = await serve(desktop_app)
    backend_runner, backend_url = await serve(backend_app)
    controller = Controller(output / "app", Desktop(desktop_url, "ui-fixture-token", output / "app/desktop"), backend_url + "/v1", 0)
    app_runner, app_url = await serve(web_app(controller))
    controller.port = int(app_url.rsplit(":", 1)[1])
    browser = None
    errors = []
    try:
        await controller.start()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(executable_path=str(args.browser),
                headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
            page = await browser.new_page(viewport={"width": 1280, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(app_url)
            await expect(page.locator("#status")).to_have_text("就绪")
            await expect(page.locator("#screen")).to_be_visible()
            await page.locator("#prompt").fill("Run the UI pause/resume fixture.")
            await page.locator("#start").click()
            await asyncio.wait_for(held[2].wait(), 30)
            assert len(actions) == 1
            first_task = controller.state["id"]
            first_thread = controller.state["thread_id"]
            await expect(page.locator("#start")).to_be_disabled()
            await expect(page.locator("#refresh")).to_be_disabled()
            await page.locator("#pause").click()
            await asyncio.wait_for(cancelled[2].wait(), 10)
            await expect(page.locator("#status")).to_have_text("已暂停")
            await expect(page.locator("#resume")).to_be_enabled()
            assert controller.desktop.paused and paused
            await asyncio.sleep(2)
            assert len(actions) == 1
            prior = controller.desktop.latest["observation_id"]
            await page.locator("#refresh").click()
            deadline = time.monotonic() + 5
            while controller.desktop.latest["observation_id"] == prior and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert controller.desktop.latest["observation_id"] != prior
            await page.screenshot(path=str(output / "paused.png"), full_page=True)
            await page.locator("#resume").click()
            await expect(page.locator("#status")).to_have_text("已结束 · 请核对结果", timeout=30000)
            assert controller.state["id"] == first_task and controller.state["thread_id"] == first_thread
            assert len(actions) == 2 and controller.desktop.paused
            await expect(page.locator("#files summary")).to_contain_text("ui-fixture.txt")
            await page.locator("#files summary").click()
            await expect(page.locator("#files")).to_contain_text(file_text.strip())
            assert await page.evaluate("window.untrustedExecuted === undefined")
            assert await page.locator("#files script, #messages script").count() == 0
            result["completed_state"] = controller.snapshot()
            await page.screenshot(path=str(output / "completed.png"), full_page=True)
            await page.locator("#prompt").fill("Run the UI stop fixture.")
            await page.locator("#start").click()
            await asyncio.wait_for(held[5].wait(), 30)
            await page.locator("#stop").click()
            await asyncio.wait_for(cancelled[5].wait(), 10)
            await expect(page.locator("#status")).to_have_text("已停止")
            await expect(page.locator("#start")).to_be_enabled()
            await expect(page.locator("#resume")).to_be_disabled()
            await asyncio.sleep(2)
            assert len(actions) == 2 and controller.desktop.paused and paused
            assert not errors, errors
            await page.screenshot(path=str(output / "stopped.png"), full_page=True)
            result.update(passed=True, actions=actions, backend_requests=len(requests),
                          paused_hold_seconds=2, stopped_hold_seconds=2,
                          both_backend_requests_cancelled=True, same_task_and_thread_resumed=True,
                          untrusted_text_rendered_literally=True, browser_errors=errors)
            await browser.close()
            browser = None
    except Exception as exc:
        result["error"] = repr(exc)
        result["state"] = controller.snapshot()
        if browser:
            with contextlib.suppress(Exception):
                await browser.close()
    finally:
        await controller.close()
        await app_runner.cleanup()
        await backend_runner.cleanup()
        await desktop_runner.cleanup()
        result["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result.get(k) for k in ("passed", "elapsed_seconds", "error")}), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
