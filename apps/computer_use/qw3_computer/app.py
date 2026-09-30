"""Local task UI and lifecycle around Codex app-server and Computer MCP."""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import secrets
import signal
import sys
import time
import uuid

from aiohttp import web

from .bridge import create_app as bridge_app
from .codex import CodexClient, write_config
from .desktop import Desktop, DesktopError, mcp_observation


class Controller:
    def __init__(self, runtime, desktop, backend, port):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.desktop = desktop
        self.backend = backend
        self.port = port
        self.token = secrets.token_hex(32)
        token_file = self.runtime / "mcp-token.txt"
        token_file.write_text(self.token)
        token_file.chmod(0o600)
        self.state_path = self.runtime / "task.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"status": "idle"}
        if self.state["status"] in ("starting", "running", "verifying", "paused"):
            self.state["status"] = "paused"
            self.state["notice"] = "应用已重新启动。请核对当前桌面后继续。"
        self.state.setdefault("messages", [])
        self.state.setdefault("files", [])
        self.ready = False
        self.client = None
        self.consume_task = None
        self.bridge = None
        self.action_count = 0
        self.control_lock = asyncio.Lock()
        self.save()

    def save(self):
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        temp.replace(self.state_path)

    def snapshot(self):
        return self.state | {"ready": self.ready, "observation": self.desktop.latest,
                             "uncertain_actions": self.desktop.unresolved(), "desktop_paused": self.desktop.paused}

    async def start(self):
        await self.desktop.start()
        self.bridge = web.AppRunner(bridge_app(self.backend, trace_dir=self.runtime / "inference"), handler_cancellation=True)
        await self.bridge.setup()
        site = web.TCPSite(self.bridge, "127.0.0.1", 0)
        await site.start()
        bridge_port = site._server.sockets[0].getsockname()[1]
        write_config(self.runtime / "codex", f"http://127.0.0.1:{bridge_port}/v1", sys.executable,
                     [str(Path(__file__).with_name("mcp_proxy.py")), "--endpoint", f"http://127.0.0.1:{self.port}",
                      "--token-file", str(self.runtime / "mcp-token.txt")])
        self.client = CodexClient(self.runtime / "codex")
        await self.client.start()
        self.consume_task = asyncio.create_task(self.consume())
        self.ready = True

    async def consume(self):
        with (self.runtime / "codex-events.jsonl").open("a") as log:
            while True:
                event = await self.client.events.get()
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                log.flush()
                method = event.get("method")
                params = event.get("params", {})
                if method == "app/disconnected":
                    self.ready = False
                    if self.state["status"] in ("starting", "running", "verifying", "paused"):
                        self.state.update(status="paused", notice="Codex 连接已断开。重启应用后可核对桌面并继续。")
                    with contextlib.suppress(DesktopError):
                        await self.desktop.control("pause")
                    self.save()
                    return
                if params.get("threadId") != self.state.get("thread_id"):
                    continue
                if method == "turn/started" and self.state["status"] == "starting":
                    self.state.update(turn_id=params["turn"]["id"], status="running")
                elif method == "item/agentMessage/delta":
                    self.state["draft"] = self.state.get("draft", "") + params["delta"]
                elif method == "item/completed":
                    item = params["item"]
                    if item["type"] == "agentMessage":
                        self.state["messages"].append(item.get("text", ""))
                        self.state["draft"] = ""
                elif method == "turn/completed" and params["turn"]["id"] == self.state.get("turn_id"):
                    status = params["turn"]["status"]
                    if self.state["status"] not in ("paused", "stopped"):
                        self.state["status"] = "verifying"
                    self.state["turn_status"] = status
                    self.state["turn_error"] = params["turn"].get("error")
                    self.state["current_action"] = None
                    try:
                        await self.desktop.control("pause")
                        await self.desktop.observe()
                        self.state["files"] = (await self.desktop.request("GET", "/files"))["files"]
                    except DesktopError as exc:
                        self.state["notice"] = str(exc)
                        self.state["evidence_error"] = str(exc)
                    if self.state["status"] == "verifying":
                        self.state["status"] = "completed" if status == "completed" else "failed"
                    self.state["finished_at"] = time.time()
                self.save()

    async def start_task(self, prompt, acknowledge_uncertain=False):
        async with self.control_lock:
            if not self.ready:
                raise DesktopError("应用尚未就绪。")
            if self.state["status"] in ("starting", "running", "verifying", "paused"):
                raise DesktopError("请先停止或继续当前任务。")
            if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 12000:
                raise DesktopError("请输入 1–12000 字的任务描述。")
            if self.desktop.unresolved():
                await self.desktop.control("resume", acknowledge_uncertain)
                await self.desktop.control("pause")
            self.action_count = 0
            self.state = {"id": uuid.uuid4().hex, "prompt": prompt, "status": "starting",
                          "messages": [], "files": [], "started_at": time.time(), "current_action": None}
            self.desktop.task_id = self.state["id"]
            self.save()
            try:
                await self.desktop.control("resume")
                observation = await self.desktop.observe()
                self.state["thread_id"] = await self.client.new_thread()
                description = json.dumps({k: v for k, v in observation.items() if k != "image"}, ensure_ascii=False)
                text = prompt + "\n\nInitial desktop observation (coordinates use screenshot pixels):\n" + description
                text += "\nSave requested test deliverables inside C:\\Qw3Computer\\outputs."
                self.state["turn_id"] = await self.client.turn(self.state["thread_id"], text,
                                                               ["data:image/png;base64," + observation["image"]])
                if self.state["status"] == "starting":
                    self.state["status"] = "running"
                self.save()
            except BaseException:
                self.state["status"] = "failed"
                self.save()
                with contextlib.suppress(DesktopError):
                    await self.desktop.control("pause")
                raise
            return self.snapshot()

    async def control(self, command, acknowledge_uncertain=False):
        async with self.control_lock:
            if command not in ("pause", "stop", "resume"):
                raise DesktopError("Unknown control command")
            if command == "resume":
                if self.state["status"] != "paused":
                    raise DesktopError("只有暂停的任务可以继续。")
                try:
                    await self.desktop.control("resume", acknowledge_uncertain)
                    observation = await self.desktop.observe()
                    self.desktop.task_id = self.state["id"]
                    # App-server restart requires loading the persisted Codex thread.
                    await self.client.rpc("thread/resume", {"threadId": self.state["thread_id"]})
                    text = "The user resumed the task. Reconcile the current desktop against the task history. "
                    text += "Do not blindly repeat any interrupted action. Current observation:\n"
                    text += json.dumps({k: v for k, v in observation.items() if k != "image"})
                    self.state.update(status="starting", notice="", draft="")
                    self.action_count = 0
                    self.save()
                    self.state["turn_id"] = await self.client.turn(self.state["thread_id"], text,
                                                                   ["data:image/png;base64," + observation["image"]])
                    if self.state["status"] == "starting":
                        self.state["status"] = "running"
                except BaseException:
                    self.state["status"] = "paused"
                    self.save()
                    with contextlib.suppress(DesktopError):
                        await self.desktop.control("pause")
                    raise
            else:
                if command == "pause" and self.state["status"] not in ("starting", "running", "paused"):
                    raise DesktopError("当前没有可暂停的任务。")
                running = self.state["status"] in ("running", "starting")
                self.state["status"] = "paused" if command == "pause" else "stopped"
                self.save()
                await self.desktop.control(command)
                if running and self.state.get("turn_id"):
                    await self.client.interrupt(self.state["thread_id"], self.state["turn_id"])
                await self.desktop.observe()
            self.save()
            return self.snapshot()

    async def tool(self, name, arguments):
        if self.state["status"] == "running":
            self.action_count += 1
            if self.action_count > 100:
                await self.control("pause")
                raise DesktopError("Tool-call limit reached; task paused for review")
        if name == "observe":
            return mcp_observation(await self.desktop.observe())
        if name != "act":
            raise DesktopError("Unknown desktop tool")
        if self.state["status"] != "running":
            raise DesktopError("Task is not running; wait for user resume")
        self.state["current_action"] = arguments.get("action")
        self.save()
        try:
            result = await self.desktop.act(arguments)
            if result.get("no_progress"):
                self.state["status"] = "paused"
                self.state["notice"] = "连续操作没有改变桌面，任务已暂停。"
                await self.client.interrupt(self.state["thread_id"], self.state["turn_id"])
            return mcp_observation(result["observation"], result)
        except BaseException:
            if self.desktop.paused:
                self.state["status"] = "paused"
                self.state["notice"] = "操作结果需要核对，请检查桌面后继续。"
            raise
        finally:
            self.state["current_action"] = None
            self.save()

    async def close(self):
        self.ready = False
        if self.state["status"] in ("starting", "running", "verifying"):
            self.state.update(status="paused", notice="应用已关闭。重启后请核对桌面并继续。")
            self.save()
        if self.consume_task:
            self.consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.consume_task
        if self.desktop.session:
            with contextlib.suppress(DesktopError):
                await self.desktop.control("pause")
        if self.client:
            await self.client.close()
        await self.desktop.close()
        if self.bridge:
            await self.bridge.cleanup()


def web_app(controller):
    @web.middleware
    async def local_only(request, handler):
        if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
            raise web.HTTPForbidden(text="Unexpected host")
        if request.method == "POST" and request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        origin = request.headers.get("Origin")
        if origin and origin != f"http://{request.host}":
            raise web.HTTPForbidden(text="Unexpected origin")
        try:
            return await handler(request)
        except (DesktopError, ValueError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=409)

    app = web.Application(middlewares=[local_only], client_max_size=1024 * 1024)

    async def index(request):
        return web.FileResponse(Path(__file__).with_name("ui.html"))

    async def state(request):
        return web.json_response(controller.snapshot())

    async def screenshot(request):
        latest = controller.desktop.latest
        if not latest:
            raise web.HTTPNotFound()
        return web.FileResponse(controller.desktop.runtime / "screenshots" / (latest["sha256"] + ".png"),
                                headers={"Cache-Control": "no-store"})

    async def start(request):
        body = await request.json()
        return web.json_response(await controller.start_task(body.get("prompt"), body.get("acknowledge_uncertain", False)))

    async def control(request):
        body = await request.json()
        return web.json_response(await controller.control(body["command"], body.get("acknowledge_uncertain", False)))

    async def observe(request):
        if not controller.desktop.paused:
            raise DesktopError("运行期间请使用暂停接管；刷新观察会使正在推理的坐标失效。")
        await controller.desktop.observe()
        return web.json_response(controller.snapshot())

    async def tool(request):
        if not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + controller.token):
            raise web.HTTPUnauthorized()
        body = await request.json()
        try:
            result = await controller.tool(body.get("name"), body.get("arguments", {}))
        except DesktopError as exc:
            result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        return web.json_response(result)

    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/screenshot", screenshot)
    app.router.add_post("/api/task", start)
    app.router.add_post("/api/control", control)
    app.router.add_post("/api/observe", observe)
    app.router.add_post("/internal/tool", tool)
    return app


async def run(args):
    desktop = Desktop(args.desktop, args.desktop_token.read_text().strip(), args.runtime / "desktop")
    controller = Controller(args.runtime, desktop, args.backend, args.port)
    runner = web.AppRunner(web_app(controller), handler_cancellation=True)
    try:
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", args.port).start()
        await controller.start()
        print(f"QW3 Computer ready: http://127.0.0.1:{args.port}", flush=True)
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGTERM, stopped.set)
        await stopped.wait()
    finally:
        await controller.close()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--backend", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--desktop-token", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
