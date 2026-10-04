"""Local task UI and lifecycle around Codex app-server and Computer MCP."""
import argparse
import asyncio
import contextlib
import json
import math
from pathlib import Path
import re
import secrets
import signal
import sys
import time
import uuid
from urllib.parse import urlparse

from aiohttp import web

from .bridge import create_app as bridge_app
from .action_validation import executor_rejection_problem
from .codex import CodexClient, write_config
from .desktop import Desktop, DesktopError, DesktopRejected, DesktopValidationError, mcp_observation, model_observation
from .software import Software
from .execution import Execution
from .device_protocol import capabilities
from .device_registry import RegisteredDesktop


class Controller:
    def __init__(self, runtime, desktop, backend, port, backend_token=None, trace_backend=False,
                 tool_call_limit=100, output_policy="outputs", *, backend_timeout_seconds=600,
                 backend_read_timeout_seconds=180, heartbeat_seconds=30, thinking=False, software=None,
                 workflow=False, device_platform="windows", model_output_tokens=4096):
        if type(thinking) is not bool:
            raise ValueError("Thinking mode must be a boolean")
        self.thinking = thinking
        if type(model_output_tokens) is not int or not 256<=model_output_tokens<=32768:
            raise ValueError('Model output limit must be256..32768 tokens')
        self.model_output_tokens=model_output_tokens
        self.software = software
        if type(tool_call_limit) is not int or not 1 <= tool_call_limit <= 10000:
            raise ValueError("Tool-call limit must be an integer between 1 and 10000")
        if output_policy not in ("outputs", "task"):
            raise ValueError("Output policy must be outputs or task")
        waits = (heartbeat_seconds, backend_read_timeout_seconds, backend_timeout_seconds)
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in waits)
                or not 0 < heartbeat_seconds <= backend_read_timeout_seconds <= backend_timeout_seconds <= 14400):
            raise ValueError("Require positive heartbeat <= backend read timeout <= total timeout <= 14400")
        self.backend_timeout_seconds = backend_timeout_seconds
        self.backend_read_timeout_seconds = backend_read_timeout_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.tool_call_limit = tool_call_limit
        self.output_policy = output_policy
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.desktop = desktop
        self.workflow = Execution(desktop, self.runtime) if workflow else None
        self.device_platform = device_platform
        self.device_capabilities = capabilities(device_platform)
        if self.workflow:
            desktop.execution = self.workflow
        self.backend = backend
        self.backend_token = backend_token
        self.model_context_window = 1048576
        self.trace_backend = trace_backend
        self.port = port
        self.token = secrets.token_hex(32)
        token_file = self.runtime / "mcp-token.txt"
        token_file.write_text(self.token)
        token_file.chmod(0o600)
        self.state_path = self.runtime / "task.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "status": "idle", "tool_calls_used": 0, "tool_call_limit": tool_call_limit,
            "output_policy": output_policy, "thinking": thinking}
        self.state.setdefault("thinking", False)
        if self.state["status"] in ("starting", "running", "verifying", "paused"):
            self.state["status"] = "paused"
            self.state["notice"] = "应用已重新启动。请核对当前桌面后继续。"
        self.state.setdefault("messages", [])
        self.state.setdefault("files", [])
        self.ready = False
        self.client = None
        self.consume_task = None
        self.finish_interrupt_task = None
        self.bridge = None
        self.control_lock = asyncio.Lock()
        self.tool_lock = asyncio.Lock()
        self.recovery_clock = time.monotonic
        self.save()

    def save(self):
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        temp.replace(self.state_path)

    def snapshot(self):
        return self.state | {"ready": self.ready, "observation": self.desktop.latest,
                             'model_output_tokens':self.model_output_tokens,
                             'model_context_window':self.model_context_window,
                             "device_capabilities": self.device_capabilities,
                             "devices": self.desktop.catalogue() if isinstance(self.desktop, RegisteredDesktop) else None,
                             "workflow": self.workflow.status() if self.workflow else None,
                             "uncertain_actions": self.desktop.unresolved(), "desktop_paused": self.desktop.paused,
                             "software": self.software.snapshot(self.state.get("id")) if self.software else None}

    def budget_remaining(self):
        used, limit = self.state.get("tool_calls_used"), self.state.get("tool_call_limit")
        if (type(used) is not int or type(limit) is not int or
                not 0 <= used <= limit <= 10000 or limit < 1):
            # Older clients did not persist this count. Guessing it would grant
            # a fresh budget to a partly executed task after an app upgrade.
            raise DesktopError("任务缺少有效的累计调用记录，请停止旧任务后新建任务。")
        return limit - used

    def require_saved_inference_mode(self):
        if self.state.get('model_output_tokens',4096)!=self.model_output_tokens:
            raise DesktopError('当前模型输出预算与任务检查点不一致，请使用原预算继续或新建任务。')
        if self.state.get('model_context_window',1048576)!=self.model_context_window:
            raise DesktopError('当前模型上下文容量与任务检查点不一致，请使用原配置继续或新建任务。')
        identity = self.desktop.selected if isinstance(self.desktop, RegisteredDesktop) else self.desktop.endpoint if hasattr(self.desktop, "endpoint") else None
        if self.state.get("device_identity") is not None and self.state["device_identity"] != identity:
            raise DesktopError("当前设备与任务检查点不一致，请切回任务的原设备。")
        if self.state.get("execution_mode", "legacy") != ("verified" if self.workflow else "legacy"):
            raise DesktopError("该任务的执行流程配置发生变化，请用原流程继续。")
        if self.state.get("device_platform", "windows") != self.device_platform:
            raise DesktopError("任务记录的设备平台与当前选择不同，请切回原设备后继续。")
        if self.state.get("thinking", False) != self.thinking:
            raise DesktopError("当前推理模式与该任务记录不同。请用原模式继续，或停止旧任务后新建任务。")
        selected = str(self.software.root) if self.software else None
        if self.state.get("software_root") != selected:
            raise DesktopError("该任务的系统配置权限或工作目录发生变化，请用原配置继续。")

    async def start(self):
        await self.discover_backend_context()
        observation = await self.desktop.start()
        if isinstance(observation, dict) and observation.get("capabilities"):
            self.device_capabilities = observation["capabilities"]
        self.bridge = web.AppRunner(bridge_app(self.backend, trace_dir=self.runtime / "inference",
                                              backend_token=self.backend_token, trace_backend=self.trace_backend,
                                              backend_timeout_seconds=self.backend_timeout_seconds,
                                              backend_read_timeout_seconds=self.backend_read_timeout_seconds,
                                              heartbeat_seconds=self.heartbeat_seconds,
                                              allow_thinking=self.thinking,
                                              max_tokens=self.model_output_tokens,
                                              allowed_tools={"mcp__computer__observe", "mcp__computer__act"}
                                              | ({"mcp__computer__workflow"} if self.workflow else set())
                                              | ({"mcp__computer__software"} if self.software else set())),
                                    handler_cancellation=True)
        await self.bridge.setup()
        site = web.TCPSite(self.bridge, "127.0.0.1", 0)
        await site.start()
        bridge_port = site._server.sockets[0].getsockname()[1]
        proxy_args = [str(Path(__file__).with_name("mcp_proxy.py")), "--endpoint", f"http://127.0.0.1:{self.port}",
                      "--token-file", str(self.runtime / "mcp-token.txt"), "--platform", self.device_platform]
        if self.software:
            proxy_args.append("--software")
        if self.workflow:
            proxy_args.append("--workflow")
        write_config(self.runtime / "codex", f"http://127.0.0.1:{bridge_port}/v1", sys.executable,
                     proxy_args, thinking=self.thinking, software=self.software is not None, workflow=self.workflow is not None,
                     context_window=self.model_context_window)
        self.client = CodexClient(self.runtime / "codex")
        await self.client.start()
        self.consume_task = asyncio.create_task(self.consume())
        self.ready = True

    async def discover_backend_context(self):
        import aiohttp
        headers={'Authorization':'Bearer '+self.backend_token} if self.backend_token else {}
        try:
            async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(self.backend.rstrip('/')+'/models',headers=headers) as response:
                    if response.status!=200:return
                    body=await response.json()
            models=body.get('data',[])
            if len(models)==1:
                context=models[0].get('context_window',models[0].get('context_length'))
                if type(context) is int and 4096<=context<=16777216:self.model_context_window=context
        except (aiohttp.ClientError,TimeoutError,ValueError,TypeError,AttributeError):
            # Older compatible backends omit metadata; preserve their original
            # configured contract. Do not leak credentials into UI or history.
            pass

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
                    if self.software:
                        await self.software.pause(self.state.get("id"))
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
                    if self.state.get("completion_origin") == "checked_workflow":
                        # The verified tool already ended this task. Interrupting
                        # the remaining narration cannot invalidate its receipt.
                        self.state["turn_status"] = status
                        self.state["turn_error"] = params["turn"].get("error")
                        self.state["current_action"] = None
                        self.save()
                        continue
                    if self.state["status"] not in ("paused", "stopped"):
                        self.state["status"] = "verifying"
                    self.state["turn_status"] = status
                    self.state["turn_error"] = params["turn"].get("error")
                    self.state["current_action"] = None
                    self.state.pop("evidence_error", None)
                    try:
                        try:
                            await self.desktop.control("pause")
                        finally:
                            if self.software:
                                await self.software.pause(self.state.get("id"))
                        if self.software:
                            jobs = self.software.snapshot(self.state.get("id"))["jobs"]
                            self.state["software_verified"] = bool(jobs) and all(
                                j["status"] == "completed" and j.get("verification", {}).get("passed") is True for j in jobs)
                            if not self.state["software_verified"]:
                                self.state.update(status="paused", notice="环境配置尚未通过工具校验，不能认定任务完成。请检查安装进度或错误。")
                        await self.desktop.observe()
                        self.state["files"] = (await self.desktop.request("GET", "/files"))["files"]
                    except DesktopError as exc:
                        self.state["notice"] = str(exc)
                        self.state["evidence_error"] = str(exc)
                        if self.state["status"] == "verifying":
                            self.state["status"] = "paused"
                    if self.state["status"] == "verifying":
                        if status != "completed":
                            self.state["status"] = "failed"
                        elif self.workflow and not (self.software and self.state.get("software_verified")):
                            declared = self.workflow.completion(self.desktop.latest)
                            if declared and declared.get("verified"):
                                self.state.update(status="completed", verification_scope=declared["verification_scope"])
                            else:
                                self.state.update(status="paused", notice="模型已结束，但完成条件尚未验证。请核对结果或继续验证。")
                        else:
                            self.state["status"] = "completed"
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
            self.archive_task()
            self.state = {"id": uuid.uuid4().hex, "prompt": prompt, "status": "starting",
                          "messages": [], "files": [], "started_at": time.time(), "current_action": None,
                          "tool_calls_used": 0, "tool_call_limit": self.tool_call_limit,
                          "output_policy": self.output_policy, "thinking": self.thinking}
            self.state["execution_mode"] = "verified" if self.workflow else "legacy"
            self.state['model_context_window'] = self.model_context_window
            self.state['model_output_tokens']=self.model_output_tokens
            self.state["device_platform"] = self.device_platform
            self.state["device_identity"] = self.desktop.selected if isinstance(self.desktop, RegisteredDesktop) else getattr(self.desktop, "endpoint", None)
            self.state["software_root"] = str(self.software.root) if self.software else None
            self.desktop.task_id = self.state["id"]
            self.save()
            try:
                await self.desktop.control("resume")
                observation = await self.desktop.observe()
                self.state["thread_id"] = await self.client.new_thread()
                description = json.dumps(model_observation(observation), ensure_ascii=False)
                text = prompt + "\n\nInitial desktop observation (prefer click_control for a matching visible control_id; visually estimated actions use coordinate_space=normalized_1000):\n" + description
                text += "\nSelected device capabilities:\n" + json.dumps(self.device_capabilities)
                if self.software:
                    text += "\nAuthorized local software workspace:\n" + json.dumps(self.software.inspect(self.state["id"]), ensure_ascii=False)
                if self.state["output_policy"] == "outputs":
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

    def archive_task(self):
        id = self.state.get("id")
        if isinstance(id, str) and re.fullmatch(r"[a-f0-9]{32}", id):
            folder = self.runtime / "task-history"
            folder.mkdir(exist_ok=True)
            (folder / (id + ".json")).write_text(json.dumps(self.state, ensure_ascii=False, indent=2))

    async def select_device(self, id):
        async with self.control_lock:
            if not isinstance(self.desktop, RegisteredDesktop):
                raise DesktopError("This application was started with one fixed device")
            if self.software:
                raise DesktopError("Stop the software workspace application before changing devices")
            if self.state["status"] in ("starting", "running", "verifying", "paused"):
                raise DesktopError("请先结束当前任务；暂停的任务仍保留设备所有权。")
            self.archive_task()
            observation = await self.desktop.select(id)
            self.ready = False
            if self.consume_task:
                self.consume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError): await self.consume_task
            if self.client: await self.client.close()
            if self.bridge: await self.bridge.cleanup()
            self.device_platform = self.desktop.platform
            self.device_capabilities = observation.get("capabilities", capabilities(self.device_platform))
            self.state = {"status":"idle", "messages":[], "files":[], "thinking":self.thinking,
                          "tool_calls_used":0, "tool_call_limit":self.tool_call_limit}
            self.save()
            await self.start()
            return self.snapshot()

    async def followup(self, task_id, completed_turn_id, event_id, prompt):
        """Continue an explicitly identified completed review on the same history.

        Used by the local owner/harness for newly arrived task information.
        It cannot recover a pause, uncertainty, failed turn or exhausted budget.
        Persist the event before sending it so a lost reply cannot duplicate it.
        """
        async with self.control_lock:
            if not isinstance(event_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", event_id):
                raise DesktopError("Invalid follow-up event identity")
            if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 12000:
                raise DesktopError("Follow-up needs a 1–12000 character prompt")
            if task_id != self.state.get("id"):
                raise DesktopError("Follow-up task identity does not match")
            request = {"event_id": event_id, "completed_turn_id": completed_turn_id, "prompt": prompt}
            previous = next((e for e in self.state.get("followups", []) if e["event_id"] == event_id), None)
            if previous:
                if any(previous.get(k) != v for k, v in request.items()):
                    raise DesktopError("Follow-up identity was reused with different content")
                if previous.get("delivery") != "accepted":
                    raise DesktopError("Follow-up delivery is uncertain; inspect before recovery")
                return self.snapshot()
            if (not self.ready or self.state.get("status") != "completed"
                    or completed_turn_id != self.state.get("turn_id")
                    or self.state.get("turn_status") != "completed"
                    or self.state.get("evidence_error") or not self.desktop.paused
                    or self.desktop.unresolved()):
                raise DesktopError("Follow-up requires the identified completed, verified and paused turn")
            if self.budget_remaining() == 0:
                raise DesktopError("任务调用额度已用尽；后续消息不会重置额度。")
            self.require_saved_inference_mode()
            receipt = request | {"delivery": "pending", "requested_at": time.time(),
                                 "tool_calls_used_before": self.state["tool_calls_used"]}
            self.state.setdefault("followups", []).append(receipt)
            self.state.update(status="starting", notice="", draft="")
            self.state.pop("completion_origin", None)
            self.state.pop("finished_at", None)
            self.state.pop("turn_id", None)
            self.save()
            try:
                await self.desktop.control("resume")
                observation = await self.desktop.observe()
                self.desktop.task_id = task_id
                await self.client.rpc("thread/resume", {"threadId": self.state["thread_id"]})
                description = json.dumps(model_observation(observation), ensure_ascii=False)
                text = prompt + "\nContinue the same task using its existing history. Current desktop observation:\n" + description
                turn_id = await self.client.turn(self.state["thread_id"], text,
                                                ["data:image/png;base64," + observation["image"]])
                self.state["turn_id"] = turn_id
                receipt.update(delivery="accepted", turn_id=turn_id)
                if self.state["status"] == "starting":
                    self.state["status"] = "running"
                self.save()
            except BaseException:
                receipt["delivery"] = "uncertain"
                self.state.update(status="paused", notice="后续消息投递未确认，请核对后恢复。")
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
                if self.budget_remaining() == 0:
                    raise DesktopError("任务调用额度已用尽；暂停或重启不会重置额度。")
                self.require_saved_inference_mode()
                try:
                    await self.desktop.control("resume", acknowledge_uncertain)
                    observation = await self.desktop.observe()
                    self.desktop.task_id = self.state["id"]
                    # App-server restart requires loading the persisted Codex thread.
                    await self.client.rpc("thread/resume", {"threadId": self.state["thread_id"]})
                    text = "The user resumed the task. Reconcile the current desktop against the task history. "
                    text += "Do not blindly repeat any interrupted action. Current observation:\n"
                    text += json.dumps(model_observation(observation))
                    if self.workflow:
                        text += "\nDurable checked/pending steps (do not replay them):\n" + json.dumps(self.workflow.status(), ensure_ascii=False)
                    self.state.update(status="starting", notice="", draft="")
                    # An explicit user resume grants a new recovery attempt,
                    # but never a new cumulative tool-call budget.
                    self.state.pop("gui_recovery", None)
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
                try:
                    await self.desktop.control(command)
                finally:
                    if self.software:
                        await self.software.pause(self.state.get("id"))
                if running and self.state.get("turn_id"):
                    await self.client.interrupt(self.state["thread_id"], self.state["turn_id"])
                await self.desktop.observe()
            self.save()
            return self.snapshot()

    def reset_gui_recovery(self):
        self.state.pop("gui_recovery", None)
        self.save()

    async def note_gui_rejection(self, problem, task_id):
        """Bound repeated known non-deliveries; observe alone cannot reset it."""
        if self.state["status"] != "running" or self.state.get("id") != task_id:
            return {"paused": self.state["status"] == "paused"}
        guard = self.state.setdefault("gui_recovery", {})
        guard.setdefault("first_rejection_at", time.time())
        guard["rejected_calls"] = guard.get("rejected_calls", 0) + 1
        guard["last_error"] = problem
        self.save()
        if guard["rejected_calls"] >= 3:
            guard["paused_reason"] = "repeated_rejection"
            self.state["notice"] = "任务连续三次未能发出有效操作，已暂停等待检查或接管。"
            self.save()
            await self.control("pause")
        return {"rejected_calls": guard["rejected_calls"], "limit": 3,
                "paused": self.state["status"] == "paused"}

    async def note_gui_observation(self, observation, task_id, *, minimum_unchanged_seconds=30):
        if self.state["status"] != "running" or self.state.get("id") != task_id:
            return None
        # An installation job can be making real progress while its desktop
        # is unchanged. Its own bounded job polling and receipts govern it.
        if self.software and any(j["status"] in ("running", "queued")
                for j in self.software.snapshot(task_id)["jobs"]):
            return None
        guard = self.state.setdefault("gui_recovery", {})
        token = json.dumps([observation.get("foreground"),
                            observation.get("progress_sha256", observation.get("sha256"))], sort_keys=True)
        now = self.recovery_clock()
        if guard.get("observation_token") != token:
            guard.update(observation_token=token, unchanged_since_monotonic=now, unchanged_observations=1)
        else:
            guard["unchanged_observations"] += 1
        self.save()
        # A single slow inference is not an observe-only loop. Real changing
        # foreground content resets this window; taskbar clocks do not.
        if guard["unchanged_observations"] >= 3 and now - guard["unchanged_since_monotonic"] >= minimum_unchanged_seconds:
            guard["paused_reason"] = "unchanged_observation_loop"
            self.state["notice"] = "反复查看的窗口持续没有变化，已暂停。请确认是否仍需等待或接管。"
            self.save()
            # Interrupting this model tool can cancel its HTTP request. Persist
            # the cause before pause/interrupt, while running remains running.
            # Acceptance still requires the actual paused device/retired task.
            await self.control("pause")
            return {"status": "paused", "reason": guard["paused_reason"],
                    "unchanged_seconds": now - guard["unchanged_since_monotonic"],
                    "hint": "No desktop input was sent. Repeated unchanged observations were paused for user review."}
        return None

    async def tool(self, name, arguments):
        # Serialize model calls across input, observations and step mutations.
        # User pause/stop uses the separate control lock and remains available.
        async with self.tool_lock:
            return await self._tool(name, arguments)

    async def _tool(self, name, arguments):
        if name not in ("observe", "act") and not (name == "software" and self.software) and not (name == "workflow" and self.workflow):
            raise DesktopError("Unknown desktop tool")
        exhausted = False
        async with self.control_lock:
            if self.state["status"] == "running":
                exhausted = self.budget_remaining() == 0
                if not exhausted:
                    # Reserve before dispatch, including rejected/failed calls.
                    # Counts survive pause, resume and process restart.
                    self.state["tool_calls_used"] += 1
                else:
                    self.state["notice"] = "任务调用额度已用尽，请核对当前结果。"
                self.save()
        if exhausted:
            await self.control("pause")
            raise DesktopError("Tool-call limit reached; task paused for review")
        task_id = self.state.get("id")
        if name == "observe":
            observation = await self.desktop.observe()
            recovery = await self.note_gui_observation(observation, task_id)
            return mcp_observation(observation, recovery) | ({"isError": True} if recovery else {})
        if self.state["status"] != "running":
            raise DesktopError("Task is not running; wait for user resume")
        if name == "workflow":
            result = await self.workflow.invoke(arguments)
            if result.get("status") == "takeover_requested":
                # Keep pending steps and uncertain receipts intact. A pause is
                # acknowledged by the device before it is reported as safe.
                self.state["takeover"] = {"reason": result["reason"], "requested_at": time.time()}
                self.state["notice"] = "任务需要接管，已暂停并保留进度。"
                self.save()
                await self.control("pause")
                result.update(status="paused", task_completed=False, desktop_paused=self.desktop.paused)
            if arguments.get("operation") == "finish" and result.get("verified"):
                result["task_completed"] = await self.complete_checked_workflow(task_id, result)
            if "observation" in result:
                if result.get("status") == "waiting":
                    recovery = await self.note_gui_observation(result["observation"], task_id,
                                                              minimum_unchanged_seconds=120)
                    if recovery:
                        result["recovery"] = recovery
                return mcp_observation(result["observation"], result)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
        if name == "software":
            result = await self.software.invoke(self.state["id"], arguments)
            self.reset_gui_recovery()
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": result.get("status") in ("failed", "uncertain")}
        self.state["current_action"] = arguments.get("action") if isinstance(arguments, dict) else None
        self.save()
        try:
            result = await self.desktop.act(arguments)
            if self.state.get("id") == task_id:
                self.reset_gui_recovery()
            if result.get("no_progress"):
                self.state["status"] = "paused"
                self.state["notice"] = "连续操作没有改变桌面，任务已暂停。"
                await self.client.interrupt(self.state["thread_id"], self.state["turn_id"])
            if result.get("app_focus", {}).get("verified") is False:
                # Launch was delivered exactly once, but its postcondition
                # failed. Return the current screen as a recoverable tool error;
                # do not label delivery uncertain or silently relaunch the app.
                return mcp_observation(result["observation"], result | {
                    "error": "Application launch was dispatched, but target focus was not confirmed. "
                             "Use the resulting screen to recover; do not type until the intended window is foreground."
                }) | {"isError": True}
            return mcp_observation(result["observation"], result)
        except DesktopValidationError as exc:
            recovery = await self.note_gui_rejection(exc.problem, task_id)
            failure = {
                "status": "invalid_arguments", "dispatched": False, "error": exc.problem,
                "recovery": recovery}
            # Target revalidation can capture a fresh observation before a
            # later step check rejects the call. Expose those IDs to the model.
            latest = self.desktop.latest
            if (latest and isinstance(arguments, dict)
                    and arguments.get("observation_id") is not None
                    and arguments.get("observation_id") != latest.get("observation_id")):
                return mcp_observation(self.desktop.with_image(latest), failure) | {"isError": True}
            return {"isError": True, "content": [{"type": "text", "text": json.dumps(failure, ensure_ascii=False)}]}
        except DesktopRejected as exc:
            # The guest explicitly confirms that no input was dispatched.
            # Return a current screen so recovery need not spend a model turn
            # requesting observe. The rejected action ID stays reserved.
            try:
                observation = exc.observation or await self.desktop.observe()
            except DesktopError:
                raise exc
            if exc.evidence_refreshed:
                # A valid action whose old view changed is a synchronization
                # response, not another malformed model call. Keep previous
                # format-error counts; never replay input or erase budgets.
                problem={'code':'evidence_refreshed','path':'$.observation_id','message':str(exc),
                         'hint':'Use this fresh observation to reassess only the next target. No input was sent.'}
                stalled=await self.note_gui_observation(observation,task_id)
                recovery=stalled or {'evidence_refreshed':True,
                    'rejected_calls':self.state.get('gui_recovery',{}).get('rejected_calls',0),'limit':3,
                    'paused':self.state['status']=='paused'}
            else:
                problem = executor_rejection_problem(str(exc))
                recovery = await self.note_gui_rejection(problem, task_id)
            return mcp_observation(observation, {"action_id": arguments.get("action_id"),
                "status": "rejected", "dispatched": False, "error": str(exc), "problem": problem,
                "recovery": recovery}) | {"isError": True}
        except BaseException:
            if self.desktop.paused:
                self.state["status"] = "paused"
                self.state["notice"] = "操作结果需要核对，请检查桌面后继续。"
            raise
        finally:
            self.state["current_action"] = None
            self.save()

    async def complete_checked_workflow(self, task_id, result):
        """End a checked GUI task without requiring another inference round."""
        async with self.control_lock:
            if self.state.get("id") != task_id or self.state["status"] != "running":
                return False
            declared = self.workflow.completion(result["observation"])
            if not declared or not declared.get("verified") or self.desktop.unresolved():
                return False
            if self.software:
                jobs = self.software.snapshot(task_id)["jobs"]
                if not jobs or not all(j["status"] == "completed" and
                        j.get("verification", {}).get("passed") is True for j in jobs):
                    return False
                self.state["software_verified"] = True
            # Stop inputs before recording completion. Failed evidence collection
            # leaves the task running/paused for recovery, never a false pass.
            await self.desktop.control("pause")
            if self.software:
                await self.software.pause(task_id)
            self.state["files"] = (await self.desktop.request("GET", "/files"))["files"]
            self.state.update(status="completed", completion_origin="checked_workflow",
                              verification_scope=declared["verification_scope"], finished_at=time.time(),
                              notice="已验证声明的可见结果，操作已结束。业务结果仍需独立验收。")
            self.save()
            thread, turn = self.state.get("thread_id"), self.state.get("turn_id")
            if thread and turn:
                async def interrupt_narration():
                    # Yield so the verified tool receipt can be returned first.
                    await asyncio.sleep(0.1)
                    if (self.state.get("id") == task_id and self.state.get("turn_id") == turn
                            and self.state.get("completion_origin") == "checked_workflow"):
                        try:
                            await self.client.interrupt(thread, turn)
                        except Exception:
                            self.state["completion_interrupt_error"] = "Remaining model narration could not be interrupted"
                            self.save()
                self.finish_interrupt_task = asyncio.create_task(interrupt_narration())
            return True

    async def close(self):
        self.ready = False
        if self.state["status"] in ("starting", "running", "verifying"):
            self.state.update(status="paused", notice="应用已关闭。重启后请核对桌面并继续。")
            self.save()
        if self.consume_task:
            self.consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.consume_task
        if self.finish_interrupt_task:
            self.finish_interrupt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.finish_interrupt_task
        if self.desktop.session:
            with contextlib.suppress(DesktopError):
                await self.desktop.control("pause")
        if self.client:
            await self.client.close()
        if self.software:
            await self.software.close()
        await self.desktop.close()
        if self.workflow:
            self.workflow.close()
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

    async def select_device(request):
        body = await request.json()
        return web.json_response(await controller.select_device(body.get("device_id")))

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

    async def followup(request):
        body = await request.json()
        return web.json_response(await controller.followup(body.get("task_id"), body.get("completed_turn_id"),
                                                          body.get("event_id"), body.get("prompt")))

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
    app.router.add_post("/api/device", select_device)
    app.router.add_get("/api/screenshot", screenshot)
    app.router.add_post("/api/task", start)
    app.router.add_post("/api/control", control)
    app.router.add_post("/api/followup", followup)
    app.router.add_post("/api/observe", observe)
    app.router.add_post("/internal/tool", tool)
    return app


async def run(args):
    software_root = getattr(args, "software_root", None)
    if software_root and (sys.platform != "win32" or urlparse(args.desktop).hostname not in {"localhost", "127.0.0.1"}):
        raise ValueError("Software mode requires the native Windows client with its local desktop executor")
    if getattr(args, "device_registry", None):
        if software_root: raise ValueError("Registry mode and software workspace mode are separate permissions")
        desktop = RegisteredDesktop(args.device_registry, args.selected_device, args.runtime / "devices")
        args.device_platform = desktop.platform
    else:
        if args.desktop_token is None: raise ValueError("A desktop token or a device registry is required")
        desktop = Desktop(args.desktop, args.desktop_token.read_text().strip(), args.runtime / "desktop")
    token = args.backend_token.read_text(encoding="utf-8-sig").strip() if args.backend_token else None
    if args.backend_token and not token:
        raise ValueError("Backend token file is empty")
    controller = Controller(args.runtime, desktop, args.backend, args.port, backend_token=token,
                            trace_backend=args.trace_backend, tool_call_limit=args.tool_call_limit,
                            output_policy=args.output_policy, backend_timeout_seconds=args.backend_timeout,
                            backend_read_timeout_seconds=args.backend_read_timeout,
                            thinking=getattr(args, "thinking", False),
                            software=Software(software_root) if software_root else None,
                            workflow=getattr(args, "execution_mode", "legacy") == "verified",
                            model_output_tokens=getattr(args,'model_output_tokens',4096),
                            device_platform=getattr(args, "device_platform", "windows"))
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
    parser.add_argument("--backend-token", type=Path, help="Optional private VM relay token file")
    parser.add_argument("--backend-timeout", type=int, default=600, help="Total model request deadline in seconds")
    parser.add_argument("--backend-read-timeout", type=int, default=180, help="Model socket read deadline in seconds")
    parser.add_argument("--desktop", default="http://127.0.0.1:18082")
    parser.add_argument("--desktop-token", type=Path)
    parser.add_argument("--device-registry", type=Path, help="Owner-created local device catalogue; includes token file references")
    parser.add_argument("--selected-device", help="Initial registry device identity")
    parser.add_argument("--device-platform", choices=("windows", "linux", "android"), default="windows")
    parser.add_argument("--execution-mode", choices=("legacy", "verified"), default="legacy",
                        help="Verified mode adds conditions, bounded waits and durable steps; legacy retains old execution")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace-backend", action="store_true", help="Tag QW3 requests for retrieval diagnostics")
    parser.add_argument("--thinking", action="store_true", help="Enable local thinking for new tasks; defaults to off")
    parser.add_argument('--model-output-tokens',type=int,default=4096,
                        help='Bound combined reasoning and output per inference request; saved in the task checkpoint')
    parser.add_argument("--software-root", type=Path,
                        help="Explicitly enable reviewed native Windows installation workflows in this local workspace")
    parser.add_argument("--tool-call-limit", type=int, default=100,
                        help="Lifetime observe/act budget for new tasks; resume keeps the saved budget")
    parser.add_argument("--output-policy", choices=("outputs", "task"), default="outputs",
                        help="Use the legacy outputs folder hint or only the task's own path instructions")
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
