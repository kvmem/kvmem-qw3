#!/usr/bin/env python3
"""Run original OSWorld task/evaluator with Claude Code's native agent loop.

Only a visible-desktop MCP tool is exposed. The local HTTP relay records exact
API timing and forwards native Messages requests without altering history.
"""
import argparse
import base64
import contextlib
import datetime
import gzip
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from request_archive import GroupedRequestWriter
from claude_desktop_mcp import validate_action

ROOT = Path(__file__).resolve().parents[2]
OSWORLD = ROOT / "third_party/OSWorld-V2"
sys.path.insert(0, str(OSWORLD))
SCRIPT = Path(__file__).resolve().parent

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)

def append(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")

def client_stop_reason(trial, deadline, now=None, now_ns=None):
    now = time.monotonic() if now is None else now
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if now > deadline:
        return "timeout"
    for reason, delivered in (("action_limit", trial.limit_delivered_ns),
                              ("episode_end", trial.episode_delivered_ns)):
        if delivered is not None and now_ns - delivered > 5_000_000_000:
            return reason
    return None

def action_code(args):
    validate_action(args)
    action = args["action"]
    coordinate = args.get("coordinate")
    point = ""
    if coordinate is not None:
        if len(coordinate) != 2 or any(not isinstance(x, (float, int)) or not 0 <= x <= 999 for x in coordinate):
            raise ValueError("coordinates must be within 0-999")
        point = f"{min(1919, int(coordinate[0] * 1920 / 999))}, {min(1079, int(coordinate[1] * 1080 / 999))}"
    methods = {"left_click": "click", "double_click": "doubleClick", "right_click": "rightClick", "mouse_move": "moveTo"}
    if action in methods:
        if not point:
            raise ValueError("this action requires a coordinate")
        return f"pyautogui.{methods[action]}({point})"
    if action == "left_click_drag":
        if not point:
            raise ValueError("drag requires a coordinate")
        return f"pyautogui.dragTo({point}, duration=0.5)"
    if action == "type":
        text = args.get("text", "").replace("\r\n", "\n").replace("\r", "\n")
        return "\npyautogui.press('enter')\n".join(f"pyautogui.typewrite({part!r})" for part in text.split("\n"))
    if action == "key":
        keys = args.get("keys", [])
        if not keys or any(not k for k in keys):
            raise ValueError("key requires a non-empty keys[] array of non-empty PyAutoGUI key names")
        return "pyautogui.hotkey(" + ", ".join(repr(k) for k in keys) + ")" if len(keys) > 1 else f"pyautogui.press({keys[0]!r})"
    if action in ("scroll", "hscroll"):
        return f"pyautogui.{action}({int(args.get('pixels', 0))})"
    if action == "wait":
        return "WAIT"
    if action == "terminate":
        return "FAIL" if args.get("status") == "failure" else "DONE"
    if action == "screenshot":
        return "pyautogui.screenshot()"
    raise ValueError("unsupported action")

class Trial:
    def __init__(self, directory, arm, server_port, max_steps):
        self.directory, self.arm, self.server_port = directory, arm, server_port
        self.max_steps = max_steps
        self.step = self.api_id = 0
        self.done = False
        self.lock = threading.Lock()
        self.last_observation_ns = None
        self.limit_delivered_ns = None
        self.episode_ended_ns = None
        self.episode_delivered_ns = None
        self.env = None
        self.writer_error = None
        self.recording_format = os.environ.get("OSWORLD_REQUEST_RECORDING", "gzip")
        if self.recording_format not in ("gzip", "zstd-groups"):
            raise ValueError("OSWORLD_REQUEST_RECORDING must be gzip or zstd-groups")
        self.artifacts = queue.Queue(maxsize=4)
        self.writer = threading.Thread(target=self.write_artifacts, daemon=True)
        self.writer.start()

    def write_artifacts(self):
        archive = None
        while True:
            item = self.artifacts.get()
            try:
                if item is None:
                    if archive: archive.close()
                    return
                if self.writer_error:
                    continue
                path, raw = item
                if self.recording_format == "zstd-groups" and path.name.startswith("request_"):
                    if archive is None: archive = GroupedRequestWriter(self.directory)
                    archive.write(path.name, raw)
                else:
                    with gzip.open(path, "wb", compresslevel=1) as output:
                        output.write(raw)
            except Exception as exc:
                self.writer_error = str(exc)
                logging.exception("Artifact recording failed")
                if item is None: return
            finally:
                self.artifacts.task_done()

    def setup(self, task_id):
        from desktop_env.desktop_env import DesktopEnv
        from task_loader import load_task_from_file
        self.task = load_task_from_file(str(OSWORLD / f"evaluation_examples/task_class/task_{task_id}.py"))
        self.env = DesktopEnv(provider_name="docker", path_to_vm=str(OSWORLD / "docker_vm_data/osworld-v2-ubuntu-x86.qcow2"),
            action_space="pyautogui", headless=True, require_a11y_tree=False,
            enable_proxy=False, client_password=os.environ.get("OSWORLD_CLIENT_PASSWORD", "osworld-public-evaluation"),
            force_disable_vnc=True, force_disable_recording=True, cache_dir=str(self.directory / "env_cache"))
        self.observation = self.env.reset(task_config=self.task, seed=0)
        self.started_at = utc()
        save(self.directory / "task.json", {"id": task_id, "instruction": self.task.instruction, "seed": 0})

    def computer(self, args):
        validate_action(args)
        from mm_agents.qwen35vl_agent import process_image
        with self.lock:
            if self.done:
                return {"content": [{"type": "text", "text": "The desktop episode has ended. Return your final answer."}], "isError": True}
            started_ns = time.monotonic_ns()
            initial = self.step == 0 and self.last_observation_ns is None and args["action"] == "screenshot"
            if not initial:
                if self.step >= self.max_steps:
                    self.done = True
                    return {"content": [{"type": "text", "text": "The 500-action evaluation limit has been reached. Stop now."}], "isError": True}
                code = action_code(args)
                self.step += 1
                self.observation, _, self.done, info = self.env.step(code, pause=0.5)
                if self.step >= self.max_steps:
                    self.done = True
            else:
                code, info = "initial_observation", {}
            png = self.observation["screenshot"]
            screenshot = self.directory / f"step_{self.step:04d}.png"
            screenshot.write_bytes(png)
            encoded = process_image(png)
            finished_ns = time.monotonic_ns()
            row = {"step": self.step, "time": utc(), "action": args, "code": code, "done": self.done,
                   "started_ns": started_ns, "observation_ready_ns": finished_ns,
                   "decision_wait_ms": None if self.last_observation_ns is None else (started_ns - self.last_observation_ns) / 1e6,
                   "tool_wall_ms": (finished_ns - started_ns) / 1e6, "screenshot": screenshot.name,
                   "env_timing": getattr(self.env, "last_step_timing", {}), "info": info}
            append(self.directory / "steps.jsonl", row)
            self.last_observation_ns = finished_ns
            if self.done:
                self.episode_ended_ns = finished_ns
            save(self.directory / "progress.json", {"arm": self.arm, "steps": self.step, "done": self.done, "updated_at": utc()})
            text = "Episode ended; return your final answer." if self.done else f"Desktop observation after action {self.step}. Coordinates use 0-999."
            return {"content": [{"type": "text", "text": text}, {"type": "image", "mimeType": "image/png", "data": encoded}]}

    def computer_delivered(self):
        # Latch the first completed response. Repeated calls after DONE must
        # not postpone the exit deadline, and an in-flight action must finish.
        with self.lock:
            now = time.monotonic_ns()
            if self.done and self.episode_delivered_ns is None:
                self.episode_delivered_ns = now
            if self.step >= self.max_steps and self.limit_delivered_ns is None:
                self.limit_delivered_ns = now

def make_handler(trial):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_):
            pass
        def json_response(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            self.json_response({"status": "ok"})
        def do_POST(self):
            started_ns = time.monotonic_ns()
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body_read_ns = time.monotonic_ns()
            if self.path == "/computer":
                try:
                    self.json_response(trial.computer(json.loads(raw)))
                    trial.computer_delivered()
                except Exception as exc:
                    append(trial.directory / "tool_errors.jsonl", {"time": utc(), "error": str(exc)})
                    self.json_response({"content": [{"type": "text", "text": str(exc)}], "isError": True})
                return
            if self.path.split("?")[0] not in ("/v1/messages", "/v1/messages/count_tokens"):
                self.json_response({"error": "unsupported endpoint"}, 404)
                return
            with trial.lock:
                trial.api_id += 1
                api_id, step = trial.api_id, trial.step
                episode_ended_ns = trial.episode_ended_ns
            record = {"id": api_id, "after_step": step, "path": self.path, "time": utc(), "started_ns": started_ns, "request_bytes": len(raw),
                      "request_body_read_ms": (body_read_ns-started_ns)/1e6,
                      "after_episode_end": episode_ended_ns is not None, "episode_ended_ns": episode_ended_ns}
            connection = http.client.HTTPConnection("127.0.0.1", trial.server_port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
            headers["Content-Type"] = "application/json"
            chunks = []
            try:
                forward_started_ns = time.monotonic_ns()
                connection.request("POST", self.path, body=raw, headers=headers)
                record["request_forward_ms"] = (time.monotonic_ns()-forward_started_ns)/1e6
                response = connection.getresponse()
                record["status"] = response.status
                record["headers_ns"] = time.monotonic_ns()
                self.send_response(response.status)
                content_type = response.getheader("Content-Type", "application/json")
                self.send_header("Content-Type", content_type)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                # Preserve SSE boundaries and forward immediately. Parsing each
                # already-complete data line is used only for timing/usage.
                for line in response:
                    received = time.monotonic_ns()
                    chunks.append(line)
                    self.wfile.write(line)
                    self.wfile.flush()
                    if line.startswith(b"data: "):
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "message_start":
                            record["input_usage"] = event.get("message", {}).get("usage", {})
                        delta = event.get("delta", {})
                        if event.get("type") == "content_block_delta" and any(delta.get(k) for k in ("text", "thinking", "partial_json")):
                            record.setdefault("first_content_ns", received)
                            if delta.get("partial_json"):
                                record.setdefault("first_tool_args_ns", received)
                        if event.get("type") == "message_delta":
                            record["output_usage"] = event.get("usage", {})
                            record["stop_reason"] = delta.get("stop_reason")
                        if event.get("type") == "error":
                            record["stream_error"] = event
            except Exception as exc:
                record["relay_error"] = str(exc)
            finally:
                record["finished_ns"] = time.monotonic_ns()
                if "first_content_ns" in record:
                    record["ttft_ms"] = (record["first_content_ns"] - started_ns) / 1e6
                record["total_ms"] = (record["finished_ns"] - started_ns) / 1e6
                append(trial.directory / "api.jsonl", record)
                connection.close()
                # Persist complete native requests after forwarding, outside
                # the measured first-token path. The bounded queue prevents
                # unbounded recording memory.
                trial.artifacts.put((trial.directory / f"request_{api_id:04d}.json.gz", raw))
                trial.artifacts.put((trial.directory / f"response_{api_id:04d}.sse.gz", b"".join(chunks)))
    return Handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["qw3", "kvmem"], required=True)
    parser.add_argument("--server-port", type=int, default=18080)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--task", default="052")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--workspace", type=Path, help="Empty shared agent directory outside the source repository")
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    trial = Trial(args.output, args.arm, args.server_port, args.max_steps)
    httpd = proc = workspace_temp = None
    result = {"arm": args.arm, "task": args.task, "status": "setting_up", "started_at": utc()}
    save(args.output / "result.json", result)
    try:
        if args.workspace is None:
            workspace_temp = tempfile.TemporaryDirectory(prefix="qw3-osworld-claude-", dir="/tmp")
            workspace = Path(workspace_temp.name)
        else:
            workspace = args.workspace.resolve()
            if workspace == ROOT or ROOT in workspace.parents:
                raise ValueError("Agent workspace must be outside the source repository")
            workspace.mkdir(parents=True, exist_ok=True)
            if any(workspace.iterdir()):
                raise ValueError("Agent workspace must be empty before each arm")
        trial.setup(args.task)
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(trial))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        config = args.output / "claude_config"; config.mkdir(exist_ok=True)
        settings = {"autoUpdatesChannel": "stable", "permissions": {"allow": ["mcp__desktop__computer_use"], "deny": ["Agent", "Task"]}}
        save(args.output / "claude_settings.json", settings)
        save(args.output / "mcp.json", {"mcpServers": {"desktop": {"command": sys.executable, "args": [str(SCRIPT / "claude_desktop_mcp.py"), f"http://127.0.0.1:{args.port}"]}}})
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy") and not k.startswith(("ANTHROPIC_", "CLAUDE_", "DISABLE_COMPACT"))}
        env.update(ANTHROPIC_BASE_URL=f"http://127.0.0.1:{args.port}", ANTHROPIC_API_KEY="local-qw3", CLAUDE_CONFIG_DIR=str(config),
            GIT_CEILING_DIRECTORIES=str(workspace.parent),
            CLAUDE_CODE_MAX_OUTPUT_TOKENS="32768", MAX_THINKING_TOKENS="0", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1",
            CLAUDE_CODE_MAX_CONTEXT_TOKENS="2097152" if args.arm == "kvmem" else "262144")
        if args.arm == "kvmem":
            env["DISABLE_COMPACT"] = "1"
        else:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "262144"
        prompt = (trial.task.instruction + "\n\nOperate only through the desktop MCP computer_use tool. First obtain a screenshot. "
                  "Coordinates use 0-999 in each axis. Every action returns a fresh screenshot. "
                  "Do not access host files or other tools. Work until the task is complete, then call terminate and give a short final answer. "
                  "The evaluation permits at most 500 desktop actions.")
        command = ["claude", "--print", "--verbose", "--output-format", "stream-json", "--include-partial-messages", "--include-hook-events",
                   "--model", "Qwen3.8-27B[1m]", "--tools", "", "--disallowedTools", "Agent,Task", "--allowedTools", "mcp__desktop__computer_use",
                   "--strict-mcp-config", "--mcp-config", str(args.output / "mcp.json"), "--setting-sources", "", "--settings", str(args.output / "claude_settings.json"),
                   "--disable-slash-commands", "--no-chrome", "--debug-file", str(args.output / "claude.debug.log"), "--", prompt]
        save(args.output / "client_manifest.json", {"command": command, "version": subprocess.check_output(["claude", "--version"], text=True).strip(),
            "environment": {k: v for k, v in env.items() if k.startswith(("CLAUDE_", "ANTHROPIC_", "DISABLE_COMPACT", "MAX_THINKING"))},
            "native_history": True, "custom_summary": False, "subagents": False, "seed": 0, "max_actions": args.max_steps,
            "workspace": str(workspace), "git_ceiling": env["GIT_CEILING_DIRECTORIES"], "request_recording": trial.recording_format,
            "driver_sources": {name: hashlib.sha256((SCRIPT/name).read_bytes()).hexdigest() for name in
                ("run_claude_pair.py", "claude_desktop_mcp.py", "request_archive.py")}})
        result["status"] = "running"; save(args.output / "result.json", result)
        with (args.output / "claude.stream.jsonl").open("w") as stdout, (args.output / "claude.stderr.log").open("w") as stderr:
            proc = subprocess.Popen(command, cwd=workspace, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            deadline = time.monotonic() + args.timeout
            while proc.poll() is None:
                if trial.writer_error:
                    raise RuntimeError("Artifact recording failed: " + trial.writer_error)
                # Let the last action and screenshot finish before stopping
                # the client, so evaluation never races an in-flight action.
                reason = client_stop_reason(trial, deadline)
                if reason:
                    result["termination"] = reason
                    result["budget_stop_ns"] = time.monotonic_ns()
                    os.killpg(proc.pid, signal.SIGTERM)
                    break
                time.sleep(1)
            try: code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); code = proc.wait()
        result.update(status="evaluating", client_exit_code=code, steps=trial.step)
        client_results = [json.loads(line) for line in (args.output / "claude.stream.jsonl").read_text().splitlines() if line.startswith('{"type":"result"')]
        result["client_result_errors"] = [x.get("errors", x.get("result", "")) for x in client_results if x.get("is_error")]
        episode_finished = trial.done and trial.episode_delivered_ns is not None
        result.update(episode_ended_ns=trial.episode_ended_ns, episode_delivered_ns=trial.episode_delivered_ns,
                      post_episode_client_error=episode_finished and bool(code or result["client_result_errors"]))
        if trial.step == 0 or (result["client_result_errors"] and not episode_finished):
            raise RuntimeError("Claude Code did not complete a valid desktop trajectory")
        save(args.output / "result.json", result)
        # Finish an already dispatched desktop action before evaluation, and
        # reject any late tool call after the client has exhausted its budget.
        with trial.lock:
            trial.done = True
            result["score"] = trial.env.evaluate()
        result["status"] = "complete" if episode_finished or code == 0 or result.get("termination") in ("action_limit", "timeout") else "client_error"
    except Exception as exc:
        logging.exception("Claude Code trial failed")
        result.update(status="error", error=str(exc))
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        if httpd: httpd.shutdown(); httpd.server_close()
        if trial.env: trial.env.close()
        trial.artifacts.put(None); trial.artifacts.join()
        if trial.writer_error:
            result.update(status="error", error="Artifact recording failed: " + trial.writer_error)
        if workspace_temp: workspace_temp.cleanup()
        result.update(finished_at=utc(), steps=trial.step)
        save(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete" else 1

if __name__ == "__main__":
    raise SystemExit(main())
