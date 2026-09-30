#!/usr/bin/env python3
"""Read-only live view of a thinking-compaction campaign (stdlib only).

Reads existing artifacts without touching the frozen experiment or inference
service. Rates count original target tokens, including discarded thinking;
draft proposals and summary-child tokens are not target output.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from thinking_campaign_metrics import RequestParser


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def short(text, limit=12000):
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else "…（仅显示末尾内容）\n" + text[-limit:]


class Tail:
    """Consume only complete new lines; reset consumers on rotation/truncation."""
    def __init__(self, path):
        self.path = path
        self.identity = None
        self.offset = 0
        self.partial = b""

    def poll(self):
        try:
            st = self.path.stat()
            identity = (st.st_dev, st.st_ino)
            reset = identity != self.identity or st.st_size < self.offset
            if reset:
                self.identity, self.offset, self.partial = identity, 0, b""
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
            lines = (self.partial + data).split(b"\n")
            self.partial = lines.pop()
            return reset, [line.decode(errors="replace") for line in lines]
        except OSError:
            return False, []


def request_metrics(row):
    gen = row.get("generation", {})
    events = row.get("compactions", [])
    removed = sum(e.get("original_tokens", 0) - e.get("retained_tokens", 0)
                  for e in events if e.get("status") == "compacted")
    original = gen.get("decoded", 0) + removed
    overhead = sum(sum(e.get(k, 0) for k in ("summary_ms", "rollback_ms", "replay_ms"))
                   for e in events) / 1000
    return {
        "index": row.get("request_index"), "original_tokens": original,
        "retained_tokens": gen.get("decoded", 0), "removed_tokens": removed,
        "decode_s": gen.get("decode_sec", 0),
        "wall_s": row.get("accounting", {}).get("total_ms", 0) / 1000,
        "overhead_s": overhead,
        "summary_s": sum(e.get("summary_ms", 0) for e in events) / 1000,
        "compactions": sum(e.get("status") == "compacted" for e in events),
        "context": row.get("output_context_tokens"),
        "status": row.get("accounting", {}).get("terminal_status"),
        "events": events,
    }


def rates(rows):
    keys = ("original_tokens", "retained_tokens", "removed_tokens", "decode_s",
            "wall_s", "overhead_s", "summary_s", "compactions")
    totals = {key: sum(r.get(key, 0) for r in rows) for key in keys}
    n = totals["original_tokens"]
    for name, denominator in (("decode_tps", totals["decode_s"]),
                              ("decode_only_tps", totals["decode_s"] - totals["overhead_s"]),
                              ("request_tps", totals["wall_s"])):
        totals[name] = n / denominator if denominator > 0 else None
    totals["requests"] = len(rows)
    return totals


class StreamView:
    def __init__(self):
        self.items = deque(maxlen=80)
        self.blocks = {}
        self.active = False
        self.message_id = None
        self.materialized = set()
        self.request_count = 0
        self.prompt_tokens = None
        self.active_chars = 0

    def add_block(self, block, label=None):
        kind = block.get("type", "text")
        if kind == "thinking":
            text, label = block.get("thinking", ""), label or "Thinking / 摘要"
        elif kind == "tool_use":
            text, label = block.get("input", {}), label or block.get("name", "工具调用")
            if isinstance(text, dict):
                text = "\n\n".join(f"{key}:\n{short(value)}" for key, value in text.items())
        elif kind == "tool_result":
            text, label = block.get("content", ""), "工具结果"
        else:
            text, label = block.get("text", ""), label or "Assistant"
        if text:
            self.items.append({"kind": kind, "label": label,
                               "text": short(text), "request": self.request_count})

    def line(self, raw):
        try:
            row = json.loads(raw)
        except ValueError:
            return
        event = row.get("event") or {}
        kind = event.get("type")
        if kind == "message_start":
            self.active = True
            self.blocks = {}
            self.active_chars = 0
            self.request_count += 1
            self.message_id = event.get("message", {}).get("id")
            self.prompt_tokens = event.get("message", {}).get("usage", {}).get("input_tokens")
        elif kind == "content_block_start":
            self.blocks[event.get("index", 0)] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta":
            block = self.blocks.setdefault(event.get("index", 0), {})
            delta = event.get("delta") or {}
            for key in ("thinking", "text", "partial_json"):
                if isinstance(delta.get(key), str):
                    self.active_chars += len(delta[key])
                    block[key] = short(block.get(key, "") + delta[key])
        elif kind == "message_stop":
            self.active = False
            # Claude emits authoritative assistant blocks; some clients emit
            # only stream events, so retain blocks until the next message.
        message = row.get("message") or {}
        if row.get("type") == "assistant":
            for block in message.get("content", []):
                if isinstance(block, dict):
                    # Claude can emit one assistant record per content block.
                    signature = (message.get("id"), hashlib.sha256(
                        json.dumps(block, sort_keys=True).encode()).digest())
                    if signature not in self.materialized:
                        self.add_block(block)
                        self.materialized.add(signature)
            self.blocks = {}
        elif row.get("type") == "user":
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    self.add_block(block)
        elif row.get("type") == "result":
            self.active = False
            self.items.append({"kind": "result", "label": "Agent 结束",
                               "text": short(row.get("result") or row.get("subtype")),
                               "request": self.request_count})

    def transcript(self):
        items = list(self.items)
        for block in self.blocks.values():
            text = block.get("thinking") or block.get("text") or block.get("partial_json")
            if text:
                items.append({"kind": block.get("type", "text"),
                              "label": "接收中" if self.active else "已接收",
                              "text": short(text), "request": self.request_count})
        return items[-80:]


class Attempt:
    def __init__(self, root, artifact, server):
        self.root, self.artifact, self.server = root, artifact, server
        self.tails = {"server": Tail(server), "stream": Tail(artifact / "claude.stream.jsonl"),
                      "debug": Tail(artifact / "claude.debug.log")}
        self.parser = RequestParser()
        self.stream = StreamView()
        self.logs = deque(maxlen=70)
        self.summaries = deque(maxlen=50)
        self.request_started = None

    def poll(self):
        reset, lines = self.tails["server"].poll()
        if reset:
            self.parser = RequestParser()
            self.logs.clear()
            self.summaries.clear()
        for line in lines:
            try:
                self.parser.line(line)
                if "[thinking-summary-output]" in line:
                    output = json.loads(line.split("[thinking-summary-output]", 1)[1])
                    output["request"] = len(self.parser.rows) + 1
                    output["text"] = short(output.get("text", ""))
                    self.summaries.append(output)
            except (ValueError, TypeError):
                pass
            self.logs.append(short(line, 2400))
        reset, lines = self.tails["stream"].poll()
        if reset:
            self.stream = StreamView()
        for line in lines:
            self.stream.line(line)
        reset, lines = self.tails["debug"].poll()
        if reset:
            self.request_started = None
        for line in lines:
            if "[API REQUEST] /v1/messages source=sdk" in line:
                self.request_started = timestamp(line.split(" ", 1)[0])
        return [request_metrics(row) for row in self.parser.rows if "generation" in row]


class Dashboard:
    def __init__(self, root):
        self.root = root.resolve()
        self.attempts = {}
        self.lock = threading.Lock()

    def within(self, value):
        path = Path(value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact is outside the experiment")
        return path

    def snapshot(self, selected=None):
        with self.lock:
            return self._snapshot(selected)

    def _snapshot(self, selected):
        now = time.time()
        manifest = read_json(self.root / "manifest.json", {})
        state = read_json(self.root / "state.json", {})
        live = read_json(self.root / "live.json", {})
        results = read_json(self.root / "results.json", [])
        tasks = manifest.get("tasks", [])
        task_ids = [t["task_id"] for t in tasks]
        selected = selected or state.get("task") or (task_ids[0] if task_ids else None)
        if selected not in task_ids:
            raise ValueError("unknown task")
        all_rows, selected_rows, selected_attempts = [], [], []
        task_rows = {t: [] for t in task_ids}
        task_attempts = {}
        for root in sorted(self.root.glob("round-*/*/attempt-*")):
            task = root.parent.name
            if task not in task_rows:
                continue
            service = read_json(root / "service.json", {})
            server = service.get("log")
            artifacts = sorted((root / "runs" / task).glob("*"))
            if not server or not artifacts:
                continue
            artifact = self.within(artifacts[-1])
            if root not in self.attempts:
                self.attempts[root] = Attempt(root, artifact, self.within(server))
            attempt = self.attempts[root]
            rows = attempt.poll()
            task_attempts[task] = attempt
            for row in rows:
                row.update(task=task, attempt=root.name, round=root.parent.parent.name)
            all_rows.extend(rows)
            task_rows[task].extend(rows)
            if task == selected:
                selected_rows.extend(rows)
                selected_attempts.append(attempt)
        attempt = selected_attempts[-1] if selected_attempts else None
        artifact_status = read_json(attempt.artifact / "status.json", {}) if attempt else {}
        active_task = state.get("task") == selected and state.get("phase") == "running_trial"
        process_alive = bool(state.get("supervisor_pid") and Path(f'/proc/{state["supervisor_pid"]}').exists())
        active = bool(active_task and process_alive and attempt and attempt.stream.active)
        elapsed = max(0, now - attempt.request_started) if active and attempt.request_started else None
        resource = {}
        if attempt:
            try:
                with (attempt.root / "resources.jsonl").open("rb") as stream:
                    stream.seek(max(0, (attempt.root / "resources.jsonl").stat().st_size - 16384))
                    for line in reversed(stream.read().splitlines()):
                        try:
                            resource = json.loads(line)
                            break
                        except ValueError:
                            continue
            except OSError:
                pass
        queue = []
        for index, task in enumerate(tasks, 1):
            task_id = task["task_id"]
            finished = [r for r in results if r.get("task") == task_id and r.get("valid")]
            expected = sum(s.get("task") == task_id for s in manifest.get("schedule", [])) or 1
            final = finished[-1] if finished else {}
            complete = len(finished) >= expected
            phase = "queued"
            if complete:
                phase = "passed" if all(r.get("passed") for r in finished) else "failed"
            elif state.get("task") == task_id:
                phase = state.get("phase", "queued")
                if phase == "running_trial":
                    phase = "running" if process_alive else "offline"
            current_context, elapsed_s = None, None
            if not complete:
                rows = task_rows[task_id]
                current_context = rows[-1].get("context") if rows else None
                current_attempt = task_attempts.get(task_id)
                if current_attempt and current_attempt.stream.active:
                    current_context = current_attempt.stream.prompt_tokens or current_context
                if state.get("task") == task_id and live.get("task") == task_id:
                    elapsed_s = live.get("elapsed_sec")
                    # Interpolate only while the same trial is running. Frozen
                    # results and paused/failed trials must not keep ticking.
                    sampled_at = timestamp(live.get("updated_at"))
                    if elapsed_s is not None and sampled_at and phase == "running":
                        elapsed_s += max(0, now - sampled_at)
            queue.append({"index": index, "task": task_id, "title": task.get("title"),
                          "phase": phase, "completed": len(finished), "total": expected,
                          "requests": len(task_rows[task_id]),
                          "reward": final.get("reward"),
                          "final_context_tokens": final.get("final_context_tokens") if complete else None,
                          "current_context_tokens": current_context,
                          "total_elapsed_sec": final.get("total_elapsed_sec") if complete else None,
                          "elapsed_sec": elapsed_s,
                          "agent_elapsed_sec": final.get("agent_elapsed_sec") if complete else None,
                          "f2p_passed": final.get("f2p_passed"), "f2p_total": final.get("f2p_total"),
                          "p2p_passed": final.get("p2p_passed"), "p2p_total": final.get("p2p_total")})
        last_modified = modified(attempt.artifact / "claude.stream.jsonl") if attempt else None
        running_tokens, running_wall = 0, 0
        for row in selected_rows:
            running_tokens += row["original_tokens"]
            running_wall += row["wall_s"]
            row["cumulative_request_tps"] = running_tokens / running_wall if running_wall > 0 else None
        return {
            "now": now, "experiment": self.root.name,
            "config": {k: manifest.get(k) for k in ("model", "seeds", "speculative_method",
                "summary_enabled", "summary_context", "summary_min_tokens", "selection_budget", "generation_reserve")},
            "phase": state.get("phase"), "supervisor_alive": process_alive,
            "heartbeat_age_s": now - timestamp(live.get("updated_at")) if timestamp(live.get("updated_at")) else None,
            "completed": sum(bool(r.get("valid")) for r in results),
            "passed": sum(bool(r.get("valid") and r.get("passed")) for r in results),
            "total": manifest.get("total_trials", len(task_ids)), "tasks": queue,
            "totals": rates(all_rows), "selected_totals": rates(selected_rows),
            "selected": selected, "active_task": state.get("task"),
            "current": {"active": active, "request": attempt.stream.request_count if attempt else None,
                "elapsed_s": elapsed, "phase": artifact_status.get("state"),
                "prompt_tokens": attempt.stream.prompt_tokens if attempt else None,
                "received_chars": attempt.stream.active_chars if attempt else 0,
                "output_age_s": now - last_modified if last_modified else None,
                "instant_tps": None,
                "instant_note": ("当前服务缓冲 thinking；逐 token 输出和瞬时吞吐不可用。收到输出即更新。"
                    if manifest.get("summary_enabled", True) else
                    "Thinking 摘要已关闭。吞吐按已完成请求统计；输出随客户端日志更新。")},
            "requests": selected_rows[-160:],
            "transcript": attempt.stream.transcript() if attempt else [],
            "summaries": list(attempt.summaries) if attempt else [],
            "server_log": list(attempt.logs) if attempt else [],
            "resources": resource,
        }


def make_handler(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    body = Path(__file__).with_suffix(".html").read_bytes()
                    mime = "text/html; charset=utf-8"
                elif url.path == "/api/state":
                    task = parse_qs(url.query).get("task", [None])[0]
                    body = json.dumps(dashboard.snapshot(task), ensure_ascii=False, allow_nan=False).encode()
                    mime = "application/json; charset=utf-8"
                elif url.path == "/report":
                    body = (dashboard.root / "report.md").read_bytes()
                    mime = "text/plain; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'")
                self.end_headers()
                self.wfile.write(body)
            except ValueError as exc:
                self.send_error(400, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *args):
            if len(args) < 2 or str(args[1]) != "200":
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8322)
    args = parser.parse_args()
    if not (args.experiment / "manifest.json").is_file():
        parser.error("experiment must contain manifest.json")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Dashboard(args.experiment)))
    print(f"Thinking dashboard: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
