#!/usr/bin/env python3
"""Tiny OpenAI-compatible capture server for debugging clients.

It logs every request to JSONL and returns deterministic dummy responses for
the common OpenAI endpoints used by coding clients:

  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/responses

The goal is not model quality. It is to see exactly what the client sends:
path, headers, body, stream flag, max_tokens/max_completion_tokens, and which
API shape it expects.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


def now_s() -> int:
    return int(time.time())


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def extract_prompt(data: Dict[str, Any]) -> str:
    if isinstance(data.get("prompt"), str):
        return data["prompt"]
    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item.get("content"), str):
                            parts.append(item["content"])
                return "\n".join(parts)
    inp = data.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list) and inp:
        return dump_json(inp)
    return ""


class CaptureHandler(BaseHTTPRequestHandler):
    server_version = "qw3-opencode-capture/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[capture] " + fmt % args + "\n")

    @property
    def capture_path(self) -> Path:
        return self.server.capture_path  # type: ignore[attr-defined]

    def write_json(self, status: int, obj: Any) -> None:
        body = dump_json(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_sse_event(self, obj: Any, event: Optional[str] = None) -> None:
        if event:
            self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(b"data: ")
        self.wfile.write(dump_json(obj).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:  # noqa: BLE001
            data = {"_invalid_json": str(exc), "_raw": raw.decode("utf-8", "replace")}
        record = {
            "ts": time.time(),
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": data,
            "raw_body": raw.decode("utf-8", "replace"),
        }
        with self.capture_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.write(
            "[capture] POST "
            f"{self.path} stream={data.get('stream')} "
            f"max_tokens={data.get('max_tokens')} "
            f"max_completion_tokens={data.get('max_completion_tokens')} "
            f"model={data.get('model')}\n"
        )
        return data if isinstance(data, dict) else {"_body": data}

    def do_GET(self) -> None:  # noqa: N802
        record = {
            "ts": time.time(),
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
        }
        with self.capture_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.path == "/health":
            self.write_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self.write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qw3-capture",
                            "object": "model",
                            "created": now_s(),
                            "owned_by": "qw3",
                        }
                    ],
                },
            )
            return
        self.write_json(404, {"error": f"unknown GET path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        data = self.read_json_body()
        if self.path == "/v1/chat/completions":
            self.handle_chat(data)
            return
        if self.path == "/v1/completions":
            self.handle_completion(data)
            return
        if self.path == "/v1/responses":
            self.handle_responses(data)
            return
        self.write_json(
            200,
            {
                "ok": True,
                "note": "captured unsupported POST path",
                "path": self.path,
                "body_keys": sorted(data.keys()),
            },
        )

    def handle_chat(self, data: Dict[str, Any]) -> None:
        prompt = extract_prompt(data)
        text = f"capture-ok: received {len(prompt)} chars"
        model = data.get("model") or "qw3-capture"
        created = now_s()
        req_id = f"chatcmpl-capture-{created}"
        if data.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.write_sse_event(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for piece in [text[:12], text[12:]]:
                if not piece:
                    continue
                self.write_sse_event(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            self.wfile.write(
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.write_json(
            200,
            {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def handle_completion(self, data: Dict[str, Any]) -> None:
        prompt = extract_prompt(data)
        text = f"capture-ok: completion received {len(prompt)} chars"
        model = data.get("model") or "qw3-capture"
        created = now_s()
        if data.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.write_sse_event(
                {
                    "id": f"cmpl-capture-{created}",
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "text": text, "logprobs": None, "finish_reason": None}
                    ],
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.write_json(
            200,
            {
                "id": f"cmpl-capture-{created}",
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "text": text, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def handle_responses(self, data: Dict[str, Any]) -> None:
        prompt = extract_prompt(data)
        text = f"capture-ok: responses received {len(prompt)} chars"
        model = data.get("model") or "qw3-capture"
        created = now_s()
        resp_id = f"resp_capture_{created}"
        if data.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.write_sse_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "created_at": created,
                        "model": model,
                        "status": "in_progress",
                    },
                },
                event="response.created",
            )
            self.write_sse_event(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_capture",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                },
                event="response.output_text.delta",
            )
            self.write_sse_event(
                {
                    "type": "response.completed",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "created_at": created,
                        "model": model,
                        "status": "completed",
                    },
                },
                event="response.completed",
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.write_json(
            200,
            {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model,
                "status": "completed",
                "output_text": text,
                "output": [
                    {
                        "id": "msg_capture",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--capture", default="/tmp/opencode_capture.jsonl")
    args = ap.parse_args()

    capture = Path(args.capture)
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_text("", encoding="utf-8")

    server = ThreadingHTTPServer((args.host, args.port), CaptureHandler)
    server.capture_path = capture  # type: ignore[attr-defined]
    print(f"capture server listening on http://{args.host}:{args.port}", flush=True)
    print(f"writing requests to {capture}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
