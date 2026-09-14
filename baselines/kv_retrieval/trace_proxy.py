"""Common transparent Messages timing/archive proxy for all pilot methods."""
import gzip
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

from kv_baselines.frontend import meaningful_content, action_content


class TraceProxy(ThreadingHTTPServer):
    def __init__(self, address, backend, out):
        self.backend = urlsplit(backend)
        if self.backend.scheme != "http":
            raise ValueError("local HTTP backend required")
        self.out = Path(out)
        (self.out / "inputs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        super().__init__(address, ProxyHandler)

    def record(self, row):
        with self.lock, (self.out / "requests.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def forward(self):
        start = time.perf_counter()
        row = {"id": uuid.uuid4().hex, "received_at": time.time(), "path": self.path,
               "method": self.command, "api_first_content_s": None, "api_first_action_s": None,
               "status": "running"}
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in {"host", "connection", "accept-encoding"}}
        connection = http.client.HTTPConnection(self.server.backend.hostname,
            self.server.backend.port, timeout=3600)
        sent_headers = False
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            row["http_status"] = response.status
            self.send_response(response.status)
            for k, v in response.getheaders():
                if k.lower() not in {"connection", "transfer-encoding", "content-length", "server", "date"}:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            sent_headers = True
            is_sse = "text/event-stream" in response.getheader("Content-Type", "")
            # SSE is line-delimited. Forward each line immediately; no buffering
            # until an event or tool call finishes. Non-stream data is unmodified.
            while True:
                chunk = response.readline() if is_sse else response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if is_sse and chunk.startswith(b"data: "):
                    try:
                        event = json.loads(chunk[6:])
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if row["api_first_content_s"] is None and meaningful_content(event):
                        row["api_first_content_s"] = time.perf_counter() - start
                    if row["api_first_action_s"] is None and action_content(event):
                        row["api_first_action_s"] = time.perf_counter() - start
                    if event.get("type") == "error":
                        row["stream_error"] = event
                    if event.get("type") == "message_stop":
                        row["message_stop"] = True
                    if event.get("type") == "message_start":
                        row["input_usage"] = event.get("message", {}).get("usage")
                        row["message_id"] = event.get("message", {}).get("id")
                    if (event.get("type") == "content_block_start" and
                            event.get("content_block", {}).get("type") == "tool_use"):
                        row.setdefault("first_tool_call_s", time.perf_counter() - start)
                        row.setdefault("tool_calls", []).append(event["content_block"])
                    if event.get("type") == "message_delta":
                        row["output_usage"] = event.get("usage")
                        row["stop_reason"] = event.get("delta", {}).get("stop_reason")
            row["status"] = "completed" if (response.status < 400 and "stream_error" not in row
                and (not is_sse or row.get("message_stop"))) else "failed"
        except Exception as error:
            row.update(status="failed", error=repr(error))
            if not sent_headers:
                self.send_error(502, "local baseline backend unavailable")
        finally:
            row["request_wall_s"] = time.perf_counter() - start
            connection.close()
            self.close_connection = True
            if body:
                # Archive after measured streaming has finished. Keep original
                # bytes, including compaction requests, for offline classification.
                row["body_sha256"] = hashlib.sha256(body).hexdigest()
                row["body_path"] = "inputs/" + row["id"] + ".json.gz"
                (self.server.out / row["body_path"]).write_bytes(gzip.compress(body, compresslevel=1))
            self.server.record(row)

    do_GET = do_POST = do_HEAD = forward

    def log_message(self, *_):
        pass
