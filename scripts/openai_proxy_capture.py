#!/usr/bin/env python3
"""Transparent OpenAI-compatible HTTP proxy with JSONL request/response capture.

This is intended for fixed-input KVMem debugging. Point a client such as
OpenHands at this proxy, point the proxy at a real qw3 server, then replay the
captured JSONL against baseline/KVMem variants.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve invalid payloads.
        return {"_invalid_json": str(exc), "_raw": raw.decode("utf-8", "replace")}


def text_preview(raw: bytes, limit: int = 4096) -> str:
    text = raw.decode("utf-8", "replace")
    return text if len(text) <= limit else text[:limit] + f"... <truncated {len(text) - limit} chars>"


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "qw3-openai-proxy-capture/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[proxy] " + fmt % args + "\n")

    @property
    def capture_path(self) -> Path:
        return self.server.capture_path  # type: ignore[attr-defined]

    @property
    def upstream(self) -> str:
        return self.server.upstream  # type: ignore[attr-defined]

    @property
    def timeout_s(self) -> float:
        return self.server.timeout_s  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        self.proxy()

    def do_POST(self) -> None:  # noqa: N802
        self.proxy()

    def proxy(self) -> None:
        req_start = time.monotonic()
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        upstream = urlparse(self.upstream)
        path = self.path
        if upstream.path and upstream.path != "/":
            prefix = upstream.path.rstrip("/")
            if not path.startswith(prefix + "/") and path != prefix:
                path = prefix + path

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() != "host"
        }
        headers["Host"] = upstream.netloc

        status = 502
        reason = "Bad Gateway"
        resp_headers: list[tuple[str, str]] = []
        resp_body = b""
        error: str | None = None
        try:
            conn_cls = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
            conn = conn_cls(upstream.hostname, upstream.port, timeout=self.timeout_s)
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            reason = resp.reason
            resp_headers = resp.getheaders()
            resp_body = resp.read()
            conn.close()
        except Exception as exc:  # noqa: BLE001 - report upstream failure to client and capture it.
            error = str(exc)
            resp_body = dump_json({"error": f"proxy upstream error: {error}"}).encode("utf-8")
            resp_headers = [("Content-Type", "application/json")]

        self.send_response(status, reason)
        sent_len = False
        for k, v in resp_headers:
            lk = k.lower()
            if lk in HOP_BY_HOP:
                continue
            if lk == "content-length":
                sent_len = True
                self.send_header(k, str(len(resp_body)))
            else:
                self.send_header(k, v)
        if not sent_len:
            self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

        elapsed_ms = (time.monotonic() - req_start) * 1000.0
        record = {
            "ts": time.time(),
            "elapsed_ms": elapsed_ms,
            "method": self.command,
            "path": self.path,
            "upstream_path": path,
            "request_headers": {k: v for k, v in self.headers.items()},
            "request_body": decode_json(body),
            "request_raw": body.decode("utf-8", "replace"),
            "response_status": status,
            "response_headers": {k: v for k, v in resp_headers},
            "response_body": decode_json(resp_body),
            "response_raw": resp_body.decode("utf-8", "replace"),
            "response_preview": text_preview(resp_body),
            "error": error,
        }
        with self.capture_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.write(
            f"[proxy] {self.command} {self.path} -> {status} "
            f"{elapsed_ms:.1f}ms bytes={len(resp_body)}\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture and forward OpenAI-compatible HTTP traffic.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--upstream", required=True, help="Upstream base URL, e.g. http://127.0.0.1:8090")
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    args.capture.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(args.upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"invalid --upstream: {args.upstream}")

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.upstream = args.upstream.rstrip("/")  # type: ignore[attr-defined]
    server.capture_path = args.capture  # type: ignore[attr-defined]
    server.timeout_s = args.timeout  # type: ignore[attr-defined]
    print(
        f"proxy listening on http://{args.host}:{args.port}, "
        f"upstream={server.upstream}, capture={args.capture}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
