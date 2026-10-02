"""Expose the loopback model only to the owned Windows container, with a token.

This host-only VM test relay binds the Docker bridge gateway, never 0.0.0.0.
It is not a public or LAN model service. Stop it after native client tests.
"""
import argparse
import asyncio
import base64
import hashlib
import re
import time
import ipaddress
import json
from pathlib import Path
import secrets
import subprocess

import aiohttp
from aiohttp import web


def request_evidence(raw):
    """Inspect only input identity; do not retain prompts, pixels or credentials."""
    evidence = {"wire_sha256": hashlib.sha256(raw).hexdigest(), "body_bytes": len(raw)}
    try:
        body = json.loads(raw)
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        evidence["canonical_request_sha256"] = hashlib.sha256(canonical).hexdigest()
        tag = body.get("kvmem_trace_tag")
        if isinstance(tag, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", tag):
            evidence["backend_trace_tag"] = tag
        evidence["latest_image_sha256"] = None
        for message in reversed(body.get("messages", [])):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in reversed(content):
                if part.get("type") != "image_url":
                    continue
                image = part.get("image_url", {})
                url = image.get("url") if isinstance(image, dict) else image
                if isinstance(url, str) and url.startswith("data:image/png;base64,"):
                    evidence["latest_image_sha256"] = hashlib.sha256(base64.b64decode(url.split(",", 1)[1], validate=True)).hexdigest()
                    return evidence
        return evidence
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        # Invalid requests still reach the same backend validation as before.
        evidence["inspection_error"] = type(exc).__name__
        return evidence


def create_app(backend, allowed_peer, token, *, receipt_path=None,
               backend_timeout_seconds=600, backend_read_timeout_seconds=180):
    if not 0 < backend_read_timeout_seconds <= backend_timeout_seconds:
        raise ValueError("Require positive read timeout <= total timeout")
    @web.middleware
    async def authorize(request, handler):
        if request.remote != allowed_peer or not secrets.compare_digest(
                request.headers.get("Authorization", ""), "Bearer " + token):
            raise web.HTTPUnauthorized()
        return await handler(request)

    app = web.Application(middlewares=[authorize], client_max_size=64 * 1024 * 1024)
    key = web.AppKey("upstream", aiohttp.ClientSession)

    async def session(app):
        async with aiohttp.ClientSession(trust_env=False,
                                         timeout=aiohttp.ClientTimeout(total=backend_timeout_seconds,
                                             sock_read=backend_read_timeout_seconds)) as client:
            app[key] = client
            yield
    app.cleanup_ctx.append(session)

    receipt_path = Path(receipt_path) if receipt_path is not None else None
    if receipt_path is not None:
        with receipt_path.open("x", encoding="utf-8"):
            pass
    sequence = 0

    def record(data):
        if receipt_path is not None:
            with receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(data, sort_keys=True) + "\n")

    async def forward(request):
        nonlocal sequence
        observed = receipt_path is not None and request.path == "/v1/chat/completions"
        entry = None
        if observed:
            sequence += 1
            entry = dict(request_index=sequence, host_received_monotonic_ns=time.monotonic_ns(),
                         host_received_unix_ns=time.time_ns(), status="receiving")
        try:
            raw = await request.read()
            if observed:
                processing_start = time.monotonic_ns()
                entry.update(request_evidence(raw))
                entry.update(receipt_inspection_ns=time.monotonic_ns()-processing_start, status="forwarding",
                             host_forwarding_monotonic_ns=time.monotonic_ns())
                record(entry | {"event": "request_forwarding"})
            async with app[key].request(request.method, backend + request.path, data=raw,
                                        headers={"Content-Type": "application/json"}) as upstream:
                if observed:
                    entry["upstream_http_status"] = upstream.status
                response = web.StreamResponse(status=upstream.status,
                                              headers={"Content-Type": upstream.headers.get("Content-Type", "application/json"),
                                                       "Cache-Control": "no-cache"})
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if observed:
                        entry.setdefault("first_response_byte_monotonic_ns", time.monotonic_ns())
                    await response.write(chunk)
                await response.write_eof()
                if observed:
                    entry["status"] = "complete"
                return response
        except asyncio.CancelledError:
            if observed:
                entry["status"] = "cancelled"
            raise
        except Exception as exc:
            if observed:
                entry.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            if observed:
                record(entry | {"event": "request_terminal", "host_finished_monotonic_ns": time.monotonic_ns(),
                                "host_finished_unix_ns": time.time_ns()})

    app.router.add_get("/health", forward)
    app.router.add_post("/v1/chat/completions", forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--backend-timeout", type=float, default=600)
    parser.add_argument("--backend-read-timeout", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.backend_read_timeout <= args.backend_timeout <= 14400:
        parser.error("Require positive read timeout <= total timeout <= 14400")
    info = json.loads(subprocess.check_output(["docker", "inspect", "qw3-computer-windows"], text=True))[0]
    assert info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") == "codex-goal-20260930"
    assert info["State"]["Running"]
    network = info["NetworkSettings"]["Networks"]["bridge"]
    gateway, peer = network["Gateway"], network["IPAddress"]
    assert ipaddress.ip_address(gateway).is_private and ipaddress.ip_address(peer).is_private
    run = args.run_dir.resolve(); run.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    token_file = run / "model-token.txt"
    token_file.write_text(token); token_file.chmod(0o600)
    (run / "manifest.json").write_text(json.dumps({"container_id": info["Id"], "bind": gateway,
                                                  "allowed_peer": peer, "port": args.port,
                                                  "backend": "http://127.0.0.1:18080",
                                                  "backend_timeout_seconds": args.backend_timeout,
                                                  "backend_read_timeout_seconds": args.backend_read_timeout}, indent=2))
    print(f"VM-only authenticated model relay: http://{gateway}:{args.port}", flush=True)
    web.run_app(create_app("http://127.0.0.1:18080", peer, token, receipt_path=run / "requests.jsonl",
                backend_timeout_seconds=args.backend_timeout,
                backend_read_timeout_seconds=args.backend_read_timeout), host=gateway, port=args.port,
                handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
