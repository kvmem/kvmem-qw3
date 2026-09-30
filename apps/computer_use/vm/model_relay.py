"""Expose the loopback model only to the owned Windows container, with a token.

This host-only VM test relay binds the Docker bridge gateway, never 0.0.0.0.
It is not a public or LAN model service. Stop it after native client tests.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import secrets
import subprocess

import aiohttp
from aiohttp import web


def create_app(backend, allowed_peer, token):
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
                                         timeout=aiohttp.ClientTimeout(total=600, sock_read=180)) as client:
            app[key] = client
            yield
    app.cleanup_ctx.append(session)

    async def forward(request):
        raw = await request.read()
        async with app[key].request(request.method, backend + request.path, data=raw,
                                    headers={"Content-Type": "application/json"}) as upstream:
            response = web.StreamResponse(status=upstream.status,
                                          headers={"Content-Type": upstream.headers.get("Content-Type", "application/json"),
                                                   "Cache-Control": "no-cache"})
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response

    app.router.add_get("/health", forward)
    app.router.add_post("/v1/chat/completions", forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18084)
    args = parser.parse_args()
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
                                                  "backend": "http://127.0.0.1:18080"}, indent=2))
    print(f"VM-only authenticated model relay: http://{gateway}:{args.port}", flush=True)
    web.run_app(create_app("http://127.0.0.1:18080", peer, token), host=gateway, port=args.port,
                handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
