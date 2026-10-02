"""Scripted capacity proof for the bridge and relay, without a model or desktop.

The generated PNGs are neutral test data. Success here proves transport only;
it is never enrolled as an agent task or used to measure model accuracy.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import random
import struct
import time
import zlib

import aiohttp
from aiohttp import web

from qw3_computer.bridge import create_app as bridge_app
from qw3_computer.limits import MAX_MODEL_REQUEST_BYTES
from qw3_computer.protocol import convert_request, read_sse
from vm.model_relay import create_app as relay_app


def canonical_digest(body):
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def synthetic_png():
    width, height = 1280, 800
    pixels = random.Random(20261002).randbytes(width * height * 3)
    rows = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


async def serve(app, runners):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    runners.append(runner)
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])


async def qualify(output):
    started = time.monotonic()
    report = dict(kind="scripted_bridge_relay_capacity_proof", passed=False,
                  real_model=False, desktop_actions=False, formal_run=False,
                  model_request_limit_bytes=MAX_MODEL_REQUEST_BYTES, checks={})
    runners = []
    try:
        image = synthetic_png()
        url = "data:image/png;base64," + base64.b64encode(image).decode()
        count = 17
        request = dict(model="qw3-computer", stream=True, store=False,
                       instructions="Neutral scripted transport qualification; no model evaluation.",
                       input=[dict(role="user", content=[dict(type="input_text", text=f"Synthetic frame {i}"),
                                  dict(type="input_image", image_url=url)]) for i in range(count)], tools=[])
        wire = json.dumps(request).encode()
        expected = canonical_digest(convert_request(request, "qw3").body)
        assert len(wire) > 64 * 1024 * 1024
        seen = []
        upstream = web.Application(client_max_size=MAX_MODEL_REQUEST_BYTES)

        async def backend(incoming):
            body = await incoming.json()
            seen.append(dict(canonical_request_sha256=canonical_digest(body), body_bytes=incoming.content_length))
            if body["messages"][-1]["content"] == "return scripted backend error":
                return web.json_response({"error": "scripted mandatory spans exceed budget"}, status=413)
            assert seen[-1]["canonical_request_sha256"] == expected
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(incoming)
            for data in [dict(choices=[dict(index=0, delta=dict(content="transport-control-ok"), finish_reason=None)]),
                         dict(choices=[dict(index=0, delta={}, finish_reason="stop")]),
                         dict(choices=[], usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))]:
                await response.write(("data: " + json.dumps(data) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        upstream.router.add_post("/v1/chat/completions", backend)
        backend_url = await serve(upstream, runners)
        relay_url = await serve(relay_app(backend_url, "127.0.0.1", "qualification-token",
                                         receipt_path=output / "relay.jsonl"), runners)
        bridge_url = await serve(bridge_app(relay_url + "/v1", backend_token="qualification-token",
                                           trace_dir=output / "trace"), runners)
        old_url = await serve(bridge_app(relay_url + "/v1", backend_token="qualification-token",
                                        max_request_bytes=64 * 1024 * 1024), runners)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180), trust_env=False) as client:
            async with client.post(relay_url + "/v1/chat/completions", data=wire) as response:
                report["checks"]["unauthenticated_request_rejected"] = response.status == 401 and not seen
            async with client.post(old_url + "/v1/responses", data=wire,
                                   headers={"Content-Type": "application/json"}) as response:
                report["checks"]["old_64_mib_limit_rejects_before_backend"] = response.status == 413 and not seen
                await response.read()
            async with client.post(bridge_url + "/v1/responses", data=wire,
                                   headers={"Content-Type": "application/json"}) as response:
                assert response.status == 200, await response.text()
                events = [json.loads(data) async for data in read_sse(response.content)]
            report["checks"]["large_request_completed_without_input_changes"] = (
                len(seen) == 1 and seen[0]["canonical_request_sha256"] == expected
                and events[-1]["type"] == "response.completed"
                and events[-1]["response"]["output"][0]["content"][0]["text"] == "transport-control-ok")
            error_request = dict(request, input=[dict(role="user", content="return scripted backend error")])
            async with client.post(bridge_url + "/v1/responses", json=error_request) as response:
                error = await response.json()
                report["checks"]["backend_error_retained_without_success"] = (
                    response.status == 502 and error["error"]["upstream_status"] == 413
                    and "scripted mandatory spans exceed budget" in error["error"]["message"])
        rows = [json.loads(line) for line in (output / "relay.jsonl").read_text().splitlines()]
        terminals = [row for row in rows if row["event"] == "request_terminal"]
        report["checks"]["relay_terminal_binds_exact_large_request"] = (
            len(terminals) == 2 and terminals[0]["status"] == "complete"
            and terminals[0]["body_bytes"] > 64 * 1024 * 1024
            and terminals[0]["canonical_request_sha256"] == expected)
        metrics = [json.loads(path.read_text()) for path in (output / "trace").glob("*/metrics.json")]
        errors = [item for item in metrics if item["status"] == "http_error"]
        report["checks"]["failure_receipt_contains_bounded_backend_reason"] = (
            len(errors) == 1 and errors[0]["http_status"] == 413
            and "scripted mandatory spans exceed budget" in errors[0]["backend_error_response"]
            and errors[0]["backend_error_truncated"] is False)
        report.update(input_wire_bytes=len(wire), image_count=count, image_sha256=hashlib.sha256(image).hexdigest(),
                      canonical_request_sha256=expected, backend_requests=seen)
        report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for runner in reversed(runners):
            await runner.cleanup()
        report["elapsed_seconds"] = time.monotonic() - started
        root = Path(__file__).resolve().parents[1]
        report["source_files"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), root / "vm/model_relay.py", root / "qw3_computer/limits.py",
                         root / "qw3_computer/bridge.py", root / "qw3_computer/protocol.py"]}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(qualify(args.output))
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
