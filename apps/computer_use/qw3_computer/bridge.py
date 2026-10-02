"""Loopback Responses SSE adapter. One active QW3 inference request at a time."""

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
import time
import uuid

import aiohttp
from aiohttp import web

from .protocol import ProtocolError, convert_request, read_sse, require

LOG = logging.getLogger(__name__)
BACKEND = web.AppKey("backend", aiohttp.ClientSession)


class ResponseStream:
    def __init__(self, response, model, names):
        self.http = response
        self.names = names
        self.sequence = 0
        self.items = []
        self.text_item = None
        self.calls = {}
        self.usage = None
        self.finish_reason = None
        self.response = {"id": "resp_" + uuid.uuid4().hex, "object": "response",
                         "created_at": int(time.time()), "model": model,
                         "status": "in_progress", "output": [], "error": None}

    async def emit(self, kind, **fields):
        event = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        await self.http.write(("event: " + kind + "\ndata: " +
                               json.dumps(event, ensure_ascii=False) + "\n\n").encode())

    async def start(self):
        await self.emit("response.created", response=self.response)
        await self.emit("response.in_progress", response=self.response)

    async def chunk(self, chunk):
        require(isinstance(chunk, dict), "backend chunk must be an object")
        require(not chunk.get("error"), "backend reported an error")
        if chunk.get("usage"):
            u = chunk["usage"]
            self.usage = {"input_tokens": u.get("prompt_tokens", 0),
                          "output_tokens": u.get("completion_tokens", 0),
                          "total_tokens": u.get("total_tokens", 0)}
        choices = chunk.get("choices", [])
        require(isinstance(choices, list), "backend choices must be a list")
        require(len(choices) <= 1, "multiple backend choices are unsupported")
        if not choices:
            return
        choice = choices[0]
        require(isinstance(choice, dict), "backend choice must be an object")
        delta = choice.get("delta", {})
        require(isinstance(delta, dict), "backend delta must be an object")
        require(self.finish_reason is None, "backend emitted a choice after finish")
        require(not delta.get("reasoning_content"), "backend ignored enable_thinking=false")
        if delta.get("content"):
            require(isinstance(delta["content"], str), "backend text must be a string")
            if self.text_item is None:
                self.text_item = {"id": "msg_" + uuid.uuid4().hex, "type": "message",
                                  "role": "assistant", "status": "in_progress", "content": []}
                await self.emit("response.output_item.added", output_index=0, item=self.text_item)
                part = {"type": "output_text", "text": "", "annotations": []}
                self.text_item["content"].append(part)
                await self.emit("response.content_part.added", item_id=self.text_item["id"],
                                output_index=0, content_index=0, part=part)
            self.text_item["content"][0]["text"] += delta["content"]
            await self.emit("response.output_text.delta", item_id=self.text_item["id"],
                            output_index=0, content_index=0, delta=delta["content"])
        for call in delta.get("tool_calls", []):
            require(isinstance(call, dict), "backend tool call must be an object")
            index = call.get("index", 0)
            require(type(index) is int and 0 <= index < 32, "invalid tool index")
            merged = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                require(isinstance(call["id"], str), "backend call_id must be a string")
                require(not merged["id"] or merged["id"] == call["id"], "backend changed call_id")
                merged["id"] = call["id"]
            for key in ("name", "arguments"):
                merged[key] += call.get("function", {}).get(key, "")
        if choice.get("finish_reason"):
            require(self.finish_reason is None, "duplicate backend finish event")
            self.finish_reason = choice["finish_reason"]

    async def finish(self):
        require(self.finish_reason in ("stop", "tool_calls", "length"),
                "backend stream has no supported finish_reason")
        require(self.finish_reason != "tool_calls" or self.calls, "backend returned no tool calls")
        if self.text_item:
            item = self.text_item
            part = item["content"][0]
            await self.emit("response.output_text.done", item_id=item["id"], output_index=0,
                            content_index=0, text=part["text"])
            await self.emit("response.content_part.done", item_id=item["id"], output_index=0,
                            content_index=0, part=part)
            item["status"] = "incomplete" if self.finish_reason == "length" else "completed"
            await self.emit("response.output_item.done", output_index=0, item=item)
            self.items.append(item)
        # Never release a partial/invalid action to Codex. Even with a valid
        # JSON prefix, a length-truncated generation must not execute a tool.
        if self.finish_reason != "length":
            validated = []
            ids = set()
            for _, call in sorted(self.calls.items()):
                require(call["id"] and call["id"] not in ids, "duplicate or empty backend call_id")
                ids.add(call["id"])
                require(call["name"] in self.names, "backend called an unknown tool")
                require(isinstance(json.loads(call["arguments"]), dict), "tool arguments must be an object")
                namespace, name = self.names[call["name"]]
                item = {"id": "fc_" + uuid.uuid4().hex, "type": "function_call",
                        "call_id": call["id"], "name": name, "arguments": call["arguments"],
                        "status": "completed"}
                if namespace:
                    item["namespace"] = namespace
                validated.append(item)
            for item in validated:
                index = len(self.items)
                await self.emit("response.output_item.added", output_index=index,
                                item=dict(item, status="in_progress", arguments=""))
                await self.emit("response.function_call_arguments.delta", output_index=index,
                                item_id=item["id"], delta=item["arguments"])
                await self.emit("response.function_call_arguments.done", output_index=index,
                                item_id=item["id"], arguments=item["arguments"])
                await self.emit("response.output_item.done", output_index=index, item=item)
                self.items.append(item)
        status = "incomplete" if self.finish_reason == "length" else "completed"
        self.response.update(status=status, output=self.items, usage=self.usage)
        if status == "incomplete":
            self.response["incomplete_details"] = {"reason": "max_output_tokens"}
        await self.emit("response." + status, response=self.response)

    async def fail(self, message):
        self.response.update(status="failed", output=self.items,
                             error={"code": "backend_error", "message": message})
        await self.emit("response.failed", response=self.response)


def create_app(backend_url, backend_model="qw3", public_model="qw3-computer", max_tokens=4096, trace_dir=None,
               backend_token=None, trace_backend=False, *, backend_timeout_seconds=600,
               backend_read_timeout_seconds=180, heartbeat_seconds=30):
    if trace_backend and trace_dir is None:
        raise ValueError("Backend trace tags require a local trace directory")
    if not 0 < heartbeat_seconds <= backend_read_timeout_seconds <= backend_timeout_seconds:
        raise ValueError("Require positive heartbeat <= read timeout <= total timeout")
    app = web.Application(client_max_size=64 * 1024 * 1024)
    lock = asyncio.Lock()

    async def sessions(app):
        headers = {"Authorization": "Bearer " + backend_token} if backend_token else {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=backend_timeout_seconds,
                                         sock_read=backend_read_timeout_seconds),
                                         trust_env=False, headers=headers) as session:
            app[BACKEND] = session
            yield

    app.cleanup_ctx.append(sessions)

    async def responses(request):
        try:
            raw = await request.json()
            require(raw.get("model") == public_model, "unknown local model")
            converted = convert_request(raw, backend_model, max_tokens)
        except (ProtocolError, ValueError, TypeError, AttributeError) as exc:
            return web.json_response({"error": {"type": "invalid_request_error", "message": str(exc)}}, status=400)
        if lock.locked():
            return web.json_response({"error": {"message": "another local inference request is active"}}, status=409)
        async with lock:
            trace = None
            if trace_dir is not None:
                request_id = uuid.uuid4().hex
                trace = Path(trace_dir) / request_id
                if trace_backend:
                    converted.body["kvmem_trace_tag"] = request_id
                trace.mkdir(parents=True)
                # Keep exact local inputs for paired on/off replay. This opt-in
                # trace belongs to the application runtime, never source control.
                (trace / "request.json").write_text(json.dumps(converted.body, ensure_ascii=False))
            started = time.monotonic()
            metrics = {"started_at": time.time(), "status": "in_progress"}
            if trace_backend:
                metrics["backend_trace_tag"] = converted.body["kvmem_trace_tag"]
            output = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                 "Cache-Control": "no-cache"})
            stream = ResponseStream(output, public_model, converted.names)
            heartbeat = None

            async def progress():
                # Connection liveness only. Never invent usage or release a
                # partial tool while the backend is doing silent prefill.
                while True:
                    await asyncio.sleep(heartbeat_seconds)
                    await stream.emit("response.in_progress", response=stream.response)

            async def stop_heartbeat():
                if heartbeat is not None:
                    heartbeat.cancel()
                    with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                        await heartbeat

            try:
                async with app[BACKEND].post(backend_url.rstrip("/") + "/chat/completions",
                                                json=converted.body) as upstream:
                    if upstream.status != 200:
                        metrics.update(status="http_error", http_status=upstream.status)
                        return web.json_response({"error": {"message": f"QW3 returned HTTP {upstream.status}"}}, status=502)
                    await output.prepare(request)
                    await stream.start()
                    heartbeat = asyncio.create_task(progress())
                    done = False
                    async for data in read_sse(upstream.content):
                        metrics.setdefault("first_event_seconds", time.monotonic() - started)
                        if data == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(data)
                        if trace is not None:
                            with (trace / "backend.jsonl").open("a") as log:
                                log.write(data + "\n")
                        await stream.chunk(chunk)
                    require(done, "backend stream ended without [DONE]")
                    await stop_heartbeat()
                    await stream.finish()
                    metrics.update(status=stream.response["status"], usage=stream.usage)
                    await output.write_eof()
                    return output
            except asyncio.CancelledError:
                metrics["status"] = "cancelled"
                # aiohttp handler_cancellation closes the upstream request even
                # when the client disconnects during a long, silent prefill.
                LOG.info("local request cancelled; upstream connection closed")
                raise
            except (ProtocolError, ValueError, TypeError, KeyError, IndexError,
                    aiohttp.ClientError, TimeoutError) as exc:
                metrics.update(status="failed", error=str(exc))
                LOG.warning("local backend failed: %s", exc)
                await stop_heartbeat()
                if not output.prepared:
                    return web.json_response({"error": {"message": str(exc)}}, status=502)
                try:
                    await stream.fail(str(exc))
                    await output.write_eof()
                except (ConnectionError, RuntimeError):
                    pass
                return output
            finally:
                await stop_heartbeat()
                if trace is not None:
                    metrics["elapsed_seconds"] = time.monotonic() - started
                    (trace / "metrics.json").write_text(json.dumps(metrics, indent=2))

    app.router.add_post("/v1/responses", responses)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--backend", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--backend-model", default="qw3")
    parser.add_argument("--backend-token", type=Path, help="Optional token file for the private VM model relay")
    parser.add_argument("--trace-dir", type=Path, help="Retain exact local inference requests and metrics")
    parser.add_argument("--trace-backend", action="store_true", help="Send request tags for QW3 retrieval diagnostics")
    args = parser.parse_args()
    if args.trace_backend and args.trace_dir is None:
        parser.error("--trace-backend requires --trace-dir")
    token = args.backend_token.read_text(encoding="utf-8-sig").strip() if args.backend_token else None
    if args.backend_token and not token:
        parser.error("backend token file is empty")
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(args.backend, args.backend_model, backend_token=token,
                           trace_dir=args.trace_dir, trace_backend=args.trace_backend), host="127.0.0.1",
                port=args.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
