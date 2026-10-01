"""Exercise pinned Codex's context boundary using scripted usage, not a GPU benchmark."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from aiohttp import web

from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, "http://127.0.0.1:"+str(site._server.sockets[0].getsockname()[1])


async def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests, events = [], []
    result = {"kind": "scripted_context_boundary_control_plane", "real_long_context_inference": False,
              "simulated_input_usage": 950100, "passed": False}

    async def backend(request):
        body = await request.json()
        requests.append(body)
        (output / f"backend-{len(requests):02}.json").write_text(json.dumps(body, indent=2))
        if len(requests) > 6:
            raise web.HTTPInternalServerError(text="Diagnostic request cap exceeded")
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        if len(requests) == 1:
            name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("observe"))
            delta = {"tool_calls": [{"index": 0, "id": "boundary-observe", "type": "function",
                                      "function": {"name": name, "arguments": "{}"}}]}
            finish = "tool_calls"
        else:
            delta = {"content": "The fixture observation was received. The diagnostic is complete; no desktop action was performed."}
            finish = "stop"
        for choice in ({"index": 0, "delta": delta, "finish_reason": None},
                       {"index": 0, "delta": {}, "finish_reason": finish}):
            await stream.write(("data: "+json.dumps({"choices": [choice]})+"\n\n").encode())
        usage = {"prompt_tokens": 950100 if len(requests) == 1 else 500, "completion_tokens": 20,
                 "total_tokens": 950120 if len(requests) == 1 else 520}
        await stream.write(("data: "+json.dumps({"choices": [], "usage": usage})+"\n\n").encode())
        await stream.write(b"data: [DONE]\n\n")
        return stream

    backend_app = web.Application()
    backend_app.router.add_post("/v1/chat/completions", backend)
    backend_runner, backend_url = await serve(backend_app)
    bridge_runner, bridge_url = await serve(create_app(backend_url+"/v1", trace_dir=output/"bridge"))
    write_config(output/"codex", bridge_url+"/v1", sys.executable,
                 [str(Path(__file__).with_name("tests")/"fixture_mcp.py")])
    client = CodexClient(output/"codex")
    try:
        await client.start()
        thread = await client.new_thread()
        turn = await client.turn(thread, "Observe the fixture once, then finish the diagnostic.")
        async with asyncio.timeout(50):
            while True:
                event = await client.events.get()
                events.append(event)
                if event.get("method") == "turn/completed" and event["params"]["turn"]["id"] == turn:
                    result["turn"] = event["params"]["turn"]
                    break
                if event.get("method") == "app/disconnected":
                    raise RuntimeError("Codex disconnected")
        result["request_count"] = len(requests)
        result["compaction_events"] = [e for e in events if e.get("method") in ("item/started", "item/completed")
            and e.get("params", {}).get("item", {}).get("type") == "contextCompaction"]
        result["completed_compactions"] = sum(e["method"] == "item/completed" for e in result["compaction_events"])
        result["request_images"] = [sum(part.get("type") == "image_url" for message in body["messages"]
            if isinstance(message.get("content"), list) for part in message["content"]) for body in requests]
        result["request_tool_counts"] = [len(body.get("tools", [])) for body in requests]
        result["passed"] = (result["turn"]["status"] == "completed" and result["completed_compactions"] == 1
                            and len(requests) == 3 and result["request_images"] == [0, 1, 0]
                            and result["request_tool_counts"][1] == 0)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        await client.close()
        await bridge_runner.cleanup()
        await backend_runner.cleanup()
        (output/"events.json").write_text(json.dumps(events, indent=2))
        (output/"result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ("turn","compaction_events")}))
    return result["passed"]


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output)) else 1)
