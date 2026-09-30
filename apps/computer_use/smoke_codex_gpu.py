"""Real local-model image/tool smoke. Uses a fixture MCP, not a desktop VM."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import time

from aiohttp import web
from PIL import Image

from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bridge = web.AppRunner(create_app(args.backend), handler_cancellation=True)
    await bridge.setup()
    site = web.TCPSite(bridge, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    write_config(output / "codex", f"http://127.0.0.1:{port}/v1", sys.executable,
                 [str(Path(__file__).resolve().parent / "tests/fixture_mcp.py")])
    client = CodexClient(output / "codex")
    started = time.monotonic()
    result = {"backend": args.backend, "kind": "real_model_fixture_mcp", "passed": False}
    try:
        await client.start()
        thread = await client.new_thread()
        image = io.BytesIO()
        Image.new("RGB", (128, 96), "red").save(image, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
        turn = await client.turn(thread,
            "This is a protocol test. First name the color of my attached image. Then call the computer observe tool "
            "exactly twice, sequentially. After receiving both results, report their observation numbers and "
            "the color shown in the tool screenshots. Do not ask questions. Do not use any other tools.", [data_url])
        result.update(thread_id=thread, turn_id=turn)
        with (output / "events.jsonl").open("w") as events:
            while True:
                event = await asyncio.wait_for(client.events.get(), 240)
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                method = event.get("method")
                if method == "item/agentMessage/delta":
                    result.setdefault("first_text_seconds", time.monotonic() - started)
                if method == "item/completed":
                    item = event["params"]["item"]
                    if item["type"] == "mcpToolCall":
                        result.setdefault("tool_calls", []).append({k: v for k, v in item.items()
                                                                   if k not in ("result", "content")})
                    if item["type"] == "agentMessage":
                        result.setdefault("messages", []).append(item.get("text", ""))
                        print(item.get("text", ""), flush=True)
                if method == "turn/completed" and event["params"]["turn"]["id"] == turn:
                    result["turn"] = event["params"]["turn"]
                    texts = "\n".join(result.get("messages", [])).lower()
                    result["passed"] = (result["turn"]["status"] == "completed"
                        and len(result.get("tool_calls", [])) == 2
                        and all(t.get("status") == "completed" for t in result["tool_calls"])
                        and "red" in texts and "blue" in texts)
                    break
                if method == "app/disconnected":
                    raise RuntimeError("Codex disconnected")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        await client.close()
        await bridge.cleanup()
    print(json.dumps({k: result[k] for k in ("passed", "elapsed_seconds")}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--run-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
