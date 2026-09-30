"""Real Codex/QW3 long-history image test with a synthetic observation fixture."""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

from aiohttp import web

from compare_replay import monitor
from long_history_probe import EXPECTED, QUESTION, answer
from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config


async def run(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = json.loads((args.engine_dir / "manifest.json").read_text())
    bridge = web.AppRunner(create_app(args.backend, trace_dir=output / "inference"), handler_cancellation=True)
    await bridge.setup()
    site = web.TCPSite(bridge, "127.0.0.1", 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    write_config(output / "codex", f"http://127.0.0.1:{port}/v1", sys.executable,
                 [str(Path(__file__).resolve().parent / "tests/fixture_long_mcp.py"),
                  "--historical", str(args.historical.resolve()), "--current", str(args.current.resolve())])
    client = CodexClient(output / "codex")
    result = {"kind": "real_codex_synthetic_history_recorded_screenshots", "passed": False, "engine": engine}
    stopped = asyncio.Event()
    sampler = asyncio.create_task(monitor(engine["pid"], output / "resources.jsonl", stopped))
    started = time.monotonic()
    async def collect(turn, log):
        messages, calls = [], []
        while True:
            event = await asyncio.wait_for(client.events.get(), 600)
            log.write(json.dumps(event, ensure_ascii=False) + "\n"); log.flush()
            method = event.get("method")
            if method == "item/completed":
                item = event["params"]["item"]
                if item["type"] == "agentMessage":
                    messages.append(item.get("text", "")); print(item.get("text", "")[:500], flush=True)
                if item["type"] == "mcpToolCall":
                    calls.append({k: item.get(k) for k in ("tool", "status", "arguments")})
                    print("MCP", len(calls), item.get("status"), flush=True)
            if method == "turn/completed" and event["params"]["turn"]["id"] == turn:
                return {"turn": event["params"]["turn"], "messages": messages, "calls": calls}
            if method == "app/disconnected":
                raise RuntimeError("Codex disconnected")
    try:
        await client.start(); thread = await client.new_thread(); result["thread_id"] = thread
        with (output / "events.jsonl").open("w") as log:
            turn = await client.turn(thread, "This is a history-collection test. Call computer observe exactly eight times, "
                                     "sequentially. Count the returned observation numbers. Only collect: do not describe images "
                                     "or archive notes and do not answer any visual questions yet. After observation 8, "
                                     "reply COLLECTION_READY. Do not use other tools.")
            first = await collect(turn, log); result["collection"] = first
            assert first["turn"]["status"] == "completed" and len(first["calls"]) == 8
            assert all(call["status"] == "completed" for call in first["calls"])
            assert EXPECTED["reference"] not in "\n".join(first["messages"]), "Answer appeared in earlier assistant text"
            turn = await client.turn(thread, QUESTION + " Use only the saved conversation history; do not call tools.")
            final = await collect(turn, log); result["question"] = final
            parsed = answer(final["messages"][-1]); result["parsed_answer"] = parsed
            metrics = [json.loads(p.read_text()) for p in (output / "inference").glob("*/metrics.json")]
            result["max_prompt_tokens"] = max(m.get("usage", {}).get("input_tokens", 0) for m in metrics)
            assert result["max_prompt_tokens"] > 131072, "Codex history did not cross K128K"
            assert final["turn"]["status"] == "completed" and not final["calls"]
            last = max((output / "inference").glob("*/metrics.json"),
                       key=lambda p: json.loads(p.read_text())["started_at"]).parent
            request = json.loads((last / "request.json").read_text())
            images, texts = [], []
            for message in request["messages"]:
                content = message.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            texts.append(part["text"])
                        elif part.get("type") == "image_url":
                            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                            images.append(hashlib.sha256(raw).hexdigest())
            expected_images = [hashlib.sha256(args.current.read_bytes()).hexdigest()] * 8
            expected_images[1] = hashlib.sha256(args.historical.read_bytes()).hexdigest()
            joined = "\n".join(texts)
            assert images == expected_images, "Codex changed or dropped a historical image"
            assert joined.count("Archived application observation:") == 4800, "Archive notes were truncated"
            assert EXPECTED["reference"] not in joined and EXPECTED["name"] not in joined
            result["history_check"] = {"image_sha256": images, "archive_note_count": 4800,
                                       "answer_in_text": False}
            if engine["kvmem"] == "on":
                metadata = [row for line in (args.engine_dir / "retrieval.jsonl").read_text().splitlines()
                            if (row := json.loads(line)).get("type") == "meta"]
                meta = metadata[-1]
                assert meta["query_begin"] > 131072 and meta["selected"] < meta["block_count"]
                assert meta["query_ready"] and meta["index_ready"] and meta["scorer_fallback_reason"] == "none"
                result["retrieval"] = meta
            result["passed"] = parsed == EXPECTED
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        await client.close(); await bridge.cleanup(); stopped.set(); await sampler
    if not result["passed"]:
        raise SystemExit("Historical screenshot answer was incorrect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path, default=Path("apps/computer_use/.runtime/long-history-probe-01/historical.png"))
    parser.add_argument("--current", type=Path, default=Path("apps/computer_use/.runtime/long-history-probe-01/current.png"))
    parser.add_argument("--run-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
