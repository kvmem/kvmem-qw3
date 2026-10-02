"""Stdio MCP transport to the application's desktop boundary."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

import aiohttp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qw3_computer.tool_schema import TOOLS, SOFTWARE_TOOLS


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--software", action="store_true")
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    active = {}

    def send(message):
        print(json.dumps({"jsonrpc": "2.0", **message}, ensure_ascii=False), flush=True)

    async with aiohttp.ClientSession(trust_env=False, headers={"Authorization": "Bearer " + token},
                                     timeout=aiohttp.ClientTimeout(total=35)) as session:
        async def handle(request):
            id = request["id"]
            method = request.get("method")
            try:
                if method == "initialize":
                    result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                              "serverInfo": {"name": "qw3-computer", "version": "0.1.0"}}
                elif method == "tools/list":
                    result = {"tools": TOOLS + (SOFTWARE_TOOLS if args.software else [])}
                elif method == "ping":
                    result = {}
                elif method == "tools/call":
                    async with session.post(args.endpoint + "/internal/tool", json=request.get("params", {})) as response:
                        result = await response.json()
                        if response.status != 200:
                            raise RuntimeError(result.get("error", "Desktop gateway failed"))
                else:
                    send({"id": id, "error": {"code": -32601, "message": "Unsupported MCP method"}})
                    return
                send({"id": id, "result": result})
            except asyncio.CancelledError:
                send({"id": id, "error": {"code": -32800, "message": "Tool call cancelled; observe before resuming"}})
                raise
            except Exception as exc:
                send({"id": id, "result": {"isError": True, "content": [{"type": "text", "text": str(exc)}]}})
            finally:
                active.pop(id, None)

        try:
            while line := await asyncio.to_thread(sys.stdin.readline):
                request = json.loads(line)
                if request.get("method") == "notifications/cancelled":
                    task = active.get(request.get("params", {}).get("requestId"))
                    if task:
                        task.cancel()
                elif "id" in request:
                    active[request["id"]] = asyncio.create_task(handle(request))
        finally:
            tasks = list(active.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    # MCP JSONL is UTF-8 even when Windows' redirected-stdio code page differs.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
