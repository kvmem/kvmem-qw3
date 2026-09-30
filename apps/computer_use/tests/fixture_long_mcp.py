"""Long-history fixture using recorded screenshots; never sends desktop input."""
import argparse
import base64
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--historical", type=Path, required=True)
parser.add_argument("--current", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=600)
args = parser.parse_args()
historical = base64.b64encode(args.historical.read_bytes()).decode()
current = base64.b64encode(args.current.read_bytes()).decode()
unit = ("Archived application observation: the status panel showed an idle local workspace. "
        "No document edit, navigation, download, clipboard update or form submission occurred during this interval.\n")
count = 0
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fixture-long-history", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "observe", "description": "Collect the next numbered historical desktop observation.",
                             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}]}
    elif method == "tools/call" and request.get("params", {}).get("name") == "observe":
        count += 1
        result = {"content": [{"type": "text", "text": f"Observation {count} of 8. Archived notes follow.\n" + unit * args.repeats},
                              {"type": "image", "data": historical if count == 2 else current, "mimeType": "image/png"}]}
    elif method == "ping":
        result = {}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "Unknown fixture method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
