"""Protocol fixture only: no real desktop actions or model inference."""
import base64
import io
import json
import sys

from PIL import Image

buffer = io.BytesIO()
Image.new("RGB", (64, 48), "blue").save(buffer, format="PNG")
image = base64.b64encode(buffer.getvalue()).decode()
count = 0
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fixture-computer", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "observe", "description": "Return a fixture screenshot.",
                             "inputSchema": {"type": "object", "properties": {},
                                             "additionalProperties": False}}]}
    elif method == "tools/call":
        count += 1
        result = {"content": [{"type": "text", "text": f"fixture observation {count}"},
                              {"type": "image", "data": image, "mimeType": "image/png"}]}
    elif method == "ping":
        result = {}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {
            "code": -32601, "message": "unsupported fixture method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
