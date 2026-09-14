#!/usr/bin/env python3
"""A minimal stdio MCP transport for the experiment's isolated desktop tool."""
import json
import math
import sys
import urllib.error
import urllib.request

TOOL = {
    "name": "computer_use",
    "description": "Operate the remote desktop through its visible interface. Coordinates are normalized 0-999 in both axes. Every action returns a fresh screenshot. Start with screenshot. Use terminate only when finished or unable to continue. No shell, DOM, JavaScript or direct network access is available.",
    "inputSchema": {"type": "object", "required": ["action"], "properties": {
        "action": {"type": "string", "enum": ["screenshot", "left_click", "double_click", "right_click", "mouse_move", "left_click_drag", "type", "key", "scroll", "hscroll", "wait", "terminate"]},
        "coordinate": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "x,y normalized to 0-999"},
        "text": {"type": "string", "description": "Text to type"},
        "keys": {"type": "array", "items": {"type": "string"}, "description": "PyAutoGUI keys, for example [ctrl,l] or [enter]"},
        "pixels": {"type": "integer", "description": "Scroll clicks: positive up/right, negative down/left"},
        "status": {"type": "string", "enum": ["success", "failure"]}
    }, "additionalProperties": False}
}


def validate_action(arguments):
    """Enforce this tool's advertised schema before any desktop side effect.

    This small validator covers the schema types used above. It does not repair
    malformed model arguments or reinterpret a string as an array of keys.
    """
    def check(value, schema, path):
        kind = schema['type']
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        matches = {
            'object': isinstance(value, dict),
            'array': isinstance(value, list),
            'string': isinstance(value, str),
            'number': number and math.isfinite(value),
            'integer': number and math.isfinite(value) and int(value) == value,
        }
        if not matches[kind]:
            raise ValueError(f'{path} must be {kind}; received {type(value).__name__}')
        if 'enum' in schema and value not in schema['enum']:
            raise ValueError(f'{path} must be one of: ' + ', '.join(schema['enum']))
        if kind == 'object':
            properties = schema['properties']
            for name in schema.get('required', []):
                if name not in value:
                    raise ValueError(f'{path}.{name} is required')
            for name, item in value.items():
                if name not in properties:
                    if schema.get('additionalProperties') is False:
                        raise ValueError(f'{path} has an unsupported property: {name}')
                else:
                    check(item, properties[name], name)
        elif kind == 'array':
            if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', len(value)):
                raise ValueError(f'{path} has an invalid number of items')
            for index, item in enumerate(value):
                check(item, schema['items'], f'{path}[{index}]')

    check(arguments, TOOL['inputSchema'], 'arguments')

def main():
    endpoint = sys.argv[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        try:
            method = request.get("method")
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                          "serverInfo": {"name": "osworld-desktop", "version": "1.0"}}
            elif method == "tools/list":
                result = {"tools": [TOOL]}
            elif method == "tools/call":
                params = request["params"]
                if params["name"] != TOOL["name"]:
                    raise ValueError("unknown desktop tool")
                call = urllib.request.Request(endpoint + "/computer", data=json.dumps(params.get("arguments", {})).encode(), headers={"Content-Type": "application/json"})
                with opener.open(call, timeout=120) as response:
                    result = json.load(response)
            elif method == "ping":
                result = {}
            else:
                raise ValueError("unsupported MCP method: " + str(method))
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32603, "message": str(exc)}}
        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()
