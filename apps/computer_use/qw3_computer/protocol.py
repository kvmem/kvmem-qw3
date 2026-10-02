"""The explicit Responses subset used by our pinned Codex client.

Conversion is stateless: full input history is the replay authority. Never
resolve previous_response_id against implicit, process-local conversation state.
"""

import base64
import binascii
import json
import re
from dataclasses import dataclass


class ProtocolError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def string(value, label):
    require(isinstance(value, str), f"{label} must be a string")
    return value


def content_parts(content):
    if isinstance(content, str):
        return content
    require(isinstance(content, list), "content must be a string or list")
    parts = []
    for part in content:
        require(isinstance(part, dict), "content parts must be objects")
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append({"type": "text", "text": string(part.get("text"), "text")})
        elif kind == "input_image":
            url = string(part.get("image_url"), "image_url")
            require(not part.get("file_id"), "file_id images are not supported")
            require(re.match(r"^data:image/(png|jpeg|webp);base64,", url),
                    "images must be inline PNG/JPEG/WebP data URLs")
            try:
                require(bool(base64.b64decode(url.split(",", 1)[1], validate=True)),
                        "image data is empty")
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError("invalid image base64") from exc
            # QW3 encodes the actual pixels; Responses detail is only a client
            # hint. Keep it in the forwarded image object for auditability.
            image = {"url": url}
            if "detail" in part:
                require(part["detail"] in ("auto", "low", "high", "original"),
                        "unsupported image detail")
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        else:
            raise ProtocolError(f"unsupported content type: {kind}")
    return parts


@dataclass
class ConvertedRequest:
    body: dict
    names: dict


def convert_request(request, backend_model, max_tokens=4096, *, allow_thinking=False):
    require(isinstance(request, dict), "request must be an object")
    allowed = {"model", "instructions", "input", "tools", "tool_choice",
               "parallel_tool_calls", "reasoning", "store", "stream", "include",
               "prompt_cache_key", "client_metadata", "max_output_tokens",
               "temperature", "top_p", "text", "metadata"}
    require(not set(request) - allowed,
            f"unsupported request fields: {sorted(set(request) - allowed)}")
    require(request.get("stream") is True, "this bridge requires stream=true")
    require(request.get("store", False) is False, "server-side history storage is unsupported")
    require(request.get("include", []) in ([], ["reasoning.encrypted_content"]),
            "unsupported include value")
    reasoning = request.get("reasoning") or {}
    require(isinstance(reasoning, dict), "reasoning must be an object")
    require(set(reasoning) <= {"effort", "summary"}, "unsupported reasoning fields")
    # Opt-in medium enables QW3 thinking; it does not claim to implement an
    # OpenAI effort budget. The same max_tokens cap includes reasoning/output.
    effort = reasoning.get("effort", "none")
    require(effort == "none" or (allow_thinking and effort == "medium"),
            "only reasoning effort=none is qualified unless local thinking is enabled")
    require(reasoning.get("summary", "none") in ("none", "auto"), "reasoning summaries are unsupported")
    text = request.get("text") or {}
    require(text in ({}, {"format": {"type": "text"}}), "structured text output is unsupported")
    tools, names = [], {}

    def add_tool(tool, namespace=None):
        require(isinstance(tool, dict), "tools must be objects")
        require(tool.get("type") == "function", "only function tools are supported")
        require(not tool.get("defer_loading"), "deferred tools are unsupported")
        name = string(tool.get("name"), "tool name")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "invalid tool name")
        wire_name = f"{namespace}__{name}" if namespace else name
        require(wire_name not in names, "colliding tool names")
        require(not tool.get("strict"), "strict schema enforcement is unsupported")
        parameters = tool.get("parameters", {"type": "object", "properties": {}})
        require(isinstance(parameters, dict), "tool parameters must be an object")
        names[wire_name] = (namespace, name)
        tools.append({"type": "function", "function": {
            "name": wire_name, "description": tool.get("description", ""),
            "parameters": parameters}})

    for tool in request.get("tools", []):
        require(isinstance(tool, dict), "tools must be objects")
        if tool.get("type") == "namespace":
            namespace = string(tool.get("name"), "namespace")
            require(re.fullmatch(r"[A-Za-z0-9_-]+", namespace), "invalid namespace")
            for child in tool.get("tools", []):
                add_tool(child, namespace)
        else:
            add_tool(tool)

    messages = []
    if request.get("instructions"):
        messages.append({"role": "system", "content": string(request["instructions"], "instructions")})
    inputs = request.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    require(isinstance(inputs, list), "input must be a string or list")
    pending_calls, seen_calls = set(), set()
    for item in inputs:
        require(isinstance(item, dict), "input items must be objects")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            require(role in ("system", "developer", "user", "assistant"), "invalid message role")
            require(not pending_calls, "tool outputs are missing before the next message")
            content = content_parts(item.get("content", []))
            if (role == "assistant" and messages and messages[-1]["role"] == "assistant"
                    and "reasoning_content" in messages[-1] and messages[-1]["content"] == ""
                    and not messages[-1].get("tool_calls")):
                messages[-1]["content"] = content
            else:
                messages.append({"role": role, "content": content})
        elif kind == "reasoning":
            require(allow_thinking, "reasoning history requires local thinking support")
            require(not pending_calls, "tool outputs are missing before reasoning")
            require(not item.get("encrypted_content"), "foreign encrypted reasoning is unsupported")
            require(item.get("summary", []) == [], "reasoning summaries cannot replace raw local reasoning")
            parts = item.get("content")
            require(isinstance(parts, list) and bool(parts), "local reasoning content is missing")
            text = []
            for part in parts:
                require(isinstance(part, dict) and part.get("type") == "reasoning_text",
                        "unsupported local reasoning content")
                text.append(string(part.get("text"), "reasoning text"))
            messages.append({"role": "assistant", "content": "", "reasoning_content": "".join(text)})
        elif kind == "function_call":
            call_id = string(item.get("call_id"), "call_id")
            require(call_id and call_id not in seen_calls, "duplicate or empty call_id")
            seen_calls.add(call_id)
            pending_calls.add(call_id)
            name = string(item.get("name"), "function name")
            if item.get("namespace"):
                name = f"{item['namespace']}__{name}"
            # Prior calls may refer to a tool removed from the current catalog.
            # Keep their identity rather than discarding their history.
            arguments = string(item.get("arguments"), "arguments")
            call = {"id": call_id, "type": "function", "function": {
                "name": name, "arguments": arguments}}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            call_id = string(item.get("call_id"), "call_id")
            require(call_id in pending_calls, "orphaned or duplicate tool output")
            pending_calls.remove(call_id)
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": content_parts(item.get("output"))})
        else:
            raise ProtocolError(f"unsupported input item: {kind}")
    require(not pending_calls, "unresolved function calls")
    require(messages, "input is empty")
    limit = request.get("max_output_tokens", max_tokens)
    require(type(limit) is int and 1 <= limit <= max_tokens,
            f"max_output_tokens must be between 1 and {max_tokens}")
    body = {"model": backend_model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": limit,
            "enable_thinking": effort == "medium"}
    if tools:
        body["tools"] = tools
    choice = request.get("tool_choice", "auto")
    require(choice in ("auto", "none", "required"), "unsupported tool_choice")
    body["tool_choice"] = choice
    # Computer actions are serialized by the executor even if the model asks
    # for several; do not treat this hint as an execution guarantee.
    body["parallel_tool_calls"] = request.get("parallel_tool_calls", False)
    for key in ("temperature", "top_p"):
        if key in request:
            require(type(request[key]) in (int, float), f"{key} must be numeric")
            body[key] = request[key]
    return ConvertedRequest(body, names)


async def read_sse(content):
    """Parse events across arbitrary HTTP chunk boundaries, including CRLF."""
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    require(not data, "backend ended in the middle of an SSE event")
