"""Replay a saved inference request without a desktop or tool dispatcher."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

import aiohttp
from jsonschema import Draft202012Validator


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay_body(source, budget):
    if type(budget) is not int or budget not in (0, 8192):
        raise ValueError("Use the declared uncapped or 8192-token thinking budget")
    body = copy.deepcopy(source)
    if (body.get("enable_thinking") is not True or body.get("max_tokens") != 16384
            or body.get("stream") is not True or body.get("stream_options") != {"include_usage": True}
            or "thinking_budget" in body):
        raise ValueError("Require the original uncapped 16K streaming request")
    body["thinking_budget"] = budget
    return body


class Stream:
    def __init__(self):
        self.reasoning_chars = 0
        self.content_chars = 0
        self.tool_deltas = 0
        self.calls = {}
        self.finish_reason = None
        self.usage = None
        self.timing = None
        self.done = False
        self.events = 0

    def feed(self, data):
        if self.done:
            raise ValueError("Unexpected event after the stream ended")
        if data == b"[DONE]":
            self.done = True
            return False
        event = json.loads(data)
        if not isinstance(event, dict) or event.get("error"):
            raise ValueError("Model stream returned an error or invalid event")
        self.events += 1
        if event.get("usage") is not None:
            self.usage = event["usage"]
        if event.get("timing") is not None:
            self.timing = event["timing"]
        visible = False
        for choice in event.get("choices", []):
            if self.finish_reason is not None:
                raise ValueError("Model choice arrived after its terminal marker")
            if choice.get("index", 0) != 0:
                raise ValueError("Replay admits only the original single model choice")
            reason = choice.get("finish_reason")
            if reason is not None:
                if self.finish_reason is not None:
                    raise ValueError("Duplicate terminal model choice")
                self.finish_reason = reason
            delta = choice.get("delta", {})
            if not isinstance(delta, dict): raise ValueError("Invalid model delta")
            for key, attribute in (("reasoning_content", "reasoning_chars"), ("content", "content_chars")):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str): raise ValueError("Invalid model text delta")
                    if key == "reasoning_content" and value and (self.content_chars or self.calls):
                        raise ValueError("Reasoning arrived after final output")
                    setattr(self, attribute, getattr(self, attribute) + len(value))
                    visible = visible or bool(value)
            for call in delta.get("tool_calls", []):
                index = call.get("index")
                if type(index) is not int or not 0 <= index < 32:
                    raise ValueError("Tool stream must identify its call index")
                current = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                function = call.get("function", {})
                if call.get("id"):
                    if not isinstance(call["id"], str) or current["id"] not in ("", call["id"]):
                        raise ValueError("Model changed its tool call identity")
                    current["id"] = call["id"]
                for key, value in (("name", function.get("name", "")), ("arguments", function.get("arguments", ""))):
                    if not isinstance(value, str): raise ValueError("Invalid tool delta")
                    current[key] += value
                self.tool_deltas += 1
                visible = True
        return visible

    def summary(self, tools):
        if not self.done or self.finish_reason not in ("stop", "length", "tool_calls"):
            raise ValueError("Model stream has no confirmed terminal marker")
        if self.finish_reason == "tool_calls" and not self.calls:
            raise ValueError("Model declared tool output without a call")
        usage = self.usage
        if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                or type(usage.get("completion_tokens")) is not int or usage["prompt_tokens"] < 0
                or not 0 <= usage["completion_tokens"] <= 16384):
            raise ValueError("The actual backend output budget or usage is unconfirmed")
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        calls = []
        for index, call in sorted(self.calls.items()):
            valid = bool(call["id"] and call["name"] in schemas)
            error = None
            try:
                arguments = json.loads(call["arguments"])
                if not isinstance(arguments, dict) or not valid:
                    raise ValueError("Tool identity or arguments are invalid")
                errors = list(Draft202012Validator(schemas[call["name"]]).iter_errors(arguments))
                valid = not errors
                if errors: error = "arguments do not satisfy the original tool schema"
            except (ValueError, TypeError):
                valid = False
                error = "tool identity or argument JSON is invalid"
            # No executor exists in this module. Even a valid call is only data.
            calls.append(dict(index=index, name=call["name"], arguments_sha256=digest(call["arguments"].encode()),
                              original_schema_valid=valid, error=error, executed=False))
        return dict(finish_reason=self.finish_reason, usage=usage, backend_timing=self.timing,
                    reasoning_characters=self.reasoning_chars, content_characters=self.content_chars,
                    tool_call_delta_events=self.tool_deltas, tool_calls=calls,
                    valid_generated_tool_calls=sum(c["original_schema_valid"] for c in calls),
                    reasoning_only_length_stop=bool(self.finish_reason == "length" and self.reasoning_chars
                                                   and not self.content_chars and not self.tool_deltas),
                    answer_or_tool_output_generated=bool(self.content_chars or self.tool_deltas),
                    model_output_is_not_business_success=True, tool_calls_executed=0)


async def confirm_termination(session, endpoint, request_id, seconds=60):
    deadline = time.monotonic() + seconds
    while True:
        async with session.post(endpoint + "/v1/requests/" + request_id + "/cancel",
                                timeout=aiohttp.ClientTimeout(total=min(5, seconds))) as response:
            response.raise_for_status()
            receipt = await response.json()
        if receipt.get("cancelled") is not True:
            raise ValueError("Own model request cancellation acknowledgement is invalid")
        if receipt.get("completed") is True:
            return receipt
        if time.monotonic() >= deadline:
            raise TimeoutError("Own model request retirement remains unconfirmed")
        await asyncio.sleep(min(.2, seconds))


async def replay(source_bytes, budget, output, endpoint="http://127.0.0.1:18080", seconds=600, retire_seconds=60):
    address = urlparse(endpoint)
    if (address.scheme != "http" or address.hostname != "127.0.0.1" or address.path not in ("", "/")
            or address.username or address.password or address.query or address.fragment):
        raise ValueError("Replay contacts only the explicit local model endpoint")
    source = json.loads(source_bytes)
    body = replay_body(source, budget)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    wire = json.dumps(body, ensure_ascii=False).encode()
    (output / "request.json").write_bytes(wire)
    request_id = "qw3-readonly-thinking-replay-" + uuid.uuid4().hex
    started = time.monotonic()
    result = dict(kind="readonly_saved_input_thinking_budget_replay", started_at=time.time(), request_id=request_id,
                  source_request_sha256=digest(source_bytes), actual_request_sha256=digest(wire),
                  thinking_budget=budget, max_output_tokens=16384, tool_calls_executed=0, device_inputs=0,
                  business_success_claimed=False, original_input_preserved=True,
                  first_model_delta_seconds=None, termination_confirmed=False, status="starting")
    stream = Stream()
    post_started = False
    error = None
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            post_started = True
            async with session.post(endpoint.rstrip("/") + "/v1/chat/completions", data=wire,
                    headers={"Content-Type": "application/json", "x-qw3-request-id": request_id},
                    timeout=aiohttp.ClientTimeout(total=seconds, sock_read=seconds)) as response:
                response.raise_for_status()
                if response.content_type != "text/event-stream":
                    raise ValueError("Saved inference replay requires the original streaming response")
                with (output / "backend.jsonl").open("xb") as backend:
                    async for raw in response.content:
                        if not raw.startswith(b"data:"): continue
                        data = raw[5:].strip()
                        visible = stream.feed(data)
                        if data != b"[DONE]": backend.write(data + b"\n")
                        if visible and result["first_model_delta_seconds"] is None:
                            result["first_model_delta_seconds"] = time.monotonic() - started
                        if stream.done: break
            result.update(stream.summary(source["tools"]), status="completed")
        except BaseException as exc:
            error = exc
            result.update(status="failed", error=type(exc).__name__)
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            if post_started:
                try:
                    result["retirement_receipt"] = await confirm_termination(session, endpoint.rstrip("/"), request_id, retire_seconds)
                    result["termination_confirmed"] = True
                except BaseException as exc:
                    result.update(status="failed-retirement-unconfirmed", retirement_error=type(exc).__name__)
                    error = exc
            result["finished_at"] = time.time()
            with (output / "result.json").open("x") as file:
                file.write(json.dumps(result, indent=2) + "\n")
    if error is not None: raise error
    return result
