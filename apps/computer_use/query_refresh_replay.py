"""Replay exact saved desktop inputs to qualify query refresh, without actions."""
import asyncio
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

import thinking_budget_replay as original


ORIGINAL_SHA = "a599f1c21c0499ed1e00b7bb016bfa666fe2ff17c769ef9f419bc4e28533ca15"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact_body(source, budget):
    if (type(budget) is not int or budget != 0
            or source.get("enable_thinking") is not True
            or source.get("max_tokens") != 16384 or source.get("stream") is not True
            or source.get("stream_options") != {"include_usage": True}
            or "thinking_budget" in source or not source.get("tools")
            or any(key.startswith("kvmem_") for key in source)):
        raise ValueError("Require the exact original unmodified desktop generation profile")
    return copy.deepcopy(source)


def qualified_replayer():
    if digest(Path(original.__file__).read_bytes()) != ORIGINAL_SHA:
        raise ValueError("Original read-only transport implementation changed")
    source = inspect.getsource(original.replay)
    changes = (
        ("    body = replay_body(source, budget)\n", "    body = exact_body(source, budget)\n"),
        ('    wire = json.dumps(body, ensure_ascii=False).encode()\n',
         '    wire = json.dumps(body, ensure_ascii=False).encode()\n'
         '    if wire != source_bytes:\n'
         '        raise ValueError("Saved replay wire must preserve every original request byte")\n'),
        ('kind="readonly_saved_input_thinking_budget_replay"', 'kind="readonly_saved_input_query_refresh_replay"'),
        ('                  thinking_budget=budget, max_output_tokens=16384,',
         '                  original_generation_profile_unchanged=True, max_output_tokens=16384,'),
        ('request_id = "qw3-readonly-thinking-replay-"', 'request_id = "qw3-readonly-query-refresh-"'),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("Original read-only transport source anchor changed")
        source = source.replace(before, after, 1)
    namespace = dict(vars(original), exact_body=exact_body)
    exec(compile(source, "qualified-exact-query-refresh-replayer", "exec"), namespace)
    return namespace["replay"]


def pair_identity(inputs):
    if set(inputs) != {"seed", "pressure"}:
        raise ValueError("Require the original short and pressure requests")
    bodies = {name: json.loads(data) for name, data in inputs.items()}
    for name, body in bodies.items():
        exact_body(body, 0)
        if json.dumps(body, ensure_ascii=False).encode() != inputs[name]:
            raise ValueError("Input is not the preserved original wire representation")
        messages = body.get("messages")
        if not isinstance(messages, list) or len(messages) < 4:
            raise ValueError("Require the saved desktop root and score-query messages")
        if [m.get("role") for m in messages[:4]] != ["system", "developer", "user", "user"]:
            raise ValueError("Saved desktop task/query roles changed")
    identities = {}
    for index, label in ((2, "root_task"), (3, "score_query")):
        identity = [json.dumps(bodies[name]["messages"][index], ensure_ascii=False, sort_keys=True).encode()
                    for name in ("seed", "pressure")]
        if identity[0] != identity[1]:
            raise ValueError("Pressure request is not the same desktop task and query")
        identities[label + "_sha256"] = digest(identity[0])
    return dict(identities, input_sha256={name: digest(data) for name, data in inputs.items()})


async def sequence(inputs, output, endpoint, seconds=600, retire_seconds=60, expected_prompt_tokens=None):
    identity = pair_identity(inputs)
    replay = qualified_replayer()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    result = dict(kind="readonly_saved_desktop_query_refresh_sequence", started_at=time.time(),
                  source_identity=identity, request_order=["seed", "pressure"], rows=[],
                  original_generation_profile_unchanged=True, same_engine_endpoint=endpoint,
                  model_concurrency=1, tool_calls_executed=0, device_inputs=0,
                  business_success_claimed=False, termination_confirmed=False, status="starting")
    error = None
    try:
        for name in result["request_order"]:
            row = await replay(inputs[name], 0, output / name, endpoint=endpoint,
                               seconds=seconds, retire_seconds=retire_seconds)
            if (row["termination_confirmed"] is not True or row["status"] != "completed"
                    or row["tool_calls_executed"] != 0
                    or row["actual_request_sha256"] != identity["input_sha256"][name]):
                raise ValueError("Saved request identity, retirement or read-only scope changed")
            if expected_prompt_tokens is not None and row["usage"]["prompt_tokens"] != expected_prompt_tokens[name]:
                raise ValueError("Actual saved-input tokenization changed")
            result["rows"].append(dict(name=name, result=row,
                                       result_sha256=digest((output / name / "result.json").read_bytes())))
        result.update(status="completed", termination_confirmed=True)
    except BaseException as exc:
        error = exc
        result.update(status="halted-no-further-requests", error=type(exc).__name__,
                      unfinished_request_evidence_preserved=True)
    finally:
        result["finished_at"] = time.time()
        with (output / "sequence-result.json").open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
    if error is not None:
        raise error
    return result
