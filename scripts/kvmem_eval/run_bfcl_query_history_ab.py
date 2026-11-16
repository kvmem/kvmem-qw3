#!/usr/bin/env python3
"""Run paired, closed-loop BFCL episodes against an isolated local qw3 server.

Only simulated benchmark tools execute. Published reference calls are used for
grading, never injected as model actions. One fresh server owns each episode.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


DEFAULT_IDS = [103, 106, 107, 109, 110, 116, 131, 135]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def input_identity(messages, tools):
    # qw3 renders neither random tool-call IDs nor tool-result name fields.
    # Canonicalize JSON argument strings as the server renderer does.
    normalized = []
    for message in messages:
        item = {k: message[k] for k in ("role", "content", "reasoning_content") if k in message}
        if message.get("tool_calls"):
            item["tool_calls"] = []
            for call in message["tool_calls"]:
                function = copy.deepcopy(call["function"])
                if isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"])
                    except ValueError:
                        pass
                item["tool_calls"].append(function)
        normalized.append(item)
    return json_digest({"messages": normalized, "tools": tools})


def convert_schema(value, mapping):
    if isinstance(value, list):
        return [convert_schema(v, mapping) for v in value]
    if not isinstance(value, dict):
        return value
    return {k: mapping.get(v, v) if k == "type" and isinstance(v, str)
            else convert_schema(v, mapping) for k, v in value.items()}


def calls_to_python(tool_calls, allowed):
    """Construct only named calls with literal data; reject executable arguments."""
    calls = []
    for call in tool_calls:
        function = call["function"]
        name = function["name"]
        if name not in allowed or not name.isidentifier():
            raise ValueError(f"Unknown benchmark tool: {name}")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or any(not k.isidentifier() for k in arguments):
            raise ValueError("Expected named, literal tool arguments")
        text = name + "(" + ", ".join(f"{k}={v!r}" for k, v in arguments.items()) + ")"
        node = ast.parse(text, mode="eval").body
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        for kw in node.keywords:
            ast.literal_eval(kw.value)
        calls.append(text)
    return calls


def request_json(url, body=None, headers=None, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def compact_check(check):
    return {k: check[k] for k in ("valid", "error_type", "error_message") if k in check}


def server_config(args, group, directory, port):
    template = json.loads(args.launch_template.read_text())
    command = list(template["command"])
    command[0] = str(args.binary.resolve())
    replacements = {"--model": str(args.model.resolve()), "--host": "127.0.0.1",
        "--port": str(port), "--ctx": "131072", "--kvmem-budget": "32768",
        "--kvmem-prefill-budget": "32768", "--kvmem-guided-reselect": "off",
        "--temp": "0", "--seed": str(args.seed), "--dflash-model": str(args.dflash_model.resolve())}
    if group == "large_budget_control":
        replacements["--kvmem-budget"] = replacements["--kvmem-prefill-budget"] = "98304"
    for flag, value in replacements.items():
        assert flag in command, flag
        command[command.index(flag) + 1] = value
    command += ["--kvmem-query-history", "on" if group == "on" else "off",
        "--kvmem-query-history-decay", "0.5", "--kvmem-select-policy", "topk",
        "--kvmem-semantic-expansion", "none", "--kvmem-mandatory-parts", "system,root,query,policy",
        "--kvmem-action-attention", "on", "--kvmem-action-scope", "tool-call",
        "--kvmem-action-combine", "quota", "--kvmem-action-fraction", "0.5",
        "--kvmem-action-decay", "0.5", "--kvmem-action-interval", "16",
        "--kvmem-action-pressure", "on", "--kvmem-prefill-attention", "off",
        "--no-kvmem-recompute-query"]
    environment = {k: v for k, v in template["environment"].items()
                   if not k.startswith("QW3_KVMEM_ACTION_")}
    environment.pop("QW3_KVMEM_DUMP_WINDOWS", None)
    environment.update(QW3_KVMEM_DUMP_SCORES=str((directory / "retrieval.jsonl").resolve()),
        QW3_KVMEM_TRACE="1", QW3_KVMEM_PERF_TRACE="1", QW3_HARNESS_STABLE_REASONING="1")
    return command, environment


def run_episode(args, row, gold, functions, group, directory, execute, checker, state_check, response_check):
    directory.mkdir(parents=True, exist_ok=False)
    port = free_port()
    command, environment = server_config(args, group, directory, port)
    write_json(directory / "launch.json", {"command": command, "environment": environment})
    # Only task-specific server settings are inherited from the historical
    # launch manifest. Do not silently inherit unrelated QW3 switches.
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    env.update(environment)
    log_path = directory / "server.log"
    started = time.monotonic()
    process = None
    result = {"id": row["id"], "group": group, "status": "running",
        "query_count": len(row["question"]), "turns": [], "requests": 0,
        "completion_tokens": 0, "tool_calls": 0, "tool_execution_errors": 0}
    decoded = []
    messages = []
    accumulated_outputs = []
    instance_name = f"bfcl_live_{group}_{time.time_ns()}"
    allowed = {f["function"]["name"] for f in functions}
    write_json(directory / "status.json", result)
    try:
        with log_path.open("wb", buffering=0) as server_log:
            process = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
            write_json(directory / "process.json", {"pid": process.pid, "started_unix": time.time()})
            deadline = time.monotonic() + 180
            base_url = f"http://127.0.0.1:{port}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited during startup with {process.returncode}")
                try:
                    request_json(base_url + "/v1/models", timeout=2)
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(.5)
            print(f"START {row['id']} history={group}", flush=True)
            with (directory / "requests.jsonl").open("w") as request_file:
                for ti, user_messages in enumerate(row["question"]):
                    messages.extend(copy.deepcopy(user_messages))
                    turn_decoded = []
                    turn = {"turn_index": ti, "query": user_messages[0]["content"], "steps": [],
                        "tool_execution_errors": 0, "step_limit_reached": False}
                    for step in range(args.max_tool_steps + 1):
                        label = f"bfcl-{row['id'].rsplit('_', 1)[1]}-{group}-t{ti}-s{step}"
                        payload = {"model": args.model.name, "messages": copy.deepcopy(messages),
                            "tools": functions, "tool_choice": "auto", "temperature": 0,
                            "seed": args.seed, "max_tokens": args.max_tokens,
                            "enable_thinking": True, "thinking_budget": 4096,
                            "stream": False, "kvmem_trace_tag": label}
                        log_offset = log_path.stat().st_size
                        began = time.monotonic()
                        try:
                            response = request_json(base_url + "/v1/chat/completions", payload,
                                {"x-qw3-harness": "generic-tool-client", "x-qw3-request-id": label})
                        except RuntimeError as error:
                            # A model-generated loop can exhaust the common logical
                            # context ceiling before the tool-step ceiling. This is
                            # a task limit, not an infrastructure retry opportunity.
                            if not (str(error).startswith("HTTP 413:") and
                                    "prompt exceeds KV context" in str(error) and
                                    "request_limit=131072" in str(error)):
                                raise
                            turn["context_limit_reached"] = True
                            turn["context_limit_error"] = str(error)
                            turn["rejected_request_latency_s"] = time.monotonic() - began
                            write_json(directory / "context_limit.json", {
                                "turn_index": ti, "step": step, "label": label,
                                "error": str(error), "request": payload,
                                "input_digest": input_identity(messages, functions),
                                "model_failure_no_retry": True})
                            turn_decoded.append([])
                            print(f"CONTEXT_LIMIT {label} {error}", flush=True)
                            break
                        latency = time.monotonic() - began
                        server_delta = log_path.read_bytes()[log_offset:].decode(errors="replace")
                        choice = response["choices"][0]
                        message = choice["message"]
                        usage = response["usage"]
                        assert usage["completion_tokens"] > 0
                        tools = message.get("tool_calls") or []
                        record = {"label": label, "turn_index": ti, "step": step,
                            "input_digest": input_identity(messages, functions),
                            "latency_s": latency, "usage": usage,
                            "finish_reason": choice["finish_reason"],
                            "query_history_commits": re.findall(r"\[query-history\][^\n]*", server_delta),
                            "prefix_reuse": [line for line in server_delta.splitlines()
                                if "prefix" in line and ("hit" in line or "miss" in line)],
                            "selection_events": re.findall(r"\[kvmem-reselect-perf\][^\n]*", server_delta)}
                        request_file.write(json.dumps({**record, "request": payload, "response": response}, ensure_ascii=False) + "\n")
                        request_file.flush()
                        turn["steps"].append(record)
                        result["requests"] += 1
                        result["completion_tokens"] += usage["completion_tokens"]
                        messages.append(copy.deepcopy(message))
                        print(f"STEP {label} input={usage['prompt_tokens']} output={usage['completion_tokens']} tools={len(tools)} seconds={latency:.2f}", flush=True)
                        if not tools:
                            turn_decoded.append([])
                            if choice["finish_reason"] == "length":
                                turn["answer_truncated"] = True
                            break
                        try:
                            calls = calls_to_python(tools, allowed)
                        except (ValueError, KeyError, TypeError, SyntaxError) as error:
                            turn_decoded.append([])
                            turn["decode_error"] = str(error)
                            break
                        outputs, model_instances = execute(calls, row["initial_config"], row["involved_classes"],
                            instance_name, row["id"], long_context=True)
                        assert len(outputs) == len(tools)
                        turn_decoded.append(calls)
                        accumulated_outputs.extend(outputs)
                        result["tool_calls"] += len(calls)
                        for tool, output in zip(tools, outputs):
                            try:
                                parsed = json.loads(output)
                            except (TypeError, ValueError):
                                parsed = None
                            is_error = output.startswith("Error during execution: ") or (
                                isinstance(parsed, dict) and "error" in parsed)
                            turn["tool_execution_errors"] += int(is_error)
                            result["tool_execution_errors"] += int(is_error)
                            messages.append({"role": "tool", "tool_call_id": tool["id"],
                                "name": tool["function"]["name"], "content": output})
                        if step >= args.max_tool_steps:
                            turn["step_limit_reached"] = True
                            break
                    decoded.append(turn_decoded)
                    _, model_instances = execute([], row["initial_config"], row["involved_classes"],
                        instance_name, row["id"], long_context=True)
                    gold_outputs, gold_instances = execute(gold["ground_truth"][ti], row["initial_config"],
                        row["involved_classes"], instance_name + "_diagnostic_gold", row["id"], long_context=True)
                    state_result = state_check(model_instances, gold_instances)
                    response_result = response_check(accumulated_outputs, gold_outputs, ti)
                    turn["state_check"] = compact_check(state_result)
                    turn["response_check"] = compact_check(response_result)
                    turn["local_conditions_valid"] = state_result["valid"] and response_result["valid"]
                    result["turns"].append(turn)
                    write_json(directory / "status.json", result)
                    print(f"TURN {row['id']} history={group} turn={ti} conditions_valid={turn['local_conditions_valid']}", flush=True)
                    if turn["step_limit_reached"] or turn.get("context_limit_reached"):
                        break
            decoded.extend([[] for _ in range(len(row["question"]) - len(decoded))])
            grade = checker(decoded, gold["ground_truth"], row, "multi_turn_long_context",
                instance_name + "_official_score")
            write_json(directory / "official_grade.json", grade)
            write_json(directory / "decoded_calls.json", decoded)
            write_json(directory / "final_messages.json", messages)
            result.update(status="completed", official_grade=compact_check(grade),
                context_limit_exhausted=any(t.get("context_limit_reached") for t in result["turns"]),
                local_conditions_passed=sum(t["local_conditions_valid"] for t in result["turns"]),
                first_query_prompt_tokens=result["turns"][0]["steps"][0]["usage"]["prompt_tokens"],
                max_prompt_tokens=max(s["usage"]["prompt_tokens"] for t in result["turns"] for s in t["steps"]))
    except Exception as error:
        result.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        result["wall_seconds_including_startup"] = time.monotonic() - started
        write_json(directory / "result.json", result)
        write_json(directory / "status.json", result)
    print(f"DONE {row['id']} history={group} valid={result['official_grade']['valid']} seconds={result['wall_seconds_including_startup']:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("artifacts/bfcl_long_context_audit_20261015/source"))
    parser.add_argument("--audit-root", type=Path, default=Path("artifacts/bfcl_long_context_audit_20261015"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.8-27B-Q8_0.gguf"))
    parser.add_argument("--dflash-model", type=Path, default=Path("models/Qwen3.8-27B-DFlash2-BF16"))
    parser.add_argument("--launch-template", type=Path, default=Path("runtime/dsh_tool_guided_actionpressure50_20261010/launch.template.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", default=",".join(map(str, DEFAULT_IDS)))
    parser.add_argument("--groups", default="off,on")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-tool-steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ids = [int(x) for x in args.ids.split(",")]
    groups = args.groups.split(",")
    assert set(groups) <= {"off", "on", "large_budget_control"}
    args.output_dir.mkdir(exist_ok=True, parents=True)
    source_upstream = args.source_root / "upstream"
    sys.path.insert(0, str(source_upstream.resolve()))
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker, state_checker, response_checker

    rows = {int(r["id"].rsplit("_", 1)[1]): r for r in read_jsonl(args.source_root / "BFCL_v4_multi_turn_long_context.json")}
    gold = {r["id"]: r for r in read_jsonl(source_upstream / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json")}
    candidates = {int(r["id"].rsplit("_", 1)[1]): r for r in read_jsonl(args.audit_root / "analysis/recommended_candidates_32k.jsonl")}
    assert len(set(ids)) == len(ids) and set(ids) <= set(candidates)
    plan = {"ids": ids, "groups": groups, "seed": args.seed, "max_tokens": args.max_tokens,
        "max_tool_steps": args.max_tool_steps, "binary": str(args.binary.resolve()),
        "binary_sha256": digest(args.binary.read_bytes()), "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size, "model_mtime_ns": args.model.stat().st_mtime_ns,
        "reference_source_sha256": digest((args.source_root / "BFCL_v4_multi_turn_long_context.json").read_bytes()),
        "selection": "Preselected without model results, covering 4-7 user queries, linked chains, a mixed workflow and a shared scene.",
        "samples": [candidates[i] for i in ids],
        "protocol": "Closed-loop model tool calls, official simulator long_context=True, official full-conversation checker. One fresh server per episode; alternating pair order. Natural language answers are not graded by the official checker.",
        "grading": "official_grade is full conversation strict success. local_conditions_valid uses official state/response subchecks at that turn and may recover after an earlier failure; it is diagnostic, not official conversation accuracy."}
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        assert args.resume, "Output plan already exists; use --resume"
        assert json.loads(plan_path.read_text()) == plan, "Resume configuration changed"
    else:
        write_json(plan_path, plan)
    results = []
    for sequence, index in enumerate(ids):
        row = rows[index]
        functions = []
        for klass in row["involved_classes"]:
            for f in read_jsonl(source_upstream / "bfcl_eval/data/multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[klass]):
                functions.append({"type": "function", "function": {
                    "name": f["name"], "description": f["description"],
                    "parameters": convert_schema(f["parameters"], GORILLA_TO_OPENAPI)}})
        assert len({f["function"]["name"] for f in functions}) == len(functions)
        order = groups if sequence % 2 == 0 else list(reversed(groups))
        for group in order:
            base = args.output_dir / f"sample_{index}" / group
            attempts = [base] + sorted(base.parent.glob(base.name + "_infra_retry_*"))
            completed = [p / "result.json" for p in attempts if (p / "result.json").exists()
                         and json.loads((p / "result.json").read_text())["status"] == "completed"]
            assert len(completed) <= 1, "Multiple completed attempts"
            if completed:
                assert args.resume
                results.append(json.loads(completed[0].read_text()))
                continue
            # Failed infrastructure attempts remain visible and are never
            # overwritten or silently counted as task successes/failures.
            directory = base
            attempt = 1
            while directory.exists():
                directory = base.with_name(base.name + f"_infra_retry_{attempt}")
                attempt += 1
            result = run_episode(args, row, gold[row["id"]], functions, group, directory,
                execute_multi_turn_func_call, multi_turn_checker, state_checker, response_checker)
            results.append(result)
            write_json(args.output_dir / "results.json", results)
    print("ALL_EPISODES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
