#!/usr/bin/env python3
"""Manual GPU integration test using the actual HTTP renderer and MTP or DFlash."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO / "models/Qwen3.5-4B-Q8_0-with-MTP.gguf")
    parser.add_argument("--dflash", type=Path, help="Use DFlash instead of native MTP")
    parser.add_argument("--summary-tokens", type=int, default=None,
                        help="Omit for natural summaries; 0 disables; positive uses an explicit cap")
    args = parser.parse_args()
    enabled = args.summary_tokens != 0
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), "serve", "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(port), "--ctx", "8192",
        "--prefill-chunk", "128", "--paged-kv", "--kv-page-size", "16",
        "--kv-dtype", "fp8",
        "--no-continuous-batching", "--enable-thinking", "--thinking-budget", "1600",
        "--kvmem", "--kvmem-prefix-cache", "--kvmem-query-conditioned",
        "--kvmem-update-mode", "step", "--kvmem-repository-blocks", "256",
        "--kvmem-repository-gc", "delete", "--kvmem-block-tokens", "32",
        "--kvmem-budget", "1024", "--kvmem-prefill-budget", "1024",
        "--kvmem-gen-budget", "4096", "--kvmem-sink-blocks", "1",
        "--kvmem-recent-blocks", "4", "--kvmem-index-placement", "cpu",
        "--kvmem-cpu-gb", "1", "--kvmem-gpu-memory-ratio", "1.0",
        "--thinking-summary-min-tokens", "1024"]
    if args.dflash:
        command += ["--speculative-method", "dflash2", "--dflash-model", str(args.dflash.resolve())]
    else:
        command += ["--speculative-method", "mtp", "--mtp-chain", "4"]
    if args.summary_tokens is None:
        command += ["--thinking-summary"]
    else:
        command += ["--thinking-summary-tokens", str(args.summary_tokens)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    env["QW3_KVMEM_PREFIX_CACHE_TRACE"] = "1"
    log_path = args.output / "server.log"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    records = []
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}: {log_path}")
                try:
                    with opener.open(base + "/health", timeout=2) as response:
                        if response.status == 200: break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("server readiness timeout")

            def request(messages, **overrides):
                payload = {"model": "test", "messages": messages, "temperature": 0,
                           "max_tokens": 2400, "enable_thinking": True, **overrides}
                call = urllib.request.Request(base + "/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                    "x-qw3-harness": "generic-tool-client"})
                with opener.open(call, timeout=300) as response:
                    if not payload.get("stream"):
                        result = json.load(response)
                    else:
                        result = {"reasoning": "", "content": "", "done": False}
                        for raw in response:
                            if not raw.startswith(b"data: "): continue
                            item = raw[6:].strip()
                            if item == b"[DONE]": result["done"] = True; break
                            event = json.loads(item)
                            if event.get("error"): raise RuntimeError(str(event["error"]))
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {})
                                result["reasoning"] += delta.get("reasoning_content", "")
                                result["content"] += delta.get("content", "")
                                if choice.get("finish_reason"): result["finish_reason"] = choice["finish_reason"]
                records.append({"request": payload, "response": result})
                if "PRIVATE REASONING COMPACTION" in json.dumps(result):
                    raise AssertionError("private summary instruction leaked")
                return result

            messages = [{"role": "user", "content":
                "Review a Python memoization decorator for recursive functions: explain how to handle "
                "mutable arguments, exceptions, recursion cycles, and concurrent calls. Compare at "
                "least three designs before choosing one. Enumerate subtle races, counterexamples and "
                "memory-ordering guarantees for each approach, analyze correctness and liveness, "
                "then revisit every counterexample under your selected design. "
                "Give only the recommended design in the answer."}]
            first = request(messages)
            assistant = first["choices"][0]["message"]
            if not assistant.get("reasoning_content"):
                raise AssertionError("missing returned reasoning")
            messages += [{"role": "assistant", "content": assistant.get("content", ""),
                          "reasoning_content": assistant["reasoning_content"]},
                         {"role": "user", "content": "Now state just the main tradeoff in one sentence."}]
            request(messages, thinking_summary_tokens=0, thinking_budget=64, max_tokens=192)
            streamed = request([{"role": "user", "content":
                "Plan a robust streaming CSV parser, comparing quoting, escaped quotes, multiline "
                "fields, invalid UTF-8 and chunk boundaries in detail before giving a short recommendation."}], stream=True)
            if not streamed["done"] or not streamed["reasoning"]:
                raise AssertionError("stream did not finish with reasoning")
            request([{"role": "user", "content": "What is 17 times 23? Check the arithmetic."}],
                    thinking_budget=64, max_tokens=192)
            incomplete = request([{"role": "user", "content":
                "Work through all cases of a lock-free concurrent map algorithm before proposing a design."}],
                max_tokens=8, thinking_budget=0)
            if incomplete["choices"][0]["finish_reason"] != "length":
                raise AssertionError("incomplete reasoning lost length finish reason")

            # Exercise the protocol and buffered-tool route used by DeepSWE.
            body = {"model": "test", "max_tokens": 768, "temperature": 0,
                "thinking": {"type": "enabled", "budget_tokens": 320},
                "system": "You are Claude Code, a coding assistant.",
                "tools": [{"name": "Read", "description": "Read a source file.",
                    "input_schema": {"type": "object", "properties": {
                        "file_path": {"type": "string"}}, "required": ["file_path"]}}],
                "messages": [{"role": "user", "content":
                    "We need to review a recursive memoization decorator. Consider mutable arguments, "
                    "exception caching, recursive cycles and concurrent calls before choosing what "
                    "to inspect. Then call Read on src/cache.py. Do not invent file contents."}]}
            call = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                         "User-Agent": "claude-cli/2.1.148 (thinking compaction probe)"})
            with opener.open(call, timeout=300) as response:
                anthropic = json.load(response)
            records.append({"request": body, "response": anthropic})
            if anthropic.get("type") != "message" or not any(
                block.get("type") == "tool_use" and block.get("name") == "Read"
                for block in anthropic.get("content", [])):
                raise AssertionError("Anthropic buffered-tool route did not return the requested Read call")

            if enabled:
                # Force an unterminated summary and verify preservation without
                # replaying the original reasoning. Low minimum is test-only.
                request([{"role": "user", "content":
                    "Compare linear probing and chaining hash tables, including load factor and deletion."}],
                    thinking_summary_tokens=1, thinking_summary_min_tokens=1,
                    thinking_budget=320, max_tokens=768)
                # Close after a heartbeat from inside buffered reasoning, then
                # issue a fresh request to verify rollback after cancellation.
                body = {"model": "test", "stream": True, "temperature": 0,
                    "max_tokens": 2304, "thinking_budget": 2048, "enable_thinking": True,
                    "messages": [{"role": "user", "content":
                        "Prove correctness of a concurrent lock-free hash map with memory reclamation. "
                        "Work through every race, ABA case and memory ordering in detail before answering."}]}
                call = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "x-qw3-harness": "generic-tool-client"})
                started = time.monotonic()
                with opener.open(call, timeout=300) as response:
                    for raw in response:
                        if raw.startswith(b"data: ") and time.monotonic() - started > 1:
                            if raw[6:].strip() == b"[DONE]":
                                raise AssertionError("cancellation probe completed before disconnect")
                            break
                    else:
                        raise AssertionError("cancellation probe produced no heartbeat")
                request([{"role": "user", "content": "What is two plus three?"}],
                        max_tokens=64, thinking_budget=16)
            text = log_path.read_text(errors="replace")
            compacted = text.count("[thinking-compaction] status=compacted")
            if enabled and compacted == 0:
                raise AssertionError("no actual compaction occurred")
            if not enabled and compacted != 0:
                raise AssertionError("baseline unexpectedly compacted reasoning")
            if "kvmem prefix-cache HIT" not in text:
                raise AssertionError("compressed history did not resume prefix cache")
            if enabled and ("status=short-thinking" not in text or
                    "status=incomplete-thinking" not in text or "status=cancelled" not in text):
                raise AssertionError("fallback paths were not exercised")
            events = [dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
                      for line in text.splitlines() if "[thinking-compaction]" in line]
            for event in events:
                if event.get("status") == "short-thinking" and any(
                        float(event.get(key, -1)) != 0 for key in ("summary_ms", "rollback_ms", "replay_ms")):
                    raise AssertionError("short reasoning performed summary/rollback/replay work")
                if event.get("status") == "invalid-summary" and float(event.get("replay_ms", -1)) != 0:
                    raise AssertionError("invalid summary replayed original reasoning")
            if enabled and not any(row.get("status") == "invalid-summary" for row in events):
                raise AssertionError("invalid-summary restoration was not exercised")
            if re.search(r"POST .* -> 500|terminal_status=error", text):
                raise AssertionError("server reported an inference error")
            (args.output / "result.json").write_text(json.dumps({"passed": True,
                "model": str(args.model.resolve()), "route": "dflash2" if args.dflash else "mtp",
                "summary_tokens": args.summary_tokens, "compacted_requests": compacted,
                "events": events, "records": records}, indent=2) + "\n")
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
    print(json.dumps({"passed": True, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
