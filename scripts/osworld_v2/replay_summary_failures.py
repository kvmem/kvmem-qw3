#!/usr/bin/env python3
"""Replay saved failing ~193K-token compactions, without executing GUI actions."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/OSWorld-V2"))
from mm_agents.qwen35vl_agent import Qwen35VLAgent
from mm_agents.qw3_summary_history import SummaryHistory


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def restore_images(messages, directory):
    by_hash = {}
    for path in sorted((directory / "agent_inputs").glob("step_*.png")):
        by_hash[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    for message in messages:
        for part in message["content"]:
            if part["type"] != "image_url":
                continue
            reference = part["image_url"]
            data = by_hash[reference["sha256"]].read_bytes()
            assert len(data) == reference["decoded_bytes"]
            part["image_url"] = {"url": "data:image/png;base64," + base64.b64encode(data).decode()}
    return messages


def count_images(messages):
    return sum(part["type"] == "image_url" for message in messages for part in message["content"])


def payload(messages, max_tokens):
    return {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "top_k": 1, "enable_thinking": False, "preserve_thinking": False, "thinking_budget": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["005", "007", "008", "010", "012"])
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(QW3_OSWORLD_REQUEST_CANCEL="1", OSWORLD_OPENAI_TIMEOUT="600")
    env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update(QW3_PREFIX_CACHE_TRACE="1", QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES="1", QW3_VISION_GPU_CACHE_MIB="4096")
    base = f"http://127.0.0.1:{args.port}"
    command = [str(ROOT / "build/qw3"), "serve", "--model", str(ROOT / "models/Qwen3.8-27B-Q8_0.gguf"),
               "--vision-model", str(ROOT / "models/Qwen3.8-27B-BF16"), "--vision-device", "cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--ctx", "262144", "--kv-dtype", "fp16",
               "--prefill-chunk", "2048", "--mtp-chain", "4", "--continuous-batching", "--prefix-cache",
               "--max-active", "1", "--temp", "0", "--thinking-budget", "0", "-n", "16384"]
    log_path = args.out / "server.log"
    results = []
    with log_path.open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError("Replay server exited during startup")
                try:
                    httpx.get(base + "/health", timeout=1, trust_env=False).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise RuntimeError("Replay server did not become ready")
            for task in args.tasks:
                directory = args.source_run / "osworld/pyautogui/screenshot/gpt-qw3/tasks" / task
                old = [e for e in read_rows(directory / "history_compaction.jsonl") if e["status"] == "error"][-1]
                step = old["step_num"]
                print(json.dumps({"task": task, "step": step, "stage": "restoring_original_history"}), flush=True)
                source = restore_images(json.loads((directory / "history_summaries" / f"step_{step:04d}_input.json").read_text()), directory)
                history = source[:-1]
                assert history[-1]["role"] == "assistant" and history[-2]["role"] == "user"
                old_instruction = next(p["text"] for p in history[1]["content"] if p["type"] == "text")
                instruction = old_instruction.split("\n\nSummary of earlier interaction (historical context):\n", 1)[0]
                current_path = directory / "agent_inputs" / f"step_{step:04d}.png"
                current_image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(current_path.read_bytes()).decode()}}
                with Image.open(current_path) as im:
                    image_tokens = im.width // 32 * (im.height // 32) + 16
                agent = Qwen35VLAgent(model="gpt-qw3", base_url=base + "/v1", api_key="replay",
                                      max_tokens=16384, enable_thinking=False, preserve_thinking=False,
                                      thinking_budget=0, history_mode="summary", context_tokens=262144,
                                      summary_trigger_tokens=196608, summary_max_tokens=4096)
                agent.set_trace_dir(str(args.out / task))
                print(json.dumps({"task": task, "stage": "prefill_original_previous_action", "old_images": count_images(history)}), flush=True)
                before = len(re.findall(r"native continuous_batch:[^\n]+", log_path.read_text()))
                agent.call_llm(payload(history[:-1], 1), agent.model)
                warm_usage = agent.last_api_metadata["usage"]
                assert warm_usage["prompt_tokens"] > 180000
                policy = agent.summary_history
                policy.start_step = old["history_start_before"]
                policy.last_prompt_tokens = warm_usage["prompt_tokens"]
                agent.responses = [history[-1]["content"][0]["text"]]

                def render(start, summary):
                    if start != step:
                        return history + [{"role": "user", "content": agent._wrap_tool_response([current_image])}]
                    return [history[0], {"role": "user", "content": [current_image, {"type": "text", "text":
                        instruction + "\n\nSummary of earlier interaction (historical context):\n" + summary}]}]

                print(json.dumps({"task": task, "stage": "retry_original_failed_compaction"}), flush=True)
                replacement = policy.prepare(agent, render, step, image_tokens)
                event = policy.event
                assert event and event["status"] == "success" and policy.start_step == step
                assert count_images(replacement) == 1
                assert len(policy.summary) >= 80
                assert not re.search(r"</?tool_call|<function|</?think", policy.summary, re.I)
                response = agent.call_llm(payload(replacement, 256), agent.model)
                follow = replacement + [{"role": "assistant", "content": [{"type": "text", "text": response}]},
                                        {"role": "user", "content": agent._wrap_tool_response([current_image])}]
                agent.call_llm(payload(follow, 64), agent.model)
                lines = re.findall(r"native continuous_batch:[^\n]+", log_path.read_text())[before:]
                assert len(lines) == 3 + len(event["attempts"]), lines
                first_summary = lines[1]
                reused = int(re.search(r"reused_tokens=(\d+)", first_summary)[1])
                prompt = int(re.search(r"prompt_tokens=(\d+)", first_summary)[1])
                assert reused / prompt > 0.95, first_summary
                assert int(re.search(r"reused_tokens=(\d+)", lines[-1])[1]) > 0, lines[-1]
                assert "admission rejected" not in log_path.read_text()
                result = {"task": task, "step": step, "ok": True, "source_prompt_tokens": prompt,
                          "summary_prefix_reused_tokens": reused, "summary_chars": len(policy.summary),
                          "summary_output_tokens": event["api"]["usage"]["completion_tokens"],
                          "summary_attempts": len(event["attempts"]), "summary_wall_sec": event["wall_time_ms"] / 1000,
                          "replacement_images": count_images(replacement), "following_action_prefix_hit": True}
                results.append(result)
                (args.out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)
            (args.out / "report.json").write_text(json.dumps({"ok": True, "cases": results, "gui_actions_executed": False}, indent=2) + "\n")
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
