#!/usr/bin/env python3
"""Real-model validation of whole-history summary boundaries without GUI actions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/OSWorld-V2"))
from mm_agents.qwen35vl_agent import Qwen35VLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update(QW3_PREFIX_CACHE_TRACE="1", QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES="1",
               QW3_VISION_GPU_CACHE_MIB="4096", QW3_OSWORLD_REQUEST_CANCEL="1", OSWORLD_OPENAI_TIMEOUT="600")
    os.environ.update(QW3_OSWORLD_REQUEST_CANCEL="1", OSWORLD_OPENAI_TIMEOUT="600")
    command = [str(ROOT / "build/qw3"), "serve", "--model", str(ROOT / "models/Qwen3.8-27B-Q8_0.gguf"),
               "--vision-model", str(ROOT / "models/Qwen3.8-27B-BF16"), "--vision-device", "cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--ctx", "262144", "--kv-dtype", "fp16",
               "--prefill-chunk", "2048", "--mtp-chain", "4", "--continuous-batching", "--prefix-cache",
               "--max-active", "1", "--temp", "0", "--thinking-budget", "0", "-n", "16384"]
    log_path = args.out / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        try:
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError("canary server exited")
                try:
                    httpx.get(base + "/health", timeout=1, trust_env=False).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise RuntimeError("server did not become ready")
            agent = Qwen35VLAgent(model="gpt-qw3", base_url=base + "/v1", api_key="test",
                                  enable_thinking=False, preserve_thinking=False, thinking_budget=0,
                                  max_tokens=256, temperature=0, history_mode="summary", context_tokens=262144,
                                  summary_trigger_tokens=16384, summary_max_tokens=2048,
                                  history_n=2, image_max=2, fold_size=1)
            agent.set_trace_dir(str(args.out / "agent"))
            files = sorted(args.images.glob("step_*.png"))
            assert files
            rows = []
            for step in range(1, 19):
                response, actions = agent.predict(
                    "This is a read-only observation test. Keep the exact note 'marker 4321, /tmp/keep-me.txt'. "
                    "On every ordinary action turn, call computer_use with action=screenshot, and do nothing else. "
                    "Do not terminate the test or click or type.",
                    {"screenshot": files[(step - 1) % len(files)].read_bytes()})
                trace = agent.last_step_trace
                assert actions, response
                assert "<think>" not in response, response
                assert not trace["compaction"]["triggered"]
                assert trace["api"]["usage"]["prompt_tokens"] < trace["estimated_prompt_tokens"]
                event = trace["summary_compaction"]
                if event:
                    assert event["status"] == "success"
                    assert trace["compaction"]["active_image_count"] == 1
                    assert "4321" in agent.summary_history.summary
                    assert "/tmp/keep-me.txt" in agent.summary_history.summary
                row = {"step": step, "images": trace["compaction"]["active_image_count"],
                       "prompt_tokens": trace["api"]["usage"]["prompt_tokens"],
                       "ttft": trace["api"]["server_timing"]["server_ttft_sec"],
                       "summary": bool(event), "summary_ms": event["wall_time_ms"] if event else None}
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.out / "progress.json").write_text(json.dumps(rows, indent=2))
                if sum(r["summary"] for r in rows) >= 2 and not event and rows[-2]["summary"]:
                    break
            assert sum(r["summary"] for r in rows) >= 2
            lines = re.findall(r"native continuous_batch:[^\n]+", log_path.read_text())
            assert len(lines) == len(rows) + sum(r["summary"] for r in rows)
            index = 0
            for row in rows:
                if row["summary"]:
                    summary_line = lines[index]
                    summary_reused = int(re.search(r"reused_tokens=(\d+)", summary_line)[1])
                    summary_prompt = int(re.search(r"prompt_tokens=(\d+)", summary_line)[1])
                    assert summary_reused / summary_prompt > 0.9, summary_line
                    row["summary_reused_tokens"] = summary_reused
                    index += 1  # additional summary-generation request
                line = lines[index]
                row["reused_tokens"] = int(re.search(r"reused_tokens=(\d+)", line)[1])
                if row["step"] > 1 and not row["summary"]:
                    assert row["reused_tokens"] > 0, line
                    assert row["ttft"] < 5, row
                index += 1
            assert "admission rejected" not in log_path.read_text()
            (args.out / "report.json").write_text(json.dumps({"ok": True, "rows": rows}, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
