"""Self-contained real-model HTTP validation; owns and cleans only its server."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=18223)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "shared_gpu": True, "performance_claim": False, "responses": []}
    def save():
        (args.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "serve_infllm.py"), "--model",
        str(REPO / "models/Qwen3.8-27B-BF16"), "--out", str(args.out / "server"),
        "--port", str(args.port), "--shared-gpu", "--temperature", "0", "--max-context", "32768"]
    report["command"] = command
    save()
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=log)
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited: {server.returncode}")
            try:
                with http.open(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError("server readiness")
        def request(body, path="/v1/messages"):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            with http.open(req, timeout=240) as response:
                raw = response.read().decode()
            result = raw if body.get("stream") else json.loads(raw)
            report["responses"].append(result)
            save()
            return result
        short = {"model": "baseline", "max_tokens": 32, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Write a Python function that adds two integers."}]}
        result = request(short)
        assert result["type"] == "message" and result["usage"]["output_tokens"] == 32, result
        counting = copy.deepcopy(short)
        del counting["max_tokens"]
        assert request(counting, "/v1/messages/count_tokens")["input_tokens"] == result["usage"]["input_tokens"]
        # Repeat exact prompt after generation: restores an eligible snapshot or
        # explicitly cold-starts; never rewinds hybrid recurrent state by length.
        thinking = {**short, "stream": True, "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 2}}
        wire = request(thinking)
        assert "event: message_stop" in wire and "event: error" not in wire, wire
        assert "thinking_delta" in wire and "text_delta" in wire, wire
        fixture = json.loads((ROOT / "reports/native_guided_smoke_20260912_v4/request_1.json").read_text())
        fixture.update(stream=True, max_tokens=128)
        wire = request(fixture)
        assert "event: message_stop" in wire and "event: error" not in wire, wire
        assert '"type":"tool_use"' in wire, wire
        # Follow actual emitted assistant blocks with a new tool observation.
        events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]
        blocks, arguments = {}, {}
        for event in events:
            if event["type"] == "content_block_start":
                blocks[event["index"]] = event["content_block"]
            if event["type"] == "content_block_delta":
                idx, delta = event["index"], event["delta"]
                if "text" in delta:
                    blocks[idx]["text"] = blocks[idx].get("text", "") + delta["text"]
                if "partial_json" in delta:
                    arguments[idx] = arguments.get(idx, "") + delta["partial_json"]
        for idx, value in arguments.items():
            blocks[idx]["input"] = json.loads(value)
        uses = [b for b in blocks.values() if b["type"] == "tool_use"]
        assert uses, blocks
        fixture["messages"] += [{"role": "assistant", "content": list(blocks.values())},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"],
                "content": "No additional matches. The declaration is in the supplied source."}]}]
        fixture["max_tokens"] = 256
        wire = request(fixture)
        assert "event: error" not in wire, wire
        # Force a genuine small tail rewrite within the most recent observation.
        fixture["messages"][-1]["content"][0]["content"] += " Inspect tensor shapes as well."
        wire = request(fixture)
        assert "event: error" not in wire, wire
        rows = [json.loads(line) for line in (args.out / "server/requests.jsonl").read_text().splitlines()]
        assert all(row["status"] == "completed" for row in rows), [r["status"] for r in rows]
        assert all(n > 16 for n in rows[-1]["global_blocks_per_layer"]), rows[-1]
        assert rows[-2]["resume_origin"] > 0 and rows[-1]["resume_origin"] > 0, rows[-2:]
        assert rows[-1]["reuse_reason"] == "checkpoint", rows[-1]
        report.update(status="passed_http_generation_tool_stream_reuse_and_rewrite",
            requests_completed=len(rows), final_reuse=[{k: r[k] for k in (
                "resume_origin", "prompt_tokens", "prefill_tokens", "reuse_reason")} for r in rows[-2:]])
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
        report["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
