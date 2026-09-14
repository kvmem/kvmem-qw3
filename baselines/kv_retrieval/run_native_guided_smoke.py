"""Bounded shared-GPU validation of production BF16 and guided/no-draft paths."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=18219)
    p.add_argument("--exercise-middecode", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = ROOT / "native_build/qw3"
    report = {"correctness_only": True, "performance_claim": False, "shared_gpu": True,
              "started": time.time(), "status": "starting", "responses": []}
    def save():
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    free = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip()) / 1024
    if free < 58:
        raise RuntimeError(f"only {free:.1f} GiB free; wait for headroom")
    command = [str(binary), "serve", "--model", str(REPO / "models/Qwen3.8-27B-BF16"),
        "--host", "127.0.0.1", "--port", str(args.port), "--ctx", "32768", "--kv-dtype", "fp16",
        "--prefill-chunk", "512", "--mtp-chain", "1", "--temp", "0", "-n", "1024" if args.exercise_middecode else "32",
        "--kvmem", "--kvmem-budget", "4096", "--kvmem-gen-budget", "2048",
        "--kvmem-block-tokens", "32", "--kvmem-sink-tokens", "128", "--kvmem-recent-tokens", "512",
        "--kvmem-gpu-memory-ratio", "0.96",
        "--kvmem-cpu-gb", "4", "--kvmem-nvme-gb", "0", "--kvmem-query-conditioned",
        "--kvmem-prefix-cache", "--kvmem-query-replay", "--kvmem-guided-reselect", "both",
        "--kvmem-middecode-trigger-tokens", "128" if args.exercise_middecode else "1024",
        "--kvmem-guided-query-tokens", "256", "--kvmem-guided-thinking-tokens", "0"]
    environment = dict(os.environ)
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        environment.pop(name, None)
    environment.update(QW3_BASELINE_NO_SPECULATIVE_DECODE="1", QW3_KVMEM_GUIDED_MTP="0",
                       QW3_HARNESS_STABLE_REASONING="1")
    report["command"] = command
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    save()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{args.port}"
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=log)
    report["server_pid"] = server.pid
    save()
    try:
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"native server exited: {server.returncode}")
            try:
                with http.open(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError("native readiness deadline")
        def request(body):
            index = len(report["responses"])
            (args.out / f"request_{index}.json").write_text(json.dumps(body) + "\n")
            req = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(),
                headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
            with http.open(req, timeout=240) as response:
                result = json.load(response)
            if result.get("type") != "message":
                raise AssertionError(result)
            report["responses"].append(result)
            save()
            return result
        tools = [{"name": "Bash", "description": "Read repository files", "input_schema": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
        short = {"model": "baseline", "max_tokens": 32, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Write a Python function that adds two integers."}]}
        request(short)
        history = [{"role": "user", "content": "Locate HfSafetensorsModelSource and explain how it maps tensor names. Inspect the supplied files and answer in three sentences."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "read_1", "name": "Bash", "input": {"command": "cat source.cpp"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_1", "content":
                (REPO / "src/kvmem_gc.cpp").read_text() + "\n" + (REPO / "src/model_source.cpp").read_text()}]}]
        if args.exercise_middecode:
            history[0]["content"] = ("Explain the supplied model-source and configuration code in 30 numbered paragraphs, "
                "covering tensor names, shapes, dtypes, validation and loading. Give detailed technical explanations "
                "of at least 2000 words. Do not call tools or request more files in this diagnostic.")
        body = {"model": "baseline", "max_tokens": 32, "thinking": {"type": "disabled"},
                "tools": tools, "messages": history}
        if args.exercise_middecode:
            body["tool_choice"] = {"type": "none"}
        request(body)
        # Keep the root user task fixed. New user instructions use the native
        # user-query path; continued tool observations exercise guided refresh.
        history += [{"role": "assistant", "content": [{"type": "tool_use", "id": "read_meta", "name": "Bash", "input": {"command": "git status --short"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_meta", "content": "No modified files."}]}]
        request(body)
        # Add an actual new tool observation to cross the generation reserve.
        history += [{"role": "assistant", "content": [{"type": "tool_use", "id": "read_2", "name": "Bash", "input": {"command": "cat config.cpp"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_2", "content": (REPO / "src/qwen_config.cpp").read_text()}]}]
        if args.exercise_middecode:
            body["max_tokens"] = 1024
        request(body)
        if args.exercise_middecode:
            # The file-inspection fixture often ends with a short tool call,
            # even with a long-answer request in its root instruction. Add a
            # bounded explicit generation task so a short answer is not mistaken
            # for coverage of the mid-generation state transition. This is a
            # correctness fixture only, outside agent quality measurements.
            history.append({"role": "user", "content":
                "For this diagnostic response, do not inspect any more files or call tools. "
                "Write every integer from 1 through 400 in order, one integer per line. "
                "Do not use ranges, ellipses, code, commentary or an abbreviated answer."})
            request(body)
            matches = re.findall(r'kvmem_middecode_refreshes=(\d+)', (args.out / "server.log").read_text())
            report["max_middecode_refreshes"] = max(map(int, matches), default=0)
            if report["max_middecode_refreshes"] < 1:
                raise AssertionError("fixture did not exercise mid-decode refresh")
        report["status"] = "requests_completed_audit_logs_required"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
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
