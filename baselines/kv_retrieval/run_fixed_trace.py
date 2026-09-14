"""Fixed-input probes for state/capacity calibration, TTFT and short decode.

The default one-token mode does not measure decode. Longer bounded responses
provide free-generation decode measurements; continuations are not teacher forced.
This does not execute tools or score tasks.
Shared GPU mode explicitly invalidates performance comparisons.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import socket
import threading
import time
import urllib.request

from kv_baselines.frontend import meaningful_content

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=["infllm", "native_kvmem", "native_dense"], required=True)
    p.add_argument("--port", type=int, default=18229)
    p.add_argument("--shared-gpu", action="store_true")
    p.add_argument("--formal-ttft", action="store_true",
        help="Exclusive-GPU fixed-input measurement; does not measure task success")
    p.add_argument("--output-tokens", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--infllm-config", type=Path)
    p.add_argument("--infllm-max-gpu-gib", type=float, default=55)
    p.add_argument("--infllm-context", type=int, default=262144)
    p.add_argument("--memory-inventory", action="store_true")
    p.add_argument("--kvmem-budget", type=int, default=4096)
    p.add_argument("--kvmem-gen-budget", type=int, default=2048)
    p.add_argument("--kvmem-middecode-trigger", type=int, default=1024)
    p.add_argument("--native-gpu-memory-ratio", type=float, default=.96)
    p.add_argument("--native-context", type=int, default=262144)
    p.add_argument("--native-cpu-gb", type=float, default=24)
    p.add_argument("--required-free-gpu-gib", type=float)
    p.add_argument("--gpu-process-cap-gib", type=float, default=60)
    a = p.parse_args()
    if a.formal_ttft and a.shared_gpu:
        p.error("formal TTFT cannot use a shared GPU")
    if not 1 <= a.output_tokens <= 256:
        p.error("bounded measurement supports output limits from 1 to 256")
    a.out = a.out.resolve()
    if a.out.exists():
        raise FileExistsError("preserve existing raw runs")
    a.out.mkdir(parents=True)
    manifest_path = a.trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    def gpu_owners():
        return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], text=True).strip()
    owners = gpu_owners()
    if owners and not a.shared_gpu:
        raise RuntimeError("formal timing requires an exclusive GPU: " + owners)
    port_deadline = time.monotonic() + 60
    while True:
        try:
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                check.bind(("127.0.0.1", a.port))
            break
        except OSError:
            if time.monotonic() >= port_deadline:
                raise
            time.sleep(.25)
    free_gib = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip()) / 1024
    required_gib = a.infllm_max_gpu_gib + 2 if a.method == "infllm" else 74 if a.method == "native_dense" else (
        58 + max(0, a.kvmem_budget + a.kvmem_gen_budget - 6144) * 65536 / 2**30)
    if a.required_free_gpu_gib is not None:
        required_gib = a.required_free_gpu_gib
    if free_gib < required_gib:
        raise RuntimeError(f"capacity preflight needs {required_gib:.1f} GiB free, available {free_gib:.1f}")
    if a.method == "infllm":
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "serve_infllm.py"),
            "--model", str(REPO / "models/Qwen3.8-27B-BF16"), "--out", str(a.out / "server"),
            "--port", str(a.port), "--temperature", "0", "--max-context", str(a.infllm_context), "--max-tokens", str(a.output_tokens)]
        command += ["--max-gpu-gib", str(a.infllm_max_gpu_gib)]
        if a.memory_inventory:
            command.append("--memory-inventory")
        if a.shared_gpu:
            command.append("--shared-gpu")
        if a.infllm_config:
            command += ["--infllm-config", str(a.infllm_config.resolve())]
    else:
        command = [str(ROOT / "native_build/qw3"), "serve", "--model", str(REPO / "models/Qwen3.8-27B-BF16"),
            "--host", "127.0.0.1", "--port", str(a.port), "--ctx", str(a.native_context), "--kv-dtype", "fp16",
            "--prefill-chunk", "512", "--mtp-chain", "1" if a.method == "native_kvmem" else "0", "--temp", "0", "-n", str(a.output_tokens),
            "--thinking-budget", "28672"]
        if a.method == "native_dense":
            command += ["--continuous-batching", "--prefix-cache", "--max-active", "1"]
        if a.method == "native_kvmem":
            command += ["--kvmem", "--kvmem-budget", str(a.kvmem_budget),
                "--kvmem-gen-budget", str(a.kvmem_gen_budget), "--kvmem-block-tokens", "32",
                "--kvmem-sink-tokens", "128", "--kvmem-recent-tokens", "512",
                "--kvmem-gpu-memory-ratio", str(a.native_gpu_memory_ratio), "--kvmem-cpu-gb", str(a.native_cpu_gb), "--kvmem-nvme-gb", "0",
                "--kvmem-query-conditioned", "--kvmem-prefix-cache", "--kvmem-query-replay",
                "--kvmem-guided-reselect", "both", "--kvmem-guided-query-tokens", "256",
                "--kvmem-middecode-trigger-tokens", str(min(a.kvmem_middecode_trigger, max(1, a.kvmem_gen_budget // 2))),
                "--kvmem-guided-thinking-tokens", "0"]
        environment.update(QW3_BASELINE_NO_SPECULATIVE_DECODE="1", QW3_KVMEM_GUIDED_MTP="0",
                           QW3_HARNESS_STABLE_REASONING="1")
    result = {"status": "starting", "method": a.method, "server_command": command,
        "shared_gpu": a.shared_gpu, "performance_claim": False, "task_success_claim": False,
        "budget_matched": False, "phase": "state_and_capacity_calibration",
        "output_limit_per_request": a.output_tokens, "sustained_decode_measurement": False,
        "bounded_decode_measurement": a.output_tokens > 1, "continuation_mode": "free generation, greedy, natural EOS retained",
        "trace_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "trace_manifest": str(manifest_path.resolve()), "initial_gpu_owners": owners, "completed": 0}
    result["formal_ttft_requested"] = a.formal_ttft
    result["performance_exclusions"] = []
    if a.formal_ttft:
        result["phase"] = "fixed_input_ttft_and_bounded_decode" if a.output_tokens > 1 else "fixed_input_one_token_ttft"
    result["measurement_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.method.startswith("native_"):
        result["native_binary_sha256"] = hashlib.sha256((ROOT / "native_build/qw3").read_bytes()).hexdigest()
    result["started"] = time.time()
    result.update(gpu_process_cap_gib=a.gpu_process_cap_gib, peak_process_gpu_gib=0,
        required_free_gpu_gib=required_gib,
        native_scratch_reserve_mib=environment.get("QW3_KVMEM_SCRATCH_RESERVE_MIB", "default"))
    result["gpu_memory_measurement"] = {"source": "NVML process memory", "sampling_interval_s": 2,
        "native_hard_allocator_cap": False}
    def save():
        (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    log = (a.out / "server.log").open("w")
    server = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=log)
    result["server_pid"] = server.pid
    stop_monitor = threading.Event()
    def monitor_gpu():
        while not stop_monitor.is_set():
            try:
                owners = {int(pid): float(mib) / 1024 for pid, mib in
                    (line.split(",") for line in gpu_owners().splitlines())}
                used = owners.get(server.pid, 0)
                if a.formal_ttft and any(pid != server.pid for pid in owners):
                    reason = "another GPU process observed during measurement"
                    if reason not in result["performance_exclusions"]:
                        result["performance_exclusions"].append(reason)
                result["peak_process_gpu_gib"] = max(result["peak_process_gpu_gib"], used)
                if used > a.gpu_process_cap_gib:
                    result["resource_error"] = f"process GPU cap exceeded: {used} GiB"
                    server.terminate()
                    return
                with (a.out / "resource_samples.jsonl").open("a") as handle:
                    handle.write(json.dumps({"at": time.time(), "gpu_processes_gib": owners}) + "\n")
            except Exception as error:
                result["monitor_error"] = repr(error)
                if a.formal_ttft and "GPU monitor error" not in result["performance_exclusions"]:
                    result["performance_exclusions"].append("GPU monitor error")
            stop_monitor.wait(2)
    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    base = f"http://127.0.0.1:{a.port}"
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
        result["status"] = "running_capacity_calibration"
        save()
        selected = manifest["requests"][:a.limit or None]
        for row in selected:
            current_owners = gpu_owners()
            if not a.shared_gpu and any(int(line.split(",")[0]) != server.pid for line in current_owners.splitlines()):
                raise RuntimeError("another GPU process appeared; timing is invalid")
            raw = gzip.decompress((a.trace / row["path"]).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == row["body_sha256"]
            body = json.loads(raw)
            body.update(max_tokens=a.output_tokens, stream=True, temperature=0)
            request = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(),
                headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
            measured = {"index": row["index"], "prompt_tokens": row["prompt_tokens"],
                "api_first_content_s": None, "gpu_owners": current_owners, "status": "running"}
            start = time.perf_counter()
            wire = []
            with http.open(request, timeout=600) as response:
                for line in response:
                    line = line.decode()
                    wire.append(line)
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if measured["api_first_content_s"] is None and meaningful_content(event):
                            measured["api_first_content_s"] = time.perf_counter() - start
                        if event.get("type") == "error":
                            measured["error"] = event
                        if event.get("type") == "message_delta":
                            measured["output_tokens"] = event.get("usage", {}).get("output_tokens")
                            measured["stop_reason"] = event.get("delta", {}).get("stop_reason")
            measured["request_wall_s"] = time.perf_counter() - start
            measured["status"] = "completed" if "error" not in measured and any("event: message_stop" in s for s in wire) else "failed"
            (a.out / f"response_{row['index']:04d}.sse").write_text("".join(wire))
            with (a.out / "measurements.jsonl").open("a") as handle:
                handle.write(json.dumps(measured) + "\n")
            if measured["status"] != "completed":
                raise RuntimeError(f"request {row['index']} failed: {measured.get('error')}")
            result["completed"] += 1
            save()
            print(json.dumps({"completed": result["completed"], "prompt_tokens": row["prompt_tokens"]}), flush=True)
        result["status"] = "completed_capacity_calibration_not_a_matched_performance_result"
        if a.formal_ttft:
            result["performance_claim"] = not result["performance_exclusions"]
            result["status"] = "completed_fixed_input_ttft" if result["performance_claim"] else "completed_timing_excluded"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
        result["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
