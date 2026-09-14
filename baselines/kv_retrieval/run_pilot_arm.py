"""One complete original DeepSWE task with a frozen baseline configuration.

Only resources owned by this result directory are created/cleaned. Shared-GPU
runs are allowed for integration/quality, but never accepted as formal timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import tomllib
import urllib.request

from trace_proxy import TraceProxy

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def gpu_processes():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits"], text=True)
    return {int(parts[0]): float(parts[1]) / 1024
            for line in raw.splitlines() if (parts := line.split(",")) and len(parts) == 2}


def clean_environment():
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    return env


def server_command(method, plan, out, port):
    model = str(REPO / "models/Qwen3.8-27B-BF16")
    setting = plan["methods"][method]
    ctx = setting["context_tokens"]
    if method == "infllm":
        cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "serve_infllm.py"),
            "--model", model, "--out", str(out / "server"), "--port", str(port),
            "--max-context", str(ctx), "--max-gpu-gib", str(setting["allocator_cap_gib"]),
            "--infllm-config", str(ROOT / setting["config"]),
            "--temperature", str(plan["temperature"])]
    else:
        cmd = [str(ROOT / "native_build/qw3"), "serve", "--model", model,
            "--host", "127.0.0.1", "--port", str(port), "--ctx", str(ctx),
            "--kv-dtype", "fp16", "--mtp-chain", "1" if method == "native_kvmem" else "0",
            "--temp", str(plan["temperature"])]
        if method == "native_dense":
            cmd += ["--continuous-batching", "--prefix-cache", "--max-active", "1"]
        if method == "native_kvmem":
            cmd += ["--kvmem", "--kvmem-budget", str(setting["selection_budget"]),
                "--kvmem-gen-budget", str(setting["generation_budget"]), "--kvmem-block-tokens", "32",
                "--kvmem-sink-tokens", "128", "--kvmem-recent-tokens", "512",
                "--kvmem-gpu-memory-ratio", str(setting["gpu_planning_ratio"]),
                "--kvmem-cpu-gb", str(plan["cpu_kv_cap_gib"]), "--kvmem-nvme-gb", "0",
                "--kvmem-query-conditioned", "--kvmem-prefix-cache", "--kvmem-query-replay",
                "--kvmem-guided-reselect", "both", "--kvmem-guided-query-tokens", "256",
                "--kvmem-guided-thinking-tokens", "0", "--kvmem-middecode-trigger-tokens",
                str(setting.get("middecode_trigger_tokens", 1024))]
    cmd += ["--top-p", str(plan["top_p"]), "--top-k", str(plan["top_k"]),
            "--seed", str(plan["seed"]), "--prefill-chunk", str(plan["prefill_chunk"]),
            "--max-tokens" if method == "infllm" else "-n", str(plan["max_output_tokens"]),
            "--thinking-budget", str(plan["thinking_budget"])]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=["infllm", "native_kvmem", "native_dense"], required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--port", type=int, default=18241)
    p.add_argument("--shared-gpu", action="store_true")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if not plan.get("frozen"):
        raise ValueError("pilot plan must be frozen after held-out calibration")
    for relative, expected in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen source changed: " + relative)
    weights = json.loads((ROOT / plan["model_manifest"]).read_text())
    for name, expected in weights["files"].items():
        actual = (Path(weights["model"]) / name).stat()
        if (actual.st_size, actual.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
            raise ValueError("model file changed since SHA256 audit: " + name)
    tasks = json.loads((ROOT / "tasks_pilot4.json").read_text())
    task = tasks["tasks"][a.task_index]
    task_dir = REPO / "benchmark/deep-swe/tasks" / task["task_id"]
    metadata = tomllib.loads((task_dir / "task.toml").read_text())
    network_mode = metadata["agent"]["network_mode"]
    if network_mode not in {"no-network", "full", "full-network"}:
        raise ValueError("unhandled official network policy: " + network_mode)
    for image in (metadata["environment"]["docker_image"], task["verifier_image"]):
        subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)
    others = gpu_processes()
    if others and not a.shared_gpu:
        raise RuntimeError("formal timing requires exclusive GPU")
    free_gib = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True).strip()) / 1024
    if free_gib < plan["methods"][a.method]["minimum_free_gpu_gib"]:
        raise RuntimeError("insufficient spare GPU memory")
    # A just-finished arm may have exited before its listening socket settles.
    # Wait only for our requested port; never stop another service to obtain it.
    port_deadline = time.monotonic() + 60
    while True:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", a.port))
            break
        except OSError:
            if time.monotonic() >= port_deadline:
                raise
            time.sleep(1)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "plan.json").write_bytes(a.plan.read_bytes())
    command = server_command(a.method, plan, a.out, a.port)
    if a.method == "infllm" and a.shared_gpu:
        command.append("--shared-gpu")
    result = {"status": "starting", "task": task, "method": a.method, "shared_gpu": a.shared_gpu,
        "performance_eligible": not a.shared_gpu, "initial_gpu_owners": others,
        "server_command": command, "started": time.time(), "peak_process_gpu_gib": 0,
        "plan_sha256": hashlib.sha256(a.plan.read_bytes()).hexdigest()}
    result["gpu_memory_measurement"] = {"source": "NVML process memory", "sampling_interval_s": 2,
        "native_hard_allocator_cap": False, "ceiling_enforcement": "terminate if a sampled value exceeds the configured ceiling"}
    def save():
        temp = a.out / "result.json.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(a.out / "result.json")
    save()
    env = clean_environment()
    env.update(QW3_BASELINE_NO_SPECULATIVE_DECODE="1", QW3_KVMEM_GUIDED_MTP="0",
               QW3_HARNESS_STABLE_REASONING="1")
    if a.method == "native_kvmem" and "scratch_reserve_mib" in plan["methods"][a.method]:
        env["QW3_KVMEM_SCRATCH_RESERVE_MIB"] = str(plan["methods"][a.method]["scratch_reserve_mib"])
    owner = hashlib.sha256(str(a.out).encode()).hexdigest()[:16]
    network_name = "kv-baseline-" + owner
    network_created = False
    server = cell = proxy = None
    stop_monitor = threading.Event()
    monitor = None
    log = (a.out / "server.log").open("w")
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    try:
        net_cmd = ["docker", "network", "create", "--driver", "bridge", "--label", "com.qw3.kv-baseline.owner=" + owner]
        if network_mode == "no-network":
            net_cmd.append("--internal")
        subprocess.run(net_cmd + [network_name], check=True, stdout=subprocess.DEVNULL, env=env)
        network_created = True
        info = json.loads(subprocess.check_output(["docker", "network", "inspect", network_name], text=True))[0]
        gateway = info["IPAM"]["Config"][0]["Gateway"]
        result["docker_network"] = network_name
        server = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=log)
        result["server_pid"] = server.pid
        def monitor_gpu():
            while not stop_monitor.is_set():
                try:
                    owners = gpu_processes()
                    if server.poll() is not None:
                        if cell is not None and cell.poll() is None:
                            result["resource_error"] = f"model server exited: {server.returncode}"
                            cell.terminate()
                        return
                    used = owners.get(server.pid, 0)
                    result["peak_process_gpu_gib"] = max(result["peak_process_gpu_gib"], used)
                    if not a.shared_gpu and any(pid != server.pid for pid in owners):
                        result["performance_eligible"] = False
                        result["performance_exclusion"] = "another GPU process appeared"
                    if used > plan["gpu_process_cap_gib"]:
                        result["resource_error"] = f"GPU process exceeded {plan['gpu_process_cap_gib']} GiB cap: {used}"
                        server.terminate()
                        return
                    with (a.out / "resource_samples.jsonl").open("a") as f:
                        f.write(json.dumps({"at": time.time(), "gpu_processes_gib": owners}) + "\n")
                except Exception as error:
                    result["monitor_error"] = repr(error)
                    result["performance_eligible"] = False
                stop_monitor.wait(2)
        monitor = threading.Thread(target=monitor_gpu, daemon=True)
        monitor.start()
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited at startup: {server.returncode}")
            try:
                with http.open(f"http://127.0.0.1:{a.port}/health", timeout=2) as reply:
                    if reply.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError("model readiness")
        proxy = TraceProxy((gateway, a.port + 1), f"http://127.0.0.1:{a.port}", a.out / "proxy")
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        setting = plan["methods"][a.method]
        compact = a.method == "native_dense"
        env.update(QWEN38_CLAUDE_API_BASE=f"http://{gateway}:{a.port + 1}",
            QWEN38_CLAUDE_MODEL_ID="Qwen3.8-27B-BF16[1m]", QWEN38_PRECISION_LABEL="BF16 weights / FP16 KV",
            QWEN38_CLAUDE_AGENT_TIMEOUT=str(plan["agent_timeout_s"]),
            QWEN38_CLAUDE_AGENT_MEMORY="16g", QWEN38_CLAUDE_CONTEXT_TOKENS=str(setting["context_tokens"]),
            QWEN38_CLAUDE_DISABLE_COMPACT="0" if compact else "1",
            QWEN38_CLAUDE_AUTO_COMPACT_WINDOW=str(setting["context_tokens"]) if compact else "",
            QWEN38_CLAUDE_CONDITION=a.method, QWEN38_VERIFIER_IMAGE=task["verifier_image"],
            QWEN38_SHARED_VLLM_LOG=str(a.out / "server.log"),
            CLAUDE_AGENT_RUNNER=str(ROOT / "harness/container_claude_agent.sh"),
            QWEN38_AGENT_NETWORK=network_name,
            QWEN38_AGENT_NETWORK_POLICY="docker_internal_bridge_no_wan" if network_mode == "no-network" else "docker_bridge_wan_audited")
        result.update(status="running_agent", agent_started=time.time())
        save()
        cell = subprocess.Popen(["bash", str(ROOT / "harness/run_deepswe_cell.sh"), task["task_id"], str(a.out / "cell")],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (a.out / "cell.console.log").open("w") as console, (a.out / "client_events.jsonl").open("w") as events:
            for line in cell.stdout:
                console.write(line)
                console.flush()
                try:
                    event = json.loads(line)
                    events.write(json.dumps({"observed_at": time.time(), "event": event}) + "\n")
                    events.flush()
                except ValueError:
                    pass
        result["cell_exit_code"] = cell.wait()
        summary = a.out / "cell/cell_summary.json"
        result["cell_summary"] = json.loads(summary.read_text()) if summary.exists() else None
        result["status"] = "completed" if result["cell_exit_code"] == 0 and summary.exists() else "infrastructure_failed"
        if result["cell_summary"]:
            if result["cell_summary"].get("reward") is None:
                result["status"] = "verifier_failed"
            if result["cell_summary"].get("agent_exit") != 0:
                result["status"] = "agent_failed"
        if "resource_error" in result:
            result["status"] = "resource_failed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if cell is not None and cell.poll() is None:
            cell.terminate()  # the snapshotted harness trap cleans its owned containers
            try:
                cell.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cell.kill()
                cell.wait()
        if proxy:
            proxy.shutdown()
            proxy.server_close()
        stop_monitor.set()
        if monitor:
            monitor.join(timeout=5)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
        if network_created:
            subprocess.run(["docker", "network", "rm", network_name], check=False, stdout=subprocess.DEVNULL)
        result["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
