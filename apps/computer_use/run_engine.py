"""Start the serialized GPU profile; requires exclusive use of the local GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["none", "mtp", "dflash2"], default="dflash2")
    parser.add_argument("--kvmem", choices=["on", "off"], default="on")
    parser.add_argument("--kvmem-cpu-gib", type=int, default=48,
                        help="CPU tier capacity; 1M context with native MTP needs more than 32 GiB")
    parser.add_argument("--prefix-reuse", choices=["auto", "off"], default="auto",
                        help="off gives cold fixed-input replay on both KVMem sides")
    parser.add_argument("--visual-history", choices=["pin-all", "retrievable"], default="pin-all")
    parser.add_argument("--guided-refresh", choices=["off", "both"], default="off",
                        help="both refreshes same-task history at KVMem boundaries; requires DFlash or MTP")
    parser.add_argument("--trace-retrieval", action="store_true")
    parser.add_argument("--trace-memory", action="store_true",
                        help="Separate diagnostic: trace CUDA tensor allocations and workspace capacities")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--lock-timeout", type=float, default=0,
                        help="Wait briefly for a previous owned engine's GPU lock to be released")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.kvmem == "on" and args.guided_refresh != "off" and args.method == "none":
        parser.error("--guided-refresh both requires --method dflash2 or mtp when KVMem is on")
    root = Path(__file__).resolve().parents[2]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    locks = [Path("/tmp/qw3-computer-gpu.lock")]
    shared = root.parent / "qw3-dflash2/benchmark/dflash2/long_running_gpu.lock"
    if shared.exists():
        locks.append(shared)
    handles = []
    for path in locks:
        handle = path.open("rb" if path == shared else "a")
        deadline = time.monotonic() + args.lock_timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit(f"GPU experiment lock is busy: {path}")
                time.sleep(0.2)
        os.set_inheritable(handle.fileno(), True)
        handles.append(handle)
    deadline = time.monotonic() + args.lock_timeout
    while True:
        users = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                         "--format=csv,noheader"], text=True).strip()
        if not users:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("GPU has existing compute processes; refusing to start: " + users)
        # A terminated process can briefly remain visible to the driver after
        # its file locks disappear. Never start over its outstanding context.
        time.sleep(0.2)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used",
                                   "--format=csv,noheader"], text=True).strip()
    binary = root / "build-dflash-integration/qw3"
    command = [str(binary), "serve", "--model", str(root / "models/Qwen3.8-27B-Q8_0.gguf"),
               "--vision-model", str(root / "models/Qwen3.8-27B-BF16"), "--vision-device", "cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--ctx", "1048576",
               "--kv-dtype", "fp8", "--speculative-method", args.method,
               "--prefill-chunk", "2048"]
    if args.kvmem == "on":
        command += ["--kvmem", "--kvmem-block-tokens", "128",
               "--kvmem-budget", "131072", "--kvmem-gen-budget", "65536",
               "--kvmem-cpu-gb", str(args.kvmem_cpu_gib),
               "--kvmem-prefill-budget", "131072",
               "--kvmem-query-conditioned", "--kvmem-query-replay",
               "--kvmem-retrieval-method", "sub-block-mean-k",
               "--kvmem-semantic-expansion", "message"]
        if args.guided_refresh != "off":
            command += ["--kvmem-guided-reselect", args.guided_refresh]
        if args.prefix_reuse == "auto":
            command += ["--kvmem-prefix-cache"]
    if args.method == "dflash2":
        command += ["--dflash-model", str(root / "models/Qwen3.8-27B-DFlash2-BF16")]
    elif args.method == "mtp":
        command += ["--mtp-chain", "4"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    if args.method == "dflash2":
        env["QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS"] = "7"
    if args.kvmem == "on" and args.visual_history == "retrievable":
        env["QW3_KVMEM_RETRIEVABLE_VISUAL_HISTORY"] = "1"
    if args.trace_retrieval:
        env["QW3_KVMEM_TRACE"] = "1"
        env["QW3_KVMEM_PREFIX_CACHE_TRACE"] = "1"
        env["QW3_KVMEM_DUMP_SCORES"] = str(run / "retrieval.jsonl")
    if args.trace_memory:
        env["QW3_CUDA_TENSOR_ALLOC_TRACE"] = "1"
        env["QW3_CUDA_WORKSPACE_TRACE"] = "1"
    manifest = {"command": command, "qw3_env": {k: v for k, v in env.items() if k.startswith("QW3_")},
                "gpu": gpu, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "kvmem": args.kvmem, "prefix_reuse": args.kvmem == "on" and args.prefix_reuse == "auto",
                "visual_history": args.visual_history,
                "guided_refresh": args.guided_refresh if args.kvmem == "on" else "off",
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pid": os.getpid()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = (run / "engine.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
