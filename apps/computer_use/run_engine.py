"""Start the serialized GPU profile; requires exclusive use of the local GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["none", "mtp", "dflash2"], default="dflash2")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
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
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(handle.fileno(), True)
        handles.append(handle)
    users = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader"], text=True).strip()
    if users:
        raise SystemExit("GPU has existing compute processes; refusing to start")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used",
                                   "--format=csv,noheader"], text=True).strip()
    binary = root / "build-dflash-integration/qw3"
    command = [str(binary), "serve", "--model", str(root / "models/Qwen3.8-27B-Q8_0.gguf"),
               "--vision-model", str(root / "models/Qwen3.8-27B-BF16"), "--vision-device", "cuda",
               "--host", "127.0.0.1", "--port", str(args.port), "--ctx", "1048576",
               "--kv-dtype", "fp8", "--speculative-method", args.method,
               "--prefill-chunk", "2048", "--kvmem", "--kvmem-block-tokens", "128",
               "--kvmem-budget", "131072", "--kvmem-gen-budget", "65536",
               "--kvmem-cpu-gb", "32",
               "--kvmem-prefill-budget", "131072", "--kvmem-prefix-cache",
               "--kvmem-query-conditioned", "--kvmem-query-replay",
               "--kvmem-retrieval-method", "sub-block-mean-k",
               "--kvmem-semantic-expansion", "message"]
    if args.method == "dflash2":
        command += ["--dflash-model", str(root / "models/Qwen3.8-27B-DFlash2-BF16")]
    elif args.method == "mtp":
        command += ["--mtp-chain", "4"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    if args.method == "dflash2":
        env["QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS"] = "7"
    manifest = {"command": command, "qw3_env": {k: v for k, v in env.items() if k.startswith("QW3_")},
                "gpu": gpu, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "pid": os.getpid()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = (run / "engine.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
