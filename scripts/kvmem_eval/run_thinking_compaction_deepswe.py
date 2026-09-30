#!/usr/bin/env python3
"""Paired DeepSWE pilot: bounded KVMem baseline vs transactional thinking summary.

Fresh service/container per task and arm, one GPU workload at a time. Waits for
an idle GPU without stopping other workloads. Runs the state oracle before any
agent trial. --prepare-only freezes inputs without launching CUDA or containers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark/claude_deepswe_ab"))
import batch  # noqa: E402

CONDITION = batch.KVMEM_128_GEN64_CONDITION


def state(root, phase, **fields):
    batch.atomic_json(root / "state.json", {"phase": phase, "updated_at": batch.utc_now(), **fields})


def gpu_usage():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    return [[int(part.strip()) for part in line.split(",")] for line in output.splitlines()]


def evidence(arm):
    path = arm / "qw3_server.timed.log"
    text = path.read_text(errors="replace") if path.exists() else ""
    pools, compactions = [], []
    for line in text.splitlines():
        if "kvmem bounded history=" not in line and "[thinking-compaction]" not in line:
            continue
        row = dict(re.findall(r"([a-z_]+)=([^\s]+)", line))
        for key, value in list(row.items()):
            try:
                row[key] = float(value) if "." in value else int(value)
            except ValueError:
                pass
        (pools if "kvmem bounded history=" in line else compactions).append(row)
    metrics = batch.read_json(arm / "metrics.json", {}) or {}
    reward = metrics.get("reward") or {}
    counts = [reward.get(key) for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")]
    quality = reward.get("reward") == 1 and all(type(value) is int and value >= 0 for value in counts) and \
        counts[0] == counts[1] and counts[2] == counts[3]
    return {
        "passed": quality,
        "reward": reward,
        "agent_elapsed_sec": metrics.get("agent_elapsed_sec"),
        "request_wall_sec": (metrics.get("server") or {}).get("request_wall_sec"),
        "pool_at_last_request": pools[-1] if pools else None,
        "peak_live_slots": max((row.get("live", 0) for row in pools), default=None),
        "minimum_ready_slots": min((row.get("ready", 0) for row in pools), default=None),
        "compactions": compactions,
        "compacted_requests": sum(row.get("status") == "compacted" for row in compactions),
        "skipped_requests": sum(row.get("status") == "skipped" for row in compactions),
        "discarded_thinking_tokens": sum(row.get("original_tokens", 0) - row.get("retained_tokens", 0)
            for row in compactions if row.get("status") == "compacted"),
        "summary_ms": sum(row.get("summary_ms", 0) for row in compactions),
        "rollback_ms": sum(row.get("rollback_ms", 0) for row in compactions),
        "replay_ms": sum(row.get("replay_ms", 0) for row in compactions),
        "metrics_path": str(arm / "metrics.json"),
    }


def comparison(results):
    pairs = {}
    for result in results:
        pairs.setdefault(result["task"], {})[result["arm"]] = result
    rows = []
    for task, pair in pairs.items():
        if set(pair) != {"baseline", "summary"}:
            continue
        baseline, summary = pair["baseline"], pair["summary"]
        row = {"task": task, "baseline_passed": baseline["passed"],
               "summary_passed": summary["passed"],
               "compacted_requests": summary["compacted_requests"],
               "skipped_requests": summary["skipped_requests"]}
        for key in ("agent_elapsed_sec", "request_wall_sec", "peak_live_slots", "minimum_ready_slots"):
            a, b = baseline[key], summary[key]
            row[key + "_summary_minus_baseline"] = b - a if a is not None and b is not None else None
        a, b = baseline["pool_at_last_request"], summary["pool_at_last_request"]
        row["final_pool_summary_minus_baseline"] = {
            key: b[key] - a[key] for key in ("live", "ready", "writing", "retired", "releasing", "gpu_kv_bytes")
            if a and b and key in a and key in b}
        rows.append(row)
    return rows


def prepare(args):
    root = args.experiment.resolve()
    if (root / "manifest.json").exists():
        manifest = batch.read_json(root / "manifest.json")
        for name, digest in manifest["binaries"].items():
            if batch.sha256(root / "bin" / name) != digest:
                raise RuntimeError(f"frozen binary changed: {name}")
        return root, manifest
    catalog = batch.read_json(args.tasks_file)
    tasks = []
    for row in catalog["tasks"]:
        task = {**batch.task_metadata(row["task_id"]), "verifier_image": row["verifier_image"]}
        subprocess.run(["docker", "image", "inspect", task["verifier_image"]],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tasks.append(task)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    binaries = {}
    for source, name in [(args.binary, "qw3"), (args.state_probe, "state-probe"),
                         (Path(__file__).with_name("test_thinking_compaction_http.py"), "test-http.py")]:
        shutil.copy2(source, root / "bin" / name)
        binaries[name] = batch.sha256(root / "bin" / name)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for relative in ("src", "include", "tests", "CMakeLists.txt",
                         "scripts/kvmem_eval/run_thinking_compaction_deepswe.py",
                         "scripts/kvmem_eval/test_thinking_compaction_http.py",
                         "docs/kvmem_thinking_compaction.md"):
            archive.add(REPO / relative, arcname=relative)
    manifest = {
        "schema": "qw3.thinking_compaction_deepswe.v1", "created_at": batch.utc_now(),
        "tasks": tasks, "tasks_file": str(args.tasks_file.resolve()),
        "tasks_sha256": batch.sha256(args.tasks_file), "seed": args.seed,
        "model": str(batch.MODEL), "binaries": binaries,
        "source_sha256": batch.sha256(root / "source.tar.gz"),
        "summary_tokens": args.summary_tokens, "summary_min_tokens": args.summary_min_tokens,
        "summary_length_mode": "explicit-cap" if args.summary_tokens else "natural",
        "context": 524288, "repository_blocks": 16384, "block_tokens": 32,
        "selection_budget": 131072, "generation_reserve": 65536, "gc": "delete",
        "cpu_gb": 48, "index": "cpu", "mtp_chain": 4,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "thinking_budget": 28672, "presence_penalty": 0.0, "repetition_penalty": 1.0},
        "note": "Diagnostic pilot, not a representative success-rate estimate. Original target generation budget is unchanged; private summary compute is extra. Final live/ready snapshot precedes service teardown. Fixed allocation capacity is not live occupancy.",
    }
    batch.atomic_json(root / "manifest.json", manifest)
    for index, task in enumerate(tasks):
        for arm in ("baseline", "summary"):
            trial = root / task["task_id"] / arm
            setup = trial / "setup"
            setup.mkdir(parents=True, exist_ok=True)
            for path in (batch.SERVER, batch.CELL_RUNNER, batch.AGENT_RUNNER):
                shutil.copy2(path, setup / path.name)
            cap = args.summary_tokens if arm == "summary" else 0
            flags = ["--kvmem-index-placement", "cpu", "--kvmem-repository-blocks", "16384",
                     "--kvmem-repository-gc", "delete", "--thinking-summary-tokens", str(cap),
                     "--thinking-summary-min-tokens", str(args.summary_min_tokens)]
            if arm == "summary":
                flags.append("--thinking-summary")
            wrapper = setup / "binary-wrapper.sh"
            wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(str(root / "bin/qw3")) +
                               ' "$@" ' + shlex.join(flags) + "\n")
            wrapper.chmod(0o755)
            trial_manifest = {
                "tasks": [task], "seed": args.seed, "sampling": manifest["sampling"],
                "claude_model_id": "Qwen3.8-27B[1m]", "kvmem": {"context": 524288},
                "execution": {"agent_container_memory": "16g"}, "thinking_arm": arm,
            }
            batch.atomic_json(trial / "manifest.json", trial_manifest)
    state(root, "prepared", tasks=len(tasks), trials=len(tasks) * 2)
    return root, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=REPO / "build-query-attention-probe-cuda/qw3")
    parser.add_argument("--state-probe", type=Path, default=REPO / "build-query-attention-probe-cuda/qw3-thinking-compaction-probe")
    parser.add_argument("--tasks-file", type=Path, default=REPO / "benchmark/claude_deepswe_ab/tasks_kvmem_128k_gen64k_4.json")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--summary-tokens", type=int, default=0,
                        help="Optional explicit cap; 0 lets the summary terminate naturally")
    parser.add_argument("--summary-min-tokens", type=int, default=1024)
    parser.add_argument("--port", type=int, default=18973)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.summary_tokens <= 1024 or args.summary_min_tokens < 0:
        parser.error("summary cap must be in [0,1024] and minimum nonnegative")
    root, manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps({"phase": "prepared", "root": str(root)})); return 0
    if (root / "results.json").exists():
        raise RuntimeError("experiment already has results; use a new output directory")
    while True:
        usage = gpu_usage()
        if usage and usage[0][1] < 1000 and usage[0][2] > 60000:
            break
        state(root, "waiting_for_gpu", gpu_usage=usage, supervisor_pid=os.getpid())
        time.sleep(15)
    state(root, "state_oracle", supervisor_pid=os.getpid())
    with (root / "state-oracle.log").open("w") as log:
        subprocess.run([str(root / "bin/state-probe"), str(REPO / "models/Qwen3.5-4B-Q8_0-with-MTP.gguf")],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    for arm in ("baseline", "summary"):
        state(root, "http_oracle", arm=arm, supervisor_pid=os.getpid())
        with (root / f"http-oracle-{arm}.log").open("w") as log:
            command = [sys.executable, str(root / "bin/test-http.py"),
                       "--binary", str(root / "bin/qw3"),
                       "--model", str(REPO / "models/Qwen3.5-4B-Q8_0-with-MTP.gguf"),
                       "--output", str(root / f"http-oracle-{arm}")]
            if arm == "baseline":
                command += ["--summary-tokens", "0"]
            elif manifest["summary_tokens"]:
                command += ["--summary-tokens", str(manifest["summary_tokens"])]
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = []
    for index, task in enumerate(manifest["tasks"]):
        order = ("baseline", "summary") if index % 2 == 0 else ("summary", "baseline")
        for arm_name in order:
            trial = root / task["task_id"] / arm_name
            os.environ["QW3_BINARY_PATH"] = str(trial / "setup/binary-wrapper.sh")
            network_policy = batch.network_policy_for_task(task)
            state(root, "running_deepswe", task=task["task_id"], arm=arm_name, supervisor_pid=os.getpid())
            with batch.DockerNetwork(trial, f"thinking-{index}-{arm_name}",
                                     allow_wan=network_policy == "docker_bridge_wan_audited") as network:
                service = batch.Service(trial, CONDITION, index, args.port, manifest["seed"],
                    manifest["sampling"], bind_host=network.gateway, agent_network=network.name,
                    context=manifest["context"], kvmem_cpu_gb=manifest["cpu_gb"], kvmem_guided_query_tokens=512)
                try:
                    service.wait_ready()
                    status = batch.run_arm(trial, task, CONDITION, service)
                    artifact = trial / "runs" / task["task_id"] / CONDITION
                    results.append({"task": task["task_id"], "arm": arm_name, "exit": status,
                                    **evidence(artifact)})
                    batch.atomic_json(root / "results.json", results)
                    batch.atomic_json(root / "comparison.json", comparison(results))
                finally:
                    service.stop()
    state(root, "finished", trials=len(results), results=str(root / "results.json"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"thinking compaction experiment failed: {error}", file=sys.stderr)
        if "--experiment" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--experiment") + 1]).resolve()
            state(root, "failed", reason=str(error))
        raise
