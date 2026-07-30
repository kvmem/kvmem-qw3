#!/usr/bin/env python3
"""Launch qw3 and run the canonical LongMemEval-S harness.

This is the single reusable experiment orchestrator.  It owns only process
lifecycle and parameterization; all prompt construction, API requests, grading,
and result generation remain in run_eval.py.  Existing historical shell scripts
under /data are intentionally left untouched for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Start qw3 with a KVMem configuration and run run_eval.py.")
    ap.add_argument("--tag", required=True,
                    help="stable experiment name used for logs and result files")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands and manifest without starting processes")
    ap.add_argument("--force", action="store_true",
                    help="allow replacing this tag's serve/run logs")

    # Stable experiment envelope.
    ap.add_argument("--binary", type=Path, default=ROOT / "build/qw3")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "models/Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--data", type=Path,
                    default=Path("/data/chaidi/kvmem_eval/data/longmemeval_s.json"))
    ap.add_argument(
        "--eval-script", type=Path,
        default=ROOT / "scripts/kvmem_eval/run_eval.py",
        help="evaluation client to run after the server is healthy; default "
             "keeps the canonical one-shot harness unchanged")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data/chaidi/kvmem_eval/results"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--prefill-chunk", type=int, default=2048)

    # KVMem variables. Defaults match the full-500 32K mean-k baseline.
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--gen-budget", type=int, default=32768)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=0)
    ap.add_argument("--method", choices=("retrieval", "h2o", "recency"),
                    default="retrieval")
    ap.add_argument("--retrieval-method",
                    choices=("mean-k", "per-token", "sub-block-mean-k",
                             "key-direction-fixed4",
                             "key-direction-adaptive"),
                    default="mean-k")
    ap.add_argument("--subblocks", type=int, default=4)
    ap.add_argument("--subblock-reduce", choices=("max", "sum"), default="max")
    ap.add_argument(
        "--adaptive-gain-1to2", type=float, default=0.10,
        help="minimum residual-reduction gain for adaptive 1->2 prototypes")
    ap.add_argument(
        "--adaptive-gain-2to4", type=float, default=0.06,
        help="minimum residual-reduction gain for adaptive 2->4 prototypes")
    ap.add_argument("--update-mode", choices=("step", "interval"), default="step")
    ap.add_argument(
        "--optimization-level",
        choices=("default", "kvmem_init", "opt_1", "opt_2", "opt_3"),
        default="kvmem_init",
        help="monotonic KVMem storage/tiering profile used for matched A/B "
             "runs; 'default' omits the deprecated compatibility flag")
    ap.add_argument("--query-conditioned", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--gpu-memory-ratio", type=float, default=0.5)
    ap.add_argument("--cpu-gb", type=float, default=64.0)
    ap.add_argument("--nvme-gb", type=float, default=256.0)
    ap.add_argument("--nvme-dir", type=Path, default=None,
                    help="default: /data/qw3_kvmem_eval_nvme_<tag>")

    # Generation/evaluation variables.
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--thinking-budget", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mtp-chain", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test prefix length; omit for all 500 samples")
    ap.add_argument("--indices", default=None,
                    help="comma-separated dataset indices passed to run_eval.py")
    ap.add_argument("--read-timeout", type=float, default=3600.0)
    ap.add_argument("--health-timeout", type=float, default=900.0)
    ap.add_argument("--server-extra-arg", action="append", default=[],
                    help="additional single qw3 argument; repeat as needed")
    ap.add_argument("--eval-extra-arg", action="append", default=[],
                    help="additional single run_eval.py argument; repeat as needed")
    return ap.parse_args()


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str], Path]:
    nvme_dir = args.nvme_dir or Path(f"/data/qw3_kvmem_eval_nvme_{args.tag}")
    server = [
        str(args.binary), "serve",
        "--model", str(args.model),
        "--ctx", str(args.ctx),
        "--kv-dtype", args.kv_dtype,
        "--kvmem",
        "--kvmem-block-tokens", str(args.block_tokens),
        "--kvmem-budget", str(args.budget),
        "--kvmem-gen-budget", str(args.gen_budget),
        "--kvmem-sink-blocks", str(args.sink_blocks),
        "--kvmem-recent-blocks", str(args.recent_blocks),
        "--kvmem-method", args.method,
        "--kvmem-retrieval-method", args.retrieval_method,
        "--kvmem-update-mode", args.update_mode,
        "--kvmem-gpu-memory-ratio", str(args.gpu_memory_ratio),
        "--kvmem-cpu-gb", str(args.cpu_gb),
        "--kvmem-nvme-gb", str(args.nvme_gb),
        "--kvmem-nvme-dir", str(nvme_dir),
        "--thinking-budget", str(args.thinking_budget),
        "--prefill-chunk", str(args.prefill_chunk),
        "--temp", str(args.temperature),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.query_conditioned:
        server.append("--kvmem-query-conditioned")
    if args.retrieval_method == "sub-block-mean-k":
        server += ["--kvmem-subblocks", str(args.subblocks),
                   "--kvmem-subblock-reduce", args.subblock_reduce]
    if args.retrieval_method == "key-direction-adaptive":
        server += [
            "--kvmem-adaptive-gain-1to2", str(args.adaptive_gain_1to2),
            "--kvmem-adaptive-gain-2to4", str(args.adaptive_gain_2to4),
        ]
    if args.optimization_level != "default":
        server += ["--kvmem-optimization-level", args.optimization_level]
    if args.thinking:
        server.append("--enable-thinking")
    if args.mtp:
        server += ["--native-mtp-speculate", "--mtp-chain", str(args.mtp_chain)]
    server += args.server_extra_arg

    base_url = f"http://{args.host}:{args.port}/v1"
    evaluation = [
        sys.executable, str(args.eval_script),
        "--data", str(args.data),
        "--use-all",
        "--base-url", base_url,
        "--tag", args.tag,
        "--out-dir", str(args.out_dir),
        "--model", args.model.name,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--read-timeout", str(args.read_timeout),
    ]
    if not args.thinking:
        evaluation.append("--no-thinking")
    if args.no_judge:
        evaluation.append("--no-judge")
    if args.limit is not None:
        evaluation += ["--limit", str(args.limit)]
    if args.indices:
        evaluation += ["--indices", args.indices]
    evaluation += args.eval_extra_arg
    return server, evaluation, nvme_dir


def git_sha() -> str | None:
    cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                        text=True, capture_output=True)
    return cp.stdout.strip() if cp.returncode == 0 else None


def health_ok(url: str) -> bool:
    # Explicitly disable environment proxies for localhost health checks.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2.0) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def port_is_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=80)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    args = parse_args()
    server_cmd, eval_cmd, nvme_dir = build_commands(args)
    serve_log = args.out_dir / f"{args.tag}_serve.log"
    run_log = args.out_dir / f"{args.tag}_run.log"
    manifest_path = args.out_dir / f"{args.tag}_manifest.json"
    health_url = f"http://{args.host}:{args.port}/health"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "git_sha": git_sha(),
        "cwd": str(ROOT),
        "server_command": server_cmd,
        "eval_command": eval_cmd,
        "health_url": health_url,
        "serve_log": str(serve_log),
        "run_log": str(run_log),
        "nvme_dir": str(nvme_dir),
        "judge": None if args.no_judge else os.environ.get(
            "DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "environment_flags": {
            "QW3_FATTN_NSPLIT": "1",
            "QW3_PREFILL_FA2_NSPLIT": "1",
            "QW3_KVMEM_TIMING": "1",
            **(
                {"QW3_KVMEM_PERF_TRACE":
                 os.environ["QW3_KVMEM_PERF_TRACE"]}
                if "QW3_KVMEM_PERF_TRACE" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_RECOMPUTE_QUERY": os.environ["QW3_KVMEM_RECOMPUTE_QUERY"]}
                if "QW3_KVMEM_RECOMPUTE_QUERY" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_IMMUTABLE_SOURCE_K":
                 os.environ["QW3_KVMEM_IMMUTABLE_SOURCE_K"]}
                if "QW3_KVMEM_IMMUTABLE_SOURCE_K" in os.environ
                else {}
            ),
            **(
                {"QW3_ROPE_POSITION_TRACE":
                 os.environ["QW3_ROPE_POSITION_TRACE"]}
                if "QW3_ROPE_POSITION_TRACE" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_QUERY_BOOTSTRAP":
                 os.environ["QW3_KVMEM_TRANSCRIPT_QUERY_BOOTSTRAP"]}
                if "QW3_KVMEM_TRANSCRIPT_QUERY_BOOTSTRAP" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_STABLE_INGEST":
                 os.environ["QW3_KVMEM_TRANSCRIPT_STABLE_INGEST"]}
                if "QW3_KVMEM_TRANSCRIPT_STABLE_INGEST" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_STABLE_FAST":
                 os.environ["QW3_KVMEM_TRANSCRIPT_STABLE_FAST"]}
                if "QW3_KVMEM_TRANSCRIPT_STABLE_FAST" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_RESET_FINAL_RECURRENT":
                 os.environ["QW3_KVMEM_TRANSCRIPT_RESET_FINAL_RECURRENT"]}
                if "QW3_KVMEM_TRANSCRIPT_RESET_FINAL_RECURRENT" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_RESET_EACH_RECURRENT":
                 os.environ["QW3_KVMEM_TRANSCRIPT_RESET_EACH_RECURRENT"]}
                if "QW3_KVMEM_TRANSCRIPT_RESET_EACH_RECURRENT" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_SESSION_LOCAL":
                 os.environ["QW3_KVMEM_TRANSCRIPT_SESSION_LOCAL"]}
                if "QW3_KVMEM_TRANSCRIPT_SESSION_LOCAL" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_EXACT_QUESTION":
                 os.environ["QW3_KVMEM_TRANSCRIPT_EXACT_QUESTION"]}
                if "QW3_KVMEM_TRANSCRIPT_EXACT_QUESTION" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_TOKENS":
                 os.environ["QW3_KVMEM_TRANSCRIPT_REFRESH_TOKENS"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_TOKENS" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS":
                 os.environ[
                     "QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_MEANK":
                 os.environ["QW3_KVMEM_TRANSCRIPT_REFRESH_MEANK"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_MEANK" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_RELEVANCE_ORDER":
                 os.environ[
                     "QW3_KVMEM_TRANSCRIPT_REFRESH_RELEVANCE_ORDER"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_RELEVANCE_ORDER" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_BLOCKS":
                 os.environ["QW3_KVMEM_TRANSCRIPT_REFRESH_BLOCKS"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_BLOCKS" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_CORE_TOKENS":
                 os.environ["QW3_KVMEM_TRANSCRIPT_REFRESH_CORE_TOKENS"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_CORE_TOKENS" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_NEIGHBOR_BLOCKS":
                 os.environ[
                     "QW3_KVMEM_TRANSCRIPT_REFRESH_NEIGHBOR_BLOCKS"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_NEIGHBOR_BLOCKS" in os.environ
                else {}
            ),
            **(
                {"QW3_KVMEM_TRANSCRIPT_REFRESH_SESSION_EXPAND":
                 os.environ[
                     "QW3_KVMEM_TRANSCRIPT_REFRESH_SESSION_EXPAND"]}
                if "QW3_KVMEM_TRANSCRIPT_REFRESH_SESSION_EXPAND" in os.environ
                else {}
            ),
        },
    }

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    if not args.no_judge and not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY is required unless --no-judge is used",
              file=sys.stderr)
        return 2
    if not args.binary.is_file():
        print(f"ERROR: qw3 binary not found: {args.binary}", file=sys.stderr)
        return 2
    if (not args.model.is_file() or not args.data.exists() or
            not args.eval_script.is_file()):
        print("ERROR: model, dataset, or eval script does not exist",
              file=sys.stderr)
        return 2
    if not port_is_free(args.host, args.port):
        print(f"ERROR: listen address is already in use: {args.host}:{args.port}",
              file=sys.stderr)
        return 2
    if not args.force and any(p.exists() for p in (serve_log, run_log, manifest_path)):
        print(f"ERROR: tag outputs already exist; use a new --tag or --force: {args.tag}",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    env = os.environ.copy()
    env.update({
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "QW3_FATTN_NSPLIT": "1",
        "QW3_PREFILL_FA2_NSPLIT": "1",
        "QW3_KVMEM_TIMING": "1",
    })

    server_proc: subprocess.Popen[Any] | None = None
    old_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        stop_process(server_proc)
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupted)

    try:
        with serve_log.open("w") as sf:
            server_proc = subprocess.Popen(server_cmd, cwd=ROOT, env=env,
                                           stdout=sf, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.health_timeout
            while time.monotonic() < deadline:
                if server_proc.poll() is not None:
                    print(f"ERROR: qw3 exited during startup with {server_proc.returncode}",
                          file=sys.stderr)
                    return 1
                if health_ok(health_url):
                    break
                time.sleep(2)
            else:
                print(f"ERROR: health timeout: {health_url}", file=sys.stderr)
                return 1

            print(f"qw3 healthy (pid={server_proc.pid}); starting canonical eval")
            with run_log.open("w") as rf:
                eval_cp = subprocess.run(eval_cmd, cwd=ROOT, env=env,
                                         stdout=rf, stderr=subprocess.STDOUT)
            print(f"evaluation exited rc={eval_cp.returncode}")
            return eval_cp.returncode
    except KeyboardInterrupt:
        print("interrupted; partial JSONL results are preserved", file=sys.stderr)
        return 130
    finally:
        stop_process(server_proc)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
