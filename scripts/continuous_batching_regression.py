#!/usr/bin/env python3
"""Continuous-batching correctness and efficiency regression runner.

This script starts the local OpenAI-compatible qw3 server, collects a serial
baseline with continuous batching disabled, then restarts the server with
QW3_CONTINUOUS_BATCHING=1 and sends concurrent deterministic greedy requests.

Correctness is defined as exact text equality against the serial baseline for
each prompt. The continuous-batching run also requires server logs to prove
that at least one decode step used batch >= --min-batch, that paged KV state was
ready for the batch, and that the HGEMM guard was enabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BATCH_STEP_RE = re.compile(
    r"native continuous_batch_step:.*?batch=(?P<batch>\d+).*?"
    r"paged_kv_ready=(?P<paged>true|false)"
)
SUMMARY_RE = re.compile(r"native continuous_batch:.*?max_batch=(?P<max_batch>\d+)")
EXECUTOR_RE = re.compile(
    r"native continuous_batch_executor:.*?"
    r"ragged_metadata_ready=(?P<ragged>true|false).*?"
    r"ragged_pages=(?P<pages>\d+).*?"
    r"ragged_max_seq_len=(?P<max_seq>\d+)"
)


@dataclass
class RequestResult:
    name: str
    prompt: str
    status: int
    elapsed_s: float
    text: str
    raw: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


@dataclass
class Comparison:
    name: str
    baseline_text: str
    continuous_text: str
    equal: bool
    common_prefix_chars: int


@dataclass
class ServerRun:
    mode: str
    command: List[str]
    log: str
    elapsed_s: float
    results: List[RequestResult]
    max_trace_batch: int = 0
    max_summary_batch: int = 0
    saw_paged_kv_ready: bool = False
    saw_hgemm_guard: bool = False
    saw_ragged_metadata_ready: bool = False
    max_ragged_pages: int = 0
    max_ragged_seq_len: int = 0


def prompt_catalog() -> Dict[str, str]:
    return {
        "capital": "The capital of France is",
        "math": "2 + 2 =",
        "cuda": "Write one sentence about CUDA memory coalescing.",
        "chinese": "请用一句话解释连续批处理的作用。",
    }


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def parse_name_list(raw: str) -> List[str]:
    return [x for x in raw.replace(",", " ").split() if x]


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    health = urllib.parse.urljoin(base_url + "/", "health")
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, _ = http_get(health, timeout_s=2.0)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - best-effort readiness loop
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s: {last_error}")


def http_get(url: str, timeout_s: float) -> Tuple[int, str]:
    parsed = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        body = res.read().decode("utf-8", errors="replace")
        return res.status, body
    finally:
        conn.close()


def post_completion(base_url: str,
                    name: str,
                    prompt: str,
                    max_tokens: int,
                    timeout_s: float) -> RequestResult:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    body = json.dumps(payload, ensure_ascii=False)
    headers = {"Content-Type": "application/json"}
    start = time.monotonic()
    try:
        conn.request("POST", "/v1/completions", body=body.encode("utf-8"), headers=headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8", errors="replace")
        elapsed = time.monotonic() - start
        text = ""
        error: Optional[str] = None
        try:
            data = json.loads(raw)
            if res.status == 200:
                text = data["choices"][0]["text"]
            else:
                error = data.get("error", raw)
        except Exception as exc:  # noqa: BLE001 - record bad response body
            error = f"invalid JSON response: {exc}"
        return RequestResult(name, prompt, res.status, elapsed, text, raw, error)
    except Exception as exc:  # noqa: BLE001 - request failure is a test result
        elapsed = time.monotonic() - start
        return RequestResult(name, prompt, 0, elapsed, "", "", str(exc))
    finally:
        conn.close()


def terminate_server(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    out = proc.stdout.read() if proc.stdout else ""
    err = proc.stderr.read() if proc.stderr else ""
    return out + err


def parse_server_log(log: str) -> Tuple[int, int, bool, bool, bool, int, int]:
    max_trace_batch = 0
    saw_paged_kv_ready = False
    for m in BATCH_STEP_RE.finditer(log):
        max_trace_batch = max(max_trace_batch, int(m.group("batch")))
        saw_paged_kv_ready = saw_paged_kv_ready or m.group("paged") == "true"
    max_summary_batch = 0
    for m in SUMMARY_RE.finditer(log):
        max_summary_batch = max(max_summary_batch, int(m.group("max_batch")))
    saw_hgemm_guard = (
        "continuous batching matmul guard" in log and
        "QW3_DISABLE_HGEMM=1" in log
    )
    saw_ragged_metadata_ready = False
    max_ragged_pages = 0
    max_ragged_seq_len = 0
    for m in EXECUTOR_RE.finditer(log):
        saw_ragged_metadata_ready = (
            saw_ragged_metadata_ready or m.group("ragged") == "true"
        )
        max_ragged_pages = max(max_ragged_pages, int(m.group("pages")))
        max_ragged_seq_len = max(max_ragged_seq_len, int(m.group("max_seq")))
    return (
        max_trace_batch,
        max_summary_batch,
        saw_paged_kv_ready,
        saw_hgemm_guard,
        saw_ragged_metadata_ready,
        max_ragged_pages,
        max_ragged_seq_len,
    )


def run_server_case(*,
                    mode: str,
                    binary: Path,
                    model: Path,
                    host: str,
                    port: int,
                    prompts: Dict[str, str],
                    max_tokens: int,
                    ctx_size: int,
                    prefill_chunk: Optional[int],
                    timeout_s: int,
                    concurrent: bool,
                    continuous_batching: bool,
                    max_active: int,
                    extra_args: Sequence[str]) -> ServerRun:
    env = os.environ.copy()
    if continuous_batching:
        env["QW3_CONTINUOUS_BATCHING"] = "1"
        env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
        env["QW3_CONTINUOUS_BATCHING_MAX_ACTIVE"] = str(max_active)
    else:
        env.pop("QW3_CONTINUOUS_BATCHING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", None)
    env.setdefault("QW3_MATMUL", "mmq")

    cmd = [
        str(binary),
        "serve",
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--ctx",
        str(ctx_size),
    ]
    if prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(prefill_chunk)])
    cmd.extend(extra_args)

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log = ""
    results: List[RequestResult] = []
    base_url = f"http://{host}:{port}"
    try:
        wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        if concurrent:
            with futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                futs = [
                    pool.submit(
                        post_completion, base_url, name, prompt, max_tokens,
                        float(timeout_s),
                    )
                    for name, prompt in prompts.items()
                ]
                for fut in futs:
                    results.append(fut.result())
        else:
            for name, prompt in prompts.items():
                results.append(
                    post_completion(base_url, name, prompt, max_tokens, float(timeout_s))
                )
    finally:
        log = terminate_server(proc)

    elapsed = time.monotonic() - start
    (
        max_trace_batch,
        max_summary_batch,
        saw_paged,
        saw_hgemm,
        saw_ragged,
        max_ragged_pages,
        max_ragged_seq_len,
    ) = parse_server_log(log)
    return ServerRun(
        mode=mode,
        command=cmd,
        log=log,
        elapsed_s=elapsed,
        results=results,
        max_trace_batch=max_trace_batch,
        max_summary_batch=max_summary_batch,
        saw_paged_kv_ready=saw_paged,
        saw_hgemm_guard=saw_hgemm,
        saw_ragged_metadata_ready=saw_ragged,
        max_ragged_pages=max_ragged_pages,
        max_ragged_seq_len=max_ragged_seq_len,
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def print_result(prefix: str, result: RequestResult) -> None:
    status = "ok" if result.ok else "FAIL"
    print(
        f"{status:4s} {prefix:10s} {result.name:10s} "
        f"status={result.status:3d} elapsed={result.elapsed_s:.3f}s "
        f"chars={len(result.text):4d}"
    )
    if result.error:
        print(f"     error: {result.error}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run continuous-batching correctness and efficiency checks."
    )
    ap.add_argument("--qw3", default="./build/qw3", help="qw3 binary")
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 selects a free port")
    ap.add_argument("--out-json", default="/tmp/qw3_continuous_batching_regression.json")
    ap.add_argument(
        "--prompts",
        type=parse_name_list,
        default=parse_name_list("capital math cuda chinese"),
        help="prompt names: capital math cuda chinese",
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--prefill-chunk", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-active", type=int, default=2)
    ap.add_argument("--min-batch", type=int, default=2)
    ap.add_argument(
        "--require-ragged-metadata",
        action="store_true",
        help="require at least one batched decode step with ragged metadata ready",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra argument passed to qw3 serve; repeat for multiple args",
    )
    args = ap.parse_args(argv)

    binary = Path(args.qw3)
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"qw3 binary not found: {binary}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.max_active < 1:
        raise SystemExit("--max-active must be positive")
    if args.min_batch < 1:
        raise SystemExit("--min-batch must be positive")

    catalog = prompt_catalog()
    unknown = [name for name in args.prompts if name not in catalog]
    if unknown:
        raise SystemExit(
            "unknown prompt(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(catalog))
        )
    prompts = {name: catalog[name] for name in args.prompts}
    port = args.port if args.port else free_port()

    print(
        f"running continuous batching regression: prompts={len(prompts)} "
        f"max_tokens={args.max_tokens} port={port}"
    )
    baseline = run_server_case(
        mode="baseline",
        binary=binary,
        model=model,
        host=args.host,
        port=port,
        prompts=prompts,
        max_tokens=args.max_tokens,
        ctx_size=args.ctx,
        prefill_chunk=args.prefill_chunk,
        timeout_s=args.timeout,
        concurrent=False,
        continuous_batching=False,
        max_active=args.max_active,
        extra_args=args.extra_arg,
    )
    for result in baseline.results:
        print_result("baseline", result)

    continuous = run_server_case(
        mode="continuous",
        binary=binary,
        model=model,
        host=args.host,
        port=port,
        prompts=prompts,
        max_tokens=args.max_tokens,
        ctx_size=args.ctx,
        prefill_chunk=args.prefill_chunk,
        timeout_s=args.timeout,
        concurrent=True,
        continuous_batching=True,
        max_active=args.max_active,
        extra_args=args.extra_arg,
    )
    for result in continuous.results:
        print_result("continuous", result)

    baseline_by_name = {r.name: r for r in baseline.results}
    continuous_by_name = {r.name: r for r in continuous.results}
    comparisons: List[Comparison] = []
    for name in prompts:
        lhs = baseline_by_name.get(name)
        rhs = continuous_by_name.get(name)
        if not lhs or not rhs or not lhs.ok or not rhs.ok:
            continue
        comparisons.append(
            Comparison(
                name=name,
                baseline_text=lhs.text,
                continuous_text=rhs.text,
                equal=lhs.text == rhs.text,
                common_prefix_chars=common_prefix_len(lhs.text, rhs.text),
            )
        )

    failed_runs = [r for r in baseline.results + continuous.results if not r.ok]
    failed_comparisons = [c for c in comparisons if not c.equal]
    max_batch = max(continuous.max_trace_batch, continuous.max_summary_batch)
    failed_requirements: List[str] = []
    if max_batch < args.min_batch:
        failed_requirements.append(
            f"max continuous batch {max_batch} < required {args.min_batch}"
        )
    if not continuous.saw_paged_kv_ready:
        failed_requirements.append("did not observe paged_kv_ready=true")
    if not continuous.saw_hgemm_guard:
        failed_requirements.append("did not observe QW3_DISABLE_HGEMM=1 guard")
    if args.require_ragged_metadata and not continuous.saw_ragged_metadata_ready:
        failed_requirements.append("did not observe ragged_metadata_ready=true")

    summary = {
        "config": {
            "qw3": str(binary),
            "model": str(model),
            "host": args.host,
            "port": port,
            "prompts": list(prompts.keys()),
            "max_tokens": args.max_tokens,
            "ctx": args.ctx,
            "prefill_chunk": args.prefill_chunk,
            "max_active": args.max_active,
            "min_batch": args.min_batch,
        },
        "status": {
            "failed_runs": len(failed_runs),
            "comparisons": len(comparisons),
            "failed_comparisons": len(failed_comparisons),
            "failed_requirements": failed_requirements,
            "continuous_max_trace_batch": continuous.max_trace_batch,
            "continuous_max_summary_batch": continuous.max_summary_batch,
            "continuous_saw_paged_kv_ready": continuous.saw_paged_kv_ready,
            "continuous_saw_hgemm_guard": continuous.saw_hgemm_guard,
            "continuous_saw_ragged_metadata_ready": (
                continuous.saw_ragged_metadata_ready
            ),
            "continuous_max_ragged_pages": continuous.max_ragged_pages,
            "continuous_max_ragged_seq_len": continuous.max_ragged_seq_len,
            "baseline_elapsed_s": baseline.elapsed_s,
            "continuous_elapsed_s": continuous.elapsed_s,
        },
        "baseline": {
            "command": baseline.command,
            "results": [asdict(r) for r in baseline.results],
            "log": baseline.log,
        },
        "continuous": {
            "command": continuous.command,
            "results": [asdict(r) for r in continuous.results],
            "log": continuous.log,
        },
        "comparisons": [asdict(c) for c in comparisons],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "continuous evidence: "
        f"trace_max_batch={continuous.max_trace_batch} "
        f"summary_max_batch={continuous.max_summary_batch} "
        f"paged_kv_ready={continuous.saw_paged_kv_ready} "
        f"hgemm_guard={continuous.saw_hgemm_guard} "
        f"ragged_metadata_ready={continuous.saw_ragged_metadata_ready} "
        f"ragged_pages={continuous.max_ragged_pages} "
        f"ragged_max_seq_len={continuous.max_ragged_seq_len}"
    )
    print(
        f"elapsed: baseline={baseline.elapsed_s:.3f}s "
        f"continuous={continuous.elapsed_s:.3f}s"
    )
    print(f"wrote summary: {out_path}")

    if failed_runs:
        print(f"failed runs: {len(failed_runs)}", file=sys.stderr)
        for r in failed_runs:
            print(f"  {r.name}: {r.error}", file=sys.stderr)
    if failed_comparisons:
        print(f"failed output comparisons: {len(failed_comparisons)}", file=sys.stderr)
        for c in failed_comparisons:
            print(
                f"  {c.name}: common_prefix={c.common_prefix_chars}",
                file=sys.stderr,
            )
    if failed_requirements:
        print("failed continuous-batching requirements:", file=sys.stderr)
        for item in failed_requirements:
            print(f"  {item}", file=sys.stderr)

    return 1 if failed_runs or failed_comparisons or failed_requirements else 0


if __name__ == "__main__":
    raise SystemExit(main())
