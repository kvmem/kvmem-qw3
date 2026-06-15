#!/usr/bin/env python3
"""Regression for MTP routed through the continuous-batching worker."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import continuous_batching_benchmark as cbb
import continuous_batching_regression as cbr


MTP_SPEC_RE = re.compile(
    r"native mtp_spec_summary:.*?"
    r"enabled=(?P<enabled>true|false).*?"
    r"batches=(?P<batches>\d+).*?"
    r"drafted=(?P<drafted>\d+).*?"
    r"accepted=(?P<accepted>\d+).*?"
    r"rejected=(?P<rejected>\d+).*?"
    r"rollbacks=(?P<rollbacks>\d+).*?"
    r"acceptance=(?P<acceptance>[0-9.]+).*?"
    r"batched_verify_batches=(?P<batched_verify_batches>\d+).*?"
    r"batched_verify_tokens=(?P<batched_verify_tokens>\d+)"
)


@dataclass
class MtpSummary:
    enabled: bool = False
    batches: int = 0
    drafted: int = 0
    accepted: int = 0
    rejected: int = 0
    rollbacks: int = 0
    acceptance: float = 0.0
    batched_verify_batches: int = 0
    batched_verify_tokens: int = 0


@dataclass
class MtpRun:
    mode: str
    request: cbb.BenchRequest
    elapsed_s: float
    command: list[str]
    saw_continuous_mtp: bool
    saw_route_continuous: bool
    mtp_summary: MtpSummary
    log: str


@dataclass
class MtpConcurrentRun:
    requests: list[cbb.BenchRequest]
    elapsed_s: float
    command: list[str]
    continuous_mtp_count: int
    route_continuous_count: int
    mtp_summaries: list[MtpSummary]
    log: str


def parse_mtp_summary(log: str) -> MtpSummary:
    matches = list(MTP_SPEC_RE.finditer(log))
    if not matches:
        return MtpSummary()
    m = matches[-1]
    return MtpSummary(
        enabled=m.group("enabled") == "true",
        batches=int(m.group("batches")),
        drafted=int(m.group("drafted")),
        accepted=int(m.group("accepted")),
        rejected=int(m.group("rejected")),
        rollbacks=int(m.group("rollbacks")),
        acceptance=float(m.group("acceptance")),
        batched_verify_batches=int(m.group("batched_verify_batches")),
        batched_verify_tokens=int(m.group("batched_verify_tokens")),
    )


def make_env(continuous: bool, extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    if continuous:
        env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    else:
        env.pop("QW3_CONTINUOUS_BATCHING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
    env.update(extra_env)
    return env


def run_server_case(*,
                    mode: str,
                    continuous: bool,
                    binary: Path,
                    model: Path,
                    host: str,
                    port: int,
                    prompt: str,
                    max_tokens: int,
                    ctx_size: int,
                    prefill_chunk: int,
                    timeout_s: int,
                    chain: int,
                    kv_dtype: str,
                    extra_env: Dict[str, str]) -> MtpRun:
    cmd = [
        str(binary),
        "serve",
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx",
        str(ctx_size),
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--mtp-chain",
        str(chain),
        "--prefill-chunk",
        str(prefill_chunk),
        "--kv-dtype",
        kv_dtype,
    ]
    if continuous:
        cmd.extend(["--continuous-batching", "--max-active", "2"])
    proc = subprocess.Popen(
        cmd,
        env=make_env(continuous, extra_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks, log_threads = cbb.start_log_drain(proc)
    base_url = f"http://{host}:{port}"
    start = time.monotonic()
    request: Optional[cbb.BenchRequest] = None
    log = ""
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        request = cbb.post_completion(
            base_url, mode, prompt, max_tokens, True, float(timeout_s)
        )
    finally:
        log = cbb.terminate_drained_server(proc, log_chunks, log_threads)
    elapsed = time.monotonic() - start
    if request is None:
        request = cbb.BenchRequest(mode, prompt, 0, elapsed, "", 0, 0,
                                   "request was not sent")
    return MtpRun(
        mode=mode,
        request=request,
        elapsed_s=elapsed,
        command=cmd,
        saw_continuous_mtp="native continuous_mtp:" in log,
        saw_route_continuous="route=continuous" in log,
        mtp_summary=parse_mtp_summary(log),
        log=log,
    )


def run_continuous_concurrent_case(*,
                                   binary: Path,
                                   model: Path,
                                   host: str,
                                   port: int,
                                   prompt: str,
                                   concurrency: int,
                                   max_tokens: int,
                                   ctx_size: int,
                                   prefill_chunk: int,
                                   timeout_s: int,
                                   chain: int,
                                   kv_dtype: str,
                                   extra_env: Dict[str, str]) -> MtpConcurrentRun:
    cmd = [
        str(binary),
        "serve",
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx",
        str(ctx_size),
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--continuous-batching",
        "--max-active",
        str(max(1, concurrency)),
        "--mtp-chain",
        str(chain),
        "--prefill-chunk",
        str(prefill_chunk),
        "--kv-dtype",
        kv_dtype,
    ]
    proc = subprocess.Popen(
        cmd,
        env=make_env(True, extra_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks, log_threads = cbb.start_log_drain(proc)
    base_url = f"http://{host}:{port}"
    start = time.monotonic()
    requests: list[cbb.BenchRequest] = []
    log = ""
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        prompts = {
            f"mtp{i + 1}": f"{prompt}\nRequest {i + 1}."
            for i in range(concurrency)
        }
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [
                pool.submit(
                    cbb.post_completion,
                    base_url,
                    name,
                    text,
                    max_tokens,
                    True,
                    float(timeout_s),
                )
                for name, text in prompts.items()
            ]
            for fut in futs:
                requests.append(fut.result())
    finally:
        log = cbb.terminate_drained_server(proc, log_chunks, log_threads)
    summaries = [
        MtpSummary(
            enabled=m.group("enabled") == "true",
            batches=int(m.group("batches")),
            drafted=int(m.group("drafted")),
            accepted=int(m.group("accepted")),
            rejected=int(m.group("rejected")),
            rollbacks=int(m.group("rollbacks")),
            acceptance=float(m.group("acceptance")),
            batched_verify_batches=int(m.group("batched_verify_batches")),
            batched_verify_tokens=int(m.group("batched_verify_tokens")),
        )
        for m in MTP_SPEC_RE.finditer(log)
    ]
    return MtpConcurrentRun(
        requests=requests,
        elapsed_s=time.monotonic() - start,
        command=cmd,
        continuous_mtp_count=log.count("native continuous_mtp:"),
        route_continuous_count=log.count("route=continuous"),
        mtp_summaries=summaries,
        log=log,
    )


def parse_env(raw: Sequence[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid --env value: {item!r}")
        env[key] = value
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--out-json", default="/tmp/qw3_mtp_continuous_regression.json")
    ap.add_argument("--prompt", default="Write one sentence about CUDA memory coalescing.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--kv-dtype", default="fp8", choices=("fp16", "fp8", "q8", "fp32"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--concurrent-continuous", type=int, default=0)
    ap.add_argument(
        "--skip-text-compare",
        action="store_true",
        help="only require successful runs and MTP/continuous evidence",
    )
    args = ap.parse_args(argv)

    binary = Path(args.qw3)
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"qw3 binary not found: {binary}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    port = args.port if args.port else cbr.free_port()
    extra_env = parse_env(args.env)

    legacy = run_server_case(
        mode="legacy",
        continuous=False,
        binary=binary,
        model=model,
        host=args.host,
        port=port,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        ctx_size=args.ctx,
        prefill_chunk=args.prefill_chunk,
        timeout_s=args.timeout,
        chain=args.chain,
        kv_dtype=args.kv_dtype,
        extra_env=extra_env,
    )
    continuous = run_server_case(
        mode="continuous",
        continuous=True,
        binary=binary,
        model=model,
        host=args.host,
        port=port,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        ctx_size=args.ctx,
        prefill_chunk=args.prefill_chunk,
        timeout_s=args.timeout,
        chain=args.chain,
        kv_dtype=args.kv_dtype,
        extra_env=extra_env,
    )
    concurrent_run: Optional[MtpConcurrentRun] = None
    if args.concurrent_continuous > 0:
        concurrent_run = run_continuous_concurrent_case(
            binary=binary,
            model=model,
            host=args.host,
            port=port,
            prompt=args.prompt,
            concurrency=args.concurrent_continuous,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx,
            prefill_chunk=args.prefill_chunk,
            timeout_s=args.timeout,
            chain=args.chain,
            kv_dtype=args.kv_dtype,
            extra_env=extra_env,
        )

    failures: list[str] = []
    if not legacy.request.ok:
        failures.append(f"legacy request failed: {legacy.request.error}")
    if not continuous.request.ok:
        failures.append(f"continuous request failed: {continuous.request.error}")
    if not continuous.saw_continuous_mtp:
        failures.append("continuous run did not log native continuous_mtp")
    if not continuous.saw_route_continuous:
        failures.append("continuous run did not report route=continuous")
    if not continuous.mtp_summary.enabled:
        failures.append("continuous run did not report enabled MTP summary")
    if continuous.mtp_summary.batched_verify_batches <= 0:
        failures.append("continuous run did not use batched MTP verify")
    if not args.skip_text_compare and legacy.request.text != continuous.request.text:
        failures.append("legacy and continuous generated text differ")
    if concurrent_run is not None:
        failed = [r for r in concurrent_run.requests if not r.ok]
        if failed:
            failures.append(f"concurrent continuous requests failed: {len(failed)}")
        if concurrent_run.continuous_mtp_count < args.concurrent_continuous:
            failures.append(
                "concurrent run did not log one continuous_mtp entry per request"
            )
        if concurrent_run.route_continuous_count < args.concurrent_continuous:
            failures.append(
                "concurrent run did not route every request through continuous"
            )
        if len(concurrent_run.mtp_summaries) < args.concurrent_continuous:
            failures.append(
                "concurrent run did not log one MTP summary per request"
            )
        if any(s.batched_verify_batches <= 0 for s in concurrent_run.mtp_summaries):
            failures.append("concurrent run had an MTP request without batched verify")

    print(
        "legacy: "
        f"ok={legacy.request.ok} tokens={legacy.request.completion_tokens} "
        f"batches={legacy.mtp_summary.batches} "
        f"accepted={legacy.mtp_summary.accepted} "
        f"rejected={legacy.mtp_summary.rejected} "
        f"rollbacks={legacy.mtp_summary.rollbacks} "
        f"batched_verify={legacy.mtp_summary.batched_verify_batches}"
    )
    print(
        "continuous: "
        f"ok={continuous.request.ok} tokens={continuous.request.completion_tokens} "
        f"continuous_mtp={continuous.saw_continuous_mtp} "
        f"route_continuous={continuous.saw_route_continuous} "
        f"batches={continuous.mtp_summary.batches} "
        f"accepted={continuous.mtp_summary.accepted} "
        f"rejected={continuous.mtp_summary.rejected} "
        f"rollbacks={continuous.mtp_summary.rollbacks} "
        f"batched_verify={continuous.mtp_summary.batched_verify_batches}"
    )
    if concurrent_run is not None:
        print(
            "concurrent continuous: "
            f"requests={len(concurrent_run.requests)} "
            f"continuous_mtp={concurrent_run.continuous_mtp_count} "
            f"route_continuous={concurrent_run.route_continuous_count} "
            f"summaries={len(concurrent_run.mtp_summaries)}"
        )

    out = {
        "config": vars(args),
        "status": {"failures": failures},
        "legacy": asdict(legacy),
        "continuous": asdict(continuous),
        "concurrent_continuous": (
            asdict(concurrent_run) if concurrent_run is not None else None
        ),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote summary: {out_path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
