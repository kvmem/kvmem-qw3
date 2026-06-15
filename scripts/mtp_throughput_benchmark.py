#!/usr/bin/env python3
"""Benchmark MTP throughput with and without continuous batching."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import continuous_batching_benchmark as cbb
import continuous_batching_regression as cbr
import mtp_continuous_regression as mcr


@dataclass
class BenchMode:
    name: str
    continuous: bool
    mtp: bool


@dataclass
class MtpBenchRun:
    mode: str
    concurrency: int
    elapsed_s: float
    request_wall_s: float
    output_tokens: int
    output_tokens_per_s: float
    mean_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    requests: list[cbb.BenchRequest]
    mtp_summaries: list[mcr.MtpSummary]
    continuous_mtp_count: int
    route_continuous_count: int
    command: list[str]
    log: str


def parse_modes(raw: str) -> list[str]:
    return [x for x in raw.replace(",", " ").split() if x]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[idx]


def make_env(mode: BenchMode, max_active: int, extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    if mode.continuous:
        env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    else:
        env.pop("QW3_CONTINUOUS_BATCHING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
    env.update(extra_env)
    return env


def make_command(binary: Path,
                 model: Path,
                 host: str,
                 port: int,
                 mode: BenchMode,
                 max_tokens: int,
                 ctx_size: int,
                 prefill_chunk: int,
                 chain: int,
                 kv_dtype: str,
                 max_active: int) -> list[str]:
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
        "--prefill-chunk",
        str(prefill_chunk),
        "--kv-dtype",
        kv_dtype,
    ]
    if mode.continuous:
        cmd.extend(["--continuous-batching", "--max-active", str(max_active)])
    if mode.mtp:
        cmd.extend(["--mtp-chain", str(chain)])
    return cmd


def synthetic_prompts(concurrency: int, repeat: int) -> dict[str, str]:
    unit = (
        "This deterministic benchmark prompt discusses CUDA memory coalescing, "
        "paged KV cache layout, speculative decoding acceptance, and continuous "
        "batching throughput. "
    )
    body = unit * repeat
    return {
        f"req{i + 1}": (
            body
            + f"\nRequest {i + 1}: continue with concise technical prose."
        )
        for i in range(concurrency)
    }


def run_case(*,
             mode: BenchMode,
             binary: Path,
             model: Path,
             host: str,
             port: int,
             concurrency: int,
             prompt_repeat: int,
             max_tokens: int,
             ctx_size: int,
             prefill_chunk: int,
             chain: int,
             kv_dtype: str,
             timeout_s: int,
             max_active: int,
             extra_env: Dict[str, str]) -> MtpBenchRun:
    cmd = make_command(
        binary, model, host, port, mode, max_tokens, ctx_size, prefill_chunk,
        chain, kv_dtype, max(max_active, concurrency))
    proc = subprocess.Popen(
        cmd,
        env=make_env(mode, max(max_active, concurrency), extra_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks, log_threads = cbb.start_log_drain(proc)
    base_url = f"http://{host}:{port}"
    prompts = synthetic_prompts(concurrency, prompt_repeat)
    start = time.monotonic()
    requests: list[cbb.BenchRequest] = []
    request_wall_s = 0.0
    log = ""
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        request_start = time.monotonic()
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [
                pool.submit(
                    cbb.post_completion,
                    base_url,
                    name,
                    prompt,
                    max_tokens,
                    True,
                    float(timeout_s),
                )
                for name, prompt in prompts.items()
            ]
            for fut in futs:
                requests.append(fut.result())
        request_wall_s = time.monotonic() - request_start
    finally:
        log = cbb.terminate_drained_server(proc, log_chunks, log_threads)

    elapsed_s = time.monotonic() - start
    output_tokens = sum(r.completion_tokens for r in requests if r.ok)
    latencies = [r.elapsed_s for r in requests if r.ok]
    summaries = [
        mcr.MtpSummary(
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
        for m in mcr.MTP_SPEC_RE.finditer(log)
    ]
    return MtpBenchRun(
        mode=mode.name,
        concurrency=concurrency,
        elapsed_s=elapsed_s,
        request_wall_s=request_wall_s,
        output_tokens=output_tokens,
        output_tokens_per_s=(
            output_tokens / request_wall_s if request_wall_s > 0.0 else 0.0
        ),
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        p50_latency_s=percentile(latencies, 0.50),
        p90_latency_s=percentile(latencies, 0.90),
        requests=requests,
        mtp_summaries=summaries,
        continuous_mtp_count=log.count("native continuous_mtp:"),
        route_continuous_count=log.count("route=continuous"),
        command=cmd,
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


def modes_for(names: Sequence[str]) -> list[BenchMode]:
    all_modes = {
        "continuous": BenchMode("continuous", continuous=True, mtp=False),
        "continuous_mtp": BenchMode("continuous_mtp", continuous=True, mtp=True),
        "legacy": BenchMode("legacy", continuous=False, mtp=False),
        "legacy_mtp": BenchMode("legacy_mtp", continuous=False, mtp=True),
    }
    modes: list[BenchMode] = []
    for name in names:
        if name not in all_modes:
            raise SystemExit(
                f"unknown mode {name!r}; available: "
                + ", ".join(sorted(all_modes))
            )
        modes.append(all_modes[name])
    return modes


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--out-json", default="/tmp/qw3_mtp_throughput_benchmark.json")
    ap.add_argument("--modes", type=parse_modes,
                    default=parse_modes("continuous continuous_mtp"))
    ap.add_argument("--concurrency-levels", type=cbb.parse_int_list,
                    default=cbb.parse_int_list("1 2"))
    ap.add_argument("--prompt-repeat", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--kv-dtype", default="fp8", choices=("fp16", "fp8", "q8", "fp32"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-active", type=int, default=4)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    binary = Path(args.qw3)
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"qw3 binary not found: {binary}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    port = args.port if args.port else cbr.free_port()
    extra_env = parse_env(args.env)

    runs: list[MtpBenchRun] = []
    failures: list[str] = []
    for mode in modes_for(args.modes):
        for concurrency in args.concurrency_levels:
            print(
                f"running mode={mode.name} concurrency={concurrency} "
                f"max_tokens={args.max_tokens}",
                flush=True,
            )
            run = run_case(
                mode=mode,
                binary=binary,
                model=model,
                host=args.host,
                port=port,
                concurrency=concurrency,
                prompt_repeat=args.prompt_repeat,
                max_tokens=args.max_tokens,
                ctx_size=args.ctx,
                prefill_chunk=args.prefill_chunk,
                chain=args.chain,
                kv_dtype=args.kv_dtype,
                timeout_s=args.timeout,
                max_active=args.max_active,
                extra_env=extra_env,
            )
            runs.append(run)
            failed = [r for r in run.requests if not r.ok]
            if failed:
                failures.append(f"{mode.name} concurrency={concurrency} failed={len(failed)}")
            accepted = sum(s.accepted for s in run.mtp_summaries)
            drafted = sum(s.drafted for s in run.mtp_summaries)
            batched_verify = sum(s.batched_verify_batches for s in run.mtp_summaries)
            print(
                f"{mode.name:15s} conc={concurrency:2d} "
                f"wall={run.request_wall_s:.3f}s "
                f"out_tok/s={run.output_tokens_per_s:.2f} "
                f"tokens={run.output_tokens} "
                f"mean={run.mean_latency_s:.3f}s "
                f"mtp_accept={accepted}/{drafted} "
                f"batched_verify={batched_verify} "
                f"continuous_mtp={run.continuous_mtp_count}",
                flush=True,
            )

    out = {
        "config": vars(args),
        "status": {"failures": failures},
        "runs": [asdict(r) for r in runs],
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
