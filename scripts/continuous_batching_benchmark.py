#!/usr/bin/env python3
"""Benchmark continuous batching decode variants.

The script starts qw3 once per variant, sends deterministic /v1/completions
requests, and records request latency, output-token throughput, and batching
evidence from server logs. It is intentionally separate from the correctness
regression script so longer throughput sweeps can evolve without weakening the
exact-parity checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import continuous_batching_regression as cbr


@dataclass
class BenchRequest:
    name: str
    prompt: str
    status: int
    elapsed_s: float
    text: str
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


@dataclass
class BenchVariant:
    name: str
    continuous: bool
    concurrent: bool
    env: Dict[str, str]


@dataclass
class BenchRun:
    variant: str
    command: List[str]
    elapsed_s: float
    request_wall_s: float
    requests: List[BenchRequest]
    output_tokens: int
    tokens_per_s: float
    mean_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    max_trace_batch: int
    max_summary_batch: int
    saw_body_batch_ready: bool
    saw_body_batch_mode: bool
    saw_ragged_metadata_ready: bool
    log: str


def post_completion(base_url: str,
                    name: str,
                    prompt: str,
                    max_tokens: int,
                    timeout_s: float) -> BenchRequest:
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
        completion_tokens = 0
        total_tokens = 0
        error: Optional[str] = None
        try:
            data = json.loads(raw)
            if res.status == 200:
                text = data["choices"][0]["text"]
                usage = data.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or 0)
            else:
                error = data.get("error", raw)
        except Exception as exc:  # noqa: BLE001
            error = f"invalid JSON response: {exc}"
        return BenchRequest(
            name=name,
            prompt=prompt,
            status=res.status,
            elapsed_s=elapsed,
            text=text,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        return BenchRequest(name, prompt, 0, elapsed, "", 0, 0, str(exc))
    finally:
        conn.close()


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[idx]


def run_variant(*,
                variant: BenchVariant,
                binary: Path,
                model: Path,
                host: str,
                port: int,
                prompts: Dict[str, str],
                max_tokens: int,
                ctx_size: int,
                prefill_chunk: Optional[int],
                timeout_s: int,
                max_active: int,
                extra_args: Sequence[str]) -> BenchRun:
    env = os.environ.copy()
    env.setdefault("QW3_MATMUL", "mmq")
    if variant.continuous:
        env["QW3_CONTINUOUS_BATCHING"] = "1"
        env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
        env["QW3_CONTINUOUS_BATCHING_MAX_ACTIVE"] = str(max_active)
        env.update(variant.env)
    else:
        env.pop("QW3_CONTINUOUS_BATCHING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_BODY_BATCH", None)
        env.pop("QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH", None)

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

    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start = time.monotonic()
    log = ""
    requests: List[BenchRequest] = []
    base_url = f"http://{host}:{port}"
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        request_start = time.monotonic()
        if variant.concurrent:
            with futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                futs = [
                    pool.submit(
                        post_completion,
                        base_url,
                        name,
                        prompt,
                        max_tokens,
                        float(timeout_s),
                    )
                    for name, prompt in prompts.items()
                ]
                for fut in futs:
                    requests.append(fut.result())
        else:
            for name, prompt in prompts.items():
                requests.append(
                    post_completion(base_url, name, prompt, max_tokens, float(timeout_s))
                )
        request_wall_s = time.monotonic() - request_start
    finally:
        log = cbr.terminate_server(proc)

    elapsed_s = time.monotonic() - start
    (
        max_trace_batch,
        max_summary_batch,
        _saw_paged,
        _saw_hgemm,
        saw_body,
        saw_body_mode,
        saw_ragged,
        _max_ragged_pages,
        _max_ragged_seq_len,
    ) = cbr.parse_server_log(log)
    latencies = [r.elapsed_s for r in requests if r.ok]
    output_tokens = sum(r.completion_tokens for r in requests if r.ok)
    tokens_per_s = output_tokens / request_wall_s if request_wall_s > 0 else 0.0
    return BenchRun(
        variant=variant.name,
        command=cmd,
        elapsed_s=elapsed_s,
        request_wall_s=request_wall_s,
        requests=requests,
        output_tokens=output_tokens,
        tokens_per_s=tokens_per_s,
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        p50_latency_s=percentile(latencies, 0.50),
        p90_latency_s=percentile(latencies, 0.90),
        max_trace_batch=max_trace_batch,
        max_summary_batch=max_summary_batch,
        saw_body_batch_ready=saw_body,
        saw_body_batch_mode=saw_body_mode,
        saw_ragged_metadata_ready=saw_ragged,
        log=log,
    )


def variants_for(names: Sequence[str]) -> List[BenchVariant]:
    all_variants = {
        "plain": BenchVariant("plain", continuous=False, concurrent=False, env={}),
        "continuous": BenchVariant("continuous", continuous=True, concurrent=True, env={}),
        "body": BenchVariant(
            "body",
            continuous=True,
            concurrent=True,
            env={
                "QW3_CONTINUOUS_BATCHING_BODY_BATCH": "1",
                "QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH": "0",
            },
        ),
        "recurrent": BenchVariant(
            "recurrent",
            continuous=True,
            concurrent=True,
            env={
                "QW3_CONTINUOUS_BATCHING_BODY_BATCH": "1",
                "QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH": "1",
            },
        ),
    }
    result: List[BenchVariant] = []
    for name in names:
        if name not in all_variants:
            raise SystemExit(
                f"unknown variant {name!r}; available: "
                + ", ".join(sorted(all_variants))
            )
        result.append(all_variants[name])
    return result


def parse_variants(raw: str) -> List[str]:
    return [x for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark qw3 continuous batching variants.")
    ap.add_argument("--qw3", default="./build/qw3", help="qw3 binary")
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 selects a free port")
    ap.add_argument("--out-json", default="/tmp/qw3_continuous_batching_benchmark.json")
    ap.add_argument(
        "--prompts",
        type=cbr.parse_name_list,
        default=cbr.parse_name_list("capital math"),
        help="prompt names from continuous_batching_regression.py",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-active", type=int, default=2)
    ap.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("plain continuous body recurrent"),
        help="variants: plain continuous body recurrent",
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

    catalog = cbr.prompt_catalog()
    unknown = [name for name in args.prompts if name not in catalog]
    if unknown:
        raise SystemExit(
            "unknown prompt(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(catalog))
        )
    prompts = {name: catalog[name] for name in args.prompts}
    port = args.port if args.port else cbr.free_port()

    runs: List[BenchRun] = []
    for variant in variants_for(args.variants):
        print(f"running benchmark variant={variant.name} prompts={len(prompts)}")
        run = run_variant(
            variant=variant,
            binary=binary,
            model=model,
            host=args.host,
            port=port,
            prompts=prompts,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx,
            prefill_chunk=args.prefill_chunk,
            timeout_s=args.timeout,
            max_active=args.max_active,
            extra_args=args.extra_arg,
        )
        runs.append(run)
        failed = [r for r in run.requests if not r.ok]
        status = "FAIL" if failed else "ok"
        print(
            f"{status:4s} {run.variant:10s} "
            f"wall={run.request_wall_s:.3f}s "
            f"tok/s={run.tokens_per_s:.2f} "
            f"tokens={run.output_tokens} "
            f"mean={run.mean_latency_s:.3f}s "
            f"p50={run.p50_latency_s:.3f}s "
            f"p90={run.p90_latency_s:.3f}s "
            f"batch={max(run.max_trace_batch, run.max_summary_batch)} "
            f"body={run.saw_body_batch_mode} "
            f"ragged={run.saw_ragged_metadata_ready}"
        )
        for req in run.requests:
            if not req.ok:
                print(f"     {req.name}: {req.error}")

    out = {
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
            "variants": args.variants,
        },
        "runs": [asdict(run) for run in runs],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote summary: {out_path}")

    return 1 if any(not req.ok for run in runs for req in run.requests) else 0


if __name__ == "__main__":
    raise SystemExit(main())
