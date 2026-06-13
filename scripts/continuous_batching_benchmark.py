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
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import continuous_batching_regression as cbr


SUMMARY_RE = re.compile(
    r"native continuous_batch: request=(?P<request>\d+) "
    r"prompt_tokens=(?P<prompt>\d+) "
    r"prefill=(?P<prefill>[0-9.]+)s .*? "
    r"decoded=(?P<decoded>\d+) "
    r"decode=(?P<decode>[0-9.]+)s .*? "
    r".*?max_batch=(?P<max_batch>\d+)"
)

GENERATE_RE = re.compile(
    r"native generate:\s+"
    r"prompt_tokens=(?P<prompt>\d+)\s+"
    r"prefill=(?P<prefill>[0-9.]+)s"
    r"(?:\s+\([0-9.]+\s+tok/s\))?\s+"
    r"decoded=(?P<decoded>\d+)\s+"
    r"decode=(?P<decode>[0-9.]+)s"
)

TIMING_RE = re.compile(
    r"native continuous_batch_timing:\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"kernel_batch=(?P<kernel_batch>\d+)\s+"
    r"total=(?P<total>[0-9.]+)s"
)

TIMING_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[0-9.]+)s")


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
    ctx_size: int
    input_target_tokens: int
    concurrency: int
    command: List[str]
    elapsed_s: float
    request_wall_s: float
    requests: List[BenchRequest]
    output_tokens: int
    tokens_per_s: float
    prefill_tokens: int
    prefill_s: float
    prefill_tokens_per_s: float
    decode_tokens: int
    decode_s: float
    decode_tokens_per_s: float
    decode_step_tokens: int
    decode_step_s: float
    decode_step_tokens_per_s: float
    decode_prepare_s: float
    decode_metadata_s: float
    decode_embed_s: float
    decode_layers_s: float
    decode_recurrent_s: float
    decode_recurrent_state_s: float
    decode_attention_s: float
    decode_qkv_s: float
    decode_kv_append_s: float
    decode_attn_kernel_s: float
    decode_attn_output_s: float
    decode_ffn_s: float
    decode_final_s: float
    decode_lm_head_s: float
    decode_argmax_s: float
    decode_post_s: float
    mean_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    max_trace_batch: int
    max_summary_batch: int
    saw_body_batch_ready: bool
    saw_body_batch_mode: bool
    saw_ragged_metadata_ready: bool
    log: str


@dataclass
class SummaryMetrics:
    prompt_tokens: int
    prefill_s: float
    decoded_tokens: int
    decode_s: float
    max_batch: int


@dataclass
class TimingMetrics:
    mode: str
    batch_tokens: int
    kernel_batch: int
    total_s: float
    prepare_s: float = 0.0
    metadata_s: float = 0.0
    embed_s: float = 0.0
    layers_s: float = 0.0
    recurrent_s: float = 0.0
    recurrent_state_s: float = 0.0
    attention_s: float = 0.0
    qkv_s: float = 0.0
    kv_append_s: float = 0.0
    attn_kernel_s: float = 0.0
    attn_output_s: float = 0.0
    ffn_s: float = 0.0
    final_s: float = 0.0
    lm_head_s: float = 0.0
    argmax_s: float = 0.0
    post_s: float = 0.0


def start_log_drain(proc: subprocess.Popen[str]) -> tuple[List[str], List[threading.Thread]]:
    chunks: List[str] = []
    lock = threading.Lock()

    def drain(stream: object) -> None:
        if stream is None:
            return
        try:
            for line in stream:  # type: ignore[operator]
                with lock:
                    chunks.append(line)
        except Exception:  # noqa: BLE001 - best-effort log collection
            return

    threads: List[threading.Thread] = []
    for stream in (proc.stdout, proc.stderr):
        t = threading.Thread(target=drain, args=(stream,), daemon=True)
        t.start()
        threads.append(t)
    return chunks, threads


def terminate_drained_server(proc: subprocess.Popen[str],
                             chunks: List[str],
                             threads: List[threading.Thread]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    for t in threads:
        t.join(timeout=2)
    return "".join(chunks)


def parse_int_list(raw: str) -> List[int]:
    values = [int(x) for x in raw.replace(",", " ").split() if x]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected positive integer list")
    return values


def synthetic_prompts(count: int, repeat: int) -> Dict[str, str]:
    unit = (
        "This is deterministic benchmark context about paged KV, continuous "
        "batching, FlashInfer attention, recurrent state isolation, and "
        "OpenAI compatible serving. "
    )
    body = unit * repeat
    return {
        f"req{i + 1}": (
            body
            + f"\nRequest {i + 1}: answer with concise benchmark output."
        )
        for i in range(count)
    }


def synthetic_prompts_for_target(count: int, target_tokens: int) -> Dict[str, str]:
    # Empirically this deterministic unit tokenizes at about 28 tokens per
    # repeat for Qwen3.6-27B. The benchmark records actual prompt_tokens from
    # server logs, so the target is only used to size the synthetic prompt.
    repeat = max(1, int(round(target_tokens / 28.0)))
    return synthetic_prompts(count, repeat)


def catalog_prompts(names: Sequence[str], count: int) -> Dict[str, str]:
    catalog = cbr.prompt_catalog()
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise SystemExit(
            "unknown prompt(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(catalog))
        )
    selected: Dict[str, str] = {}
    for i in range(count):
        name = names[i % len(names)]
        selected[f"{name}_{i + 1}"] = catalog[name]
    return selected


def post_completion(base_url: str,
                    name: str,
                    prompt: str,
                    max_tokens: int,
                    ignore_eos: bool,
                    timeout_s: float) -> BenchRequest:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
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


def parse_summary_metrics(log: str) -> List[SummaryMetrics]:
    summaries: List[SummaryMetrics] = []
    for m in SUMMARY_RE.finditer(log):
        summaries.append(
            SummaryMetrics(
                prompt_tokens=int(m.group("prompt")),
                prefill_s=float(m.group("prefill")),
                decoded_tokens=int(m.group("decoded")),
                decode_s=float(m.group("decode")),
                max_batch=int(m.group("max_batch")),
            )
        )
    for m in GENERATE_RE.finditer(log):
        summaries.append(
            SummaryMetrics(
                prompt_tokens=int(m.group("prompt")),
                prefill_s=float(m.group("prefill")),
                decoded_tokens=int(m.group("decoded")),
                decode_s=float(m.group("decode")),
                max_batch=1,
            )
        )
    return summaries


def parse_timing_metrics(log: str) -> List[TimingMetrics]:
    timings: List[TimingMetrics] = []
    for m in TIMING_RE.finditer(log):
        line_end = log.find("\n", m.start())
        line = log[m.start(): line_end if line_end >= 0 else len(log)]
        fields = {fm.group("key"): float(fm.group("value"))
                  for fm in TIMING_FIELD_RE.finditer(line)}
        timings.append(
            TimingMetrics(
                mode=m.group("mode"),
                batch_tokens=int(m.group("batch")),
                kernel_batch=int(m.group("kernel_batch")),
                total_s=float(m.group("total")),
                prepare_s=fields.get("prepare", 0.0),
                metadata_s=fields.get("metadata", 0.0),
                embed_s=fields.get("embed", 0.0),
                layers_s=fields.get("layers", 0.0),
                recurrent_s=fields.get("recurrent", 0.0),
                recurrent_state_s=fields.get("recurrent_state", 0.0),
                attention_s=fields.get("attention", 0.0),
                qkv_s=fields.get("qkv", 0.0),
                kv_append_s=fields.get("kv_append", 0.0),
                attn_kernel_s=fields.get("attn_kernel", 0.0),
                attn_output_s=fields.get("attn_output", 0.0),
                ffn_s=fields.get("ffn", 0.0),
                final_s=fields.get("final", 0.0),
                lm_head_s=fields.get("lm_head", 0.0),
                argmax_s=fields.get("argmax", 0.0),
                post_s=fields.get("post", 0.0),
            )
        )
    return timings


def sum_timing(timings: Sequence[TimingMetrics], field: str) -> float:
    return sum(float(getattr(t, field)) for t in timings)


def take_timing_slice(timings: Sequence[TimingMetrics],
                      start: int,
                      target_tokens: int) -> tuple[List[TimingMetrics], int]:
    if target_tokens <= 0:
        return [], start
    selected: List[TimingMetrics] = []
    cursor = start
    tokens = 0
    while cursor < len(timings) and tokens < target_tokens:
        t = timings[cursor]
        selected.append(t)
        tokens += t.batch_tokens
        cursor += 1
    return selected, cursor


def make_command(binary: Path,
                 model: Path,
                 host: str,
                 port: int,
                 max_tokens: int,
                 ctx_size: int,
                 prefill_chunk: Optional[int],
                 extra_args: Sequence[str]) -> List[str]:
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
    return cmd


def make_env(variant: BenchVariant,
             max_active: int,
             trace: bool,
             timing: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QW3_MATMUL", "mmq")
    env.setdefault("QW3_DISABLE_HGEMM", "1")
    if variant.continuous:
        env["QW3_CONTINUOUS_BATCHING"] = "1"
        if trace:
            env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
        else:
            env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
        if timing:
            env["QW3_CONTINUOUS_BATCHING_TIMING"] = "1"
        else:
            env.pop("QW3_CONTINUOUS_BATCHING_TIMING", None)
        env["QW3_CONTINUOUS_BATCHING_MAX_ACTIVE"] = str(max_active)
        env.update(variant.env)
    else:
        env.pop("QW3_CONTINUOUS_BATCHING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TRACE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_TIMING", None)
        env.pop("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", None)
        env.pop("QW3_CONTINUOUS_BATCHING_BODY_BATCH", None)
        env.pop("QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH", None)
    return env


def send_requests(base_url: str,
                  prompts: Dict[str, str],
                  max_tokens: int,
                  ignore_eos: bool,
                  timeout_s: int,
                  concurrent: bool) -> tuple[float, List[BenchRequest]]:
    request_start = time.monotonic()
    requests: List[BenchRequest] = []
    if concurrent:
        with futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futs = [
                pool.submit(
                    post_completion,
                    base_url,
                    name,
                    prompt,
                    max_tokens,
                    ignore_eos,
                    float(timeout_s),
                )
                for name, prompt in prompts.items()
            ]
            for fut in futs:
                requests.append(fut.result())
    else:
        for name, prompt in prompts.items():
            requests.append(
                post_completion(
                    base_url, name, prompt, max_tokens, ignore_eos,
                    float(timeout_s))
            )
    return time.monotonic() - request_start, requests


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
                trace: bool,
                timing: bool,
                ignore_eos: bool,
                extra_args: Sequence[str]) -> BenchRun:
    env = make_env(variant, max_active, trace, timing)
    cmd = make_command(
        binary, model, host, port, max_tokens, ctx_size, prefill_chunk, extra_args
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks, log_threads = start_log_drain(proc)
    start = time.monotonic()
    log = ""
    requests: List[BenchRequest] = []
    base_url = f"http://{host}:{port}"
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        request_wall_s, requests = send_requests(
            base_url, prompts, max_tokens, ignore_eos, timeout_s,
            variant.concurrent
        )
    finally:
        log = terminate_drained_server(proc, log_chunks, log_threads)

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
        *_prefill_metrics,
    ) = cbr.parse_server_log(log)
    latencies = [r.elapsed_s for r in requests if r.ok]
    output_tokens = sum(r.completion_tokens for r in requests if r.ok)
    tokens_per_s = output_tokens / request_wall_s if request_wall_s > 0 else 0.0
    prefill_tokens = 0
    prefill_s = 0.0
    decode_tokens = 0
    decode_s = 0.0
    summary_max_batch = max_summary_batch
    for summary in parse_summary_metrics(log):
        prefill_tokens += summary.prompt_tokens
        prefill_s += summary.prefill_s
        decode_tokens += summary.decoded_tokens
        decode_s += summary.decode_s
        summary_max_batch = max(summary_max_batch, summary.max_batch)
    timing_metrics = parse_timing_metrics(log)
    decode_step_tokens = sum(t.batch_tokens for t in timing_metrics)
    decode_step_s = sum(t.total_s for t in timing_metrics)
    saw_body_from_timing = any(t.mode == "body_batch_fp16" for t in timing_metrics)
    max_timing_kernel_batch = max(
        (t.kernel_batch for t in timing_metrics),
        default=0,
    )
    return BenchRun(
        variant=variant.name,
        ctx_size=ctx_size,
        input_target_tokens=0,
        concurrency=len(prompts),
        command=cmd,
        elapsed_s=elapsed_s,
        request_wall_s=request_wall_s,
        requests=requests,
        output_tokens=output_tokens,
        tokens_per_s=tokens_per_s,
        prefill_tokens=prefill_tokens,
        prefill_s=prefill_s,
        prefill_tokens_per_s=(
            prefill_tokens / prefill_s if prefill_s > 0.0 else 0.0
        ),
        decode_tokens=decode_tokens,
        decode_s=decode_s,
        decode_tokens_per_s=decode_tokens / decode_s if decode_s > 0.0 else 0.0,
        decode_step_tokens=decode_step_tokens,
        decode_step_s=decode_step_s,
        decode_step_tokens_per_s=(
            decode_step_tokens / decode_step_s if decode_step_s > 0.0 else 0.0
        ),
        decode_prepare_s=sum_timing(timing_metrics, "prepare_s"),
        decode_metadata_s=sum_timing(timing_metrics, "metadata_s"),
        decode_embed_s=sum_timing(timing_metrics, "embed_s"),
        decode_layers_s=sum_timing(timing_metrics, "layers_s"),
        decode_recurrent_s=sum_timing(timing_metrics, "recurrent_s"),
        decode_recurrent_state_s=sum_timing(timing_metrics, "recurrent_state_s"),
        decode_attention_s=sum_timing(timing_metrics, "attention_s"),
        decode_qkv_s=sum_timing(timing_metrics, "qkv_s"),
        decode_kv_append_s=sum_timing(timing_metrics, "kv_append_s"),
        decode_attn_kernel_s=sum_timing(timing_metrics, "attn_kernel_s"),
        decode_attn_output_s=sum_timing(timing_metrics, "attn_output_s"),
        decode_ffn_s=sum_timing(timing_metrics, "ffn_s"),
        decode_final_s=sum_timing(timing_metrics, "final_s"),
        decode_lm_head_s=sum_timing(timing_metrics, "lm_head_s"),
        decode_argmax_s=sum_timing(timing_metrics, "argmax_s"),
        decode_post_s=sum_timing(timing_metrics, "post_s"),
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        p50_latency_s=percentile(latencies, 0.50),
        p90_latency_s=percentile(latencies, 0.90),
        max_trace_batch=max(max_trace_batch, max_timing_kernel_batch),
        max_summary_batch=summary_max_batch,
        saw_body_batch_ready=saw_body or saw_body_from_timing,
        saw_body_batch_mode=saw_body_mode or saw_body_from_timing,
        saw_ragged_metadata_ready=saw_ragged or saw_body_from_timing,
        log=log,
    )


def build_run_from_requests(*,
                            variant: BenchVariant,
                            ctx_size: int,
                            input_target_tokens: int,
                            command: List[str],
                            request_wall_s: float,
                            requests: List[BenchRequest],
                            summaries: Sequence[SummaryMetrics],
                            timings: Sequence[TimingMetrics],
                            full_log: str) -> BenchRun:
    latencies = [r.elapsed_s for r in requests if r.ok]
    output_tokens = sum(r.completion_tokens for r in requests if r.ok)
    tokens_per_s = output_tokens / request_wall_s if request_wall_s > 0 else 0.0
    prefill_tokens = sum(s.prompt_tokens for s in summaries)
    prefill_s = sum(s.prefill_s for s in summaries)
    decode_tokens = sum(s.decoded_tokens for s in summaries)
    decode_s = sum(s.decode_s for s in summaries)
    decode_step_tokens = sum(t.batch_tokens for t in timings)
    decode_step_s = sum(t.total_s for t in timings)
    max_summary_batch = max((s.max_batch for s in summaries), default=0)
    saw_body_from_timing = any(t.mode == "body_batch_fp16" for t in timings)
    max_timing_kernel_batch = max(
        (t.kernel_batch for t in timings),
        default=0,
    )
    return BenchRun(
        variant=variant.name,
        ctx_size=ctx_size,
        input_target_tokens=input_target_tokens,
        concurrency=len(requests),
        command=command,
        elapsed_s=0.0,
        request_wall_s=request_wall_s,
        requests=requests,
        output_tokens=output_tokens,
        tokens_per_s=tokens_per_s,
        prefill_tokens=prefill_tokens,
        prefill_s=prefill_s,
        prefill_tokens_per_s=(
            prefill_tokens / prefill_s if prefill_s > 0.0 else 0.0
        ),
        decode_tokens=decode_tokens,
        decode_s=decode_s,
        decode_tokens_per_s=decode_tokens / decode_s if decode_s > 0.0 else 0.0,
        decode_step_tokens=decode_step_tokens,
        decode_step_s=decode_step_s,
        decode_step_tokens_per_s=(
            decode_step_tokens / decode_step_s if decode_step_s > 0.0 else 0.0
        ),
        decode_prepare_s=sum_timing(timings, "prepare_s"),
        decode_metadata_s=sum_timing(timings, "metadata_s"),
        decode_embed_s=sum_timing(timings, "embed_s"),
        decode_layers_s=sum_timing(timings, "layers_s"),
        decode_recurrent_s=sum_timing(timings, "recurrent_s"),
        decode_recurrent_state_s=sum_timing(timings, "recurrent_state_s"),
        decode_attention_s=sum_timing(timings, "attention_s"),
        decode_qkv_s=sum_timing(timings, "qkv_s"),
        decode_kv_append_s=sum_timing(timings, "kv_append_s"),
        decode_attn_kernel_s=sum_timing(timings, "attn_kernel_s"),
        decode_attn_output_s=sum_timing(timings, "attn_output_s"),
        decode_ffn_s=sum_timing(timings, "ffn_s"),
        decode_final_s=sum_timing(timings, "final_s"),
        decode_lm_head_s=sum_timing(timings, "lm_head_s"),
        decode_argmax_s=sum_timing(timings, "argmax_s"),
        decode_post_s=sum_timing(timings, "post_s"),
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        p50_latency_s=percentile(latencies, 0.50),
        p90_latency_s=percentile(latencies, 0.90),
        max_trace_batch=max_timing_kernel_batch,
        max_summary_batch=max_summary_batch,
        saw_body_batch_ready=saw_body_from_timing,
        saw_body_batch_mode=saw_body_from_timing,
        saw_ragged_metadata_ready=saw_body_from_timing,
        log=full_log,
    )


def run_reused_variant_matrix(*,
                              variant: BenchVariant,
                              binary: Path,
                              model: Path,
                              host: str,
                              port: int,
                              ctx_size: int,
                              input_targets: Sequence[int],
                              concurrency_levels: Sequence[int],
                              max_tokens: int,
                              prefill_chunk: Optional[int],
                              timeout_s: int,
                              max_active: int,
                              trace: bool,
                              timing: bool,
                              ignore_eos: bool,
                              extra_args: Sequence[str]) -> List[BenchRun]:
    max_concurrency = max(concurrency_levels)
    env = make_env(variant, max(max_active, max_concurrency), trace, timing)
    cmd = make_command(
        binary, model, host, port, max_tokens, ctx_size, prefill_chunk, extra_args
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks, log_threads = start_log_drain(proc)
    base_url = f"http://{host}:{port}"
    pending: List[tuple[int, float, List[BenchRequest]]] = []
    start = time.monotonic()
    log = ""
    try:
        cbr.wait_for_health(base_url, timeout_s=min(120.0, float(timeout_s)))
        for input_target in input_targets:
            for concurrency in concurrency_levels:
                prompts = synthetic_prompts_for_target(concurrency, input_target)
                wall_s, requests = send_requests(
                    base_url, prompts, max_tokens, ignore_eos, timeout_s,
                    variant.concurrent
                )
                pending.append((input_target, wall_s, requests))
                print(
                    f"completed requests variant={variant.name} ctx={ctx_size} "
                    f"input_target={input_target} concurrency={concurrency} "
                    f"wall={wall_s:.3f}s",
                    flush=True,
                )
    finally:
        log = terminate_drained_server(proc, log_chunks, log_threads)
    elapsed_s = time.monotonic() - start
    summaries = parse_summary_metrics(log)
    timing_metrics = parse_timing_metrics(log)
    runs: List[BenchRun] = []
    cursor = 0
    timing_cursor = 0
    for input_target, wall_s, requests in pending:
        next_cursor = cursor + len(requests)
        run_summaries = summaries[cursor:next_cursor]
        cursor = next_cursor
        target_decode_tokens = sum(s.decoded_tokens for s in run_summaries)
        run_timings, timing_cursor = take_timing_slice(
            timing_metrics, timing_cursor, target_decode_tokens
        )
        run = build_run_from_requests(
            variant=variant,
            ctx_size=ctx_size,
            input_target_tokens=input_target,
            command=cmd,
            request_wall_s=wall_s,
            requests=requests,
            summaries=run_summaries,
            timings=run_timings,
            full_log=log,
        )
        run.elapsed_s = elapsed_s
        runs.append(run)
    return runs


def variants_for(names: Sequence[str]) -> List[BenchVariant]:
    all_variants = {
        "plain": BenchVariant("plain", continuous=False, concurrent=False, env={}),
        "continuous": BenchVariant(
            "continuous",
            continuous=True,
            concurrent=True,
            env={"QW3_CONTINUOUS_BATCHING_BODY_BATCH": "0"},
        ),
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
    ap.add_argument(
        "--ctx-sizes",
        type=parse_int_list,
        default=None,
        help="space/comma separated context sizes; overrides --ctx",
    )
    ap.add_argument(
        "--concurrency-levels",
        type=parse_int_list,
        default=None,
        help="space/comma separated concurrent request counts",
    )
    ap.add_argument(
        "--synthetic-repeat",
        type=int,
        default=0,
        help="repeat a deterministic text unit this many times per request",
    )
    ap.add_argument(
        "--input-token-targets",
        type=parse_int_list,
        default=None,
        help="space/comma separated synthetic input token targets",
    )
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-active", type=int, default=2)
    ap.add_argument(
        "--trace",
        action="store_true",
        help="enable verbose continuous batching trace logs",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="enable continuous batching timing logs and decode-step throughput",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="request benchmark generations to continue until max_tokens",
    )
    ap.add_argument(
        "--reuse-server",
        action="store_true",
        help="load one server per variant/ctx and run all input/concurrency cases",
    )
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

    port = args.port if args.port else cbr.free_port()
    ctx_sizes = args.ctx_sizes if args.ctx_sizes else [args.ctx]
    concurrency_levels = (
        args.concurrency_levels if args.concurrency_levels else [len(args.prompts)]
    )
    input_targets = args.input_token_targets if args.input_token_targets else [0]

    runs: List[BenchRun] = []
    for ctx_size in ctx_sizes:
        for variant in variants_for(args.variants):
            if args.reuse_server and args.input_token_targets:
                matrix_runs = run_reused_variant_matrix(
                    variant=variant,
                    binary=binary,
                    model=model,
                    host=args.host,
                    port=port,
                    ctx_size=ctx_size,
                    input_targets=input_targets,
                    concurrency_levels=concurrency_levels,
                    max_tokens=args.max_tokens,
                    prefill_chunk=args.prefill_chunk,
                    timeout_s=args.timeout,
                    max_active=args.max_active,
                    trace=args.trace,
                    timing=args.timing,
                    ignore_eos=args.ignore_eos,
                    extra_args=args.extra_arg,
                )
                runs.extend(matrix_runs)
                for run in matrix_runs:
                    failed = [r for r in run.requests if not r.ok]
                    status = "FAIL" if failed else "ok"
                    print(
                        f"{status:4s} {run.variant:10s} "
                        f"ctx={run.ctx_size:6d} "
                        f"input={run.input_target_tokens:6d} "
                        f"conc={run.concurrency:2d} "
                        f"wall={run.request_wall_s:.3f}s "
                        f"out_tok/s={run.tokens_per_s:.2f} "
                        f"prefill_tok/s={run.prefill_tokens_per_s:.2f} "
                        f"decode_tok/s={run.decode_tokens_per_s:.2f} "
                        f"decode_step_tok/s={run.decode_step_tokens_per_s:.2f} "
                        f"layers={run.decode_layers_s:.3f}s "
                        f"recurrent={run.decode_recurrent_s:.3f}s "
                        f"attention={run.decode_attention_s:.3f}s "
                        f"ffn={run.decode_ffn_s:.3f}s "
                        f"final={run.decode_final_s:.3f}s "
                        f"prefill_tokens={run.prefill_tokens} "
                        f"decode_tokens={run.decode_tokens} "
                        f"batch={max(run.max_trace_batch, run.max_summary_batch)}"
                    )
                continue
            for concurrency in concurrency_levels:
                prompts = (
                    synthetic_prompts(concurrency, args.synthetic_repeat)
                    if args.synthetic_repeat > 0
                    else catalog_prompts(args.prompts, concurrency)
                )
                max_active = max(args.max_active, concurrency)
                print(
                    f"running benchmark variant={variant.name} "
                    f"ctx={ctx_size} concurrency={concurrency}"
                )
                run = run_variant(
                    variant=variant,
                    binary=binary,
                    model=model,
                    host=args.host,
                    port=port,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                    ctx_size=ctx_size,
                    prefill_chunk=args.prefill_chunk,
                    timeout_s=args.timeout,
                    max_active=max_active,
                    trace=args.trace,
                    timing=args.timing,
                    ignore_eos=args.ignore_eos,
                    extra_args=args.extra_arg,
                )
                runs.append(run)
                failed = [r for r in run.requests if not r.ok]
                status = "FAIL" if failed else "ok"
                print(
                    f"{status:4s} {run.variant:10s} "
                    f"ctx={run.ctx_size:6d} conc={run.concurrency:2d} "
                    f"wall={run.request_wall_s:.3f}s "
                    f"out_tok/s={run.tokens_per_s:.2f} "
                    f"prefill_tok/s={run.prefill_tokens_per_s:.2f} "
                    f"decode_tok/s={run.decode_tokens_per_s:.2f} "
                    f"decode_step_tok/s={run.decode_step_tokens_per_s:.2f} "
                    f"layers={run.decode_layers_s:.3f}s "
                    f"recurrent={run.decode_recurrent_s:.3f}s "
                    f"attention={run.decode_attention_s:.3f}s "
                    f"ffn={run.decode_ffn_s:.3f}s "
                    f"final={run.decode_final_s:.3f}s "
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
            "prompts": (
                []
                if args.input_token_targets
                else list(catalog_prompts(args.prompts, max(concurrency_levels)).keys())
            ),
            "max_tokens": args.max_tokens,
            "ctx": args.ctx,
            "ctx_sizes": ctx_sizes,
            "concurrency_levels": concurrency_levels,
            "synthetic_repeat": args.synthetic_repeat,
            "prefill_chunk": args.prefill_chunk,
            "max_active": args.max_active,
            "timing": args.timing,
            "ignore_eos": args.ignore_eos,
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
