#!/usr/bin/env python3
"""Regression for delayed admission in continuous MTP serving.

The test starts one long MTP continuous request, waits briefly, then sends a
short request. Success requires the short request to complete before the long
request finishes, proving the MTP scheduler admits pending requests while an
active request is already decoding.
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
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


ADMIT_RE = re.compile(r"native continuous_mtp_admit_pending: admitted=(?P<n>\d+)")
CHAT_DONE_RE = re.compile(
    r"\[qw3-serve\] #(?P<rid>\d+) chat\(stream.*?\).*?route=continuous"
)


@dataclass
class StreamResult:
    name: str
    status: int
    first_byte_s: float
    elapsed_s: float
    chunks: int
    raw_size: int
    error: Optional[str] = None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get(url: str, timeout_s: float) -> tuple[int, str]:
    parsed = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        return res.status, res.read().decode("utf-8", errors="replace")
    finally:
        conn.close()


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, _ = http_get(urllib.parse.urljoin(base_url + "/", "health"), 2.0)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def drain(pipe, chunks: list[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            chunks.append(line)
    finally:
        pipe.close()


def terminate_server(proc: subprocess.Popen[str],
                     chunks: list[str],
                     threads: list[threading.Thread]) -> str:
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


def post_stream_chat(base_url: str,
                     name: str,
                     prompt: str,
                     max_tokens: int,
                     timeout_s: float,
                     ignore_eos: bool = False) -> StreamResult:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    body = json.dumps({
        "model": "qw3",
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
    }).encode("utf-8")
    start = time.monotonic()
    first_byte_s = -1.0
    chunks = 0
    raw_size = 0
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        if res.status != 200:
            raw = res.read().decode("utf-8", errors="replace")
            return StreamResult(
                name, res.status, first_byte_s, time.monotonic() - start,
                chunks, len(raw), raw,
            )
        while True:
            data = res.readline()
            if not data:
                break
            if first_byte_s < 0:
                first_byte_s = time.monotonic() - start
            raw_size += len(data)
            if data.startswith(b"data: "):
                chunks += 1
            if data.strip() == b"data: [DONE]":
                break
        return StreamResult(
            name, res.status, first_byte_s, time.monotonic() - start,
            chunks, raw_size, None,
        )
    except Exception as exc:  # noqa: BLE001
        return StreamResult(
            name, 0, first_byte_s, time.monotonic() - start, chunks, raw_size,
            str(exc),
        )
    finally:
        conn.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--kv-dtype", default="fp8", choices=("fp16", "fp8", "q8", "fp32"))
    ap.add_argument("--long-tokens", type=int, default=256)
    ap.add_argument("--short-tokens", type=int, default=16)
    ap.add_argument("--delay", type=float, default=0.75)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-json", default="/tmp/qw3_mtp_delayed_admission.json")
    args = ap.parse_args(argv)

    binary = Path(args.qw3)
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"qw3 binary not found: {binary}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    port = args.port if args.port else free_port()
    base_url = f"http://{args.host}:{port}"

    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    env.setdefault("QW3_MATMUL", "mmq")
    env.setdefault("QW3_DISABLE_HGEMM", "1")
    cmd = [
        str(binary),
        "serve",
        "--model", str(model),
        "--host", args.host,
        "--port", str(port),
        "--ctx", str(args.ctx),
        "--prefill-chunk", str(args.prefill_chunk),
        "--continuous-batching",
        "--max-active", "2",
        "--kv-dtype", args.kv_dtype,
        "--mtp-chain", str(args.chain),
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_chunks: list[str] = []
    log_threads = [
        threading.Thread(target=drain, args=(proc.stdout, log_chunks), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, log_chunks), daemon=True),
    ]
    for t in log_threads:
        t.start()

    long_prompt = (
        "请持续写一个末日生存小说片段，保持中文叙事，包含环境、行动和对话。"
        "不要提前结束，继续扩展细节。"
    )
    short_prompt = "用一句话回答：2+2等于几？"
    results: list[StreamResult] = []
    short_wall_elapsed = -1.0
    log = ""
    try:
        wait_for_health(base_url, min(float(args.timeout), 120.0))
        print(
            "running delayed MTP admission regression: "
            f"port={port} delay={args.delay}s"
        )
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            long_fut = pool.submit(
                post_stream_chat, base_url, "long", long_prompt,
                args.long_tokens, float(args.timeout), True,
            )
            time.sleep(args.delay)
            short_submit_t = time.monotonic()
            short_fut = pool.submit(
                post_stream_chat, base_url, "short", short_prompt,
                args.short_tokens, float(args.timeout), False,
            )
            long = long_fut.result()
            short = short_fut.result()
            short_wall_elapsed = time.monotonic() - short_submit_t
            results = [long, short]
            time.sleep(0.25)
    finally:
        log = terminate_server(proc, log_chunks, log_threads)

    admitted = sum(int(m.group("n")) for m in ADMIT_RE.finditer(log))
    chat_done = list(CHAT_DONE_RE.finditer(log))
    failures: list[str] = []
    for result in results:
        ok = result.status == 200 and result.error is None and result.chunks > 0
        print(
            f"{'ok' if ok else 'FAIL':4s} {result.name:5s} "
            f"status={result.status:3d} first_byte={result.first_byte_s:.3f}s "
            f"elapsed={result.elapsed_s:.3f}s chunks={result.chunks}"
        )
        if not ok:
            failures.append(f"{result.name} request failed: {result.error}")
    if len(results) == 2:
        long, short = results
        if args.delay + short_wall_elapsed >= long.elapsed_s:
            failures.append(
                "short delayed request did not finish before the long request"
            )
    if admitted <= 0:
        failures.append("did not observe continuous_mtp_admit_pending trace")

    summary = {
        "config": vars(args),
        "command": cmd,
        "results": [asdict(r) for r in results],
        "short_wall_elapsed": short_wall_elapsed,
        "admitted": admitted,
        "chat_done_count": len(chat_done),
        "failed": failures,
        "log": log,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"admitted_pending={admitted} chat_done={len(chat_done)}")
    print(f"wrote summary: {out}")
    if failures:
        print("failed delayed admission requirements:")
        for item in failures:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
