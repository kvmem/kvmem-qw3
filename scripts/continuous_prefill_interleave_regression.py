#!/usr/bin/env python3
"""Regression check for continuous-batching prefill/decode interleaving.

The test starts qw3 with continuous batching and trace logs enabled, sends a
long prompt first, then sends a short prompt while the long request is still in
chunked prefill. Success requires the short request to complete before the long
request's final prefill chunk is logged.
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
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


PREFILL_RE = re.compile(
    r"native continuous_prefill_chunk: request=(?P<request>\d+) "
    r"offset=(?P<offset>\d+) total=(?P<total>\d+) "
    r"chunk=(?P<chunk>\d+) final=(?P<final>true|false)"
)
SUMMARY_RE = re.compile(r"native continuous_batch: request=(?P<request>\d+)\b")
DECODE_RE = re.compile(r"native continuous_batch_step:.*?batch=(?P<batch>\d+)")


@dataclass
class RequestResult:
    name: str
    status: int
    elapsed_s: float
    text: str
    raw: str
    error: Optional[str] = None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def post_completion(base_url: str,
                    name: str,
                    prompt: str,
                    max_tokens: int,
                    timeout_s: float) -> RequestResult:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    start = time.monotonic()
    try:
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
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
        except Exception as exc:  # noqa: BLE001
            error = f"invalid JSON response: {exc}"
        return RequestResult(name, res.status, elapsed, text, raw, error)
    except Exception as exc:  # noqa: BLE001
        return RequestResult(name, 0, time.monotonic() - start, "", "", str(exc))
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


def analyze_log(log: str) -> dict:
    prefill = []
    for m in PREFILL_RE.finditer(log):
        prefill.append({
            "request": int(m.group("request")),
            "offset": int(m.group("offset")),
            "total": int(m.group("total")),
            "chunk": int(m.group("chunk")),
            "final": m.group("final") == "true",
            "pos": m.start(),
        })
    summaries = [
        {"request": int(m.group("request")), "pos": m.start()}
        for m in SUMMARY_RE.finditer(log)
    ]
    decode_steps = [
        {"batch": int(m.group("batch")), "pos": m.start()}
        for m in DECODE_RE.finditer(log)
    ]
    long_nonfinal = [
        item for item in prefill
        if item["request"] == 1 and not item["final"]
    ]
    long_final = [
        item for item in prefill
        if item["request"] == 1 and item["final"]
    ]
    short_final_prefill = [
        item for item in prefill
        if item["request"] == 2 and item["final"]
    ]
    short_decode_before_long_final = bool(
        short_final_prefill and long_final and any(
            step["pos"] > short_final_prefill[0]["pos"] and
            step["pos"] < long_final[-1]["pos"]
            for step in decode_steps
        )
    )
    return {
        "prefill_chunks": prefill,
        "summaries": summaries,
        "decode_steps": decode_steps,
        "long_nonfinal_chunks": len(long_nonfinal),
        "short_decode_before_long_final_prefill": short_decode_before_long_final,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check continuous batching prefill/decode interleaving."
    )
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--delay", type=float, default=0.10)
    ap.add_argument("--long-words", type=int, default=1600)
    ap.add_argument("--out-json", default="/tmp/qw3_continuous_prefill_interleave.json")
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
    env["QW3_CONTINUOUS_BATCHING"] = "1"
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    env["QW3_CONTINUOUS_BATCHING_MAX_ACTIVE"] = "2"
    env.setdefault("QW3_MATMUL", "mmq")

    cmd = [
        str(binary),
        "serve",
        "--model", str(model),
        "--host", args.host,
        "--port", str(port),
        "-n", str(args.max_tokens),
        "--temp", "0",
        "--ctx", str(args.ctx),
        "--prefill-chunk", str(args.prefill_chunk),
        "--continuous-batching",
        "--max-active", "2",
        "--paged-kv",
    ]
    long_prompt = " ".join(["word"] * args.long_words)
    short_prompt = "2 + 2 ="

    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    results = []
    log = ""
    try:
        wait_for_health(base_url, min(float(args.timeout), 120.0))
        print(
            "running prefill interleave regression: "
            f"port={port} long_words={args.long_words} "
            f"prefill_chunk={args.prefill_chunk}"
        )
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            long_fut = pool.submit(
                post_completion, base_url, "long", long_prompt,
                args.max_tokens, float(args.timeout),
            )
            time.sleep(args.delay)
            short_fut = pool.submit(
                post_completion, base_url, "short", short_prompt,
                args.max_tokens, float(args.timeout),
            )
            results = [long_fut.result(), short_fut.result()]
    finally:
        log = terminate_server(proc)

    analysis = analyze_log(log)
    failed = []
    for result in results:
        ok = result.status == 200 and result.error is None
        print(
            f"{'ok' if ok else 'FAIL':4s} {result.name:5s} "
            f"status={result.status:3d} elapsed={result.elapsed_s:.3f}s "
            f"chars={len(result.text):4d}"
        )
        if not ok:
            failed.append(f"{result.name} request failed: {result.error}")
    if analysis["long_nonfinal_chunks"] < 2:
        failed.append("long request did not show at least two non-final prefill chunks")
    if not analysis["short_decode_before_long_final_prefill"]:
        failed.append("short request did not start decode before long final prefill chunk")

    summary = {
        "config": {
            "qw3": str(binary),
            "model": str(model),
            "host": args.host,
            "port": port,
            "ctx": args.ctx,
            "prefill_chunk": args.prefill_chunk,
            "max_tokens": args.max_tokens,
            "delay": args.delay,
            "long_words": args.long_words,
        },
        "results": [asdict(r) for r in results],
        "analysis": analysis,
        "failed": failed,
        "log": log,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "interleave evidence: "
        f"long_nonfinal_chunks={analysis['long_nonfinal_chunks']} "
        f"short_decode_before_long_final_prefill="
        f"{analysis['short_decode_before_long_final_prefill']}"
    )
    print(f"wrote summary: {out_path}")
    if failed:
        print("failed prefill interleave requirements:")
        for item in failed:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
