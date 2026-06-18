#!/usr/bin/env python3
"""Phase 1 prefix-cache LATENCY benchmark — multi-turn append scenario.

Measures end-to-end latency saved by prefix caching when the prefix is
byte-identical and only a NEW question is appended at the end (the strict
token-prefix / multi-turn-append case v1 supports).

For each target shared-prefix length L in {16k, 32k, 64k} tokens:

  WARM (QW3_PREFIX_CACHE=1), one server:
    turn1 = <context_L> + Q1            -> cold prefill, COMMITS aligned prefix
    turn2 = <context_L> + Q1 + A1 + Q2  -> strict EXTENSION -> HITS, prefills
                                           only the small appended tail
  COLD (QW3_PREFIX_CACHE=0), one server:
    turn2 (same prompt)                 -> full cold prefill of all len2 tokens

Reported per L (apples-to-apples, identical turn2 prompt + identical KV length):
  * prefill_saved = cold.prefill_s - warm.prefill_s   (server-trace authoritative)
  * e2e_saved     = cold.e2e_s     - warm.e2e_s       (client wall-clock)
  * reused_tokens (from the warm hit trace) and reused fraction

Server trace parsed:
  native continuous_batch: request=.. prompt_tokens=.. prefill=X.XXXs ...
  prefix_cache hit id=.. req=.. reused_tokens=.. pages=..
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple


def wait_for_server(base_url: str, timeout_s: float) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                              timeout=2.0)
            conn.request("GET", "/v1/models")
            res = conn.getresponse()
            res.read()
            conn.close()
            if res.status < 500:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def post_completion(base_url: str, prompt: str, max_tokens: int,
                    timeout_s: float) -> Tuple[int, str, float, str]:
    """Returns (status, text, e2e_seconds, err)."""
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                      timeout=timeout_s)
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/completions", body=body, headers=headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8", errors="replace")
        e2e = time.monotonic() - t0
        if res.status != 200:
            return res.status, "", e2e, raw
        data = json.loads(raw)
        return 200, data["choices"][0]["text"], e2e, ""
    except Exception as exc:  # noqa: BLE001
        return 0, "", time.monotonic() - t0, str(exc)
    finally:
        conn.close()


def start_server(binary: str, model: str, port: int, ctx: int,
                 env_extra: Dict[str, str], extra_args: List[str],
                 log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = ([binary, "serve", "--model", model, "--port", str(port),
            "--continuous-batching", "--max-active", "2", "--ctx", str(ctx)]
           + extra_args)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, text=True)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def read_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def build_context(target_tokens: int) -> str:
    """A long, varied document sized to ~target_tokens (~1.4 tok/word for this
    filler). Numbered lines avoid a degenerate pure-repeat prompt."""
    base = [
        "The subsystem coordinator records each transaction with a monotonic "
        "sequence number and a wall-clock stamp for later reconciliation.",
        "Memory bandwidth, cache locality, and kernel launch overhead jointly "
        "determine throughput long before microarchitectural details matter.",
        "When the scheduler admits a request it pins the relevant pages and "
        "increments the reference count to keep them off the free stack.",
        "Recurrent layers fold their state forward while attention layers scan "
        "the full key and value history accumulated up to the current step.",
        "Eviction selects the least recently used entry whose reference count "
        "has fallen to zero, returning its physical pages to the shared pool.",
    ]
    # ~ each line averages ~25 tokens; aim a bit high then the actual count is
    # reported from the server trace anyway.
    approx_tok_per_line = 25
    n_lines = max(8, target_tokens // approx_tok_per_line)
    lines = []
    for i in range(n_lines):
        lines.append(f"[note {i:05d}] " + base[i % len(base)])
    return "Background document for reference:\n" + "\n".join(lines) + "\n\n"


# Parse all per-request prefill traces in order.
RE_CB = re.compile(
    r"native continuous_batch: request=(\d+) prompt_tokens=(\d+) "
    r"prefill=([\d.]+)s")
RE_HIT = re.compile(
    r"prefix_cache hit .*?req=(\d+) reused_tokens=(\d+)")


def parse_prefills(trace: str) -> List[Tuple[int, int, float]]:
    """Returns list of (request_id, prompt_tokens, prefill_s) in log order."""
    out = []
    for m in RE_CB.finditer(trace):
        out.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return out


def parse_hits(trace: str) -> Dict[int, int]:
    """request_id -> reused_tokens."""
    return {int(m.group(1)): int(m.group(2)) for m in RE_HIT.finditer(trace)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/qw3")
    ap.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--port", type=int, default=8139)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("--req-timeout", type=float, default=600.0)
    ap.add_argument("--lengths", default="16000,32000,64000")
    ap.add_argument("--logdir", default="/tmp/prefix_cache_latency")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    # Build the turn1/turn2 prompt pair per target length.
    prompts: Dict[int, Tuple[str, str]] = {}
    for L in lengths:
        ctx = build_context(L)
        q1 = "Question: Summarize the key invariant described above.\nAnswer:"
        a1 = (" The coordinator keeps a monotonic sequence and pins pages by "
              "refcount.")
        q2 = "\n\nQuestion: Now restate it in a single short sentence.\nAnswer:"
        turn1 = ctx + q1
        turn2 = ctx + q1 + a1 + q2  # strict text extension of turn1
        prompts[L] = (turn1, turn2)

    failures: List[str] = []

    # ---- WARM run (cache ON) --------------------------------------------
    warm_log = os.path.join(args.logdir, "warm.log")
    proc = start_server(args.binary, args.model, args.port, args.ctx,
                        {"QW3_PREFIX_CACHE_TRACE": "1",
                         "QW3_CONTINUOUS_BATCHING_TRACE": "1"},
                        ["--prefix-cache"], warm_log)
    # request log order -> (L, turn) so we can map prefill traces back.
    warm_order: List[Tuple[int, int]] = []
    warm_e2e: Dict[Tuple[int, int], float] = {}
    try:
        if not wait_for_server(base_url, args.boot_timeout):
            print("FAIL: warm server did not become ready")
            print(read_text(warm_log)[-2000:])
            stop_server(proc)
            return 1
        for L in lengths:
            turn1, turn2 = prompts[L]
            s, _t, e2e, err = post_completion(base_url, turn1, args.max_tokens,
                                              args.req_timeout)
            if s != 200:
                failures.append(f"warm L={L} turn1 failed: {s} {err}")
            warm_order.append((L, 1))
            warm_e2e[(L, 1)] = e2e
            s, _t, e2e, err = post_completion(base_url, turn2, args.max_tokens,
                                              args.req_timeout)
            if s != 200:
                failures.append(f"warm L={L} turn2 failed: {s} {err}")
            warm_order.append((L, 2))
            warm_e2e[(L, 2)] = e2e
    finally:
        time.sleep(0.5)
        stop_server(proc)

    # ---- COLD run (cache OFF) -------------------------------------------
    cold_log = os.path.join(args.logdir, "cold.log")
    proc = start_server(args.binary, args.model, args.port, args.ctx,
                        {"QW3_CONTINUOUS_BATCHING_TRACE": "1"},
                        [], cold_log)
    cold_order: List[Tuple[int, int]] = []
    cold_e2e: Dict[Tuple[int, int], float] = {}
    try:
        if not wait_for_server(base_url, args.boot_timeout):
            print("FAIL: cold server did not become ready")
            print(read_text(cold_log)[-2000:])
            stop_server(proc)
            return 1
        for L in lengths:
            _turn1, turn2 = prompts[L]
            s, _t, e2e, err = post_completion(base_url, turn2, args.max_tokens,
                                              args.req_timeout)
            if s != 200:
                failures.append(f"cold L={L} turn2 failed: {s} {err}")
            cold_order.append((L, 2))
            cold_e2e[(L, 2)] = e2e
    finally:
        time.sleep(0.5)
        stop_server(proc)

    # ---- Correlate traces ------------------------------------------------
    warm_trace = read_text(warm_log)
    cold_trace = read_text(cold_log)
    warm_pf = parse_prefills(warm_trace)   # in order: [t1_L0, t2_L0, t1_L1, ...]
    cold_pf = parse_prefills(cold_trace)   # in order: [t2_L0, t2_L1, t2_L2]
    warm_hits = parse_hits(warm_trace)

    # Map warm prefill traces to (L, turn) by log order.
    warm_pf_by_key: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    for key, rec in zip(warm_order, warm_pf):
        warm_pf_by_key[key] = rec
    cold_pf_by_key: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    for key, rec in zip(cold_order, cold_pf):
        cold_pf_by_key[key] = rec

    print("=== prefix cache latency benchmark (multi-turn append) ===")
    print(f"max_tokens={args.max_tokens} ctx={args.ctx}")
    header = (f"{'target_L':>9} {'prompt_tok':>10} {'reused':>8} {'reuse%':>7} "
              f"{'cold_pf':>9} {'warm_pf':>9} {'pf_saved':>9} {'pf_x':>6} "
              f"{'cold_e2e':>9} {'warm_e2e':>9} {'e2e_saved':>10} {'e2e%':>6}")
    print(header)
    print("-" * len(header))
    for L in lengths:
        w2 = warm_pf_by_key.get((L, 2))
        c2 = cold_pf_by_key.get((L, 2))
        if not w2 or not c2:
            print(f"{L:>9}  (missing trace: warm={bool(w2)} cold={bool(c2)})")
            failures.append(f"L={L}: missing prefill trace")
            continue
        w_rid, w_ptok, w_pf = w2
        c_rid, c_ptok, c_pf = c2
        reused = warm_hits.get(w_rid, 0)
        if reused == 0:
            failures.append(f"L={L}: warm turn2 took NO prefix hit "
                            f"(reused_tokens=0) -- extension did not match")
        reuse_pct = 100.0 * reused / w_ptok if w_ptok else 0.0
        pf_saved = c_pf - w_pf
        pf_x = (c_pf / w_pf) if w_pf > 1e-9 else float("inf")
        we = warm_e2e.get((L, 2), float("nan"))
        ce = cold_e2e.get((L, 2), float("nan"))
        e2e_saved = ce - we
        e2e_pct = 100.0 * e2e_saved / ce if ce else 0.0
        print(f"{L:>9} {w_ptok:>10} {reused:>8} {reuse_pct:>6.1f}% "
              f"{c_pf:>8.3f}s {w_pf:>8.3f}s {pf_saved:>8.3f}s {pf_x:>5.1f}x "
              f"{ce:>8.3f}s {we:>8.3f}s {e2e_saved:>9.3f}s {e2e_pct:>5.1f}%")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: every warm turn2 hit the cached prefix; see savings above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
