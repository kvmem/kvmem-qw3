#!/usr/bin/env python3
"""Phase 1 prefix-cache eviction + regression harness.

Two checks, both on the continuous-batching serve path (cache enabled with the
--prefix-cache CLI flag; QW3_PREFIX_CACHE_TRACE env stays on only to parse the
diagnostic log lines):

  1. EVICTION under a tiny global KV pool (--kv-pool-pages). The server hardcodes
     the prefix-cache page budget to unlimited, so reclamation is forced through
     the REAL pool-exhaustion path. Flood the server with many DISTINCT long
     prompts (distinct system preambles -> distinct committed prefixes, no
     cross-prompt reuse in v1) until the pool fills; allocate_physical_page must
     then evict LRU refcount==0 cache entries to make room. Invariants:
       * every request returns 200 (no 429 pool-exhausted),
       * the trace shows `prefix_cache evict` lines,
       * no "global KV page pool exhausted" line (that would mean the evict loop
         failed to reclaim real pages -- the shared-page bug),
       * a refcount>0 (in-flight) entry is never the victim -- implicitly held
         because the flood is sequential, so only finished entries are evicted.

  2. REGRESSION guard: with NO --prefix-cache flag the server forces the cache
     off. Send a prompt twice; both must return 200 and the trace must contain
     NO `prefix_cache` commit/hit/evict lines at all.

Exit code 0 = all invariants held.
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
from typing import Dict, List, Tuple


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
                    timeout_s: float) -> Tuple[int, str, str]:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                      timeout=timeout_s)
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        conn.request("POST", "/v1/completions", body=body, headers=headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8", errors="replace")
        if res.status != 200:
            return res.status, "", raw
        data = json.loads(raw)
        return 200, data["choices"][0]["text"], ""
    except Exception as exc:  # noqa: BLE001
        return 0, "", str(exc)
    finally:
        conn.close()


def start_server(binary: str, model: str, port: int, env_extra: Dict[str, str],
                 extra_args: List[str], log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = ([binary, "serve", "--model", model, "--port", str(port),
            "--continuous-batching", "--max-active", "2"] + extra_args)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, text=True)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def read_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/qw3")
    ap.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--port", type=int, default=8138)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--boot-timeout", type=float, default=240.0)
    ap.add_argument("--req-timeout", type=float, default=180.0)
    ap.add_argument("--pool-pages", type=int, default=48)
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--logdir", default="/tmp/prefix_cache_eviction")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    failures: List[str] = []

    # Distinct long prompts: each has a unique system preamble paragraph, so the
    # floor-aligned committed prefix differs across prompts (no reuse in v1).
    # Each must be long enough (>= 2*page_size tokens) to commit several pages.
    fillers = [
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        "kilo lima mike november oscar papa quebec romeo sierra tango",
        "uniform victor whiskey xray yankee zulu zero one two three",
        "spring summer autumn winter monsoon drought flood blizzard frost",
        "copper silver iron nickel cobalt zinc lead tin gold platinum",
        "mercury venus earth mars jupiter saturn uranus neptune pluto",
        "violin cello viola trumpet trombone clarinet oboe bassoon flute",
        "granite basalt marble slate quartz feldspar mica gypsum shale",
    ]
    n = max(2, min(args.n_distinct, len(fillers)))
    prompts = []
    for i in range(n):
        preamble = (
            f"System note {i}: You are a precise assistant. Context tokens: "
            + (fillers[i] + " ") * 12
            + "End of context.\n\n"
        )
        prompts.append(preamble + f"Question {i}: Summarize the above in one line.\nAnswer:")

    # ---- Check 1: eviction under a tiny KV pool --------------------------
    # The server hardcodes the prefix-cache page budget to unlimited, so we
    # force reclamation through the REAL pool-exhaustion path: a tiny global KV
    # pool (--kv-pool-pages). Flooding distinct long prompts fills the pool;
    # allocate_physical_page must then evict LRU refcount==0 cache entries to
    # make room. This is exactly the path the shared-page eviction bug lived in.
    evic_log = os.path.join(args.logdir, "evict.log")
    proc = start_server(args.binary, args.model, args.port,
                        {"QW3_PREFIX_CACHE_TRACE": "1"},
                        ["--prefix-cache",
                         "--kv-pool-pages", str(args.pool_pages)],
                        evic_log)
    try:
        if not wait_for_server(base_url, args.boot_timeout):
            print("FAIL: eviction server did not become ready")
            print(read_text(evic_log)[-2000:])
            stop_server(proc)
            return 1
        for i, p in enumerate(prompts):
            s, _txt, err = post_completion(base_url, p, args.max_tokens,
                                           args.req_timeout)
            if s != 200:
                failures.append(f"flood req#{i} failed: {s} {err}")
        # Re-send the FIRST prompt: it may have been evicted (LRU) -> cold
        # re-prefill must still succeed (no use-after-free of freed pages).
        s, _txt, err = post_completion(base_url, prompts[0], args.max_tokens,
                                       args.req_timeout)
        if s != 200:
            failures.append(f"post-eviction re-send failed: {s} {err}")
    finally:
        time.sleep(0.5)
        stop_server(proc)

    evic_trace = read_text(evic_log)
    commits = len(re.findall(r"prefix_cache commit ", evic_trace))
    evicts = len(re.findall(r"prefix_cache evict ", evic_trace))
    exhausted = "global KV page pool exhausted" in evic_trace

    if commits < 2:
        failures.append(f"eviction: expected >=2 commits to fill the pool, "
                        f"saw {commits}")
    if evicts < 1:
        failures.append(f"eviction: expected >=1 prefix_cache evict, saw {evicts} "
                        f"(pool-pages={args.pool_pages} may be too large for "
                        f"the per-prompt prefix size)")
    # The whole point: eviction must reclaim pages so NO request 429s. A
    # `pool exhausted` line means the evict loop failed to free real pages
    # (the shared-page bug) -- a hard failure.
    if exhausted:
        failures.append("eviction: 'global KV page pool exhausted' appeared -- "
                        "eviction did not reclaim real pages (shared-page bug)")

    # ---- Check 2: cache OFF regression guard -----------------------------
    # No --prefix-cache flag -> the server forces QW3_PREFIX_CACHE off. Tracing
    # stays on; if any cache activity is logged, the flag gate leaked.
    off_log = os.path.join(args.logdir, "off.log")
    proc = start_server(args.binary, args.model, args.port,
                        {"QW3_PREFIX_CACHE_TRACE": "1"},
                        [], off_log)
    try:
        if not wait_for_server(base_url, args.boot_timeout):
            print("FAIL: cache-off server did not become ready")
            print(read_text(off_log)[-2000:])
            stop_server(proc)
            return 1
        for k in range(2):
            s, _txt, err = post_completion(base_url, prompts[0], args.max_tokens,
                                           args.req_timeout)
            if s != 200:
                failures.append(f"cache-off req#{k} failed: {s} {err}")
    finally:
        time.sleep(0.5)
        stop_server(proc)

    off_trace = read_text(off_log)
    off_commits = len(re.findall(r"prefix_cache commit ", off_trace))
    off_hits = len(re.findall(r"prefix_cache hit ", off_trace))
    off_evicts = len(re.findall(r"prefix_cache evict ", off_trace))
    if off_commits or off_hits or off_evicts:
        failures.append(f"regression: no --prefix-cache flag but cache still "
                        f"active (commits={off_commits} hits={off_hits} "
                        f"evicts={off_evicts}) -- flag gate leaked")

    print("=== prefix cache eviction + regression ===")
    print(f"kv_pool_pages={args.pool_pages} distinct_prompts={n}")
    print(f"eviction trace: commits={commits} evicts={evicts} "
          f"pool_exhausted={exhausted}")
    print(f"cache-off trace: commits={off_commits} hits={off_hits} "
          f"evicts={off_evicts}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\n--- eviction trace tail ---")
        print(evic_trace[-2000:])
        return 1
    print("\nPASS: LRU eviction reclaims pages without crash; cache-off is inert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
