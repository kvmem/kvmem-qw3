#!/usr/bin/env python3
"""Focused Phase-D gate: CB + MTP + kvmem on the continuous-batching path.

Runs only the server-side continuous cases that exercise the changed code
(window-aware ragged verify, narrowed hard error). Avoids the single-request
CLI cases in kvmem_e2e_regression.py so iteration only reloads the model a few
times. Checks:

  1. plain CB+MTP baseline succeeds (reference output).
  2. kvmem CB+MTP at identity budget, forced onto the ragged route, is
     byte-identical to the plain baseline per request.
  3. kvmem CB+MTP with the opt-in LAYERED verifier still hard-errors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402


def long_prompt(tag: str, words: int) -> str:
    body = " ".join(["alpha beta gamma delta epsilon"] * (words // 5 + 1))
    return f"{tag}\n{body}\nSummarize the passage in one short sentence."


def run_prefix_probe(qw3: Path, model: Path, ctx: int, max_tokens: int,
                     timeout_s: int, mtp: bool, kvmem: bool = False):
    """Start one serve, post the SAME prompt twice SEQUENTIALLY with the prefix
    cache on, and report commit/hit trace evidence + the prefill ops of each
    request (the second should reuse the committed prefix => fewer prefill ops).
    """
    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", "fp16", "--prefill-chunk", "256",
        "--continuous-batching", "--max-active", "2",
        "--prefix-cache",
    ]
    if kvmem:
        cmd += ["--kvmem", "--kvmem-block-tokens", "16",
                "--kvmem-budget", "131072", "--kvmem-method", "recency"]
    if mtp:
        cmd += ["--native-mtp-speculate", "--mtp-chain", "2"]
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    env["QW3_PREFIX_CACHE"] = "1"
    env["QW3_PREFIX_CACHE_TRACE"] = "1"
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    prompt = long_prompt("PFX-SHARED-PREAMBLE", 200)
    bodies = []
    try:
        e2e.wait_for_health(host, port, min(120.0, float(timeout_s)))
        # sequential: request 1 fully completes (and commits) before request 2.
        for _ in range(2):
            _, body = e2e.post_completion(host, port, prompt, max_tokens,
                                          float(timeout_s))
            bodies.append(body)
            time.sleep(0.5)
    finally:
        log = e2e.terminate_server(proc)
    texts = [e2e.parse_completion_text(b) for b in bodies]
    return {
        "log": log,
        "texts": texts,
        "commit": "prefix_cache commit" in log,
        "hit": "prefix_cache hit" in log,
        "reused": "reused_tokens=" in log,
        "ok_status": all(t != "" for t in texts),
        "identical": len(texts) == 2 and texts[0] == texts[1],
        "cmd": cmd,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_cb_mtp_gate.json")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    prompts = [
        "Answer in five words: what is paged KV?",
        "Answer in five words: why use batching?",
    ]
    parity_env = {
        "QW3_FATTN_NSPLIT": "1",
        "QW3_PREFILL_FA2_NSPLIT": "1",
        "QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS": "1",
    }

    results = []
    failures: List[str] = []

    plain = e2e.run_continuous_case(
        "plain_continuous_mtp",
        qw3, model, prompts, args.max_tokens, args.ctx, args.timeout,
        ["--native-mtp-speculate", "--mtp-chain", "2"],
        expect_success=True,
        extra_env=parity_env,
        enable_kvmem=False,
    )
    results.append(plain)
    if not plain.ok:
        failures.append(f"plain CB+MTP baseline failed: {plain.error}")

    kvmem = e2e.run_continuous_case(
        "kvmem_continuous_mtp",
        qw3, model, prompts, args.max_tokens, args.ctx, args.timeout,
        ["--native-mtp-speculate", "--mtp-chain", "2",
         "--kvmem", "--kvmem-block-tokens", "16",
         "--kvmem-budget", "131072", "--kvmem-method", "recency"],
        expect_success=True,
        extra_env=parity_env,
        enable_kvmem=False,  # flags supplied explicitly above
    )
    results.append(kvmem)
    if not kvmem.ok:
        failures.append(f"kvmem CB+MTP (ragged) failed: {kvmem.error}")
    elif plain.ok and kvmem.completions != plain.completions:
        failures.append(
            "kvmem CB+MTP (identity budget) diverged from plain CB+MTP:\n"
            f"  plain={plain.completions!r}\n  kvmem={kvmem.completions!r}"
        )

    # ragged-route evidence: the kvmem run's server log should show a batched
    # verify (route=continuous + native continuous_mtp). If it silently fell to
    # the single verifier the parity check still holds, but we want the ragged
    # path covered, so surface it.
    saw_continuous_mtp = "native continuous_mtp:" in kvmem.stdout
    if kvmem.ok and not saw_continuous_mtp:
        print("WARN: kvmem run did not log native continuous_mtp marker")

    layered = e2e.run_continuous_case(
        "kvmem_continuous_mtp_layered_hard_error",
        qw3, model, ["Answer briefly: hello"],
        args.max_tokens, args.ctx, args.timeout,
        ["--native-mtp-speculate", "--mtp-chain", "2",
         "--kvmem", "--kvmem-block-tokens", "16",
         "--kvmem-budget", "131072", "--kvmem-method", "recency"],
        expect_success=False,
        expect_error_substring="cannot be combined with the layered MTP",
        extra_env={"QW3_CONTINUOUS_MTP_LAYERED_VERIFY": "1"},
        enable_kvmem=False,
    )
    results.append(layered)
    if not layered.ok:
        failures.append(f"layered hard-error guard failed: {layered.error}")

    # Scenario #3: sparse-budget coherence. A genuinely tight window (budget=64
    # tokens => 4 blocks of 16) plus frequent reselection (interval=8) and 2
    # concurrent requests with rejections must NOT trip the window-underflow
    # throw or otherwise crash. Output WILL differ from plain (sparse visibility
    # changes it), so we only require successful completion + no underflow throw.
    sparse_prompts = [long_prompt("SPARSE-A", 240), long_prompt("SPARSE-B", 240)]
    sparse = e2e.run_continuous_case(
        "kvmem_continuous_mtp_sparse",
        qw3, model, sparse_prompts, max(args.max_tokens, 64), max(args.ctx, 2048),
        args.timeout,
        ["--native-mtp-speculate", "--mtp-chain", "4",
         "--kvmem", "--kvmem-block-tokens", "16", "--kvmem-budget", "64",
         "--kvmem-sink-blocks", "1", "--kvmem-recent-blocks", "1",
         "--kvmem-method", "recency", "--kvmem-interval", "8"],
        expect_success=True,
        extra_env={"QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS": "1"},
        enable_kvmem=False,
    )
    results.append(sparse)
    underflow = "true base precedes window tail" in sparse.stdout
    if not sparse.ok:
        failures.append(f"sparse kvmem CB+MTP failed: {sparse.error}")
    if underflow:
        failures.append("sparse kvmem CB+MTP tripped the window-underflow throw")

    # Scenario #4: prefix-cache double-prefill probe on the CB+MTP path. Sends a
    # shared-prefix prompt twice sequentially; reports commit/hit evidence. In a
    # pure-MTP serve the commit may not fire (commit lives on the non-MTP prefill
    # path), so this is informational: if a hit occurs the second output must
    # still equal the first (Task-6 prefill_offset fix => no double-prefill).
    pfx = run_prefix_probe(qw3, model, max(args.ctx, 2048),
                           max(args.max_tokens, 32), args.timeout, mtp=True)
    print(f"prefix-probe(MTP): commit={pfx['commit']} hit={pfx['hit']} "
          f"reused={pfx['reused']} ok_status={pfx['ok_status']} "
          f"identical={pfx['identical']}")
    if not pfx["ok_status"]:
        failures.append("prefix-cache CB+MTP probe: a request returned no text")
    if pfx["hit"] and not pfx["identical"]:
        failures.append(
            "prefix-cache CB+MTP hit produced different output on the 2nd "
            "request (possible double-prefill / offset bug)")

    for r in results:
        status = "ok" if r.ok else "FAIL"
        print(f"{status:4s} {r.name:38s} elapsed={r.elapsed_s:.1f}s "
              f"status={r.status} error={r.error or '-'}")
    print(f"plain completions: {plain.completions!r}")
    print(f"kvmem completions: {kvmem.completions!r}")

    out = {"ok": not failures, "failures": failures,
           "results": [e2e.asdict(r) for r in results]}
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"wrote {args.out_json}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: CB + MTP + kvmem gate green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
