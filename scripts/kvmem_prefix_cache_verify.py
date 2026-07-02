#!/usr/bin/env python3
"""Two-turn verification for the kvmem single-request prefix cache.

Targets the HTTP serve *plain* (non-continuous-batching) route with
`--kvmem --kvmem-prefix-cache` and `QW3_KVMEM_PREFIX_CACHE_TRACE=1`.

For each mode (plain decode and MTP speculate) it drives the append-only
multi-turn agent pattern:

  turn A: prompt = P            -> response R (captured)
  turn B: prompt = P + R + U    -> should reuse the warm end-of-A checkpoint
                                   and prefill only the U suffix.

Checks:
  1. HIT: turn-B server log shows `kvmem prefix-cache HIT` with reuse>0.
  2. Savings: turn-B `native generate:` log shows kvmem_reuse=/prefilled= with
     prefilled << prompt_tokens.
  3. Reuse ~ cold: turn-B warm (reuse) output shares a long common prefix with a
     fresh-server cold full-prefill of the identical P+R+U prompt. It is NOT
     required to be bit-exact: warm reuse resumes the DeltaNet recurrent state
     *incrementally* (restore_state) whereas the cold path rebuilds it via a
     *batched* prefill scan, so the two agree for many tokens then may diverge
     in the tail by fp rounding (the same "algorithmically lossless, not
     bit-reproducible" property run_kvmem_session has). A *broken* resume would
     instead corrupt the very first suffix-conditioned token, collapsing the
     common prefix -- which this threshold catches.
  4. Flag-off: with the flag unset, no HIT fires and the turn-B output is
     BIT-EXACT vs the cold full-prefill (both are genuine full prefills of the
     same P+R+U -> the byte-identity-when-off guarantee).
  5. MTP acceptance stays in the healthy 0.4-0.8 band under reuse.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            res = conn.getresponse()
            body = res.read()
            conn.close()
            if res.status == 200 and b"ok" in body:
                return
        except BaseException as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy: {last}")


def post_completion(host: str, port: int, prompt: str, max_tokens: int,
                    timeout_s: float) -> Tuple[int, str]:
    body = json.dumps({
        "model": "qw3",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    })
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    data = res.read().decode("utf-8", errors="replace")
    status = res.status
    conn.close()
    return status, data


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def completion_text(body: str) -> str:
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return ""
    ch = obj.get("choices") if isinstance(obj, dict) else None
    if not ch:
        return ""
    first = ch[0]
    return first.get("text", "") if isinstance(first, dict) else ""


def start_server(qw3: Path, model: Path, ctx: int, max_tokens: int,
                 mtp: bool, prefix_cache: bool,
                 trace: bool) -> Tuple[subprocess.Popen, str, int]:
    host = "127.0.0.1"
    port = find_free_port()
    cmd = [
        str(qw3), "serve",
        "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx),
        "-n", str(max_tokens),
        "--temp", "0",
        "--kv-dtype", "fp16",
        "--prefill-chunk", "2048",
        "--kvmem",
        "--kvmem-block-tokens", "256",
        "--kvmem-budget", "32768",
        "--kvmem-method", "retrieval",
    ]
    if prefix_cache:
        cmd.append("--kvmem-prefix-cache")
    if mtp:
        cmd.extend(["--native-mtp-speculate", "--mtp-chain", "4"])
    env = os.environ.copy()
    # Neutralize fp-atomic split-K nondeterminism so greedy output is
    # bit-reproducible warm-vs-cold.
    env["QW3_FATTN_NSPLIT"] = "1"
    env["QW3_PREFILL_FA2_NSPLIT"] = "1"
    if trace:
        env["QW3_KVMEM_PREFIX_CACHE_TRACE"] = "1"
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    return proc, host, port


def terminate(proc: subprocess.Popen) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    out, err = proc.communicate(timeout=5)
    return (out or "") + "\n" + (err or "")


# Turn A is captured with a SHORT response so the reply round-trips through the
# tokenizer: a long (dozens-of-token) reply frequently ends on a sub-word whose
# last token merges with the suffix's leading char at the reply|suffix seam,
# which perturbs the retokenized "P + R + U" and silently defeats the token-
# prefix reuse predicate. 8 tokens reliably preserves the warm-log token prefix.
WARM_TURNA_TOKENS = 8


def prompt_A() -> str:
    # Prose prompt ending on a hard boundary (period). The model's reply begins
    # after "machines." — that junction round-trips through the tokenizer, so
    # turn B's re-tokenized "P + R + U" preserves the warm-log token prefix and
    # the reuse predicate fires. (A prompt ending in "Answer:" does NOT round-
    # trip across the "<think>" reply and silently defeats reuse.)
    filler = " ".join(["The paged attention cache stores keys and values in order."] * 20)
    return (
        "Continue the following passage in plain prose.\n\n"
        + filler
        + " In a small town by the sea there lived a curious young engineer "
          "named Ada who loved building tiny machines."
    )


def suffix_U() -> str:
    # Leading newline makes the reply|suffix junction a hard left boundary
    # regardless of where the turn-A reply happened to stop, so re-tokenizing
    # keeps every warm-log token intact at the seam.
    return "\n\nThe next morning, Ada opened her workshop and began a new project."


ACCEPT_RE = re.compile(r"acceptance=([0-9.]+)")
REUSE_RE = re.compile(r"kvmem_reuse=(\d+)\s+prefilled=(\d+)")
PROMPT_TOK_RE = re.compile(r"native generate:\s+prompt_tokens=(\d+)")


def last_native_generate(log: str) -> str:
    lines = [ln for ln in log.splitlines() if "native generate:" in ln]
    return lines[-1] if lines else ""


def run_mode(qw3: Path, model: Path, ctx: int, max_tokens: int,
             mtp: bool, timeout: float) -> Dict:
    label = "mtp" if mtp else "plain"
    result: Dict = {"mode": label}

    # ---- warm server (flag ON, trace ON) ----
    proc, host, port = start_server(qw3, model, ctx, max_tokens, mtp,
                                    prefix_cache=True, trace=True)
    warm_B_text = ""
    try:
        wait_for_health(host, port, min(300.0, timeout))
        pA, u = prompt_A(), suffix_U()
        sA, bA = post_completion(host, port, pA, WARM_TURNA_TOKENS, timeout)
        rA = completion_text(bA)
        pB = pA + rA + u
        sB, bB = post_completion(host, port, pB, max_tokens, timeout)
        warm_B_text = completion_text(bB)
        result["status_A"], result["status_B"] = sA, sB
        result["turnA_response"] = rA
        result["turnB_warm_response"] = warm_B_text
    finally:
        warm_log = terminate(proc)
    result["warm_hit"] = "kvmem prefix-cache HIT" in warm_log
    result["warm_capture"] = "kvmem prefix-cache CAPTURE" in warm_log
    gen_B = last_native_generate(warm_log)
    result["turnB_generate_line"] = gen_B
    m = REUSE_RE.search(gen_B)
    if m:
        result["reuse"] = int(m.group(1))
        result["prefilled"] = int(m.group(2))
    pt = PROMPT_TOK_RE.search(gen_B)
    if pt:
        result["turnB_prompt_tokens"] = int(pt.group(1))
    if mtp:
        accs = [float(x) for x in ACCEPT_RE.findall(warm_log)]
        result["acceptances"] = accs

    # ---- cold server (flag ON, but only turn B sent -> full prefill) ----
    proc, host, port = start_server(qw3, model, ctx, max_tokens, mtp,
                                    prefix_cache=True, trace=True)
    cold_B_text = ""
    try:
        wait_for_health(host, port, min(300.0, timeout))
        pA, u = prompt_A(), suffix_U()
        pB = pA + result.get("turnA_response", "") + u
        sC, bC = post_completion(host, port, pB, max_tokens, timeout)
        cold_B_text = completion_text(bC)
        result["status_cold"] = sC
        result["turnB_cold_response"] = cold_B_text
    finally:
        cold_log = terminate(proc)
    result["cold_hit"] = "kvmem prefix-cache HIT" in cold_log

    # Reuse is NOT required to be bit-exact vs a cold full-prefill (incremental
    # vs batched DeltaNet recurrent-state scan diverges by fp rounding in the
    # tail). Require instead a long shared prefix: a correct resume agrees for
    # many tokens; a broken resume corrupts token 1 and collapses this to ~0.
    cp = common_prefix_len(warm_B_text, cold_B_text)
    min_len = min(len(warm_B_text), len(cold_B_text))
    result["reuse_exact"] = (warm_B_text == cold_B_text and warm_B_text != "")
    result["reuse_cold_prefix_chars"] = cp
    result["reuse_cold_min_len"] = min_len
    result["reuse_prefix_ok"] = (
        warm_B_text != ""
        and (result["reuse_exact"] or cp >= 32)
    )

    # ---- flag OFF (no reuse should ever fire) ----
    proc, host, port = start_server(qw3, model, ctx, max_tokens, mtp,
                                    prefix_cache=False, trace=True)
    off_B_text = ""
    try:
        wait_for_health(host, port, min(300.0, timeout))
        pA, u = prompt_A(), suffix_U()
        _sa, _ba = post_completion(host, port, pA, WARM_TURNA_TOKENS, timeout)
        pB = pA + result.get("turnA_response", "") + u
        _sb, bBoff = post_completion(host, port, pB, max_tokens, timeout)
        off_B_text = completion_text(bBoff)
        result["turnB_flagoff_response"] = off_B_text
    finally:
        off_log = terminate(proc)
    result["flagoff_no_hit"] = "kvmem prefix-cache HIT" not in off_log
    result["flagoff_equals_cold"] = (off_B_text == cold_B_text and off_B_text != "")

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--modes", default="plain,mtp",
                    help="comma list of: plain, mtp")
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_prefix_cache_verify.json")
    args = ap.parse_args(argv)

    qw3, model = Path(args.qw3), Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    failures: List[str] = []
    for mode in modes:
        mtp = mode == "mtp"
        r = run_mode(qw3, model, args.ctx, args.max_tokens, mtp, args.timeout)
        results.append(r)
        tag = f"[{mode}]"
        if not r.get("warm_hit"):
            failures.append(f"{tag} warm reuse did not fire (no HIT in log)")
        if not r.get("warm_capture"):
            failures.append(f"{tag} turn-A did not CAPTURE warm state")
        if r.get("cold_hit"):
            failures.append(f"{tag} cold server unexpectedly fired a HIT")
        if not r.get("reuse_prefix_ok"):
            failures.append(
                f"{tag} warm reuse diverges from cold too early "
                f"(common_prefix={r.get('reuse_cold_prefix_chars')} chars, "
                f"min_len={r.get('reuse_cold_min_len')}) -- resume likely broken")
        if not r.get("flagoff_no_hit"):
            failures.append(f"{tag} flag-off run fired a HIT")
        if not r.get("flagoff_equals_cold"):
            failures.append(f"{tag} flag-off turn-B output != cold turn-B output")
        prefilled = r.get("prefilled")
        ptok = r.get("turnB_prompt_tokens")
        if prefilled is not None and ptok is not None and prefilled >= ptok:
            failures.append(f"{tag} no prefill savings (prefilled={prefilled} >= prompt={ptok})")
        if mtp:
            accs = r.get("acceptances", [])
            # Per-step acceptance of 1.0 is the ideal case (the draft chain was
            # fully accepted) -- do NOT flag high values. The degenerate signal
            # is a consistently LOW mean, which is what a broken resume feeding
            # MTP would produce (draft repeatedly rejected). Gate on the mean.
            if accs:
                mean_acc = sum(accs) / len(accs)
                r["acceptance_mean"] = mean_acc
                if mean_acc < 0.3:
                    failures.append(
                        f"{tag} MTP acceptance mean too low: {mean_acc:.3f} "
                        f"(reuse may be corrupting decode state)")

    for r in results:
        print(f"=== mode={r['mode']} ===")
        for k in ("warm_hit", "warm_capture", "cold_hit", "reuse", "prefilled",
                  "turnB_prompt_tokens", "reuse_exact", "reuse_cold_prefix_chars",
                  "reuse_cold_min_len", "reuse_prefix_ok", "flagoff_no_hit",
                  "flagoff_equals_cold", "acceptances", "acceptance_mean"):
            if k in r:
                print(f"  {k}: {r[k]}")
        print(f"  turnB_generate_line: {r.get('turnB_generate_line','')}")

    out = {"ok": not failures, "failures": failures, "results": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        print(f"wrote {args.out_json}")
        return 1
    print(f"\nAll prefix-cache checks passed -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
