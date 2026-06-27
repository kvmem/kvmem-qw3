#!/usr/bin/env python3
"""End-to-end KVMem regression runner.

This is intentionally model-gated and not part of default ctest. It drives the
real qw3 binary with a GGUF model and checks:

1. KVMem identity selection is byte-identical to plain decode.
2. Step update mode still preserves identity behavior.
3. Tight recency-only KVMem selection changes a codeword-recall prompt, proving
   sparse visibility actually engages.
4. Single-request KVMem + MTP succeeds.
5. Continuous batching + KVMem works without MTP.
6. Continuous batching + KVMem + MTP fails explicitly instead of silently using
   an unsafe route.
7. Tiered KVMem can run with a bounded single-request GPU page pool smaller
   than ctx, forcing prefill-time CPU offload before decode.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import http.client
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SECRET = "ZYLON-4937-KVMEM"


@dataclass
class CaseResult:
    name: str
    ok: bool
    command: List[str]
    elapsed_s: float
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    status: Optional[int] = None
    response: str = ""
    error: Optional[str] = None
    completions: List[str] = field(default_factory=list)


def parse_completion_text(body: str) -> str:
    """Extract choices[0].text from an OpenAI /v1/completions JSON body."""
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return ""
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        return first.get("text") or first.get("message", {}).get("content", "") or ""
    return ""


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            res = conn.getresponse()
            body = res.read()
            conn.close()
            if res.status == 200 and b"ok" in body:
                return
        except BaseException as exc:  # noqa: BLE001 - record and retry.
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def post_completion(host: str,
                    port: int,
                    prompt: str,
                    max_tokens: int,
                    timeout_s: float) -> Tuple[int, str]:
    body = json.dumps({
        "model": "qw3",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    })
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    res = conn.getresponse()
    data = res.read().decode("utf-8", errors="replace")
    status = res.status
    conn.close()
    return status, data


def terminate_server(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    out, err = proc.communicate(timeout=5)
    return (out or "") + "\n" + (err or "")


def run_cli(name: str,
            qw3: Path,
            model: Path,
            prompt: str,
            max_tokens: int,
            ctx: int,
            timeout_s: int,
            extra_args: Sequence[str],
            extra_env: Optional[Dict[str, str]] = None) -> CaseResult:
    cmd = [
        str(qw3),
        "--backend", "qwen-native",
        "--model", str(model),
        "--native-heavy",
        "--raw",
        "--kv-dtype", "fp16",
        "--temp", "0",
        "--top-k", "1",
        "--prefill-chunk", "256",
        "-c", str(ctx),
        "-n", str(max_tokens),
        "-p", prompt,
    ]
    cmd.extend(extra_args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        elapsed = time.monotonic() - start
        return CaseResult(
            name=name,
            ok=proc.returncode == 0,
            command=cmd,
            elapsed_s=elapsed,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
    except subprocess.TimeoutExpired as exc:
        return CaseResult(
            name=name,
            ok=False,
            command=cmd,
            elapsed_s=time.monotonic() - start,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"timeout after {timeout_s}s",
        )


def codeword_prompt() -> str:
    prefix = (
        "You are testing exact long-context recall. Do not infer the answer. "
        "Only repeat the exact secret code if it appears in the context.\n\n"
    )
    filler_a = " ".join(["alpha beta gamma delta epsilon"] * 80)
    filler_b = " ".join(["theta iota kappa lambda mu"] * 80)
    suffix = (
        "\n\nQuestion: What is the exact secret code? "
        "Answer with only the code string."
    )
    return (
        prefix
        + filler_a
        + "\n\nThe exact secret code is: "
        + SECRET
        + "\n\n"
        + filler_b
        + suffix
    )


def identity_prompt() -> str:
    return (
        "Write one concise sentence explaining why paged KV cache page tables "
        "must preserve token order."
    )


def bounded_pool_prompt() -> str:
    return " ".join(
        ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 80
    )


def compare_identity(results: List[CaseResult],
                     lhs_name: str,
                     rhs_name: str) -> Optional[str]:
    by_name = {r.name: r for r in results}
    lhs = by_name[lhs_name]
    rhs = by_name[rhs_name]
    if not lhs.ok or not rhs.ok:
        return f"{lhs_name}/{rhs_name} command failed"
    if lhs.stdout != rhs.stdout:
        return (
            f"{lhs_name} and {rhs_name} stdout differ "
            f"(lhs={lhs.stdout[:80]!r}, rhs={rhs.stdout[:80]!r})"
        )
    return None


def tier_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.kvmem_cpu_bytes > 0:
        out.extend(["--kvmem-cpu-bytes", str(args.kvmem_cpu_bytes)])
    if args.kvmem_nvme_dir:
        out.extend(["--kvmem-nvme-dir", args.kvmem_nvme_dir])
    if args.kvmem_nvme_bytes > 0:
        out.extend(["--kvmem-nvme-bytes", str(args.kvmem_nvme_bytes)])
    return out


def run_continuous_case(name: str,
                        qw3: Path,
                        model: Path,
                        prompts: Sequence[str],
                        max_tokens: int,
                        ctx: int,
                        timeout_s: int,
                        extra_args: Sequence[str],
                        expect_success: bool,
                        expect_error_substring: Optional[str] = None,
                        extra_env: Optional[Dict[str, str]] = None,
                        enable_kvmem: bool = True) -> CaseResult:
    host = "127.0.0.1"
    port = find_free_port()
    cmd = [
        str(qw3),
        "serve",
        "--model", str(model),
        "--host", host,
        "--port", str(port),
        "--ctx", str(ctx),
        "-n", str(max_tokens),
        "--temp", "0",
        "--kv-dtype", "fp16",
        "--prefill-chunk", "256",
        "--continuous-batching",
        "--max-active", str(max(2, len(prompts))),
    ]
    if enable_kvmem:
        cmd.extend([
            "--kvmem",
            "--kvmem-block-tokens", "16",
            "--kvmem-budget", "4096",
            "--kvmem-method", "recency",
        ])
    cmd.extend(extra_args)
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    if extra_env:
        env.update(extra_env)
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    response_text = ""
    status = None
    error: Optional[str] = None
    log = ""
    completions: List[str] = []
    try:
        wait_for_health(host, port, min(120.0, float(timeout_s)))
        if len(prompts) == 1:
            status, response_text = post_completion(
                host, port, prompts[0], max_tokens, float(timeout_s)
            )
            completions = [parse_completion_text(response_text)]
        else:
            with futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
                futs = [
                    ex.submit(post_completion, host, port, p, max_tokens, float(timeout_s))
                    for p in prompts
                ]
                replies = [f.result(timeout=timeout_s) for f in futs]
            status = max(s for s, _ in replies)
            response_text = "\n".join(body for _, body in replies)
            completions = [parse_completion_text(body) for _, body in replies]
    except BaseException as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        log = terminate_server(proc)
    elapsed = time.monotonic() - start

    ok = False
    combined = response_text + "\n" + log
    if expect_success:
        ok = error is None and status is not None and 200 <= status < 300
        if ok and "route=continuous" not in log:
            ok = False
            error = "did not observe route=continuous in server log"
    else:
        ok = (
            status is not None
            and status >= 400
            and (expect_error_substring is None or expect_error_substring in combined)
        )
        if not ok and error is None:
            error = "expected explicit server error was not observed"

    return CaseResult(
        name=name,
        ok=ok,
        command=cmd,
        elapsed_s=elapsed,
        returncode=proc.returncode,
        stdout=log,
        stderr="",
        status=status,
        response=response_text,
        error=error,
        completions=completions,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run KVMem end-to-end regressions.")
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_e2e_regression.json")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--skip-continuous",
        action="store_true",
        help="skip server-side continuous-batching checks",
    )
    ap.add_argument(
        "--skip-mtp",
        action="store_true",
        help="skip single-request KVMem + MTP check",
    )
    ap.add_argument(
        "--kvmem-cpu-bytes",
        type=int,
        default=0,
        help="optional CPU tier byte budget passed to all KVMem cases",
    )
    ap.add_argument(
        "--kvmem-nvme-dir",
        default="",
        help="optional NVMe tier directory passed to all KVMem cases",
    )
    ap.add_argument(
        "--kvmem-nvme-bytes",
        type=int,
        default=0,
        help="optional NVMe tier byte budget passed to all KVMem cases",
    )
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    results: List[CaseResult] = []
    failures: List[str] = []
    tier = tier_args(args)
    tier_env = {"QW3_KVMEM_TIER_TRACE": "1"} if tier else None

    prompt = identity_prompt()
    results.append(run_cli(
        "plain_identity",
        qw3,
        model,
        prompt,
        args.max_tokens,
        args.ctx,
        args.timeout,
        [],
    ))
    results.append(run_cli(
        "kvmem_identity",
        qw3,
        model,
        prompt,
        args.max_tokens,
        args.ctx,
        args.timeout,
        [
            "--kvmem",
            "--kvmem-block-tokens", "16",
            "--kvmem-budget", "4096",
            "--kvmem-method", "recency",
        ] + tier,
        tier_env,
    ))
    results.append(run_cli(
        "kvmem_identity_step",
        qw3,
        model,
        prompt,
        args.max_tokens,
        args.ctx,
        args.timeout,
        [
            "--kvmem",
            "--kvmem-block-tokens", "16",
            "--kvmem-budget", "4096",
            "--kvmem-method", "recency",
            "--kvmem-update-mode", "step",
        ] + tier,
        tier_env,
    ))

    for lhs, rhs in [
        ("plain_identity", "kvmem_identity"),
        ("plain_identity", "kvmem_identity_step"),
    ]:
        err = compare_identity(results, lhs, rhs)
        if err:
            failures.append(err)

    sparse_prompt = codeword_prompt()
    codeword_max_tokens = max(args.max_tokens, 24)
    results.append(run_cli(
        "kvmem_codeword_full",
        qw3,
        model,
        sparse_prompt,
        codeword_max_tokens,
        max(args.ctx, 2048),
        args.timeout,
        [
            "--kvmem",
            "--kvmem-block-tokens", "16",
            "--kvmem-budget", "4096",
            "--kvmem-method", "recency",
        ] + tier,
        tier_env,
    ))
    results.append(run_cli(
        "kvmem_codeword_sparse",
        qw3,
        model,
        sparse_prompt,
        codeword_max_tokens,
        max(args.ctx, 2048),
        args.timeout,
        [
            "--kvmem",
            "--kvmem-block-tokens", "16",
            "--kvmem-budget", "64",
            "--kvmem-sink-blocks", "1",
            "--kvmem-recent-blocks", "1",
            "--kvmem-method", "recency",
        ] + tier,
        tier_env,
    ))
    by_name = {r.name: r for r in results}
    full = by_name["kvmem_codeword_full"]
    sparse = by_name["kvmem_codeword_sparse"]
    if not full.ok or not sparse.ok:
        failures.append("codeword full/sparse command failed")
    elif full.stdout == sparse.stdout:
        failures.append("tight sparse KVMem output matched full-window output")
    elif SECRET not in full.stdout:
        failures.append(f"full-window codeword output did not contain {SECRET}")
    if tier and "[kvmem-tier] stage_out" not in sparse.stderr:
        failures.append("tiered sparse KVMem did not emit stage_out trace")

    if tier:
        results.append(run_cli(
            "kvmem_retrieval_stagein",
            qw3,
            model,
            sparse_prompt,
            codeword_max_tokens,
            max(args.ctx, 2048),
            args.timeout,
            [
                "--kvmem",
                "--kvmem-block-tokens", "16",
                "--kvmem-budget", "64",
                "--kvmem-sink-blocks", "1",
                "--kvmem-recent-blocks", "1",
                "--kvmem-method", "retrieval",
                "--kvmem-interval", "1",
            ] + tier,
            tier_env,
        ))
        stagein = results[-1]
        if not stagein.ok:
            failures.append("tiered retrieval stage-in command failed")
        elif "[kvmem-tier] stage_in" not in stagein.stderr:
            failures.append("tiered retrieval KVMem did not emit stage_in trace")
        elif args.kvmem_nvme_bytes > 0 and "[kvmem-tier] stage_in_async_read" not in stagein.stderr:
            failures.append("tiered retrieval KVMem did not emit async NVMe read trace")

        results.append(run_cli(
            "kvmem_bounded_gpu_pool",
            qw3,
            model,
            bounded_pool_prompt(),
            1,
            max(args.ctx, 1024),
            args.timeout,
            [
                "--prefill-chunk", "64",
                "--kvmem",
                "--kvmem-block-tokens", "16",
                "--kvmem-budget", "64",
                "--kvmem-sink-blocks", "1",
                "--kvmem-recent-blocks", "1",
                "--kvmem-method", "recency",
                "--kvmem-gpu-memory-ratio", "0.00005",
            ] + tier,
            tier_env,
        ))
        bounded = results[-1]
        if not bounded.ok:
            failures.append("bounded GPU-pool KVMem command failed")
        elif "[kvmem-tier] bounded_gpu_pool" not in bounded.stderr:
            failures.append("bounded GPU-pool KVMem did not enable bounded pool")
        elif "[kvmem-tier] stage_out" not in bounded.stderr:
            failures.append("bounded GPU-pool KVMem did not stage out during prefill")

    if not args.skip_mtp:
        results.append(run_cli(
            "kvmem_mtp",
            qw3,
            model,
            prompt,
            args.max_tokens,
            args.ctx,
            args.timeout,
            [
                "--kvmem",
                "--kvmem-block-tokens", "16",
                "--kvmem-budget", "4096",
                "--kvmem-method", "recency",
                "--native-mtp-speculate",
                "--mtp-chain", "2",
            ] + tier,
            tier_env,
        ))
        if not results[-1].ok:
            failures.append("single-request KVMem + MTP command failed")

    if not args.skip_continuous:
        prompts = [
            "Answer in five words: what is paged KV?",
            "Answer in five words: why use batching?",
        ]
        results.append(run_continuous_case(
            "kvmem_continuous",
            qw3,
            model,
            prompts,
            args.max_tokens,
            args.ctx,
            args.timeout,
            tier,
            expect_success=True,
            extra_env=tier_env,
        ))
        if not results[-1].ok:
            failures.append("KVMem continuous-batching request failed")

        # Phase-D ragged gate: CB + MTP + kvmem at an all-fit (identity) budget
        # must be byte-identical to plain CB + MTP. Drive >=2 concurrent prompts
        # so the verify batch has jobs.size()>1 and takes the window-aware ragged
        # route (jobs.size()==1 uses the single verifier). Force the ragged route
        # deterministically by lowering the ragged-verify min-token threshold
        # (default 16) to 1 -- applied to BOTH the plain baseline and the kvmem
        # run so the only variable is kvmem on/off, not the verify kernel route.
        # NSPLIT=1 neutralizes the known fp-atomic split-K nondeterminism so
        # greedy output is reproducible. A longer generation keeps both requests
        # co-batched for many verify steps.
        mtp_max_tokens = max(args.max_tokens, 48)
        mtp_args = [
            "--native-mtp-speculate",
            "--mtp-chain", "2",
            "--kvmem-budget", "131072",
        ]
        parity_env = dict(tier_env or {})
        parity_env["QW3_FATTN_NSPLIT"] = "1"
        parity_env["QW3_PREFILL_FA2_NSPLIT"] = "1"
        parity_env["QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS"] = "1"

        plain_mtp = run_continuous_case(
            "plain_continuous_mtp",
            qw3,
            model,
            prompts,
            mtp_max_tokens,
            args.ctx,
            args.timeout,
            ["--native-mtp-speculate", "--mtp-chain", "2"] + tier,
            expect_success=True,
            extra_env=parity_env,
            enable_kvmem=False,
        )
        results.append(plain_mtp)
        if not plain_mtp.ok:
            failures.append("plain continuous + MTP baseline request failed")

        kvmem_mtp = run_continuous_case(
            "kvmem_continuous_mtp",
            qw3,
            model,
            prompts,
            mtp_max_tokens,
            args.ctx,
            args.timeout,
            mtp_args + tier,
            expect_success=True,
            extra_env=parity_env,
        )
        results.append(kvmem_mtp)
        if not kvmem_mtp.ok:
            failures.append("KVMem + continuous + MTP (ragged route) request failed")
        elif plain_mtp.ok:
            if kvmem_mtp.completions != plain_mtp.completions:
                failures.append(
                    "kvmem CB+MTP (identity budget) diverged from plain CB+MTP: "
                    f"plain={plain_mtp.completions!r} kvmem={kvmem_mtp.completions!r}"
                )

        # The opt-in LAYERED verifier is the one remaining unsupported combo and
        # must still hard-error (never silently verify against the wrong KV).
        layered_env = dict(tier_env or {})
        layered_env["QW3_CONTINUOUS_MTP_LAYERED_VERIFY"] = "1"
        results.append(run_continuous_case(
            "kvmem_continuous_mtp_layered_hard_error",
            qw3,
            model,
            ["Answer briefly: hello"],
            args.max_tokens,
            args.ctx,
            args.timeout,
            ["--native-mtp-speculate", "--mtp-chain", "2"] + tier,
            expect_success=False,
            expect_error_substring="cannot be combined with the layered MTP",
            extra_env=layered_env,
        ))
        if not results[-1].ok:
            failures.append(
                "KVMem + continuous + layered-MTP hard-error guard failed"
            )

    for r in results:
        status = "ok" if r.ok else "FAIL"
        print(
            f"{status:4s} {r.name:32s} elapsed={r.elapsed_s:.2f}s "
            f"rc={r.returncode} status={r.status} error={r.error or '-'}"
        )

    out = {
        "ok": not failures,
        "failures": failures,
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        print("failed KVMem E2E requirements:")
        for f in failures:
            print(f"  - {f}")
        print(f"wrote {out_path}")
        return 1
    print(f"KVMem E2E regression passed -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
