"""Drive ./build/qw3 for one benchmark cell (plain or MTP speculative).

Reuses prompt synthesis from long_prompt_sweep.make_prompt and the qw3 + MTP
regexes from mtp_acceptance_probe. TTFT/ITL use the `approx` source by default
(ttft = prefill_s, itl = decode_s / decoded) which is faithful for qw3 because
the first decode token's argmax is produced by the prefill step.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from long_prompt_sweep import make_prompt  # type: ignore  # noqa: E402
from mtp_acceptance_probe import (  # type: ignore  # noqa: E402
    _QW3_LINE,
    _MTP_SPEC_SUMMARY,
    _MTP_SPEC_TIMINGS,
    _MTP_ACCEPT_HIST,
    _MTP_ACCEPT_HIST_ITEM,
    _MTP_CHAIN_OFFSET,
)

from .schema import TrialMeasurement, SRC_APPROX
from .vram import run_with_polling
from .config import BenchConfig


def _write_prompt(prompt: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="qw3_bench_prompt_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(prompt)
    return Path(path)


def _base_cmd(cfg: BenchConfig, ctx: int, n_decode: int, prompt_path: Path) -> list:
    return [
        cfg.qw3,
        "--backend", "qwen-native",
        "--native-heavy",
        "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", cfg.model,
        "--raw",
        "-c", str(ctx),
        "-n", str(n_decode),
        "--temp", "0",
        "--seed", "0",
        "--prompt-file", str(prompt_path),
    ]


def _approx_latency(prefill_s: float, decode_s: float, decoded: int):
    """qw3 TTFT/ITL via the `approx` source.

    ttft ~= prefill_s (first decode token argmax comes from prefill).
    itl  ~= decode_s / decoded (mean inter-token latency over the decode run).
    """
    ttft_s = prefill_s
    itl_ms = (decode_s / decoded * 1000.0) if decoded > 0 else 0.0
    return ttft_s, itl_ms


def _parse_common(out: str) -> Optional[dict]:
    m = _QW3_LINE.search(out)
    if not m:
        return None
    prompt_tokens = int(m.group(1))
    prefill_s = float(m.group(2))
    prefill_tok_s = float(m.group(3))
    decoded = int(m.group(4))
    decode_s = float(m.group(5))
    decode_tok_s = float(m.group(6))
    return {
        "prompt_tokens": prompt_tokens,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tok_s,
        "decoded": decoded,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
    }


def run_plain(cfg: BenchConfig, prompt_tokens: int, n_decode: int) -> TrialMeasurement:
    prompt = make_prompt(prompt_tokens)
    pf = _write_prompt(prompt)
    ctx = cfg.ctx_for(prompt_tokens, n_decode)
    cmd = _base_cmd(cfg, ctx, n_decode, pf)
    env = os.environ.copy()
    timeout = cfg.timeout_for(prompt_tokens, n_decode)
    try:
        proc, peak = run_with_polling(cmd, env, timeout)
    except subprocess.TimeoutExpired:
        return TrialMeasurement(ok=False, error="timeout")
    finally:
        pf.unlink(missing_ok=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return TrialMeasurement(ok=False, error=f"exit {proc.returncode}: {out.strip()[-200:]}")
    parsed = _parse_common(out)
    if not parsed:
        return TrialMeasurement(ok=False, error=f"no parse; tail={out.strip()[-200:]}")
    ttft_s, itl_ms = _approx_latency(parsed["prefill_s"], parsed["decode_s"], parsed["decoded"])
    return TrialMeasurement(
        ok=True,
        prompt_tokens=parsed["prompt_tokens"],
        decoded_tokens=parsed["decoded"],
        prefill_s=parsed["prefill_s"],
        decode_s=parsed["decode_s"],
        prefill_tok_s=parsed["prefill_tok_s"],
        decode_tok_s=parsed["decode_tok_s"],
        ttft_s=ttft_s,
        itl_ms=itl_ms,
        ttft_source=SRC_APPROX,
        itl_source=SRC_APPROX,
        peak_vram_mib=peak,
    )


def run_mtp(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
            mtp_chain: int) -> TrialMeasurement:
    prompt = make_prompt(prompt_tokens)
    pf = _write_prompt(prompt)
    ctx = cfg.ctx_for(prompt_tokens, n_decode)
    cmd = _base_cmd(cfg, ctx, n_decode, pf)
    # MTP speculative + adaptive policy (mandatory). Insert flags after binary.
    cmd[1:1] = ["--native-mtp-speculate", "--native-mtp-chain", str(mtp_chain)]
    env = os.environ.copy()
    env["QW3_MTP_POLICY"] = "adaptive"
    timeout = cfg.timeout_for(prompt_tokens, n_decode)
    try:
        proc, peak = run_with_polling(cmd, env, timeout)
    except subprocess.TimeoutExpired:
        return TrialMeasurement(ok=False, error="timeout")
    finally:
        pf.unlink(missing_ok=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return TrialMeasurement(ok=False, error=f"exit {proc.returncode}: {out.strip()[-200:]}")
    parsed = _parse_common(out)
    summary = _MTP_SPEC_SUMMARY.search(out)
    if not parsed or not summary:
        return TrialMeasurement(ok=False, error=f"no parse; tail={out.strip()[-300:]}")
    accept_rate = float(summary.group(11))

    # Per-chain-step acceptance from mtp_chain_offset lines.
    per_step = [0.0] * mtp_chain
    for step_s, _ver_s, _acc_s, acc_rate_s in _MTP_CHAIN_OFFSET.findall(out):
        step = int(step_s)
        if 1 <= step <= mtp_chain:
            per_step[step - 1] = float(acc_rate_s)

    # Accept-length histogram.
    hist = {}
    hist_match = _MTP_ACCEPT_HIST.search(out)
    if hist_match:
        for idx_s, val_s in _MTP_ACCEPT_HIST_ITEM.findall(hist_match.group(1)):
            hist[f"len{idx_s}"] = int(val_s)

    # Draft/verify timing split.
    draft_s = verify_s = None
    tmatch = _MTP_SPEC_TIMINGS.search(out)
    if tmatch:
        draft_s = float(tmatch.group(1) or 0.0)
        verify_s = float(tmatch.group(3) or 0.0)

    ttft_s, itl_ms = _approx_latency(parsed["prefill_s"], parsed["decode_s"], parsed["decoded"])
    return TrialMeasurement(
        ok=True,
        prompt_tokens=parsed["prompt_tokens"],
        decoded_tokens=parsed["decoded"],
        prefill_s=parsed["prefill_s"],
        decode_s=parsed["decode_s"],
        prefill_tok_s=parsed["prefill_tok_s"],
        decode_tok_s=parsed["decode_tok_s"],
        ttft_s=ttft_s,
        itl_ms=itl_ms,
        ttft_source=SRC_APPROX,
        itl_source=SRC_APPROX,
        accept_rate=accept_rate,
        accept_per_step=per_step,
        accept_hist=hist or None,
        mtp_draft_s=draft_s,
        mtp_verify_s=verify_s,
        peak_vram_mib=peak,
    )
