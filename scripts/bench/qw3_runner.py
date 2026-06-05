"""Drive a persistent `qw3 serve` for one benchmark cell (plain or MTP).

The 27B model is expensive to load, so — exactly like the llama-server runner —
we launch ONE `qw3 serve` per cell and loop `trials` requests against it over
the OpenAI-compatible HTTP API. The model loads once per cell instead of once
per trial.

The qw3 serve process prints the SAME per-request timing + MTP summary lines to
its stderr that the bare CLI prints (`generate_plain`/`generate_mtp` emit them
on every request), so we parse those lines per request — identical numbers to
the old per-process CLI path, no engine changes required. Each request's lines
are isolated by a unique marker we grep between.

TTFT/ITL use the `approx` source (ttft = prefill_s, itl = decode_s / decoded),
faithful for qw3 because the first decode token's argmax comes from prefill.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

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
from .vram import BackgroundVramPoller
from .config import BenchConfig


class Qw3Server:
    """Start/stop one `qw3 serve` for a cell; drain stderr into a buffer.

    The server prints `[qw3] native generate: ...` and MTP summary/hist lines to
    stderr per request. We keep a growing list of stderr lines in a background
    drain thread; each request slices out the lines emitted since its start.
    """

    def __init__(self, cfg: BenchConfig, ctx: int, n_decode: int,
                 is_mtp: bool, mtp_chain: int):
        self.cfg = cfg
        self.ctx = ctx
        self.n_decode = n_decode
        self.is_mtp = is_mtp
        self.mtp_chain = mtp_chain
        self.proc: Optional[subprocess.Popen] = None
        self._lines: List[str] = []
        self._lock = threading.Lock()
        self._drain: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.qw3_host}:{self.cfg.qw3_port}"

    def _cmd(self) -> list:
        cmd = [
            self.cfg.qw3, "serve",
            "--model", self.cfg.model,
            "--backend", "qwen-native",
            "--native-heavy",
            "--native-kernels", "cuda",
            "--native-linear-backend", "auto",
            "--host", self.cfg.qw3_host,
            "--port", str(self.cfg.qw3_port),
            "-c", str(self.ctx),
        ]
        if self.cfg.kv_dtype != "fp16":
            cmd += ["--kv-dtype", self.cfg.kv_dtype]
        if self.is_mtp:
            cmd += ["--native-mtp-speculate",
                    "--native-mtp-chain", str(self.mtp_chain)]
        return cmd

    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in self.proc.stderr:
            with self._lock:
                self._lines.append(raw)

    def line_count(self) -> int:
        with self._lock:
            return len(self._lines)

    def lines_since(self, start: int) -> str:
        with self._lock:
            return "".join(self._lines[start:])

    def __enter__(self) -> "Qw3Server":
        env = os.environ.copy()
        if self.is_mtp:
            env["QW3_MTP_POLICY"] = "adaptive"
        self.proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, start_new_session=True, env=env,
        )
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()
        self._wait_ready(self.cfg.startup_timeout_s)
        return self

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        url = f"{self.base_url}/v1/models"
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = self.lines_since(0)[-1500:]
                raise RuntimeError(
                    f"qw3 serve exited {self.proc.returncode}; tail={tail}")
            try:
                with urllib.request.urlopen(url, timeout=1.0):
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.2)
        raise TimeoutError("qw3 serve did not become ready")

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self._drain is not None:
            self._drain.join(timeout=2.0)


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
    return {
        "prompt_tokens": int(m.group(1)),
        "prefill_s": float(m.group(2)),
        "prefill_tok_s": float(m.group(3)),
        "decoded": int(m.group(4)),
        "decode_s": float(m.group(5)),
        "decode_tok_s": float(m.group(6)),
    }


def _post_completion(base_url: str, model_name: str, prompt: str,
                     n_decode: int, timeout: float) -> None:
    """POST a non-streamed raw completion (greedy). Body is discarded — the
    measurement comes from the server's stderr timing lines."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": n_decode,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _measure(server: Qw3Server, cfg: BenchConfig, prompt: str, n_decode: int,
             is_mtp: bool, mtp_chain: int) -> TrialMeasurement:
    model_name = Path(cfg.model).name
    timeout = cfg.timeout_for(server.ctx, n_decode)
    mark = server.line_count()  # stderr lines before this request
    with BackgroundVramPoller() as vram:
        try:
            _post_completion(server.base_url, model_name, prompt, n_decode, timeout)
        except Exception as exc:  # noqa: BLE001
            return TrialMeasurement(ok=False, error=f"request failed: {exc}")
        peak = vram.peak_mib
    out = server.lines_since(mark)
    parsed = _parse_common(out)
    if not parsed:
        return TrialMeasurement(
            ok=False, error=f"no timing line; tail={out.strip()[-300:]}")
    ttft_s, itl_ms = _approx_latency(
        parsed["prefill_s"], parsed["decode_s"], parsed["decoded"])
    tm = TrialMeasurement(
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
    if is_mtp:
        _parse_mtp(out, mtp_chain, tm)
    return tm


def _parse_mtp(out: str, mtp_chain: int, tm: TrialMeasurement) -> None:
    summary = _MTP_SPEC_SUMMARY.search(out)
    if not summary:
        tm.ok = False
        tm.error = (tm.error + "; " if tm.error else "") + "no mtp_spec_summary"
        return
    tm.accept_rate = float(summary.group(11))

    per_step = [0.0] * mtp_chain
    for step_s, _ver_s, _acc_s, acc_rate_s in _MTP_CHAIN_OFFSET.findall(out):
        step = int(step_s)
        if 1 <= step <= mtp_chain:
            per_step[step - 1] = float(acc_rate_s)
    tm.accept_per_step = per_step

    hist = {}
    hist_match = _MTP_ACCEPT_HIST.search(out)
    if hist_match:
        for idx_s, val_s in _MTP_ACCEPT_HIST_ITEM.findall(hist_match.group(1)):
            hist[f"len{idx_s}"] = int(val_s)
    tm.accept_hist = hist or None

    tmatch = _MTP_SPEC_TIMINGS.search(out)
    if tmatch:
        tm.mtp_draft_s = float(tmatch.group(1) or 0.0)
        tm.mtp_verify_s = float(tmatch.group(3) or 0.0)


def _run(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
         is_mtp: bool, mtp_chain: int, trials: int) -> List[TrialMeasurement]:
    prompt = make_prompt(prompt_tokens)
    ctx = cfg.ctx_for(prompt_tokens, n_decode)
    chain = mtp_chain if is_mtp else 0
    out: List[TrialMeasurement] = []
    try:
        with Qw3Server(cfg, ctx, n_decode, is_mtp, chain) as server:
            for _ in range(trials):
                out.append(_measure(server, cfg, prompt, n_decode, is_mtp, mtp_chain))
    except Exception as exc:  # noqa: BLE001
        return [TrialMeasurement(ok=False, error=f"server error: {exc}")]
    return out


def run_plain_trials(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
                     trials: int) -> List[TrialMeasurement]:
    return _run(cfg, prompt_tokens, n_decode, False, 0, trials)


def run_mtp_trials(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
                   mtp_chain: int, trials: int) -> List[TrialMeasurement]:
    return _run(cfg, prompt_tokens, n_decode, True, mtp_chain, trials)
