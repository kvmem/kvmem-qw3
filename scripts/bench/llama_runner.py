"""Drive llama-server for one benchmark cell (plain or MTP speculative).

Starts llama-server once per cell (so model-load cost is amortized over trials),
sends `trials` streamed OpenAI-compatible completion requests with
`cache_prompt=false` (forces a fresh prefill each trial, no prefix-cache
contamination), and parses:
  - throughput from the response `timings` block,
  - TTFT from the wall-clock to the first streamed content chunk (`streamed`),
  - ITL from `timings.predicted_per_second`,
  - MTP acceptance from `timings.draft_n` / `draft_n_accepted`.

Reuses prompt synthesis (long_prompt_sweep.make_prompt) and the server
lifecycle helpers from probe_llama_mtp_server.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from long_prompt_sweep import make_prompt  # type: ignore  # noqa: E402
from probe_llama_mtp_server import (  # type: ignore  # noqa: E402
    wait_until_ready,
    stop_process_group,
)

from .schema import TrialMeasurement, SRC_STREAMED
from .vram import BackgroundVramPoller
from .config import BenchConfig

_MTP_LOG_MARKERS = (
    "common_speculative_impl_draft_mtp",
    "speculative decoding context initialized",
)


class LlamaServer:
    """Start/stop a llama-server for a given ctx + optional MTP chain."""

    def __init__(self, cfg: BenchConfig, ctx: int, n_decode: int,
                 mtp_chain: Optional[int]):
        self.cfg = cfg
        self.ctx = ctx
        self.n_decode = n_decode
        self.mtp_chain = mtp_chain
        self.proc: Optional[subprocess.Popen] = None
        self.log_tail = ""
        self._mtp_initialized = False

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.llama_host}:{self.cfg.llama_port}"

    def _cmd(self) -> list:
        cmd = [
            self.cfg.llama_server,
            "-m", self.cfg.model,
            "-ngl", "all",
            "-c", str(self.ctx),
            "-n", str(self.n_decode),
            "--host", self.cfg.llama_host,
            "--port", str(self.cfg.llama_port),
            "--no-webui",
            "--no-warmup",
        ]
        if self.mtp_chain is not None:
            cmd += ["--spec-type", "draft-mtp",
                    "--spec-draft-n-max", str(self.mtp_chain),
                    "--spec-draft-n-min", str(self.mtp_chain)]
        return cmd

    def __enter__(self) -> "LlamaServer":
        self.proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        wait_until_ready(self.proc, self.base_url, self.cfg.startup_timeout_s)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            full_log = stop_process_group(self.proc)
            # MTP-init markers print at startup, so scan the whole log, not a tail.
            self._mtp_initialized = all(m in full_log for m in _MTP_LOG_MARKERS)
            self.log_tail = full_log[-4000:]

    def mtp_initialized(self) -> bool:
        return self._mtp_initialized


def _stream_completion(base_url: str, model_name: str, prompt: str,
                       n_decode: int, timeout: float) -> dict:
    """POST a streamed completion. Returns dict with text, ttft_s, timings."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": n_decode,
        "temperature": 0,
        "stream": True,
        "cache_prompt": False,        # force fresh prefill each trial
        "ignore_eos": True,           # keep decode length comparable at 1K
        "timings_per_token": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft_s = 0.0
    text_parts: List[str] = []
    timings: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                piece = choices[0].get("text") or ""
                if piece and ttft_s == 0.0:
                    ttft_s = time.perf_counter() - t0
                text_parts.append(piece)
            if chunk.get("timings"):
                timings = chunk["timings"]
    return {"text": "".join(text_parts), "ttft_s": ttft_s, "timings": timings}


def _measure(server: LlamaServer, cfg: BenchConfig, prompt: str,
             n_decode: int, is_mtp: bool, mtp_chain: int) -> TrialMeasurement:
    model_name = Path(cfg.model).name
    timeout = cfg.timeout_for(server.ctx, n_decode)
    with BackgroundVramPoller() as vram:
        try:
            res = _stream_completion(server.base_url, model_name, prompt,
                                     n_decode, timeout)
        except Exception as exc:  # noqa: BLE001
            return TrialMeasurement(ok=False, error=f"request failed: {exc}")
        peak = vram.peak_mib
    timings = res["timings"]
    if not timings:
        return TrialMeasurement(ok=False, error="no timings in response")
    prompt_n = int(timings.get("prompt_n") or 0)
    predicted_n = int(timings.get("predicted_n") or 0)
    prompt_tok_s = float(timings.get("prompt_per_second") or 0.0)
    decode_tok_s = float(timings.get("predicted_per_second") or 0.0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    itl_ms = (predicted_ms / predicted_n) if predicted_n else 0.0

    tm = TrialMeasurement(
        ok=True,
        prompt_tokens=prompt_n,
        decoded_tokens=predicted_n,
        prefill_s=prompt_ms / 1000.0,
        decode_s=predicted_ms / 1000.0,
        prefill_tok_s=prompt_tok_s,
        decode_tok_s=decode_tok_s,
        ttft_s=res["ttft_s"],
        itl_ms=itl_ms,
        ttft_source=SRC_STREAMED,
        itl_source=SRC_STREAMED,
        peak_vram_mib=peak,
    )
    if is_mtp:
        draft_n = int(timings.get("draft_n") or 0)
        draft_acc = int(timings.get("draft_n_accepted") or 0)
        tm.accept_rate = (draft_acc / draft_n) if draft_n else 0.0
        # llama-server timings expose only aggregate draft stats, not per-step.
        tm.accept_per_step = None
        tm.accept_hist = None
    return tm


def _run(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
         is_mtp: bool, mtp_chain: int, trials: int) -> List[TrialMeasurement]:
    prompt = make_prompt(prompt_tokens)
    ctx = cfg.ctx_for(prompt_tokens, n_decode)
    chain = mtp_chain if is_mtp else None
    out: List[TrialMeasurement] = []
    try:
        with LlamaServer(cfg, ctx, n_decode, chain) as server:
            for _ in range(trials):
                out.append(_measure(server, cfg, prompt, n_decode, is_mtp, mtp_chain))
    except Exception as exc:  # noqa: BLE001
        return [TrialMeasurement(ok=False, error=f"server error: {exc}")]
    if is_mtp and not server.mtp_initialized():
        for tm in out:
            tm.ok = False
            tm.error = (tm.error + "; " if tm.error else "") + "MTP not initialized"
    return out


def run_plain_trials(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
                     trials: int) -> List[TrialMeasurement]:
    return _run(cfg, prompt_tokens, n_decode, False, 0, trials)


def run_mtp_trials(cfg: BenchConfig, prompt_tokens: int, n_decode: int,
                   mtp_chain: int, trials: int) -> List[TrialMeasurement]:
    return _run(cfg, prompt_tokens, n_decode, True, mtp_chain, trials)
