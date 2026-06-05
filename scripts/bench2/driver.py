"""Bench driver: walk a prompt-length grid against pre-launched server URLs.

Each endpoint is a (label, base_url) the user already started (e.g. an fp16
server on :8080 and an fp8 server on :8082). For every (endpoint, prompt_len)
cell we run `trials` measurements, sample peak VRAM around them, and keep the
median prefill/decode. Results are written to JSON for the report renderer.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple

from . import client
from .util import make_prompt, VramPoller, smi_used_mib


@dataclass
class Cell:
    endpoint: str
    prompt_tokens: int
    n_decode: int
    ok: bool = True
    error: str = ""
    method: str = ""
    decoded_tokens: int = 0
    prefill_s_med: float = 0.0
    decode_s_med: float = 0.0
    prefill_tok_s_med: float = 0.0
    decode_tok_s_med: float = 0.0
    ttft_s_med: float = 0.0
    peak_vram_mib: int = 0
    trials: int = 0


@dataclass
class BenchResult:
    endpoints: Dict[str, str] = field(default_factory=dict)  # label -> base_url
    prompt_tokens: List[int] = field(default_factory=list)
    n_decode: int = 128
    trials: int = 3
    method: str = "stream"
    cells: List[Cell] = field(default_factory=list)


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def run_cell(label: str, base_url: str, model: str, prompt_tokens: int,
             n_decode: int, trials: int, method: str,
             timeout: float, log: Callable[[str], None]) -> Cell:
    prompt = make_prompt(prompt_tokens)
    measure = client.measure_stream if method == "stream" else client.measure_two_point
    ok_t: List[client.Timing] = []
    peak = 0
    last_err = ""
    with VramPoller() as vram:
        for _ in range(trials):
            tm = measure(base_url, model, prompt, n_decode,
                         prompt_tokens_hint=prompt_tokens, timeout=timeout)
            if tm.ok:
                ok_t.append(tm)
            else:
                last_err = tm.error
        peak = vram.peak_mib
    if not ok_t:
        return Cell(endpoint=label, prompt_tokens=prompt_tokens, n_decode=n_decode,
                    ok=False, error=last_err or "all trials failed",
                    method=method, peak_vram_mib=peak)
    cell = Cell(
        endpoint=label, prompt_tokens=prompt_tokens, n_decode=n_decode, ok=True,
        method=method, decoded_tokens=ok_t[-1].decoded_tokens,
        prefill_s_med=_median([t.prefill_s for t in ok_t]),
        decode_s_med=_median([t.decode_s for t in ok_t]),
        prefill_tok_s_med=_median([t.prefill_tok_s for t in ok_t]),
        decode_tok_s_med=_median([t.decode_tok_s for t in ok_t]),
        ttft_s_med=_median([t.ttft_s for t in ok_t]),
        peak_vram_mib=peak, trials=len(ok_t))
    log(f"    {label:10s} p={prompt_tokens:>7} n={n_decode:>5}  "
        f"prefill={cell.prefill_tok_s_med:8.1f} tok/s  "
        f"decode={cell.decode_tok_s_med:7.2f} tok/s  "
        f"ttft={cell.ttft_s_med:.3f}s  vram={cell.peak_vram_mib} MiB  ({cell.trials}/{trials})")
    return cell


def run_grid(endpoints: List[Tuple[str, str]], prompt_tokens: List[int],
             n_decode: int, trials: int, method: str, timeout: float,
             log: Callable[[str], None] = print) -> BenchResult:
    ep_map = {label: url for label, url in endpoints}
    models: Dict[str, str] = {}
    for label, url in endpoints:
        models[label] = client.wait_ready(url)
        log(f"endpoint {label}: {url}  model={models[label]}")
    result = BenchResult(endpoints=ep_map, prompt_tokens=list(prompt_tokens),
                         n_decode=n_decode, trials=trials, method=method)
    for p in prompt_tokens:
        log(f"\n[prompt_tokens={p}]")
        for label, url in endpoints:
            result.cells.append(
                run_cell(label, url, models[label], p, n_decode,
                         trials, method, timeout, log))
    return result


def to_dict(result: BenchResult) -> dict:
    d = asdict(result)
    return d
