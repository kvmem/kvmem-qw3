"""Sweep orchestrator: walk the grid, run trials, checkpoint after every row.

Both engines now run one server per cell, looping `trials` requests internally
so the 27B model loads once per cell (not once per trial):
  - qw3 runs `qw3 serve` (OpenAI-compatible HTTP) and parses per-request timing.
  - llama runs `llama-server` and reads its `timings` block.

Per cell we alternate which engine goes first to spread thermal drift, and wait
for the GPU to return near idle between engine switches so peak-VRAM readings
are not contaminated by the previous engine's teardown.

The store is saved to disk after every completed row, so a crash mid-sweep
leaves a `partial: true` JSON that the report renderer can still consume.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Callable, List

from .config import BenchConfig, host_label
from .schema import BenchStore, ResultRow, TrialMeasurement, cell_key
from .vram import wait_for_idle
from . import qw3_runner, llama_runner, utility_runner


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _qw3_plain_trials(cfg: BenchConfig, p: int, n: int) -> List[TrialMeasurement]:
    return qw3_runner.run_plain_trials(cfg, p, n, cfg.trials)


def _qw3_mtp_trials(cfg: BenchConfig, p: int, n: int, chain: int) -> List[TrialMeasurement]:
    return qw3_runner.run_mtp_trials(cfg, p, n, chain, cfg.trials)


class Orchestrator:
    def __init__(self, cfg: BenchConfig, out_json: Path, log=print,
                 resume: bool = False, force: bool = False):
        self.cfg = cfg
        self.out_json = out_json
        self.log = log
        self.force = force
        if resume and out_json.exists():
            self.store = BenchStore.load(out_json)
            self.store.config = cfg.to_dict()
            self.store.partial = True
        else:
            self.store = BenchStore(
                git_commit=_git_commit(),
                host=host_label(),
                timestamp=_now(),
                config=cfg.to_dict(),
                partial=True,
            )

    def _checkpoint(self) -> None:
        self.store.save(self.out_json)

    def _record(self, engine: str, mode: str, p: int, n: int, chain: int,
                trials: List[TrialMeasurement]) -> None:
        row = ResultRow.from_trials(engine, mode, p, n, chain, trials)
        self.store.upsert(row)
        if row.error:
            self.store.add_error(row.key, row.error)
            self.log(f"    {engine:5s} {mode:5s} chain={chain}  ERROR: {row.error[:120]}")
        else:
            extra = ""
            if mode == "mtp" and row.accept_rate is not None:
                extra = f"  accept={row.accept_rate:.3f}"
            self.log(f"    {engine:5s} {mode:5s} chain={chain}  "
                     f"prefill={row.prefill_tok_s_med:8.1f}  "
                     f"decode={row.decode_tok_s_med:7.2f}  "
                     f"ttft={row.ttft_s_med:.3f}s  vram={row.peak_vram_mib}{extra}")
        self._checkpoint()

    def _has_completed(self, engine: str, mode: str, p: int, n: int, chain: int) -> bool:
        if self.force:
            return False
        row = self.store.row_index().get(cell_key(engine, mode, p, n, chain))
        return bool(row and not row.error and row.trials > 0)

    def _run_engine_plain(self, engine: str, p: int, n: int) -> None:
        if self._has_completed(engine, "plain", p, n, 0):
            self.log(f"    {engine:5s} plain chain=0  SKIP existing")
            return
        wait_for_idle()
        if engine == "qw3":
            self._record("qw3", "plain", p, n, 0, _qw3_plain_trials(self.cfg, p, n))
        else:
            self._record("llama", "plain", p, n, 0,
                         llama_runner.run_plain_trials(self.cfg, p, n, self.cfg.trials))

    def _run_engine_mtp(self, engine: str, p: int, n: int, chain: int) -> None:
        if self._has_completed(engine, "mtp", p, n, chain):
            self.log(f"    {engine:5s} mtp   chain={chain}  SKIP existing")
            return
        wait_for_idle()
        if engine == "qw3":
            self._record("qw3", "mtp", p, n, chain,
                         _qw3_mtp_trials(self.cfg, p, n, chain))
        else:
            self._record("llama", "mtp", p, n, chain,
                         llama_runner.run_mtp_trials(self.cfg, p, n, chain, self.cfg.trials))

    def run(self) -> BenchStore:
        cfg = self.cfg
        mode_cells = (1 if cfg.run_plain else 0) + (len(cfg.mtp_chain) if cfg.run_mtp else 0)
        total_cells = len(cfg.prompt_tokens) * len(cfg.n_decode) * mode_cells * len(cfg.engines)
        self.log(f"sweep: {total_cells} cells  "
                 f"(prompts={cfg.prompt_tokens} n={cfg.n_decode} "
                 f"chains={cfg.mtp_chain} engines={cfg.engines} trials={cfg.trials})")
        self._checkpoint()
        flip = False
        for p in cfg.prompt_tokens:
            for n in cfg.n_decode:
                self.log(f"\n[cell] prompt_tokens={p} n_decode={n}")
                order = list(cfg.engines)
                if flip:
                    order = order[::-1]
                flip = not flip
                # Plain rows first (chain-independent), then each MTP chain.
                if cfg.run_plain:
                    for engine in order:
                        self._run_engine_plain(engine, p, n)
                if cfg.run_mtp:
                    for chain in cfg.mtp_chain:
                        for engine in order:
                            self._run_engine_mtp(engine, p, n, chain)
        self._run_utility()
        self.store.partial = False
        self._checkpoint()
        self.log(f"\ndone: {len(self.store.rows)} rows, "
                 f"{len(self.store.errors)} errors -> {self.out_json}")
        return self.store

    def _run_utility(self) -> None:
        cfg = self.cfg
        if not cfg.run_utility:
            return
        if not self.force and self.store.utility is not None and self.store.utility.ok:
            self.log(f"\n[utility] SKIP existing ({cfg.kv_dtype})")
            return
        self.log(f"\n[utility] kv_dtype={cfg.kv_dtype}  "
                 f"passkey lens={cfg.passkey_lens} depths={cfg.passkey_depths} "
                 f"trials={cfg.passkey_trials}  gsm8k_n={cfg.gsm8k_n}")
        wait_for_idle()
        result = utility_runner.run_utility(cfg)
        self.store.utility = result
        if not result.ok:
            self.log(f"    utility ERROR: {result.error}")
        else:
            pk_hits = sum(c.hits for c in result.passkey_cells)
            pk_tot = sum(c.trials for c in result.passkey_cells)
            acc = result.gsm8k_acc
            acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a"
            self.log(f"    passkey {pk_hits}/{pk_tot}  gsm8k {acc_s}")
        self._checkpoint()
