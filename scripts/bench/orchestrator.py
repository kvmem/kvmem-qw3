"""Sweep orchestrator: walk the grid, run trials, checkpoint after every row.

Engine asymmetry handled here:
  - qw3 runs one fresh process per trial (qw3_runner returns one measurement).
  - llama runs one server per cell, looping `trials` requests internally.

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
from . import qw3_runner, llama_runner


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _qw3_plain_trials(cfg: BenchConfig, p: int, n: int) -> List[TrialMeasurement]:
    return [qw3_runner.run_plain(cfg, p, n) for _ in range(cfg.trials)]


def _qw3_mtp_trials(cfg: BenchConfig, p: int, n: int, chain: int) -> List[TrialMeasurement]:
    return [qw3_runner.run_mtp(cfg, p, n, chain) for _ in range(cfg.trials)]


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
        self.store.partial = False
        self._checkpoint()
        self.log(f"\ndone: {len(self.store.rows)} rows, "
                 f"{len(self.store.errors)} errors -> {self.out_json}")
        return self.store
