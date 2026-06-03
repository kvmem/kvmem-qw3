"""Result schema + JSON store for the qw3 vs llama.cpp benchmark suite.

One `ResultRow` per measured cell. A cell is identified by
`(engine, mode, prompt_tokens, n_decode, mtp_chain)`. Plain rows are
chain-independent, so they use `mtp_chain=0` and are shared across all
MTP chains in the report.

The store is a single JSON object (see `BenchStore`) written incrementally
during a sweep so a crash leaves a `partial: true` file that the report
renderer can still consume.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# Latency-source tags: how a TTFT/ITL number was obtained.
SRC_APPROX = "approx"          # derived from prefill_s / decode_s aggregates
SRC_STREAMED = "streamed"      # measured from streamed SSE chunk wall-clock
SRC_INSTRUMENTED = "instrumented"  # true per-token timestamps inside the engine


def cell_key(engine: str, mode: str, prompt_tokens: int,
             n_decode: int, mtp_chain: int) -> str:
    """Stable identity for a measured cell. Plain rows pass mtp_chain=0."""
    return f"{engine}|{mode}|{prompt_tokens}|{n_decode}|{mtp_chain}"


def _med(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(xs) if xs else 0.0


def _min_pos(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and x > 0]
    return min(xs) if xs else 0.0


def _max_pos(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and x > 0]
    return max(xs) if xs else 0.0


@dataclass
class ResultRow:
    # --- identity ---
    engine: str                 # "qw3" | "llama"
    mode: str                   # "plain" | "mtp"
    prompt_tokens: int          # target prompt-token count for the cell
    n_decode: int               # requested decode tokens
    mtp_chain: int = 0          # 0 for plain rows; >=1 for MTP

    # --- actual sizes (measured, may differ from targets) ---
    actual_prompt_tokens: int = 0
    decoded_tokens: int = 0

    # --- throughput (tok/s), aggregated over trials ---
    prefill_tok_s_med: float = 0.0
    prefill_tok_s_min: float = 0.0
    prefill_tok_s_max: float = 0.0
    decode_tok_s_med: float = 0.0
    decode_tok_s_min: float = 0.0
    decode_tok_s_max: float = 0.0

    # --- latency ---
    ttft_s_med: float = 0.0
    itl_ms_med: float = 0.0
    ttft_source: str = SRC_APPROX
    itl_source: str = SRC_APPROX

    # --- MTP (null/zero for plain rows) ---
    accept_rate: Optional[float] = None
    accept_per_step: Optional[List[float]] = None   # per chain-step acceptance
    accept_hist: Optional[Dict[str, int]] = None     # {"len0": n, "len1": n, ...}
    mtp_draft_s: Optional[float] = None
    mtp_verify_s: Optional[float] = None

    # --- memory ---
    peak_vram_mib: int = 0

    # --- provenance ---
    trials: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def key(self) -> str:
        return cell_key(self.engine, self.mode, self.prompt_tokens,
                        self.n_decode, self.mtp_chain)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultRow":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_trials(cls, engine: str, mode: str, prompt_tokens: int,
                    n_decode: int, mtp_chain: int,
                    trials: List["TrialMeasurement"]) -> "ResultRow":
        """Aggregate a list of per-trial measurements into one row."""
        ok = [t for t in trials if t.ok]
        row = cls(engine=engine, mode=mode, prompt_tokens=prompt_tokens,
                  n_decode=n_decode, mtp_chain=mtp_chain, trials=len(trials))
        row.raw = [t.to_dict() for t in trials]
        if not ok:
            row.error = "; ".join(t.error for t in trials if t.error) or "all trials failed"
            return row
        row.actual_prompt_tokens = ok[0].prompt_tokens
        row.decoded_tokens = ok[0].decoded_tokens
        row.prefill_tok_s_med = _med([t.prefill_tok_s for t in ok])
        row.prefill_tok_s_min = _min_pos([t.prefill_tok_s for t in ok])
        row.prefill_tok_s_max = _max_pos([t.prefill_tok_s for t in ok])
        row.decode_tok_s_med = _med([t.decode_tok_s for t in ok])
        row.decode_tok_s_min = _min_pos([t.decode_tok_s for t in ok])
        row.decode_tok_s_max = _max_pos([t.decode_tok_s for t in ok])
        row.ttft_s_med = _med([t.ttft_s for t in ok])
        row.itl_ms_med = _med([t.itl_ms for t in ok])
        row.ttft_source = ok[0].ttft_source
        row.itl_source = ok[0].itl_source
        row.peak_vram_mib = int(_max_pos([float(t.peak_vram_mib) for t in ok]))
        if mode == "mtp":
            row.accept_rate = _med([t.accept_rate for t in ok if t.accept_rate is not None])
            steps = [t.accept_per_step for t in ok if t.accept_per_step]
            if steps:
                width = max(len(s) for s in steps)
                row.accept_per_step = [
                    _med([s[i] for s in steps if i < len(s)]) for i in range(width)
                ]
            hist: Dict[str, int] = {}
            for t in ok:
                if t.accept_hist:
                    for k, v in t.accept_hist.items():
                        hist[k] = hist.get(k, 0) + v
            row.accept_hist = hist or None
            row.mtp_draft_s = _med([t.mtp_draft_s for t in ok if t.mtp_draft_s])
            row.mtp_verify_s = _med([t.mtp_verify_s for t in ok if t.mtp_verify_s])
        return row


@dataclass
class TrialMeasurement:
    """One engine invocation. Runners produce these; schema aggregates them."""
    ok: bool
    prompt_tokens: int = 0
    decoded_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    ttft_s: float = 0.0
    itl_ms: float = 0.0
    ttft_source: str = SRC_APPROX
    itl_source: str = SRC_APPROX
    accept_rate: Optional[float] = None
    accept_per_step: Optional[List[float]] = None
    accept_hist: Optional[Dict[str, int]] = None
    mtp_draft_s: Optional[float] = None
    mtp_verify_s: Optional[float] = None
    peak_vram_mib: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchStore:
    """Top-level JSON store. Written incrementally during a sweep."""
    git_commit: str = ""
    host: str = ""
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    rows: List[ResultRow] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    partial: bool = True
    schema_version: int = SCHEMA_VERSION

    def upsert(self, row: ResultRow) -> None:
        """Replace any existing row with the same key, else append."""
        for i, existing in enumerate(self.rows):
            if existing.key == row.key:
                self.rows[i] = row
                return
        self.rows.append(row)

    def add_error(self, key: str, message: str) -> None:
        self.errors.append({"cell_key": key, "message": message})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "host": self.host,
            "timestamp": self.timestamp,
            "config": self.config,
            "rows": [r.to_dict() for r in self.rows],
            "errors": self.errors,
            "partial": self.partial,
        }

    def save(self, path: Path) -> None:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BenchStore":
        d = json.loads(Path(path).read_text())
        store = cls(
            git_commit=d.get("git_commit", ""),
            host=d.get("host", ""),
            timestamp=d.get("timestamp", ""),
            config=d.get("config", {}),
            errors=d.get("errors", []),
            partial=d.get("partial", True),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )
        store.rows = [ResultRow.from_dict(r) for r in d.get("rows", [])]
        return store

    def row_index(self) -> Dict[str, ResultRow]:
        return {r.key: r for r in self.rows}
