"""Trace I/O for immutable QW3 attention-mass JSONL files.

Reuses the segment-splitting semantics from ``scripts/attention_step_kl_curve.py``:
a new decode segment starts when the ``sample`` counter resets (decreases).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow importing sibling scripts under scripts/
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import attention_step_kl_curve as _kl  # noqa: E402


normalize = _kl.normalize
mean_dist = _kl.mean_dist
kl_divergence = _kl.kl_divergence
context_filter = _kl.context_filter
visible_filter = _kl.visible_filter
window_starts = _kl.window_starts
build_windows = _kl.build_windows


def read_trace(paths: list[Path] | Path) -> list[dict[str, Any]]:
    """Read one or more JSONL traces into decode segments (layer-mean mass)."""
    if isinstance(paths, Path):
        paths = [paths]
    return _kl.read_trace(list(paths))


def read_trace_with_meta(paths: list[Path] | Path) -> list[dict[str, Any]]:
    """Like ``read_trace`` but also keep first/last wall timestamps if present.

    Extra keys per segment (best-effort): ``t_first``, ``t_last``, ``n_rows``.
    """
    if isinstance(paths, Path):
        paths = [paths]

    segments: list[dict[str, Any]] = []
    current_layers: dict[int, list[dict[int, float]]] = defaultdict(list)
    current_seq_lens: dict[int, list[int]] = defaultdict(list)
    current_block_tokens: int | None = None
    t_first: float | None = None
    t_last: float | None = None
    n_rows = 0
    last_sample: int | None = None

    def flush() -> None:
        nonlocal current_block_tokens, t_first, t_last, n_rows
        if not current_layers:
            return
        samples: list[dict[int, float]] = []
        seq_lens: list[int] = []
        for sample in sorted(current_layers):
            samples.append(mean_dist(current_layers[sample]))
            seq_lens.append(min(current_seq_lens[sample]))
        if samples:
            seg: dict[str, Any] = {
                "samples": samples,
                "seq_lens": seq_lens,
                "block_tokens": current_block_tokens or 128,
                "n_rows": n_rows,
            }
            if t_first is not None:
                seg["t_first"] = t_first
            if t_last is not None:
                seg["t_last"] = t_last
            segments.append(seg)
        current_layers.clear()
        current_seq_lens.clear()
        current_block_tokens = None
        t_first = None
        t_last = None
        n_rows = 0

    for path in paths:
        flush()
        last_sample = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("kind") not in {"attention_mass", "kvmem_attention_mass"}:
                    continue
                sample = int(row["sample"])
                if last_sample is not None and sample < last_sample:
                    flush()
                last_sample = sample
                current_block_tokens = int(row.get("block_tokens", current_block_tokens or 128))
                block_ids = [int(x) for x in row["block_ids"]]
                mass = [float(x) for x in row["mass"]]
                current_layers[sample].append(normalize(dict(zip(block_ids, mass))))
                current_seq_lens[sample].append(int(row["seq_len"]))
                n_rows += 1
                for key in ("t", "ts", "timestamp", "wall_s", "time"):
                    if key in row:
                        try:
                            val = float(row[key])
                        except (TypeError, ValueError):
                            continue
                        if t_first is None:
                            t_first = val
                        t_last = val
                        break
        flush()
    return segments


def slice_segments(
    segments: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Inclusive slice of source segments by index."""
    return segments[start : end + 1]


def verify_sha256(path: Path, expected: str) -> bool:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected.lower()
