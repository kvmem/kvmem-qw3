"""Render a side-by-side text/markdown report from a BenchResult dict.

Endpoints (e.g. fp16 vs fp8) are columns; prompt lengths are rows. Shows
prefill tok/s, decode tok/s, TTFT, and peak VRAM, plus an fp8-vs-fp16-style
ratio when exactly two endpoints are present.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def _cells_by(result: dict) -> Dict[Tuple[str, int], dict]:
    return {(c["endpoint"], c["prompt_tokens"]): c for c in result["cells"]}


def render_text(result: dict) -> str:
    labels: List[str] = list(result["endpoints"].keys())
    prompts: List[int] = result["prompt_tokens"]
    idx = _cells_by(result)
    lines: List[str] = []
    lines.append(f"# bench2  n_decode={result['n_decode']}  trials={result['trials']}  "
                 f"method={result['method']}")
    for label in labels:
        lines.append(f"#   {label}: {result['endpoints'][label]}")
    lines.append("")

    for metric, key, fmt in (
        ("prefill tok/s", "prefill_tok_s_med", "{:>12.1f}"),
        ("decode tok/s",  "decode_tok_s_med",  "{:>12.2f}"),
        ("ttft (s)",      "ttft_s_med",        "{:>12.3f}"),
        ("peak VRAM MiB", "peak_vram_mib",     "{:>12d}"),
    ):
        lines.append(f"## {metric}")
        header = f"{'prompt_tok':>12}" + "".join(f"{l:>12}" for l in labels)
        if len(labels) == 2:
            header += f"{'ratio':>12}"
        lines.append(header)
        for p in prompts:
            row = f"{p:>12}"
            vals: List[float] = []
            for label in labels:
                c = idx.get((label, p))
                if c and c["ok"]:
                    v = c[key]
                    row += fmt.format(v)
                    vals.append(float(v))
                else:
                    row += f"{'ERR':>12}"
                    vals.append(float("nan"))
            if len(labels) == 2 and all(v == v for v in vals) and vals[0] != 0:
                row += f"{vals[1] / vals[0]:>12.3f}"
            lines.append(row)
        lines.append("")

    errs = [c for c in result["cells"] if not c["ok"]]
    if errs:
        lines.append("## errors")
        for c in errs:
            lines.append(f"  {c['endpoint']:10s} p={c['prompt_tokens']}: {c['error'][:160]}")
    return "\n".join(lines)
