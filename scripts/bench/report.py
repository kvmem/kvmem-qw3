"""Render a BenchStore into a self-contained HTML report.

No external/CDN dependencies: inline CSS + server-side inline SVG charts.
The report is organized into:
  - header (commit, host, time, config, partial flag, any errors)
  - throughput tables + charts (qw3 vs llama, speedup), faceted by n_decode
  - latency (TTFT, ITL) tables + charts
  - peak VRAM chart
  - MTP section: acceptance vs chain, per-step acceptance, accept histogram,
    draft/verify timing split

All numbers are medians over trials. Latency source (approx/streamed) is
footnoted so qw3's prefill-derived TTFT is not mistaken for a true measurement.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import BenchStore, ResultRow

# ---------------------------------------------------------------------------
# Palette + tiny SVG primitives.

_QW3_COLOR = "#2563eb"      # blue
_LLAMA_COLOR = "#dc2626"    # red
_GRID = "#e5e7eb"
_AXIS = "#9ca3af"
_TEXT = "#111827"


def _esc(s) -> str:
    return html.escape(str(s))


def _fmt(x: Optional[float], nd: int = 1) -> str:
    if x is None or x == 0:
        return "—"
    return f"{x:.{nd}f}"


def _line_chart(title: str, x_vals: List[int], series: List[Tuple[str, str, List[Optional[float]]]],
                y_label: str, width: int = 540, height: int = 300,
                x_log: bool = True) -> str:
    """series: list of (label, color, y-values aligned to x_vals)."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 34, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_y = [y for _, _, ys in series for y in ys if y is not None and y > 0]
    if not all_y or not x_vals:
        return f'<div class="chart"><div class="chart-title">{_esc(title)}</div><div class="empty">no data</div></div>'
    y_max = max(all_y) * 1.1
    y_min = 0.0

    import math
    xs = x_vals
    if x_log:
        xs_t = [math.log10(max(1, v)) for v in x_vals]
    else:
        xs_t = [float(v) for v in x_vals]
    x_lo, x_hi = min(xs_t), max(xs_t)
    x_span = (x_hi - x_lo) or 1.0

    def px(i: int) -> float:
        return pad_l + (xs_t[i] - x_lo) / x_span * plot_w

    def py(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="svg-chart" '
             f'role="img" aria-label="{_esc(title)}">']
    # y gridlines + labels (5 ticks)
    for k in range(6):
        v = y_min + (y_max - y_min) * k / 5
        y = py(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" y2="{y:.1f}" '
                     f'stroke="{_GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" '
                     f'font-size="10" fill="{_AXIS}">{v:.0f}</text>')
    # x labels
    for i, xv in enumerate(xs):
        x = px(i)
        parts.append(f'<text x="{x:.1f}" y="{height-pad_b+14:.0f}" text-anchor="middle" '
                     f'font-size="10" fill="{_AXIS}">{_fmt_xtick(xv)}</text>')
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="{_AXIS}"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="{_AXIS}"/>')
    parts.append(f'<text x="14" y="{pad_t+plot_h/2:.0f}" font-size="10" fill="{_AXIS}" '
                 f'transform="rotate(-90 14 {pad_t+plot_h/2:.0f})" text-anchor="middle">{_esc(y_label)}</text>')
    # series polylines + points
    for label, color, ys in series:
        pts = [(px(i), py(y)) for i, y in enumerate(ys) if y is not None and y > 0]
        if len(pts) >= 2:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    parts.append('</svg>')
    legend = " ".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>{_esc(l)}</span>'
        for l, c, _ in series)
    return (f'<div class="chart"><div class="chart-title">{_esc(title)}</div>'
            f'{"".join(parts)}<div class="legend">{legend}</div></div>')


def _fmt_xtick(v: int) -> str:
    if v >= 1024 and v % 1024 == 0:
        return f"{v // 1024}K"
    if v >= 1000:
        return f"{v/1000:.0f}K" if v % 1000 == 0 else f"{v}"
    return str(v)


def _bar_chart(title: str, categories: List[str],
               series: List[Tuple[str, str, List[Optional[float]]]],
               y_label: str, width: int = 540, height: int = 300) -> str:
    """Grouped bar chart. categories on x; series share each category slot."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 34, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_y = [y for _, _, ys in series for y in ys if y is not None]
    if not all_y or not categories:
        return f'<div class="chart"><div class="chart-title">{_esc(title)}</div><div class="empty">no data</div></div>'
    y_max = (max(all_y) or 1.0) * 1.15
    n_cat = len(categories)
    n_ser = len(series)
    slot = plot_w / n_cat
    bar_w = slot / (n_ser + 1)

    def py(v: float) -> float:
        return pad_t + plot_h - (v / y_max) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="svg-chart" role="img" aria-label="{_esc(title)}">']
    for k in range(6):
        v = y_max * k / 5
        y = py(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" y2="{y:.1f}" stroke="{_GRID}"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="{_AXIS}">{v:.2f}</text>')
    for ci, cat in enumerate(categories):
        slot_x = pad_l + ci * slot
        for si, (label, color, ys) in enumerate(series):
            v = ys[ci] if ci < len(ys) and ys[ci] is not None else 0.0
            bx = slot_x + (si + 0.5) * bar_w + 0.25 * bar_w
            by = py(v)
            bh = pad_t + plot_h - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.9:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{slot_x + slot/2:.1f}" y="{height-pad_b+14:.0f}" text-anchor="middle" '
                     f'font-size="10" fill="{_AXIS}">{_esc(cat)}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="{_AXIS}"/>')
    parts.append(f'<text x="14" y="{pad_t+plot_h/2:.0f}" font-size="10" fill="{_AXIS}" '
                 f'transform="rotate(-90 14 {pad_t+plot_h/2:.0f})" text-anchor="middle">{_esc(y_label)}</text>')
    parts.append('</svg>')
    legend = " ".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>{_esc(l)}</span>'
        for l, c, _ in series)
    return (f'<div class="chart"><div class="chart-title">{_esc(title)}</div>'
            f'{"".join(parts)}<div class="legend">{legend}</div></div>')


# ---------------------------------------------------------------------------
# Data shaping.

class RowLookup:
    def __init__(self, store: BenchStore):
        self.idx: Dict[str, ResultRow] = store.row_index()

    def get(self, engine: str, mode: str, p: int, n: int, chain: int) -> Optional[ResultRow]:
        from .schema import cell_key
        return self.idx.get(cell_key(engine, mode, p, n, chain))


def _series_over_prompts(lk: RowLookup, engine: str, mode: str, n: int,
                         chain: int, prompts: List[int],
                         attr: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for p in prompts:
        r = lk.get(engine, mode, p, n, chain)
        out.append(getattr(r, attr) if (r and not r.error) else None)
    return out


def _speedup_series(lk: RowLookup, mode: str, n: int, chain: int,
                    prompts: List[int], attr: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for p in prompts:
        q = lk.get("qw3", mode, p, n, chain)
        l = lk.get("llama", mode, p, n, chain)
        if q and l and not q.error and not l.error:
            lv = getattr(l, attr)
            out.append(getattr(q, attr) / lv if lv else None)
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Tables.

def _throughput_table(lk: RowLookup, prompts: List[int], n: int,
                      mode: str, chain: int) -> str:
    rows = []
    for p in prompts:
        q = lk.get("qw3", mode, p, n, chain)
        l = lk.get("llama", mode, p, n, chain)
        def cell(r, attr):
            return _fmt(getattr(r, attr)) if (r and not r.error) else "—"
        def speed(attr):
            if q and l and not q.error and not l.error and getattr(l, attr):
                return f'{getattr(q, attr)/getattr(l, attr)*100:.0f}%'
            return "—"
        rows.append(
            f"<tr><td>{_fmt_xtick(p)}</td>"
            f"<td>{cell(q,'prefill_tok_s_med')}</td><td>{cell(l,'prefill_tok_s_med')}</td>"
            f"<td class='sp'>{speed('prefill_tok_s_med')}</td>"
            f"<td>{cell(q,'decode_tok_s_med')}</td><td>{cell(l,'decode_tok_s_med')}</td>"
            f"<td class='sp'>{speed('decode_tok_s_med')}</td></tr>")
    return (
        "<table><thead><tr>"
        "<th>prompt</th>"
        "<th>qw3 prefill</th><th>llama prefill</th><th>qw3/llama</th>"
        "<th>qw3 decode</th><th>llama decode</th><th>qw3/llama</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _latency_table(lk: RowLookup, prompts: List[int], n: int,
                   mode: str, chain: int) -> str:
    rows = []
    for p in prompts:
        q = lk.get("qw3", mode, p, n, chain)
        l = lk.get("llama", mode, p, n, chain)
        def cell(r, attr, nd):
            return _fmt(getattr(r, attr), nd) if (r and not r.error) else "—"
        rows.append(
            f"<tr><td>{_fmt_xtick(p)}</td>"
            f"<td>{cell(q,'ttft_s_med',3)}</td><td>{cell(l,'ttft_s_med',3)}</td>"
            f"<td>{cell(q,'itl_ms_med',1)}</td><td>{cell(l,'itl_ms_med',1)}</td>"
            f"<td>{q.peak_vram_mib if q and not q.error else '—'}</td>"
            f"<td>{l.peak_vram_mib if l and not l.error else '—'}</td></tr>")
    return (
        "<table><thead><tr>"
        "<th>prompt</th><th>qw3 TTFT(s)</th><th>llama TTFT(s)</th>"
        "<th>qw3 ITL(ms)</th><th>llama ITL(ms)</th>"
        "<th>qw3 VRAM(MiB)</th><th>llama VRAM(MiB)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


# ---------------------------------------------------------------------------
# MTP section.

def _mtp_acceptance_chart(lk: RowLookup, prompts: List[int], n: int,
                          chains: List[int]) -> str:
    """Acceptance vs chain, one series per engine (median over prompts)."""
    import statistics
    cats = [f"chain={c}" for c in chains]
    def avg_accept(engine: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for c in chains:
            vals = []
            for p in prompts:
                r = lk.get(engine, "mtp", p, n, c)
                if r and not r.error and r.accept_rate is not None:
                    vals.append(r.accept_rate)
            out.append(statistics.median(vals) if vals else None)
        return out
    series = [("qw3", _QW3_COLOR, avg_accept("qw3")),
              ("llama", _LLAMA_COLOR, avg_accept("llama"))]
    return _bar_chart(f"MTP acceptance vs chain (n_decode={n})", cats, series,
                      "accept rate")


def _mtp_per_step_table(lk: RowLookup, prompts: List[int], n: int,
                        chains: List[int]) -> str:
    rows = []
    for c in chains:
        for p in prompts:
            r = lk.get("qw3", "mtp", p, n, c)
            if not r or r.error or not r.accept_per_step:
                continue
            steps = " ".join(f"{v:.2f}" for v in r.accept_per_step)
            split = ""
            if r.mtp_draft_s and r.mtp_verify_s:
                split = f"draft {r.mtp_draft_s:.3f}s / verify {r.mtp_verify_s:.3f}s"
            hist = ""
            if r.accept_hist:
                hist = " ".join(f"{k}={v}" for k, v in sorted(r.accept_hist.items()))
            rows.append(
                f"<tr><td>{c}</td><td>{_fmt_xtick(p)}</td>"
                f"<td>{_fmt(r.accept_rate,3)}</td><td class='mono'>{steps}</td>"
                f"<td class='mono'>{hist}</td><td class='mono'>{split}</td></tr>")
    if not rows:
        return "<p class='empty'>no qw3 MTP per-step data</p>"
    return ("<table><thead><tr><th>chain</th><th>prompt</th><th>accept</th>"
            "<th>per-step accept</th><th>accept-len hist</th>"
            "<th>draft/verify split</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


# ---------------------------------------------------------------------------
# Top-level render.

_CSS = """
:root{--qw3:#2563eb;--llama:#dc2626;}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:#111827;margin:0;padding:24px;background:#f9fafb}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:28px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:4px}
h3{font-size:14px;margin:18px 0 8px;color:#374151}
.meta{color:#6b7280;font-size:12px;margin-bottom:8px}
.meta code{background:#eef2ff;padding:1px 5px;border-radius:4px;color:#3730a3}
.warn{background:#fef3c7;border:1px solid #f59e0b;padding:8px 12px;border-radius:6px;margin:8px 0}
.err{background:#fee2e2;border:1px solid #ef4444;padding:8px 12px;border-radius:6px;margin:8px 0;font-size:12px}
table{border-collapse:collapse;margin:8px 0 16px;background:#fff;font-size:13px;
  box-shadow:0 1px 2px rgba(0,0,0,.06);border-radius:6px;overflow:hidden}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #f3f4f6}
th{background:#f3f4f6;color:#374151;font-weight:600}
td:first-child,th:first-child{text-align:left}
td.sp{color:#6b21a8;font-weight:600}
td.mono,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;text-align:left}
.charts{display:flex;flex-wrap:wrap;gap:16px}
.chart{background:#fff;border-radius:8px;padding:10px 12px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.chart-title{font-size:13px;font-weight:600;color:#374151;margin-bottom:4px}
.svg-chart{width:540px;height:300px;max-width:100%}
.legend{font-size:11px;color:#6b7280;margin-top:4px}
.lg{margin-right:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.empty{color:#9ca3af;font-style:italic;padding:20px;text-align:center}
.facet{margin:14px 0 22px}
footer{margin-top:30px;color:#9ca3af;font-size:11px}
"""


def render_report(store: BenchStore, out_path: Path) -> Path:
    cfg = store.config
    prompts = list(cfg.get("prompt_tokens", []))
    n_decodes = list(cfg.get("n_decode", []))
    chains = list(cfg.get("mtp_chain", []))
    lk = RowLookup(store)

    P: List[str] = []
    P.append(f"<!doctype html><html><head><meta charset='utf-8'>")
    P.append(f"<title>qw3 vs llama.cpp benchmark</title><style>{_CSS}</style></head><body>")
    P.append("<h1>qw3 vs llama.cpp — benchmark report</h1>")
    P.append(f"<div class='meta'>commit <code>{_esc(store.git_commit)}</code> · "
             f"host <code>{_esc(store.host)}</code> · {_esc(store.timestamp)} · "
             f"model <code>{_esc(Path(cfg.get('model','')).name)}</code> · "
             f"trials {cfg.get('trials')}</div>")
    if store.partial:
        P.append("<div class='warn'>⚠ partial run — sweep did not complete; "
                 "some cells may be missing.</div>")
    if store.errors:
        P.append(f"<div class='err'><b>{len(store.errors)} cell error(s):</b><br>" +
                 "<br>".join(_esc(f"{e['cell_key']}: {e['message'][:160]}")
                             for e in store.errors[:20]) + "</div>")

    # ---- Plain throughput ----
    P.append("<h2>Plain decode — throughput</h2>")
    for n in n_decodes:
        P.append(f"<div class='facet'><h3>n_decode = {n} (tok/s)</h3>")
        P.append(_throughput_table(lk, prompts, n, "plain", 0))
        P.append("<div class='charts'>")
        P.append(_line_chart(
            f"Decode tok/s vs prompt (n={n})", prompts,
            [("qw3", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "decode_tok_s_med")),
             ("llama", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "decode_tok_s_med"))],
            "tok/s"))
        P.append(_line_chart(
            f"Prefill tok/s vs prompt (n={n})", prompts,
            [("qw3", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "prefill_tok_s_med")),
             ("llama", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "prefill_tok_s_med"))],
            "tok/s"))
        P.append("</div></div>")

    # ---- Latency + VRAM (plain) ----
    P.append("<h2>Plain decode — latency &amp; VRAM</h2>")
    for n in n_decodes:
        P.append(f"<div class='facet'><h3>n_decode = {n}</h3>")
        P.append(_latency_table(lk, prompts, n, "plain", 0))
        P.append("<div class='charts'>")
        P.append(_line_chart(
            f"TTFT (s) vs prompt (n={n})", prompts,
            [("qw3", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "ttft_s_med")),
             ("llama", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "ttft_s_med"))],
            "seconds"))
        P.append(_line_chart(
            f"Peak VRAM (MiB) vs prompt (n={n})", prompts,
            [("qw3", _QW3_COLOR, [float(v) if v else None for v in _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "peak_vram_mib")]),
             ("llama", _LLAMA_COLOR, [float(v) if v else None for v in _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "peak_vram_mib")])],
            "MiB"))
        P.append("</div></div>")

    # ---- MTP ----
    if chains:
        P.append("<h2>MTP speculative decode</h2>")
        for n in n_decodes:
            P.append(f"<div class='facet'><h3>n_decode = {n}</h3>")
            P.append("<div class='charts'>")
            P.append(_mtp_acceptance_chart(lk, prompts, n, chains))
            # decode tok/s vs prompt, one series per chain for qw3
            chain_series = []
            palette = ["#2563eb", "#7c3aed", "#0891b2", "#059669", "#ca8a04"]
            for i, c in enumerate(chains):
                chain_series.append((f"qw3 c{c}", palette[i % len(palette)],
                                     _series_over_prompts(lk, "qw3", "mtp", n, c, prompts, "decode_tok_s_med")))
            P.append(_line_chart(f"qw3 MTP decode tok/s by chain (n={n})", prompts,
                                 chain_series, "tok/s"))
            llama_chain_series = []
            lpal = ["#dc2626", "#ea580c", "#db2777", "#b91c1c", "#9a3412"]
            for i, c in enumerate(chains):
                llama_chain_series.append((f"llama c{c}", lpal[i % len(lpal)],
                                           _series_over_prompts(lk, "llama", "mtp", n, c, prompts, "decode_tok_s_med")))
            P.append(_line_chart(f"llama MTP decode tok/s by chain (n={n})", prompts,
                                 llama_chain_series, "tok/s"))
            P.append("</div>")
            for c in chains:
                P.append(f"<h3>chain={c} throughput (tok/s)</h3>")
                P.append(_throughput_table(lk, prompts, n, "mtp", c))
            P.append("<h3>qw3 MTP detail (per-step acceptance · accept-length histogram · draft/verify split)</h3>")
            P.append(_mtp_per_step_table(lk, prompts, n, chains))
            P.append("</div>")

    P.append("<footer>TTFT/ITL source: qw3 = <i>approx</i> (ttft≈prefill_s, "
             "itl≈decode_s/decoded; faithful because qw3's first decode token "
             "argmax comes from prefill); llama = <i>streamed</i> (first-chunk "
             "wall-clock + server timings). Throughput/acceptance are medians "
             "over trials. llama-server exposes only aggregate draft acceptance, "
             "so per-step/histogram columns are qw3-only.</footer>")
    P.append("</body></html>")

    out_path.write_text("".join(P), encoding="utf-8")
    return out_path
