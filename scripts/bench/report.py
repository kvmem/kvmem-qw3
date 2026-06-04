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
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import BenchStore, ResultRow

# ---------------------------------------------------------------------------
# Palette + tiny SVG primitives.

_QW3_COLOR = "#2563eb"      # blue
_LLAMA_COLOR = "#dc2626"    # red
_CHAIN_COLORS = ["#2563eb", "#7c3aed", "#0891b2", "#059669", "#ca8a04", "#db2777"]
_LLAMA_CHAIN_COLORS = ["#dc2626", "#ea580c", "#db2777", "#b91c1c", "#9a3412", "#be123c"]
_GRID = "#e5e7eb"
_AXIS = "#9ca3af"
_TEXT = "#111827"


def _esc(s) -> str:
    return html.escape(str(s))


def _fmt(x: Optional[float], nd: int = 1) -> str:
    if x is None or x == 0:
        return "—"
    return f"{x:.{nd}f}"


def _fmt_gb(mib: Optional[float]) -> str:
    if mib is None or mib == 0:
        return "—"
    return f"{mib / 1024.0:.2f}"


def _ok(r: Optional[ResultRow]) -> bool:
    return bool(r and not r.error)


def _status(r: Optional[ResultRow]) -> str:
    if r is None:
        return "—"
    return "fail" if r.error else "ok"


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
        if "GB" in y_label:
            tick = f"{v:.2f}"
        elif y_max < 20:
            tick = f"{v:.2f}"
        elif y_max < 100:
            tick = f"{v:.1f}"
        else:
            tick = f"{v:.0f}"
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" '
                     f'font-size="10" fill="{_AXIS}">{tick}</text>')
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
        tick = f"{v:.2f}" if ("GB" in y_label or y_max < 20) else (f"{v:.1f}" if y_max < 100 else f"{v:.0f}")
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="{_AXIS}">{tick}</text>')
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
            f"<td>{_status(q)}</td><td>{_status(l)}</td>"
            f"<td>{cell(q,'prefill_tok_s_med')}</td><td>{cell(l,'prefill_tok_s_med')}</td>"
            f"<td class='sp'>{speed('prefill_tok_s_med')}</td>"
            f"<td>{cell(q,'decode_tok_s_med')}</td><td>{cell(l,'decode_tok_s_med')}</td>"
            f"<td class='sp'>{speed('decode_tok_s_med')}</td></tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>prompt</th>"
        "<th>qw3</th><th>llama</th>"
        "<th>qw3 prefill</th><th>llama prefill</th><th>qw3/llama</th>"
        "<th>qw3 decode</th><th>llama decode</th><th>qw3/llama</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


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
            f"<td>{_status(q)}</td><td>{_status(l)}</td>"
            f"<td>{cell(q,'ttft_s_med',3)}</td><td>{cell(l,'ttft_s_med',3)}</td>"
            f"<td>{cell(q,'itl_ms_med',1)}</td><td>{cell(l,'itl_ms_med',1)}</td>"
            f"<td>{_fmt_gb(q.peak_vram_mib if q and not q.error else None)}</td>"
            f"<td>{_fmt_gb(l.peak_vram_mib if l and not l.error else None)}</td></tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>prompt</th><th>qw3</th><th>llama</th><th>qw3 TTFT(s)</th><th>llama TTFT(s)</th>"
        "<th>qw3 ITL(ms)</th><th>llama ITL(ms)</th>"
        "<th>qw3 VRAM(GB)</th><th>llama VRAM(GB)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


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
    return ("<div class='table-wrap'><table><thead><tr><th>chain</th><th>prompt</th><th>accept</th>"
            "<th>per-step accept</th><th>accept-len hist</th>"
            "<th>draft/verify split</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def _mtp_chain_sweep_table(lk: RowLookup, prompts: List[int], n: int,
                           chains: List[int]) -> str:
    rows = []
    for p in prompts:
        cells = []
        q_best: Optional[ResultRow] = None
        l_best: Optional[ResultRow] = None
        for c in chains:
            q = lk.get("qw3", "mtp", p, n, c)
            l = lk.get("llama", "mtp", p, n, c)
            if _ok(q) and (q_best is None or q.decode_tok_s_med > q_best.decode_tok_s_med):
                q_best = q
            if _ok(l) and (l_best is None or l.decode_tok_s_med > l_best.decode_tok_s_med):
                l_best = l
            def pack(r: Optional[ResultRow]) -> str:
                if r is None:
                    return "—"
                if r.error:
                    return "fail"
                acc = _fmt(r.accept_rate, 3)
                return f"{_fmt(r.decode_tok_s_med, 2)} / {acc} / {_fmt_gb(r.peak_vram_mib)}"
            cells.append(f"<td>{pack(q)}</td><td>{pack(l)}</td>")
        ratio = "—"
        if q_best and l_best and l_best.decode_tok_s_med:
            ratio = f"{q_best.decode_tok_s_med / l_best.decode_tok_s_med:.2f}x"
        q_label = f"c{q_best.mtp_chain} {_fmt(q_best.decode_tok_s_med,2)}" if q_best else "—"
        l_label = f"c{l_best.mtp_chain} {_fmt(l_best.decode_tok_s_med,2)}" if l_best else "—"
        rows.append(
            f"<tr><td>{_fmt_xtick(p)}</td>{''.join(cells)}"
            f"<td class='sp'>{q_label}</td><td class='sp'>{l_label}</td><td class='sp'>{ratio}</td></tr>")
    head = "".join(
        f"<th>qw3 c{c}<br><span>tok/s / accept / GB</span></th>"
        f"<th>llama c{c}<br><span>tok/s / accept / GB</span></th>"
        for c in chains)
    return (
        "<div class='table-wrap'><table class='wide'><thead><tr><th>prompt</th>"
        + head +
        "<th>qw3 best</th><th>llama best</th><th>best ratio</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _decode_by_chain_chart(lk: RowLookup, prompts: List[int], n: int,
                           chains: List[int], engine: str) -> str:
    colors = _CHAIN_COLORS if engine == "qw3" else _LLAMA_CHAIN_COLORS
    series = []
    for i, p in enumerate(prompts):
        ys = []
        for c in chains:
            r = lk.get(engine, "mtp", p, n, c)
            ys.append(r.decode_tok_s_med if _ok(r) else None)
        series.append((f"{engine} mtp n_decode={n} prompt={_fmt_xtick(p)}", colors[i % len(colors)], ys))
    return _line_chart(
        f"{engine} MTP decode throughput vs MTP chain (n_decode={n}, output={n})",
        chains, series, "decode tok/s", x_log=False)


# ---------------------------------------------------------------------------
# Top-level render.

_CSS = """
:root{--qw3:#2563eb;--llama:#dc2626;}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:#111827;margin:0;padding:24px;background:#f6f7f9}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:30px 0 10px;border-bottom:1px solid #d1d5db;padding-bottom:5px}
h3{font-size:14px;margin:18px 0 8px;color:#374151}
.meta{color:#6b7280;font-size:12px;margin-bottom:8px}
.meta code{background:#eef2ff;padding:1px 5px;border-radius:4px;color:#3730a3}
.warn{background:#fff7ed;border:1px solid #f59e0b;padding:8px 12px;border-radius:6px;margin:8px 0}
.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0 18px}
.metric{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px}
.metric b{display:block;font-size:18px;margin-top:2px}.metric span{color:#6b7280;font-size:12px}
.table-wrap{max-width:100%;overflow:auto;border:1px solid #e5e7eb;border-radius:8px;background:white;margin:8px 0 16px}
table{width:100%;border-collapse:collapse;background:#fff;font-size:12px}
table.wide{min-width:1180px}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #f3f4f6;white-space:nowrap}
th{background:#f3f4f6;color:#374151;font-weight:600}
th span{font-size:10px;color:#6b7280;font-weight:500}
td:first-child,th:first-child{text-align:left}
td.sp{color:#6b21a8;font-weight:600}
td.mono,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;text-align:left}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px;margin:10px 0 16px}
.chart{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px}
.chart-title{font-size:13px;font-weight:600;color:#374151;margin-bottom:4px}
.svg-chart{width:100%;height:300px}
.legend{font-size:11px;color:#6b7280;margin-top:4px}
.lg{margin-right:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.empty{color:#9ca3af;font-style:italic;padding:20px;text-align:center}
.facet{margin:14px 0 22px}
.note{color:#6b7280;font-size:12px;margin:6px 0 10px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:white;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin:10px 0 12px}
.controls label{font-size:12px;color:#374151}.controls select{display:block;margin-top:3px;border:1px solid #d1d5db;border-radius:6px;padding:5px 8px;background:white;color:#111827}
.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.mini-title{font-weight:600;color:#374151;margin:10px 0 4px}
footer{margin-top:30px;color:#9ca3af;font-size:11px}
"""


def _interactive_section(store: BenchStore) -> str:
    payload = json.dumps(store.to_dict(), ensure_ascii=False).replace("</", "<\\/")
    return f"""
<h2>Interactive Results</h2>
<div class='controls'>
  <label>Engine<select id='fltEngine'><option value='all'>qw3 + llama.cpp</option><option value='qw3'>qw3</option><option value='llama'>llama.cpp</option></select></label>
  <label>Mode<select id='fltMode'><option value='all'>plain + MTP</option><option value='plain'>plain</option><option value='mtp'>MTP</option></select></label>
  <label>n_decode<select id='fltDecode'></select></label>
  <label>MTP chain<select id='fltChain'><option value='all'>all chains</option></select></label>
  <label>Metric<select id='fltMetric'>
    <option value='decode'>decode tok/s</option>
    <option value='prefill'>prefill tok/s</option>
    <option value='ttft'>TTFT seconds</option>
    <option value='tbt'>TBT ms</option>
    <option value='vram'>peak VRAM GB</option>
    <option value='accept'>MTP accept rate</option>
  </select></label>
</div>
<div id='interactiveChart'></div>
<div id='interactiveTables' class='split'></div>
<script id='benchStoreJson' type='application/json'>{payload}</script>
<script>
(function() {{
  const STORE = JSON.parse(document.getElementById('benchStoreJson').textContent);
  const rows = STORE.rows || [];
  const cfg = STORE.config || {{}};
  const prompts = [...(cfg.prompt_tokens || [])].sort((a,b)=>a-b);
  const decodes = [...(cfg.n_decode || [])].sort((a,b)=>a-b);
  const chains = [...(cfg.mtp_chain || [])].sort((a,b)=>a-b);
  const colors = ['#2563eb','#dc2626','#7c3aed','#ea580c','#0891b2','#db2777','#059669','#b91c1c','#ca8a04'];
  const labelPrompt = v => (v >= 1024 && v % 1024 === 0) ? `${{v/1024}}K` : String(v);
  const fmt = (v,d=2) => (v === null || v === undefined || !isFinite(v) || v === 0) ? '—' : Number(v).toFixed(d);
  const metricVal = (r, metric) => {{
    if (!r || r.error) return null;
    if (metric === 'decode') return r.decode_tok_s_med;
    if (metric === 'prefill') return r.prefill_tok_s_med;
    if (metric === 'ttft') return r.ttft_s_med;
    if (metric === 'tbt') return r.itl_ms_med;
    if (metric === 'vram') return r.peak_vram_mib ? r.peak_vram_mib / 1024.0 : null;
    if (metric === 'accept') return r.accept_rate;
    return null;
  }};
  const metricLabel = metric => ({{
    decode:'decode tok/s', prefill:'prefill tok/s', ttft:'TTFT seconds',
    tbt:'TBT ms', vram:'peak VRAM GB', accept:'MTP accept rate'
  }}[metric] || metric);
  const sel = id => document.getElementById(id);
  sel('fltDecode').innerHTML = decodes.map(n => `<option value="${{n}}">${{n}}</option>`).join('');
  sel('fltChain').innerHTML += chains.map(c => `<option value="${{c}}">chain=${{c}}</option>`).join('');

  function filteredRows(engine, mode, n, chain) {{
    return rows.filter(r => {{
      if (engine !== 'all' && r.engine !== engine) return false;
      if (mode !== 'all' && r.mode !== mode) return false;
      if (Number(r.n_decode) !== Number(n)) return false;
      if (r.mode === 'plain') return chain === 'all' || chain === '0';
      return chain === 'all' || Number(r.mtp_chain) === Number(chain);
    }}).sort((a,b) => {{
      if (a.engine !== b.engine) return a.engine.localeCompare(b.engine);
      if (a.mode !== b.mode) return a.mode.localeCompare(b.mode);
      if (a.mtp_chain !== b.mtp_chain) return a.mtp_chain - b.mtp_chain;
      return a.prompt_tokens - b.prompt_tokens;
    }});
  }}

  function renderTables() {{
    const engine = sel('fltEngine').value, mode = sel('fltMode').value;
    const n = sel('fltDecode').value, chain = sel('fltChain').value;
    const engines = engine === 'all' ? ['qw3','llama'] : [engine];
    const html = engines.map(e => {{
      const rs = filteredRows(e, mode, n, chain);
      const trs = rs.map(r => `<tr>
        <td>${{labelPrompt(r.prompt_tokens)}}</td>
        <td>${{r.mode}}</td><td>${{r.mode === 'mtp' ? r.mtp_chain : 0}}</td>
        <td>${{r.error ? 'fail' : 'ok'}}</td>
        <td>${{r.actual_prompt_tokens || '—'}}</td><td>${{r.decoded_tokens || '—'}}</td>
        <td>${{fmt(r.prefill_tok_s_med,1)}}</td><td>${{fmt(r.decode_tok_s_med,2)}}</td>
        <td>${{fmt(r.ttft_s_med,3)}}</td><td>${{fmt(r.itl_ms_med,2)}}</td>
        <td>${{fmt(r.peak_vram_mib ? r.peak_vram_mib/1024.0 : null,2)}}</td>
        <td>${{r.accept_rate === null || r.accept_rate === undefined ? '—' : fmt(r.accept_rate,3)}}</td>
      </tr>`).join('');
      return `<div><div class='mini-title'>${{e === 'llama' ? 'llama.cpp' : 'qw3'}} results</div>
        <div class='table-wrap'><table><thead><tr>
        <th>target prompt</th><th>mode</th><th>chain</th><th>status</th>
        <th>actual prompt</th><th>decoded</th><th>prefill tok/s</th><th>decode tok/s</th>
        <th>TTFT(s)</th><th>TBT(ms)</th><th>VRAM(GB)</th><th>accept</th>
        </tr></thead><tbody>${{trs || '<tr><td colspan="12">no rows</td></tr>'}}</tbody></table></div></div>`;
    }}).join('');
    document.getElementById('interactiveTables').innerHTML = html;
  }}

  function renderChart() {{
    const engine = sel('fltEngine').value, mode = sel('fltMode').value;
    const n = sel('fltDecode').value, chain = sel('fltChain').value;
    const metric = sel('fltMetric').value;
    const series = [];
    const rs = filteredRows(engine, mode, n, chain);
    const groups = new Map();
    for (const r of rs) {{
      const key = `${{r.engine}} ${{r.mode}} n_decode=${{r.n_decode}} chain=${{r.mode === 'mtp' ? r.mtp_chain : 0}}`;
      if (!groups.has(key)) groups.set(key, new Map());
      groups.get(key).set(r.prompt_tokens, metricVal(r, metric));
    }}
    let ci = 0;
    for (const [label, values] of groups.entries()) {{
      series.push({{label, color: colors[ci++ % colors.length], ys: prompts.map(p => values.get(p) ?? null)}});
    }}
    const vals = series.flatMap(s => s.ys).filter(v => v && isFinite(v));
    if (!vals.length) {{
      document.getElementById('interactiveChart').innerHTML = `<div class='chart'><div class='chart-title'>${{metricLabel(metric)}}: no data</div><div class='empty'>no data</div></div>`;
      return;
    }}
    const W=760,H=330,pad={{l:60,r:18,t:22,b:38}}, yMax=Math.max(...vals)*1.12;
    const xs = prompts.map(p => Math.log10(Math.max(1,p)));
    const xMin=Math.min(...xs), xMax=Math.max(...xs), xSpan=xMax-xMin || 1;
    const px=i => pad.l + (xs[i]-xMin)/xSpan*(W-pad.l-pad.r);
    const py=v => pad.t + (H-pad.t-pad.b) - v/yMax*(H-pad.t-pad.b);
    let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" class="svg-chart" role="img">`;
    for (let k=0;k<=5;k++) {{
      const v=yMax*k/5, y=py(v);
      const tick = metric === 'vram' || metric === 'accept' || yMax < 20 ? v.toFixed(2) : (yMax < 100 ? v.toFixed(1) : v.toFixed(0));
      svg += `<line x1="${{pad.l}}" y1="${{y}}" x2="${{W-pad.r}}" y2="${{y}}" stroke="#e5e7eb"/>`;
      svg += `<text x="${{pad.l-7}}" y="${{y+3}}" text-anchor="end" font-size="10" fill="#9ca3af">${{tick}}</text>`;
    }}
    prompts.forEach((p,i) => svg += `<text x="${{px(i)}}" y="${{H-12}}" text-anchor="middle" font-size="10" fill="#9ca3af">${{labelPrompt(p)}}</text>`);
    svg += `<line x1="${{pad.l}}" y1="${{pad.t}}" x2="${{pad.l}}" y2="${{H-pad.b}}" stroke="#9ca3af"/>`;
    svg += `<line x1="${{pad.l}}" y1="${{H-pad.b}}" x2="${{W-pad.r}}" y2="${{H-pad.b}}" stroke="#9ca3af"/>`;
    for (const s of series) {{
      const pts = s.ys.map((v,i)=>v ? [px(i),py(v)] : null).filter(Boolean);
      if (pts.length > 1) svg += `<polyline points="${{pts.map(p=>p.join(',')).join(' ')}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
      for (const p of pts) svg += `<circle cx="${{p[0]}}" cy="${{p[1]}}" r="3" fill="${{s.color}}"/>`;
    }}
    svg += `</svg>`;
    const legend = series.map(s => `<span class="lg"><span class="sw" style="background:${{s.color}}"></span>${{s.label}}</span>`).join(' ');
    document.getElementById('interactiveChart').innerHTML =
      `<div class='chart'><div class='chart-title'>${{metricLabel(metric)}} vs prompt (engine=${{engine}}, mode=${{mode}}, n_decode=${{n}}, chain=${{chain}})</div>${{svg}}<div class='legend'>${{legend}}</div></div>`;
  }}
  function render() {{ renderChart(); renderTables(); }}
  ['fltEngine','fltMode','fltDecode','fltChain','fltMetric'].forEach(id => sel(id).addEventListener('change', render));
  render();
}})();
</script>
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
    ok_rows = sum(1 for r in store.rows if not r.error)
    failed_rows = sum(1 for r in store.rows if r.error)
    P.append("<div class='overview'>"
             f"<div class='metric'><span>Successful rows</span><b>{ok_rows}</b></div>"
             f"<div class='metric'><span>Failed rows</span><b>{failed_rows}</b></div>"
             f"<div class='metric'><span>Prompt lengths</span><b>{len(prompts)}</b></div>"
             f"<div class='metric'><span>MTP chains</span><b>{', '.join(str(c) for c in chains) or '—'}</b></div>"
             "</div>")
    if store.partial:
        P.append("<div class='warn'>Partial run: sweep did not complete; some cells may be missing.</div>")
    if failed_rows:
        P.append("<div class='note'>Some cells failed and are shown as <b>fail</b> in the relevant tables. "
                 "Detailed failure text is kept in the JSON raw data, but is not expanded here.</div>")
    P.append(_interactive_section(store))

    # ---- Plain throughput ----
    P.append("<h2>Plain</h2>")
    for n in n_decodes:
        P.append(f"<div class='facet'><h3>mode=plain · n_decode={n} · chain=0</h3>")
        P.append(_throughput_table(lk, prompts, n, "plain", 0))
        P.append("<div class='charts'>")
        P.append(_line_chart(
            f"Decode throughput vs prompt (mode=plain, n_decode={n}, chain=0)", prompts,
            [(f"qw3 plain n_decode={n} chain=0", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "decode_tok_s_med")),
             (f"llama plain n_decode={n} chain=0", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "decode_tok_s_med"))],
            "decode tok/s"))
        P.append(_line_chart(
            f"Prefill throughput vs prompt (mode=plain, n_decode={n}, chain=0)", prompts,
            [(f"qw3 plain n_decode={n} chain=0", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "prefill_tok_s_med")),
             (f"llama plain n_decode={n} chain=0", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "prefill_tok_s_med"))],
            "prefill tok/s"))
        P.append("</div>")
        P.append(_latency_table(lk, prompts, n, "plain", 0))
        P.append("<div class='charts'>")
        P.append(_line_chart(
            f"TTFT vs prompt (mode=plain, n_decode={n}, chain=0)", prompts,
            [(f"qw3 plain n_decode={n} chain=0", _QW3_COLOR, _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "ttft_s_med")),
             (f"llama plain n_decode={n} chain=0", _LLAMA_COLOR, _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "ttft_s_med"))],
            "seconds"))
        P.append(_line_chart(
            f"Peak GPU VRAM vs prompt (mode=plain, n_decode={n}, chain=0)", prompts,
            [(f"qw3 plain n_decode={n} chain=0", _QW3_COLOR, [float(v)/1024.0 if v else None for v in _series_over_prompts(lk, "qw3", "plain", n, 0, prompts, "peak_vram_mib")]),
             (f"llama plain n_decode={n} chain=0", _LLAMA_COLOR, [float(v)/1024.0 if v else None for v in _series_over_prompts(lk, "llama", "plain", n, 0, prompts, "peak_vram_mib")])],
            "GB"))
        P.append("</div></div>")

    # ---- MTP ----
    if chains:
        P.append("<h2>MTP</h2>")
        for n in n_decodes:
            P.append(f"<div class='facet'><h3>mode=mtp · n_decode={n} · chains={', '.join(str(c) for c in chains)}</h3>")
            P.append("<h3>MTP chain sweep</h3>")
            P.append(_mtp_chain_sweep_table(lk, prompts, n, chains))
            P.append("<div class='charts'>")
            P.append(_mtp_acceptance_chart(lk, prompts, n, chains))
            chain_series = []
            for i, c in enumerate(chains):
                chain_series.append((f"qw3 mtp n_decode={n} chain={c}", _CHAIN_COLORS[i % len(_CHAIN_COLORS)],
                                     _series_over_prompts(lk, "qw3", "mtp", n, c, prompts, "decode_tok_s_med")))
            P.append(_line_chart(f"qw3 decode throughput vs prompt by MTP chain (mode=mtp, n_decode={n})",
                                 prompts, chain_series, "decode tok/s"))
            llama_chain_series = []
            for i, c in enumerate(chains):
                llama_chain_series.append((f"llama mtp n_decode={n} chain={c}", _LLAMA_CHAIN_COLORS[i % len(_LLAMA_CHAIN_COLORS)],
                                           _series_over_prompts(lk, "llama", "mtp", n, c, prompts, "decode_tok_s_med")))
            P.append(_line_chart(f"llama decode throughput vs prompt by MTP chain (mode=mtp, n_decode={n})",
                                 prompts, llama_chain_series, "decode tok/s"))
            P.append(_decode_by_chain_chart(lk, prompts, n, chains, "qw3"))
            P.append(_decode_by_chain_chart(lk, prompts, n, chains, "llama"))
            P.append("</div>")
            P.append("<h3>qw3 MTP details</h3>")
            P.append(_mtp_per_step_table(lk, prompts, n, chains))
            P.append("</div>")

    P.append("<footer>TTFT/ITL source: qw3 = <i>approx</i> (ttft≈prefill_s, "
             "itl≈decode_s/decoded; faithful because qw3's first decode token "
             "argmax comes from prefill); llama = <i>streamed</i> (first-chunk "
             "wall-clock + server timings). Main throughput/latency/acceptance values are means "
             "over successful trials; min/max fields remain in the JSON for spread checks. "
             "llama-server exposes only aggregate draft acceptance, "
             "so per-step/histogram columns are qw3-only.</footer>")
    P.append("</body></html>")

    out_path.write_text("".join(P), encoding="utf-8")
    return out_path
