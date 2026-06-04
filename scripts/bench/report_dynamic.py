"""Dynamic HTML report for the benchmark JSON store.

The generated page fetches the JSON store at page-load time instead of
embedding a snapshot. Serve the directory with `python3 -m http.server` and
refresh the browser to see the latest checkpointed benchmark rows.
"""
from __future__ import annotations

import html
from pathlib import Path


def render_dynamic_report(json_path: Path, html_path: Path) -> Path:
    data_name = html.escape(json_path.name)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qw3 vs llama.cpp benchmark</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 22px;
  background: #f6f7f9;
  color: #111827;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
h1 {{ margin: 0 0 4px; font-size: 22px; }}
h2 {{ margin: 24px 0 10px; padding-bottom: 5px; border-bottom: 1px solid #d1d5db; font-size: 17px; }}
h3 {{ margin: 18px 0 8px; font-size: 14px; color: #374151; }}
code {{ padding: 1px 5px; border-radius: 4px; background: #eef2ff; color: #3730a3; }}
.meta {{ color: #6b7280; font-size: 12px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0; }}
.toolbar button, .toolbar select {{
  border: 1px solid #d1d5db; background: white; color: #111827;
  border-radius: 6px; padding: 6px 9px; font: inherit;
}}
.pill {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e5e7eb; color: #374151; font-size: 12px; }}
.warn {{ background: #fff7ed; border: 1px solid #f59e0b; border-radius: 7px; padding: 9px 11px; margin: 10px 0; }}
.err {{ background: #fee2e2; border: 1px solid #ef4444; border-radius: 7px; padding: 9px 11px; margin: 10px 0; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
.card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
table {{ width: 100%; border-collapse: collapse; background: white; font-size: 12px; }}
th, td {{ padding: 6px 8px; border-bottom: 1px solid #f3f4f6; text-align: right; white-space: nowrap; }}
th {{ background: #f3f4f6; color: #374151; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
td:first-child, th:first-child, td.left, th.left {{ text-align: left; }}
.table-wrap {{ max-height: 520px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
.num-good {{ color: #047857; font-weight: 600; }}
.num-bad {{ color: #b91c1c; font-weight: 600; }}
.muted {{ color: #6b7280; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.chart {{ width: 100%; height: 290px; }}
.legend {{ display: flex; gap: 12px; flex-wrap: wrap; color: #6b7280; font-size: 12px; margin-top: 6px; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
footer {{ margin-top: 26px; color: #6b7280; font-size: 12px; }}
</style>
</head>
<body>
<h1>qw3 vs llama.cpp benchmark</h1>
<div class="meta" id="meta">Loading <code id="dataName">{data_name}</code>...</div>
<div class="toolbar">
  <button id="reloadBtn">Reload JSON</button>
  <label>n_decode <select id="decodeSelect"></select></label>
  <label>MTP chain <select id="chainSelect"></select></label>
  <span class="pill" id="rowCount">0 rows</span>
  <span class="pill" id="partialFlag">partial=?</span>
</div>
<div id="messages"></div>

<h2>Summary</h2>
<div id="summary" class="grid"></div>

<h2>Plain: Throughput, Latency, VRAM</h2>
<div id="plain"></div>

<h2>MTP: Throughput, Latency, VRAM</h2>
<div id="mtp"></div>

<h2>MTP Chain Comparison</h2>
<div id="chainComparison"></div>

<h2>Charts</h2>
<div class="grid">
  <div class="card"><h3>Decode throughput</h3><svg id="decodeChart" class="chart"></svg><div id="decodeLegend" class="legend"></div></div>
  <div class="card"><h3>Prefill throughput</h3><svg id="prefillChart" class="chart"></svg><div id="prefillLegend" class="legend"></div></div>
  <div class="card"><h3>TTFT</h3><svg id="ttftChart" class="chart"></svg><div id="ttftLegend" class="legend"></div></div>
  <div class="card"><h3>Peak GPU VRAM</h3><svg id="vramChart" class="chart"></svg><div id="vramLegend" class="legend"></div></div>
</div>

<h2>MTP Details</h2>
<div id="mtpDetails"></div>

<footer>
The page fetches the JSON store every time it is opened or reloaded. Use a local
HTTP server, for example <code>cd /tmp/qw3_llama_bench && python3 -m http.server 8017</code>.
TTFT/TBT source differs by engine: qw3 derives TTFT from prefill time and TBT
from decode aggregate timing; llama.cpp uses streamed first-token timing and
server timings. Rows show medians over trials. Peak VRAM is sampled via
<code>nvidia-smi</code>.
</footer>

<script>
const DEFAULT_JSON = "{data_name}";
const COLORS = {{
  "qw3 plain": "#2563eb",
  "llama plain": "#dc2626",
  "qw3 mtp": "#7c3aed",
  "llama mtp": "#ea580c",
}};
let STORE = null;

function qs(name) {{
  const u = new URL(window.location.href);
  return u.searchParams.get(name);
}}
function dataUrl() {{ return qs("data") || DEFAULT_JSON; }}
function fmt(x, d=1) {{
  if (x === null || x === undefined || !isFinite(x) || x === 0) return "—";
  return Number(x).toFixed(d);
}}
function fmtTok(n) {{
  if (n >= 1024 && n % 1024 === 0) return (n / 1024) + "K";
  return String(n);
}}
function key(engine, mode, p, n, chain) {{
  return `${{engine}}|${{mode}}|${{p}}|${{n}}|${{chain}}`;
}}
function rowIndex(rows) {{
  const m = new Map();
  for (const r of rows || []) m.set(key(r.engine, r.mode, r.prompt_tokens, r.n_decode, r.mtp_chain), r);
  return m;
}}
function ok(r) {{ return r && !r.error; }}
function prompts(store) {{ return [...(store.config.prompt_tokens || [])].sort((a,b)=>a-b); }}
function decodes(store) {{ return [...(store.config.n_decode || [])].sort((a,b)=>a-b); }}
function chains(store) {{ return [...(store.config.mtp_chain || [])].sort((a,b)=>a-b); }}
function selectedDecode() {{ return Number(document.getElementById("decodeSelect").value || 0); }}
function selectedChain() {{ return Number(document.getElementById("chainSelect").value || 0); }}

async function load() {{
  const url = dataUrl() + "?t=" + Date.now();
  document.getElementById("dataName").textContent = dataUrl();
  const res = await fetch(url, {{ cache: "no-store" }});
  if (!res.ok) throw new Error(`fetch ${{url}} failed: ${{res.status}}`);
  STORE = await res.json();
  populateSelectors();
  render();
}}

function populateSelectors() {{
  const ds = decodes(STORE);
  const cs = chains(STORE);
  const dSel = document.getElementById("decodeSelect");
  const cSel = document.getElementById("chainSelect");
  const prevD = dSel.value;
  const prevC = cSel.value;
  dSel.innerHTML = ds.map(d => `<option value="${{d}}">${{d}}</option>`).join("");
  cSel.innerHTML = cs.map(c => `<option value="${{c}}">${{c}}</option>`).join("");
  if (ds.includes(Number(prevD))) dSel.value = prevD;
  if (cs.includes(Number(prevC))) cSel.value = prevC;
}}

function render() {{
  const cfg = STORE.config || {{}};
  document.getElementById("meta").innerHTML =
    `commit <code>${{STORE.git_commit || "unknown"}}</code> · host <code>${{STORE.host || ""}}</code> · ` +
    `${{STORE.timestamp || ""}} · model <code>${{(cfg.model || "").split("/").pop()}}</code> · ` +
    `trials ${{cfg.trials || 0}} · data <code>${{dataUrl()}}</code>`;
  document.getElementById("rowCount").textContent = `${{(STORE.rows || []).length}} rows`;
  document.getElementById("partialFlag").textContent = STORE.partial ? "partial=true" : "partial=false";
  renderMessages();
  renderSummary();
  renderTables();
  renderChainComparison();
  renderCharts();
  renderMtpDetails();
}}

function renderMessages() {{
  const box = document.getElementById("messages");
  let html = "";
  if (STORE.partial) html += `<div class="warn">Benchmark is still partial or was interrupted. Refresh after more rows are written.</div>`;
  if ((STORE.errors || []).length) {{
    html += `<div class="err"><b>${{STORE.errors.length}} cell error(s)</b><br>` +
      STORE.errors.slice(-20).map(e => `${{e.cell_key}}: ${{String(e.message).slice(0, 220)}}`).join("<br>") +
      `</div>`;
  }}
  box.innerHTML = html;
}}

function metricCard(title, body) {{
  return `<div class="card"><h3>${{title}}</h3>${{body}}</div>`;
}}
function bestDecode(store, mode) {{
  const rows = (store.rows || []).filter(r => r.mode === mode && ok(r));
  if (!rows.length) return "—";
  const best = [...rows].sort((a,b)=>b.decode_tok_s_med-a.decode_tok_s_med)[0];
  return `${{best.engine}} ${{fmtTok(best.prompt_tokens)}}/${{best.n_decode}}` +
    `${{mode === "mtp" ? " c"+best.mtp_chain : ""}}: <b>${{fmt(best.decode_tok_s_med,2)}} tok/s</b>`;
}}
function renderSummary() {{
  const rows = STORE.rows || [];
  const done = rows.filter(ok).length;
  const errors = rows.filter(r => r.error).length;
  document.getElementById("summary").innerHTML =
    metricCard("Coverage", `${{done}} successful rows, ${{errors}} row errors`) +
    metricCard("Best plain decode", bestDecode(STORE, "plain")) +
    metricCard("Best MTP decode", bestDecode(STORE, "mtp")) +
    metricCard("Target grid", `${{prompts(STORE).map(fmtTok).join(", ")}} input · ${{decodes(STORE).join(", ")}} output`);
}}

function tableFor(mode, n, chain) {{
  const idx = rowIndex(STORE.rows || []);
  const rows = [];
  for (const p of prompts(STORE)) {{
    const q = idx.get(key("qw3", mode, p, n, mode === "plain" ? 0 : chain));
    const l = idx.get(key("llama", mode, p, n, mode === "plain" ? 0 : chain));
    const ratio = (a,b) => ok(a) && ok(b) && b.decode_tok_s_med ? a.decode_tok_s_med / b.decode_tok_s_med : null;
    const rr = ratio(q,l);
    rows.push(`<tr>
      <td>${{fmtTok(p)}}</td>
      <td>${{ok(q) ? q.actual_prompt_tokens : "—"}}</td>
      <td>${{ok(q) ? q.decoded_tokens : "—"}}</td>
      <td>${{ok(q) ? fmt(q.prefill_tok_s_med,1) : "—"}}</td>
      <td>${{ok(l) ? fmt(l.prefill_tok_s_med,1) : "—"}}</td>
      <td>${{ok(q) ? fmt(q.decode_tok_s_med,2) : "—"}}</td>
      <td>${{ok(l) ? fmt(l.decode_tok_s_med,2) : "—"}}</td>
      <td class="${{rr && rr >= 1 ? "num-good" : "num-bad"}}">${{rr ? fmt(rr * 100,1) + "%" : "—"}}</td>
      <td>${{ok(q) ? fmt(q.ttft_s_med,3) : "—"}}</td>
      <td>${{ok(l) ? fmt(l.ttft_s_med,3) : "—"}}</td>
      <td>${{ok(q) ? fmt(q.itl_ms_med,2) : "—"}}</td>
      <td>${{ok(l) ? fmt(l.itl_ms_med,2) : "—"}}</td>
      <td>${{ok(q) ? q.peak_vram_mib : "—"}}</td>
      <td>${{ok(l) ? l.peak_vram_mib : "—"}}</td>
      <td>${{ok(q) && q.accept_rate !== null ? fmt(q.accept_rate * 100,1) + "%" : "—"}}</td>
      <td>${{ok(l) && l.accept_rate !== null ? fmt(l.accept_rate * 100,1) + "%" : "—"}}</td>
    </tr>`);
  }}
  return `<div class="table-wrap"><table>
    <thead><tr>
      <th class="left">input</th><th>qw3 actual in</th><th>qw3 out</th>
      <th>qw3 prefill tok/s</th><th>llama prefill tok/s</th>
      <th>qw3 decode tok/s</th><th>llama decode tok/s</th><th>qw3/llama decode</th>
      <th>qw3 TTFT(s)</th><th>llama TTFT(s)</th>
      <th>qw3 TBT(ms)</th><th>llama TBT(ms)</th>
      <th>qw3 VRAM(MiB)</th><th>llama VRAM(MiB)</th>
      <th>qw3 accept</th><th>llama accept</th>
    </tr></thead><tbody>${{rows.join("")}}</tbody></table></div>`;
}}
function renderTables() {{
  const n = selectedDecode();
  const c = selectedChain();
  document.getElementById("plain").innerHTML = `<h3>n_decode=${{n}}</h3>` + tableFor("plain", n, 0);
  document.getElementById("mtp").innerHTML = `<h3>n_decode=${{n}}, chain=${{c}}</h3>` + tableFor("mtp", n, c);
}}

function renderChainComparison() {{
  const n = selectedDecode();
  const idx = rowIndex(STORE.rows || []);
  const cs = chains(STORE);
  const rows = [];
  for (const p of prompts(STORE)) {{
    const cells = [];
    for (const c of cs) {{
      const q = idx.get(key("qw3", "mtp", p, n, c));
      const l = idx.get(key("llama", "mtp", p, n, c));
      const qr = ok(q) ? `${{fmt(q.decode_tok_s_med,2)}} / ${{fmt(q.accept_rate*100,1)}}% / ${{q.peak_vram_mib}}` : (q && q.error ? "ERR" : "—");
      const lr = ok(l) ? `${{fmt(l.decode_tok_s_med,2)}} / ${{fmt(l.accept_rate*100,1)}}% / ${{l.peak_vram_mib}}` : (l && l.error ? "ERR" : "—");
      cells.push(`<td>${{qr}}</td><td>${{lr}}</td>`);
    }}
    rows.push(`<tr><td>${{fmtTok(p)}}</td>${{cells.join("")}}</tr>`);
  }}
  const head = cs.map(c => `<th>qw3 c${{c}}<br><span class="muted">tok/s / accept / MiB</span></th><th>llama c${{c}}<br><span class="muted">tok/s / accept / MiB</span></th>`).join("");
  document.getElementById("chainComparison").innerHTML = `<h3>n_decode=${{n}}</h3>
    <div class="table-wrap"><table><thead><tr><th class="left">input</th>${{head}}</tr></thead>
    <tbody>${{rows.join("")}}</tbody></table></div>`;
}}

function series(mode, engine, n, chain, attr) {{
  const idx = rowIndex(STORE.rows || []);
  return prompts(STORE).map(p => {{
    const r = idx.get(key(engine, mode, p, n, mode === "plain" ? 0 : chain));
    return ok(r) ? Number(r[attr] || 0) : null;
  }});
}}
function drawLine(svgId, legendId, title, ysList, yLabel) {{
  const svg = document.getElementById(svgId);
  const legend = document.getElementById(legendId);
  const xs = prompts(STORE);
  const W = svg.clientWidth || 520, H = 290;
  const pad = {{l:54,r:14,t:18,b:34}};
  const vals = ysList.flatMap(s => s.ys).filter(v => v && isFinite(v));
  if (!vals.length) {{ svg.innerHTML = ""; legend.innerHTML = "<span class='muted'>no data</span>"; return; }}
  const yMax = Math.max(...vals) * 1.12;
  const xLog = xs.map(x => Math.log10(Math.max(1, x)));
  const xMin = Math.min(...xLog), xMax = Math.max(...xLog), xSpan = xMax - xMin || 1;
  const px = i => pad.l + (xLog[i] - xMin) / xSpan * (W - pad.l - pad.r);
  const py = v => pad.t + (H - pad.t - pad.b) - v / yMax * (H - pad.t - pad.b);
  let out = `<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="100%">`;
  for (let k=0;k<=5;k++) {{
    const v = yMax * k / 5, y = py(v);
    out += `<line x1="${{pad.l}}" y1="${{y}}" x2="${{W-pad.r}}" y2="${{y}}" stroke="#e5e7eb"/>`;
    out += `<text x="${{pad.l-6}}" y="${{y+3}}" text-anchor="end" font-size="10" fill="#6b7280">${{v.toFixed(yMax>100?0:1)}}</text>`;
  }}
  xs.forEach((x,i) => out += `<text x="${{px(i)}}" y="${{H-12}}" text-anchor="middle" font-size="10" fill="#6b7280">${{fmtTok(x)}}</text>`);
  for (const s of ysList) {{
    const pts = s.ys.map((v,i) => v ? [px(i), py(v)] : null).filter(Boolean);
    if (pts.length > 1) out += `<polyline points="${{pts.map(p=>p.join(",")).join(" ")}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
    for (const p of pts) out += `<circle cx="${{p[0]}}" cy="${{p[1]}}" r="3" fill="${{s.color}}"/>`;
  }}
  out += `</svg>`;
  svg.innerHTML = out;
  legend.innerHTML = ysList.map(s => `<span><span class="sw" style="background:${{s.color}}"></span>${{s.label}}</span>`).join("");
}}
function renderCharts() {{
  const n = selectedDecode(), c = selectedChain();
  const base = [
    {{ label: "qw3 plain", color: COLORS["qw3 plain"], mode: "plain", engine: "qw3", chain: 0 }},
    {{ label: "llama plain", color: COLORS["llama plain"], mode: "plain", engine: "llama", chain: 0 }},
    {{ label: "qw3 MTP", color: COLORS["qw3 mtp"], mode: "mtp", engine: "qw3", chain: c }},
    {{ label: "llama MTP", color: COLORS["llama mtp"], mode: "mtp", engine: "llama", chain: c }},
  ];
  drawLine("decodeChart", "decodeLegend", "decode", base.map(s => ({{...s, ys: series(s.mode, s.engine, n, s.chain, "decode_tok_s_med")}})), "tok/s");
  drawLine("prefillChart", "prefillLegend", "prefill", base.map(s => ({{...s, ys: series(s.mode, s.engine, n, s.chain, "prefill_tok_s_med")}})), "tok/s");
  drawLine("ttftChart", "ttftLegend", "ttft", base.map(s => ({{...s, ys: series(s.mode, s.engine, n, s.chain, "ttft_s_med")}})), "seconds");
  drawLine("vramChart", "vramLegend", "vram", base.map(s => ({{...s, ys: series(s.mode, s.engine, n, s.chain, "peak_vram_mib")}})), "MiB");
}}

function renderMtpDetails() {{
  const n = selectedDecode(), c = selectedChain();
  const idx = rowIndex(STORE.rows || []);
  const rows = [];
  for (const p of prompts(STORE)) {{
    const q = idx.get(key("qw3", "mtp", p, n, c));
    const l = idx.get(key("llama", "mtp", p, n, c));
    rows.push(`<tr>
      <td>${{fmtTok(p)}}</td>
      <td>${{ok(q) && q.accept_per_step ? q.accept_per_step.map(v=>fmt(v*100,1)+"%").join(" / ") : "—"}}</td>
      <td class="mono">${{ok(q) && q.accept_hist ? Object.entries(q.accept_hist).map(([k,v])=>k+"="+v).join(" ") : "—"}}</td>
      <td>${{ok(q) ? fmt(q.mtp_draft_s,3) : "—"}}</td>
      <td>${{ok(q) ? fmt(q.mtp_verify_s,3) : "—"}}</td>
      <td>${{ok(l) && l.accept_rate !== null ? fmt(l.accept_rate*100,1)+"%" : "—"}}</td>
    </tr>`);
  }}
  document.getElementById("mtpDetails").innerHTML = `<h3>n_decode=${{n}}, chain=${{c}}</h3>
    <div class="table-wrap"><table><thead><tr>
      <th class="left">input</th><th>qw3 per-step accept</th><th class="left">qw3 accept histogram</th>
      <th>qw3 draft_s</th><th>qw3 verify_s</th><th>llama aggregate accept</th>
    </tr></thead><tbody>${{rows.join("")}}</tbody></table></div>`;
}}

document.getElementById("reloadBtn").addEventListener("click", () => load().catch(showError));
document.getElementById("decodeSelect").addEventListener("change", render);
document.getElementById("chainSelect").addEventListener("change", render);
function showError(err) {{
  document.getElementById("messages").innerHTML =
    `<div class="err"><b>Failed to load benchmark JSON.</b><br>${{String(err)}}<br>` +
    `If you opened this file directly, serve the directory with <code>python3 -m http.server</code>.</div>`;
}}
load().catch(showError);
</script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path
