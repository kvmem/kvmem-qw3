#!/usr/bin/env python3
"""Read-only live dashboard for an official Pier + MiniSweAgent run."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSWE · MiniSweAgent · KVMem</title>
<style>
:root{color-scheme:light;--ink:#162033;--muted:#6a7588;--line:#dce4ef;--blue:#1769e0;--green:#087f5b;--red:#bd3345;--amber:#a96600}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system;background:radial-gradient(circle at 12% 0,#dceeff,transparent 34%),radial-gradient(circle at 92% 0,#e9e1ff,transparent 32%),#f4f7fb}
.wrap{max-width:1600px;margin:auto;padding:18px}.glass{background:rgba(255,255,255,.80);border:1px solid rgba(255,255,255,.94);box-shadow:0 14px 42px rgba(34,58,92,.10);backdrop-filter:blur(18px);border-radius:17px}
header{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:17px 20px;margin-bottom:12px}h1{font-size:20px;margin:0}.sub{color:var(--muted);margin-top:4px}.badge{padding:6px 10px;border-radius:999px;background:#eaf3ff;color:var(--blue);font-weight:750;white-space:nowrap}
.stats{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:9px;margin-bottom:12px}.stat{padding:10px 12px}.stat b{display:block;font-size:17px}.stat span{font-size:11px;color:var(--muted)}
.table{overflow:auto;margin-bottom:12px}table{border-collapse:collapse;width:100%}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}tr.active{background:#edf6ff}.ok{color:var(--green);font-weight:750}.bad{color:var(--red);font-weight:750}.run{color:var(--blue);font-weight:750}.wait{color:var(--amber);font-weight:750}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:12px}.panel{padding:14px;min-width:0}.panel h2{font-size:14px;margin:0 0 8px}.log{height:46vh;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border:1px solid var(--line);border-radius:11px;padding:11px;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
@media(max-width:950px){.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.log{height:38vh}.wrap{padding:9px}header{align-items:flex-start}}
</style></head><body><div class="wrap">
<header class="glass"><div><h1>DeepSWE · 官方 Pier + MiniSweAgent · QW3/KVMem</h1><div class="sub" id="protocol">加载中…</div></div><span class="badge" id="live">连接中</span></header>
<div class="stats" id="stats"></div>
<div class="glass table"><table><thead><tr id="table-head"></tr></thead><tbody id="rows"></tbody></table></div>
<div class="grid"><section class="glass panel"><h2 id="agent-title">当前 MiniSweAgent 输出</h2><div class="log" id="agent"></div></section><section class="glass panel"><h2>当前 QW3 / KVMem 日志</h2><div class="log" id="server"></div></section></div>
</div><script>
let auto=true;const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const val=(v,d='—')=>v==null?d:v,fmt=n=>n==null?'—':Number(n).toLocaleString(),sec=n=>n==null?'—':Number(n).toFixed(1)+'s';
function status(r){let c=r.status==='completed'?'ok':r.status==='running'?'run':r.status==='infrastructure_error'?'bad':'wait';return `<span class="${c}">${esc(r.status)}</span>`}
function rolloutCell(r){if(r.reward!=null)return `<td class="${r.reward===1?'ok':r.reward===0?'bad':''}">${esc(r.reward)}</td>`;if(r.status==='running')return '<td class="run">运行中</td>';return '<td>—</td>'}
async function tick(){try{let response=await fetch(`/api/state?_=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);let x=await response.json();$('#live').textContent=(x.running?'LIVE':'IDLE')+' · '+new Date(x.generated_at).toLocaleTimeString();$('#protocol').textContent=`${x.method} · MiniSweAgent ${x.versions.mini_swe_agent} · Pier ${x.versions.pier} · Qwen3.8-27B · seed ${x.seed}`;
$('#protocol').textContent+=x.binary_sha?` · binary ${x.binary_sha.slice(0,8)}`:'';let s=x.summary;$('#stats').innerHTML=[['试次完成',`${s.completed_rollouts}/${s.total_rollouts}`],['试次通过',`${s.passed_rollouts}/${s.completed_rollouts||0}`],['当前任务',x.current_task||'—'],['当前请求',fmt(s.current_requests)],['当前 prompt',fmt(s.current_prompt_tokens)],['当前压缩',fmt(s.current_compactions)],['Serving 异常',fmt(s.serving_errors)]].map(v=>`<div class="glass stat"><b>${esc(v[1])}</b><span>${v[0]}</span></div>`).join('');
$('#table-head').innerHTML=['#','任务','难度','状态','历史',...x.rollout_columns.map(r=>`${r.label} / s${r.seed}`),'F2P','P2P','Partial','Steps','Compactions','Peak ctx','Avg ctx','耗时'].map(v=>`<th>${esc(v)}</th>`).join('');
$('#rows').innerHTML=x.tasks.map(r=>`<tr class="${r.task_id===x.current_task?'active':''}"><td>${r.order}</td><td><b>${esc(r.task_id)}</b></td><td>${r.difficulty.level?`L${r.difficulty.level} · ${r.difficulty.pass_rate.toFixed(1)}%`:'—'}</td><td>${status(r)}</td><td>${r.historical_reward}</td>${r.rollouts.map(rolloutCell).join('')}<td>${r.f2p_passed==null?'—':r.f2p_passed+'/'+r.f2p_total}</td><td>${r.p2p_passed==null?'—':r.p2p_passed+'/'+r.p2p_total}</td><td>${r.partial==null?'—':Number(r.partial).toFixed(4)}</td><td>${fmt(r.steps)}</td><td>${fmt(r.compactions)}</td><td>${fmt(r.peak_context_tokens)}</td><td>${fmt(r.avg_context_tokens)}</td><td>${sec(r.elapsed_sec)}</td></tr>`).join('');
$('#agent-title').textContent=`当前 MiniSweAgent 输出 · ${x.current_task||'等待中'}`;let a=$('#agent'),q=$('#server');a.textContent=x.agent_log||'等待输出…';q.textContent=x.server_log||'等待输出…';if(auto){a.scrollTop=a.scrollHeight;q.scrollTop=q.scrollHeight}}
catch(e){$('#live').textContent='连接重试 · '+String(e)}setTimeout(tick,1500)}tick();
</script></body></html>"""


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def tail(path: Path | None, limit: int = 100_000) -> str:
    if path is None:
        return ""
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - limit))
            data = stream.read()
        if size > limit:
            data = data.split(b"\n", 1)[-1]
        return data.decode(errors="replace").replace("\r", "")
    except OSError:
        return ""


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def latest_attempt(task_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    """Return the newest root or infrastructure-retry attempt descriptor."""
    path = newest(list(task_dir.rglob("attempt.json")))
    return path, (read_json(path, {}) or {}) if path else {}


def attempt_seed(attempt: dict[str, Any], default: int | None = None) -> int | None:
    """Read the rollout seed from new or legacy attempt metadata."""
    value = attempt.get("rollout_seed")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    command = attempt.get("qwen_command")
    if isinstance(command, list) and "--seed" in command:
        index = command.index("--seed") + 1
        if index < len(command):
            try:
                return int(command[index])
            except (TypeError, ValueError):
                pass
    return default


def result_attempt(result_path: Path, task_dir: Path) -> dict[str, Any]:
    """Find the attempt descriptor that owns one nested Pier result."""
    for parent in (result_path.parent, *result_path.parents):
        if parent == task_dir.parent:
            break
        candidate = parent / "attempt.json"
        if candidate.is_file():
            return read_json(candidate, {}) or {}
    return {}


def summary_peak_context_tokens(summary: dict[str, Any]) -> int | None:
    """Return the authoritative peak context from one completed Pier trial."""
    if summary.get("status") != "completed":
        return None
    result_value = summary.get("trial_result")
    if not isinstance(result_value, str):
        return None
    trial = read_json(Path(result_value), {}) or {}
    value = (trial.get("agent_result") or {}).get("peak_context_tokens")
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    return None


def rollout_results(
    task_dir: Path,
    rollout_seeds: list[int],
    attempt_path: Path | None,
    attempt: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return accepted verifier rewards aligned to the configured rollouts.

    The runner snapshots the previous accepted summary before every explicit
    rerun.  Reading those snapshots rather than every ``result.json`` avoids
    counting failed infrastructure retries as extra model attempts.
    """
    by_seed: dict[int, dict[str, Any]] = {}
    summaries = sorted(task_dir.glob("summary_before_rerun_attempt_*.json"))
    current = read_json(task_dir / "summary.json", {}) or {}
    if current.get("status") == "completed":
        summaries.append(task_dir / "summary.json")

    for path in summaries:
        summary = read_json(path, {}) or {}
        reward = summary.get("reward")
        if not isinstance(reward, (int, float)):
            continue
        seed = attempt_seed(summary)
        result_value = summary.get("trial_result")
        if seed is None and isinstance(result_value, str):
            result_path = Path(result_value)
            seed = attempt_seed(result_attempt(result_path, task_dir))
        # Legacy rollout-1 artifacts predate explicit seed metadata, but their
        # locked QW3 command used the runner's original default seed 73.
        if seed is None:
            seed = rollout_seeds[0]
        by_seed[seed] = {
            "seed": seed,
            "reward": reward,
            "status": "completed",
            "peak_context_tokens": summary_peak_context_tokens(summary),
        }

    if attempt_path is not None and process_alive(attempt.get("qwen_pid")):
        seed = attempt_seed(attempt)
        if seed is not None and seed not in by_seed:
            by_seed[seed] = {"seed": seed, "reward": None, "status": "running"}

    return [
        by_seed.get(seed, {"seed": seed, "reward": None, "status": "queued"})
        for seed in rollout_seeds
    ]


def independent_rollout_result(task_dir: Path, seed: int) -> dict[str, Any]:
    """Return one result from an independent per-rollout run directory.

    Keeping each sampling attempt in its own run directory makes duplicate
    seeds unambiguous (for example, R1/s73 and the deterministic regression
    control R2/s73) and avoids treating infrastructure retries as new model
    attempts.
    """
    summary = read_json(task_dir / "summary.json", {}) or {}
    reward = summary.get("reward")
    if summary.get("status") == "completed" and isinstance(reward, (int, float)):
        return {
            "seed": seed,
            "reward": reward,
            "status": "completed",
            "peak_context_tokens": summary_peak_context_tokens(summary),
        }
    attempt_path, attempt = latest_attempt(task_dir)
    if attempt_path is not None and process_alive(attempt.get("qwen_pid")):
        return {"seed": seed, "reward": None, "status": "running"}
    if summary.get("status") == "infrastructure_error":
        return {"seed": seed, "reward": None, "status": "infrastructure_error"}
    return {"seed": seed, "reward": None, "status": "queued"}


def process_alive(value: Any) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def difficulty_index(repo: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(repo / "benchmark/opencode_swebench/deepswe_v1_1_trials.json", {}) or {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("rows", []):
        if row.get("included_in_score") and row.get("task_name"):
            grouped.setdefault(str(row["task_name"]), []).append(row)
    anchors = ((95.0, 1), (80.0, 2), (60.0, 3), (30.0, 4), (5.0, 5))
    result = {}
    for task_id, rows in grouped.items():
        rate = 100.0 * sum(bool(row.get("passed")) for row in rows) / len(rows)
        result[task_id] = {
            "pass_rate": rate,
            "level": min(anchors, key=lambda item: abs(rate - item[0]))[1],
            "trials": len(rows),
        }
    return result


def trial_result(task_dir: Path) -> dict[str, Any]:
    for path in sorted(task_dir.rglob("result.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        value = read_json(path, {}) or {}
        if value.get("task_name") and "verifier_result" in value:
            return value
    return {}


def trajectory(task_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    path = newest(list(task_dir.rglob("mini-swe-agent.trajectory.json")))
    return path, (read_json(path, {}) or {}) if path else {}


def request_stats(log: str) -> tuple[int, int | None, int]:
    rows = re.findall(r"\[qw3-serve\] #(\d+) chat .*?prompt_tokens=(\d+)", log)
    errors = len(re.findall(r"mtp_fallback=plain|unexpected_no_reusable_checkpoint=1|terminal_status=(?:error|failed)|chat\(stream\) error=|POST .* -> 5\d\d", log))
    return (int(rows[-1][0]), int(rows[-1][1]), errors) if rows else (0, None, errors)


def state(
    repo: Path,
    rollout_runs: list[tuple[str, int, Path]],
    plan: dict[str, Any],
    difficulty: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    existing_runs = [entry for entry in rollout_runs if (entry[2] / "manifest.json").is_file()]
    active_runs = []
    for entry in existing_runs:
        candidate = entry[2]
        if any(
            process_alive(latest_attempt(candidate / str(spec["task_id"]))[1].get("qwen_pid"))
            for spec in plan.get("tasks", [])
        ):
            active_runs.append(entry)
    active_label, active_seed, run = (
        active_runs[-1]
        if active_runs
        else existing_runs[-1]
        if existing_runs
        else rollout_runs[0]
    )
    manifest = read_json(run / "manifest.json", {}) or {}
    tasks = []
    current_task = None
    current_agent = current_server = ""
    current_requests = serving_errors = current_compactions = 0
    current_prompt: int | None = None
    for order, spec in enumerate(plan.get("tasks", []), start=1):
        task_id = str(spec["task_id"])
        task_dir = run / task_id
        summary = read_json(task_dir / "summary.json", {}) or {}
        attempt_path, attempt = latest_attempt(task_dir)
        attempt_running = attempt_path is not None and process_alive(
            attempt.get("qwen_pid")
        )
        # A retry keeps the failed root attempt for auditability.  While the
        # retry is running, its newer attempt.json is the authoritative clock;
        # completed summaries remain authoritative once written.
        if summary.get("status") == "completed":
            status = "completed"
        elif summary.get("status") == "infrastructure_error":
            status = "infrastructure_error"
        elif attempt_running:
            status = "running"
        else:
            status = "queued"

        metric_root: Path | None = None
        trial: dict[str, Any] = {}
        if status == "completed" and summary.get("trial_result"):
            result_path = Path(str(summary["trial_result"]))
            trial = read_json(result_path, {}) or {}
            metric_root = result_path.parent
        elif status == "running" and attempt_path is not None:
            metric_root = attempt_path.parent
            trial = trial_result(metric_root)
        verifier = trial.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        _, traj = trajectory(metric_root) if metric_root is not None else (None, {})
        info = traj.get("info") or {}
        model_stats = info.get("model_stats") or {}
        compaction = info.get("compaction") or {}
        agent_result = trial.get("agent_result") or {}
        started = parse_time(
            summary.get("started_at")
            or attempt.get("started_at")
            or trial.get("started_at")
        )
        if status == "queued":
            started = None
        # An older failed trial remains under the task root while a newer
        # infrastructure retry is active.  Its finished_at must not freeze (or
        # make negative) the live retry clock.
        finished = parse_time(summary.get("finished_at")) if status != "running" else None
        elapsed = max(0.0, ((finished or now) - started).total_seconds()) if started else None
        steps = agent_result.get("n_agent_steps") or trial.get("n_agent_steps") or model_stats.get("api_calls")
        row = {
            "order": order,
            "task_id": task_id,
            "difficulty": difficulty.get(task_id, {}),
            "status": status,
            "historical_reward": spec.get("historical_reward"),
            "reward": rewards.get("reward", summary.get("reward")),
            "f2p_total": rewards.get("f2p_total"),
            "f2p_passed": rewards.get("f2p_passed"),
            "p2p_total": rewards.get("p2p_total"),
            "p2p_passed": rewards.get("p2p_passed"),
            "partial": rewards.get("partial"),
            "steps": steps,
            "compactions": compaction.get("count", 0),
            "peak_context_tokens": agent_result.get("peak_context_tokens"),
            "elapsed_sec": elapsed,
            "rollouts": rollout_results(
                task_dir, [active_seed], attempt_path, attempt
            ) if len(rollout_runs) == 1 else [
                independent_rollout_result(rollout_run / task_id, seed)
                for _label, seed, rollout_run in rollout_runs
            ],
        }
        completed_contexts = [
            rollout.get("peak_context_tokens")
            for rollout in row["rollouts"]
            if rollout.get("status") == "completed"
            and isinstance(rollout.get("peak_context_tokens"), int)
        ]
        if (
            len(row["rollouts"]) == 4
            and len(completed_contexts) == len(row["rollouts"])
        ):
            row["avg_context_tokens"] = round(
                sum(completed_contexts) / len(completed_contexts)
            )
        else:
            row["avg_context_tokens"] = None
        tasks.append(row)
        if status == "running":
            current_task = task_id
            agent_path = newest(list(task_dir.rglob("mini-swe-agent.txt")))
            server_path = newest(list(task_dir.rglob("qw3_server.log")))
            # Keep a wider server tail for metrics, but do not resend and
            # replace hundreds of KiB of log DOM on every browser poll.  The
            # Codex embedded browser can otherwise fall behind and keep a
            # visibly stale task row/output panel even though /api/state is
            # current.
            current_agent = tail(agent_path, 30_000)
            server_stats = tail(server_path, 100_000)
            current_server = server_stats[-30_000:]
            current_requests, current_prompt, serving_errors = request_stats(server_stats)
            current_compactions = int(row["compactions"] or 0)
            if row["peak_context_tokens"] is None:
                row["peak_context_tokens"] = current_prompt
    completed = sum(row["status"] == "completed" for row in tasks)
    passed = sum(row["reward"] == 1 for row in tasks)
    completed_rollouts = sum(
        rollout["reward"] is not None
        for row in tasks
        for rollout in row["rollouts"]
    )
    passed_rollouts = sum(
        rollout["reward"] == 1
        for row in tasks
        for rollout in row["rollouts"]
    )
    locks = manifest.get("locks") or {}
    mini = locks.get("mini_swe_agent") or {}
    pier = locks.get("pier") or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "running": current_task is not None,
        "current_task": current_task,
        "method": manifest.get("method", "official Pier + MiniSweAgent"),
        "binary_sha": (manifest.get("artifacts") or {}).get(
            "qw3_binary_sha256"
        ),
        "seed": (
            attempt_seed(latest_attempt(run / current_task)[1], active_seed)
            if current_task
            else active_seed
        ),
        "rollout_columns": [
            {"label": label, "seed": seed} for label, seed, _path in rollout_runs
        ],
        "versions": {"mini_swe_agent": mini.get("version", "—"), "pier": pier.get("version", "—")},
        "summary": {
            "total": len(tasks), "completed": completed, "passed": passed,
            "total_rollouts": len(tasks) * len(rollout_runs),
            "completed_rollouts": completed_rollouts,
            "passed_rollouts": passed_rollouts,
            "current_requests": current_requests, "current_prompt_tokens": current_prompt,
            "serving_errors": serving_errors, "current_compactions": current_compactions,
        },
        "tasks": tasks,
        "agent_log": current_agent,
        "server_log": current_server,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--rollout-run",
        action="append",
        default=[],
        metavar="LABEL:SEED:PATH",
        help=(
            "append an independent rollout column; PATH may be a planned run "
            "directory that does not exist yet"
        ),
    )
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("requestplan10.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55792)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    manifest = read_json(run / "manifest.json", {}) or {}
    manifest_seeds = manifest.get("rollout_seeds")
    base_seed = (
        int(manifest_seeds[0])
        if isinstance(manifest_seeds, list) and manifest_seeds
        else 73
    )
    rollout_runs: list[tuple[str, int, Path]] = [("R1", base_seed, run)]
    for value in args.rollout_run:
        try:
            label, seed_text, path_text = value.split(":", 2)
            rollout_runs.append((label, int(seed_text), Path(path_text).resolve()))
        except (TypeError, ValueError) as error:
            raise SystemExit(
                f"invalid --rollout-run {value!r}; expected LABEL:SEED:PATH"
            ) from error
    plan = read_json(args.plan.resolve(), {}) or {}
    selected_ids = manifest.get("task_ids")
    if isinstance(selected_ids, list) and selected_ids:
        selected = {str(task_id) for task_id in selected_ids}
        plan["tasks"] = [
            item for item in plan.get("tasks", [])
            if str(item.get("task_id")) in selected
        ]
    difficulty = difficulty_index(repo)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/api/state":
                body = json.dumps(
                    state(repo, rollout_runs, plan, difficulty), ensure_ascii=False
                ).encode()
                content_type = "application/json; charset=utf-8"
            elif self.path.split("?", 1)[0] in {"/", "/index.html"}:
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MiniSweAgent dashboard: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
