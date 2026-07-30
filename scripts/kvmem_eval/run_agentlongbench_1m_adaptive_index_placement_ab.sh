#!/usr/bin/env bash
# Controlled one-sample AgentLongBench-1M A/B for Adaptive index placement.
# The two arms are identical except for --kvmem-index-placement. Generation is
# capped at one token so the measurement isolates prompt ingestion, index
# construction, query-conditioned scoring, reselection, and query replay.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
QUESTION_ID=${QUESTION_ID:-2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021}
OUT_ROOT=${OUT_ROOT:-/data/chaidi/kvmem_eval/ab/adaptive_index_placement_1m_${STAMP}}
PORT=${PORT:-18092}
STAGING_MB=${STAGING_MB:-64}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-0.2}

mkdir -p "$OUT_ROOT"

baseline_pids=$(
  nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null |
  tr -d ' ' | paste -sd, -
)
baseline_pids=",${baseline_pids},"

monitor_pid=""
arm_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
  if [[ -n "$arm_pid" ]] && kill -0 "$arm_pid" 2>/dev/null; then
    kill "$arm_pid" 2>/dev/null || true
    wait "$arm_pid" 2>/dev/null || true
  fi
  arm_pid=""
}
trap cleanup EXIT INT TERM

start_monitor() {
  local output=$1
  (
    echo "unix_time,total_gpu_mib,qw3_pid,qw3_mib"
    while true; do
      local_total=$(
        nvidia-smi --query-gpu=memory.used \
          --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d ' '
      ) || true
      local_pid=""
      local_mem="0"
      while IFS=',' read -r pid mem; do
        pid=${pid//[[:space:]]/}
        mem=${mem//[[:space:]]/}
        [[ -z "$pid" ]] && continue
        if [[ "$baseline_pids" != *",$pid,"* ]]; then
          local_pid=$pid
          local_mem=$mem
          break
        fi
      done < <(
        nvidia-smi --query-compute-apps=pid,used_memory \
          --format=csv,noheader,nounits 2>/dev/null || true
      )
      printf '%s,%s,%s,%s\n' \
        "$(date +%s.%N)" "${local_total:-0}" \
        "$local_pid" "$local_mem"
      sleep "$MONITOR_INTERVAL"
    done
  ) >"$output" &
  monitor_pid=$!
}

stop_monitor() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
}

run_arm() {
  local placement=$1
  local arm_dir="$OUT_ROOT/$placement"
  local tag="adaptive_index_ab_${placement}_${STAMP}"
  mkdir -p "$arm_dir"
  start_monitor "$arm_dir/gpu_memory.csv"

  echo "[$(date --iso-8601=seconds)] START placement=$placement" |
    tee "$arm_dir/ab.log"
  env \
    PORT="$PORT" \
    TAG="$tag" \
    RESULT_ROOT="$arm_dir/results" \
    SERVER_LOG="$arm_dir/server.log" \
    RUN_LOG="$arm_dir/runner.log" \
    PID_FILE="$arm_dir/launcher.pid" \
    QUESTION_IDS="$QUESTION_ID" \
    EXPECTED=1 \
    MAX_TOKENS=1 \
    KVMEM_INDEX_PLACEMENT="$placement" \
    KVMEM_INDEX_STAGING_MB="$STAGING_MB" \
    QW3_KVMEM_PERF_TRACE=1 \
    QW3_KVMEM_DUMP_SCORES="$arm_dir/retrieval_scores.jsonl" \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_adaptive_10.sh" \
    >>"$arm_dir/ab.log" 2>&1 &
  arm_pid=$!
  set +e
  wait "$arm_pid"
  local status=$?
  set -e
  arm_pid=""
  stop_monitor
  if [[ "$status" -ne 0 ]]; then
    echo "placement=$placement failed with status=$status" >&2
    tail -100 "$arm_dir/ab.log" >&2 || true
    exit "$status"
  fi
  echo "[$(date --iso-8601=seconds)] DONE placement=$placement" |
    tee -a "$arm_dir/ab.log"
}

run_arm gpu
sleep 5
run_arm cpu

"$ROOT/.venv/bin/python" - "$OUT_ROOT" "$STAGING_MB" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
staging_mb = int(sys.argv[2])
summary = {"staging_mb_per_slot": staging_mb, "arms": {}}

def last_float(pattern, text):
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None

def last_int(pattern, text):
    values = re.findall(pattern, text)
    return int(values[-1]) if values else None

for placement in ("gpu", "cpu"):
    arm = root / placement
    log = (arm / "server.log").read_text(errors="replace")
    runner_log = (arm / "runner.log").read_text(errors="replace")
    memory_rows = list(csv.DictReader((arm / "gpu_memory.csv").open()))
    process_values = [
        int(row["qw3_mib"])
        for row in memory_rows
        if row.get("qw3_pid") and int(row["qw3_mib"]) > 0
    ]
    answer_path = arm / "results" / "answers.jsonl"
    answer = (
        json.loads(answer_path.read_text().splitlines()[0])
        if answer_path.exists() and answer_path.stat().st_size
        else {}
    )
    selected = []
    dump_path = arm / "retrieval_scores.jsonl"
    if dump_path.exists():
        for line in dump_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "meta":
                selected = []
            elif row.get("sel") == 1 and "b" in row:
                selected.append(int(row["b"]))

    index_gib = (
        last_float(r"\[bs-adaptive-index\].*?\bgib=([0-9.]+)", log)
        if placement == "gpu"
        else last_float(
            r"\[bs-adaptive-index\].*?host_gib=([0-9.]+)", log
        )
    )
    server_ready_mib = last_int(
        r"\[gpu-memory\] phase=server_ready .*?used_mib=(\d+)",
        runner_log,
    )
    peak_process_gpu_mib = max(process_values) if process_values else None
    arm_summary = {
        "prompt_tokens": last_int(
            r"\[qw3\] native generate: prompt_tokens=(\d+)", log
        ),
        "prefill_sec": last_float(
            r"\[qw3\] native generate:.*?prefill=([0-9.]+)s", log
        ),
        "prefill_tok_per_sec": last_float(
            r"\[qw3\] native generate:.*?\(([0-9.]+) tok/s\)", log
        ),
        "request_total_sec": (
            answer.get("timing", {}).get("total_sec")
            if isinstance(answer, dict)
            else None
        ),
        "request_ttft_sec": (
            answer.get("timing", {}).get("ttft_sec")
            if isinstance(answer, dict)
            else None
        ),
        "server_ready_gpu_mib": server_ready_mib,
        "peak_process_gpu_mib": peak_process_gpu_mib,
        "request_gpu_delta_mib": (
            peak_process_gpu_mib - server_ready_mib
            if peak_process_gpu_mib is not None
            and server_ready_mib is not None
            else None
        ),
        "index_authority_gib": index_gib,
        "index_gpu_resident_gib": index_gib if placement == "gpu" else 0.0,
        "index_finalize_ms": last_float(
            r"\[bs-adaptive-index\].*?finalize_ms=([0-9.]+)", log
        ),
        "index_upload_ms": last_float(
            r"\[bs-adaptive-upload\].*?total_ms=([0-9.]+)", log
        ),
        "incremental_h2d_gib": last_float(
            r"\[bs-adaptive-index\].*?incremental_h2d_gib=([0-9.]+)",
            log,
        ),
        "incremental_pack_ms": last_float(
            r"\[bs-adaptive-index\].*?pack_ms=([0-9.]+)", log
        ) if placement == "gpu" else None,
        "incremental_h2d_wait_ms": last_float(
            r"\[bs-adaptive-index\].*?h2d_wait_ms=([0-9.]+)", log
        ) if placement == "gpu" else None,
        "gpu_layer_score_submit_ms": last_float(
            r"\[bs-adaptive-score\].*?submit_ms=([0-9.]+)", log
        ),
        "score_ms": last_float(
            r"\[bs-mean-score-perf\].*?elapsed_ms=([0-9.]+)", log
        ),
        "stream_ms": last_float(
            r"\[bs-adaptive-stream\].*?total_ms=([0-9.]+)", log
        ),
        "stream_transferred_gib": last_float(
            r"\[bs-adaptive-stream\].*?transferred_gib=([0-9.]+)", log
        ),
        "stream_pack_ms": last_float(
            r"\[bs-adaptive-stream\].*?pack_ms=([0-9.]+)", log
        ),
        "stream_h2d_wait_ms": last_float(
            r"\[bs-adaptive-stream\].*?h2d_wait_ms=([0-9.]+)", log
        ),
        "semantic_reselection_ms": last_float(
            r"\[kvmem-reselect-perf\] kind=semantic .*?total_ms=([0-9.]+)",
            log,
        ),
        "semantic_stage_in_ms": last_float(
            r"\[kvmem-reselect-perf\] kind=semantic .*?stage_in_wall_ms=([0-9.]+)",
            log,
        ),
        "fallback_count": len(
            re.findall(r"\[kvmem-scorer\].*?\bfallback=1\b", log)
        ),
        "selected_blocks": selected,
        "selected_block_count": len(selected),
    }
    summary["arms"][placement] = arm_summary

gpu = summary["arms"]["gpu"]
cpu = summary["arms"]["cpu"]
gpu_selected = set(gpu["selected_blocks"])
cpu_selected = set(cpu["selected_blocks"])
summary["comparison"] = {
    "selected_identical": gpu_selected == cpu_selected,
    "selected_intersection": len(gpu_selected & cpu_selected),
    "selected_union": len(gpu_selected | cpu_selected),
    "peak_gpu_saved_mib": (
        gpu["peak_process_gpu_mib"] - cpu["peak_process_gpu_mib"]
        if gpu["peak_process_gpu_mib"] is not None
        and cpu["peak_process_gpu_mib"] is not None
        else None
    ),
    "score_cpu_over_gpu": (
        cpu["score_ms"] / gpu["score_ms"]
        if gpu["score_ms"] and cpu["score_ms"]
        else None
    ),
    "prefill_cpu_over_gpu": (
        cpu["prefill_sec"] / gpu["prefill_sec"]
        if gpu["prefill_sec"] and cpu["prefill_sec"]
        else None
    ),
}

(root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)

lines = [
    "# Adaptive index placement A/B",
    "",
    "| Metric | GPU index | CPU index |",
    "|---|---:|---:|",
]
for label, key, unit in (
    ("Server-ready VRAM", "server_ready_gpu_mib", "MiB"),
    ("Peak process VRAM", "peak_process_gpu_mib", "MiB"),
    ("Request VRAM delta", "request_gpu_delta_mib", "MiB"),
    ("Index authority", "index_authority_gib", "GiB"),
    ("Prefill", "prefill_sec", "s"),
    ("Prefill throughput", "prefill_tok_per_sec", "tok/s"),
    ("Index finalize", "index_finalize_ms", "ms"),
    ("Index upload", "index_upload_ms", "ms"),
    ("Incremental index H2D", "incremental_h2d_gib", "GiB"),
    ("Incremental host pack", "incremental_pack_ms", "ms"),
    ("Incremental H2D wait", "incremental_h2d_wait_ms", "ms"),
    ("GPU layer score submit", "gpu_layer_score_submit_ms", "ms"),
    ("Adaptive score", "score_ms", "ms"),
    ("CPU stream transfer", "stream_transferred_gib", "GiB"),
    ("Semantic reselection", "semantic_reselection_ms", "ms"),
):
    gv = gpu.get(key)
    cv = cpu.get(key)
    gs = "-" if gv is None else f"{gv:.3f} {unit}"
    cs = "-" if cv is None else f"{cv:.3f} {unit}"
    lines.append(f"| {label} | {gs} | {cs} |")
lines.extend(
    [
        "",
        f"- Selected blocks identical: {summary['comparison']['selected_identical']}",
        f"- GPU fallback count: {gpu['fallback_count']}",
        f"- CPU fallback count: {cpu['fallback_count']}",
    ]
)
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "A/B complete: $OUT_ROOT"
