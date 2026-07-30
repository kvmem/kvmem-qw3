#!/usr/bin/env bash
# One-sample AgentLongBench A/B for mean-K GPU residency vs CPU offload.
# The two arms are identical except for --kvmem-index-placement. Generation is
# capped at one token so TTFT primarily measures prompt ingestion + retrieval.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-/tmp/qw3_mean_index_offload_ab_${STAMP}}
PORT=${PORT:-18089}
QUESTION_ID=${QUESTION_ID:-adb0765b3c59611b3b9923d5b06e6dfddf70021a16c0ea948c219c40085642e5}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
CTX=${CTX:-655360}
KVMEM_BUDGET=${KVMEM_BUDGET:-229376}
GEN_BUDGET=${GEN_BUDGET:-32768}
STAGING_MB=${STAGING_MB:-64}
CPU_GB=${CPU_GB:-64}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
MAX_TOKENS=${MAX_TOKENS:-1}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-0.2}

mkdir -p "$OUT_ROOT"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

server_pid=""
monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup EXIT INT TERM

gpu_snapshot() {
  local phase=$1
  local target=$2
  local reading
  reading=$(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1
  )
  printf '%s,%s,%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S%z)" "$phase" "$reading" >>"$target"
}

start_monitor() {
  local target=$1
  (
    echo "unix_time,memory_used_mib,utilization_gpu_pct"
    while true; do
      reading=$(
        nvidia-smi \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | head -n 1
      ) || true
      if [[ -n "${reading:-}" ]]; then
        printf '%s,%s\n' "$(date +%s.%N)" "$reading"
      fi
      sleep "$MONITOR_INTERVAL"
    done
  ) >"$target" &
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
  local server_log="$arm_dir/server.log"
  local runner_log="$arm_dir/runner.log"
  local gpu_log="$arm_dir/gpu_samples.csv"
  local snapshots="$arm_dir/gpu_snapshots.csv"
  mkdir -p "$arm_dir/results"
  echo "timestamp,phase,memory_used_mib,utilization_gpu_pct" >"$snapshots"
  gpu_snapshot before_server "$snapshots"

  echo "[$(date --iso-8601=seconds)] START placement=$placement"
  env \
    -u QW3_KVMEM_REBUILT_STATE_DIR \
    QW3_Q8_BF16_MAIN=0 \
    QW3_KVMEM_RECOMPUTE_QUERY=1 \
    QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
    QW3_KVMEM_TRACE=1 \
    QW3_KVMEM_PERF_TRACE=1 \
    QW3_KVMEM_TIMING=1 \
    QW3_FATTN_NSPLIT=1 \
    QW3_PREFILL_FA2_NSPLIT=1 \
    QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
    "$ROOT/build/qw3" serve \
      --model "$MODEL" \
      --ctx "$CTX" --kv-dtype fp16 \
      --kvmem --kvmem-block-tokens 32 \
      --kvmem-budget "$KVMEM_BUDGET" \
      --kvmem-gen-budget "$GEN_BUDGET" \
      --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
      --kvmem-method retrieval --kvmem-retrieval-method mean-k \
      --kvmem-index-placement "$placement" \
      --kvmem-index-staging-mb "$STAGING_MB" \
      --kvmem-update-mode step --kvmem-query-conditioned \
      --kvmem-immutable-k --kvmem-gpu-memory-ratio 0.51 \
      --kvmem-cpu-gb "$CPU_GB" \
      --enable-thinking --thinking-budget 4096 \
      --prefill-chunk "$PREFILL_CHUNK" --temp 0 \
      --native-mtp-speculate --mtp-chain 4 \
      --host 127.0.0.1 --port "$PORT" \
      >"$server_log" 2>&1 &
  server_pid=$!

  local healthy=0
  for _ in $(seq 1 300); do
    if curl -fsS --noproxy '*' \
        "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      healthy=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "qw3 exited during startup; see $server_log" >&2
      tail -100 "$server_log" >&2 || true
      exit 4
    fi
    sleep 2
  done
  if [[ "$healthy" -ne 1 ]]; then
    echo "qw3 health timeout; see $server_log" >&2
    exit 4
  fi
  if ! grep -q "kvmem_index_placement=$placement" "$server_log"; then
    echo "server did not confirm placement=$placement" >&2
    exit 5
  fi

  sleep 2
  gpu_snapshot server_ready "$snapshots"
  start_monitor "$gpu_log"

  "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
      --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
      --dataset "$DATA" --manifest "$MANIFEST" --allow-custom-subset \
      --benchmark-name AgentLongBench-512K-normal100 \
      --output-root "$arm_dir/results" \
      --api-base "http://127.0.0.1:$PORT/v1" \
      --model "$(basename "$MODEL")" \
      --method "mean_index_${placement}" \
      --temperature 0 --top-p 0.95 --max-tokens "$MAX_TOKENS" \
      --context-window "$CTX" --context-safety-margin 16 \
      --timeout-sec 3600 --max-sample-sec 3600 --attempts 1 \
      --enable-thinking --seed 20260729 \
      --kvmem-retrieval-trace-metadata \
      --question-id "$QUESTION_ID" \
      2>&1 | tee "$runner_log"

  sleep 2
  gpu_snapshot post_request "$snapshots"
  stop_monitor
  cleanup
  sleep 2
  gpu_snapshot post_server "$snapshots"
  echo "[$(date --iso-8601=seconds)] DONE placement=$placement"
}

run_arm gpu
run_arm cpu

"$ROOT/.venv/bin/python" - "$OUT_ROOT" "$STAGING_MB" <<'PY'
import csv
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
staging_mb = int(sys.argv[2])
rows = {}
for placement in ("gpu", "cpu"):
    arm = root / placement
    answer = json.loads((arm / "results" / "answers.jsonl").read_text().splitlines()[0])
    snapshots = {}
    with (arm / "gpu_snapshots.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            snapshots[row["phase"]] = int(row["memory_used_mib"].strip())
    gpu_samples = []
    with (arm / "gpu_samples.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            gpu_samples.append(int(row["memory_used_mib"].strip()))
    server_log = (arm / "server.log").read_text(errors="replace")
    score_ms = [
        float(x)
        for x in re.findall(r"\[bs-mean-score-perf\].*?elapsed_ms=([0-9.]+)", server_log)
    ]
    stream = [
        {
            "transferred_mib": float(a),
            "pack_ms": float(b),
            "submit_total_ms": float(c),
        }
        for a, b, c in re.findall(
            r"\[bs-mean-stream\].*?transferred_mib=([0-9.]+)"
            r".*?pack_ms=([0-9.]+).*?submit_total_ms=([0-9.]+)",
            server_log,
        )
    ]
    rows[placement] = {
        "prompt_tokens": answer.get("prompt_tokens"),
        "actual_prompt_tokens": answer.get("actual_prompt_tokens"),
        "ttft_sec": answer["timing"]["ttft_sec"],
        "total_sec": answer["timing"]["total_sec"],
        "server_ready_mib": snapshots["server_ready"],
        "post_request_mib": snapshots["post_request"],
        "request_peak_mib": max(gpu_samples),
        "mean_score_ms": score_ms,
        "mean_score_total_ms": sum(score_ms),
        "mean_score_median_ms": statistics.median(score_ms) if score_ms else None,
        "stream_passes": stream,
    }

gpu = rows["gpu"]
cpu = rows["cpu"]
summary = {
    "config": {
        "sample_id": answer["stable_sample_id"],
        "ctx": 655360,
        "index_staging_mb_per_slot": staging_mb,
        "generation_tokens": 1,
    },
    "arms": rows,
    "delta_cpu_minus_gpu": {
        "post_request_mib": cpu["post_request_mib"] - gpu["post_request_mib"],
        "request_peak_mib": cpu["request_peak_mib"] - gpu["request_peak_mib"],
        "ttft_sec": cpu["ttft_sec"] - gpu["ttft_sec"],
        "ttft_percent": (cpu["ttft_sec"] / gpu["ttft_sec"] - 1.0) * 100.0,
        "mean_score_total_ms": (
            cpu["mean_score_total_ms"] - gpu["mean_score_total_ms"]
        ),
        "mean_score_slowdown": (
            cpu["mean_score_total_ms"] / gpu["mean_score_total_ms"]
            if gpu["mean_score_total_ms"] else None
        ),
    },
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "summary=$OUT_ROOT/summary.json"
