#!/usr/bin/env bash
# Sequential controlled A/B on a balanced AgentLongBench 512K cross-task set.
# Both arms reuse historical KV cache.  The second arm only rewinds/recomputes
# the final query suffix against the already selected KVMem window; it never
# performs a dense full-context KV rebuild.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
TAG=${TAG:-agentlongbench_512k_cross_task16_k224k_b32_20260720}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_cross_task_16/samples.jsonl}
MANIFEST=${MANIFEST:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_cross_task_16/manifest.jsonl}
RESULT_BASE=${RESULT_BASE:-/data/chaidi/kvmem_eval/results}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
MODEL=${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}
NVME_BASE=${NVME_BASE:-/data/chaidi/kvmem_eval/nvme/${TAG}}
PID_FILE=${LOG_BASE}/${TAG}.pid
RUN_LOG=${LOG_BASE}/${TAG}_runner.log

mkdir -p "$LOG_BASE" "$RESULT_BASE" "$NVME_BASE"
echo $$ >"$PID_FILE"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

gpu_compute_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
cleanup() {
  cleanup_server
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

echo "[$(timestamp)] queued tag=${TAG}; waiting for current GPU job" | tee -a "$RUN_LOG"
while true; do
  used=$(gpu_compute_mib)
  if [[ "$used" -lt 8192 ]]; then
    break
  fi
  echo "[$(timestamp)] waiting_for_gpu compute_memory_mib=${used}" | tee -a "$RUN_LOG"
  sleep 30
done

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already has a healthy server; refusing to reuse it" | tee -a "$RUN_LOG"
  exit 1
fi

run_arm() {
  local arm=$1
  local replay=$2
  local out=${RESULT_BASE}/${TAG}_${arm}
  local server_log=${LOG_BASE}/${TAG}_${arm}_server.log
  local nvme_dir=${NVME_BASE}/${arm}
  mkdir -p "$out" "$nvme_dir"

  echo "[$(timestamp)] starting arm=${arm} query_replay=${replay}" | tee -a "$RUN_LOG"
  env \
    QW3_KVMEM_RECOMPUTE_QUERY="$replay" \
    QW3_KVMEM_TRACE=1 \
    QW3_FATTN_NSPLIT=1 \
    QW3_PREFILL_FA2_NSPLIT=1 \
    QW3_KVMEM_TIMING=1 \
    "$ROOT/build/qw3" serve \
      --model "$MODEL" \
      --ctx 1048576 --kv-dtype fp16 \
      --kvmem --kvmem-block-tokens 32 --kvmem-budget 229376 \
      --kvmem-gen-budget 32768 \
      --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
      --kvmem-method retrieval --kvmem-retrieval-method mean-k \
      --kvmem-update-mode step --kvmem-query-conditioned \
      --kvmem-gpu-memory-ratio 0.51 \
      --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir "$nvme_dir" \
      --enable-thinking --thinking-budget 4096 \
      --prefill-chunk 2048 --temp 0.6 \
      --native-mtp-speculate --mtp-chain 4 \
      --host 127.0.0.1 --port "$PORT" >>"$server_log" 2>&1 &
  server_pid=$!

  local healthy=0
  for _ in $(seq 1 180); do
    if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
      healthy=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$(timestamp)] qw3 exited during startup arm=${arm}" | tee -a "$RUN_LOG"
      tail -100 "$server_log" | tee -a "$RUN_LOG"
      return 1
    fi
    sleep 5
  done
  if [[ "$healthy" -ne 1 ]]; then
    echo "[$(timestamp)] qw3 health timeout arm=${arm}" | tee -a "$RUN_LOG"
    tail -100 "$server_log" | tee -a "$RUN_LOG"
    return 1
  fi

  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
    --dataset "$DATA" --manifest "$MANIFEST" \
    --allow-custom-subset --benchmark-name AgentLongBench-512K-cross-task-16 \
    --output-root "$out" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --model "$(basename "$MODEL")" \
    --method "kvmem_mean_k_224k_b32_${arm}" \
    --temperature 0.6 --top-p 0.95 --max-tokens 32768 \
    --context-window 1048576 --timeout-sec 7200 --max-sample-sec 7200 \
    --attempts 3 --enable-thinking --seed 20260720 \
    --kvmem-retrieval-trace-metadata 2>&1 | tee -a "$RUN_LOG"

  "$ROOT/.venv/bin/python" - "$out/validation_report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("passed") or report.get("answers_unique") != 16 or report.get("eval_unique") != 16:
    raise SystemExit(f"arm validation failed: {report}")
print("arm validation passed: 16 answers and 16 evaluations")
PY
  cleanup_server
  sleep 5
}

run_arm original 0
run_arm query_replay 1

echo "[$(timestamp)] complete original=${RESULT_BASE}/${TAG}_original query_replay=${RESULT_BASE}/${TAG}_query_replay" | tee -a "$RUN_LOG"
