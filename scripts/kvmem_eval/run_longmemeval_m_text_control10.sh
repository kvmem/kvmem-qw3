#!/usr/bin/env bash
# Capture exact 224K mean-k selections for ten frozen LongMemEval-M failures,
# then densely prefill the decoded selected text and grade with the same judge.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
TAG=${TAG:-longmemeval_m_k224k_selected_text_control10_20260720}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl}
BASELINE=${BASELINE:-/data/chaidi/kvmem_eval/results/m102_2m_k224k_g32k_bt32_meank_r051_20260719_eval_20260719_063739.jsonl}
MODEL=${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}
OUTPUT=${OUTPUT:-/data/chaidi/kvmem_eval/results/${TAG}}
LOG_DIR=${LOG_DIR:-/data/chaidi/kvmem_eval/logs}
DUMP_FILE=${DUMP_FILE:-${OUTPUT}/kvmem_retrieval_dump.jsonl}
NVME_DIR=${NVME_DIR:-/data/chaidi/kvmem_eval/nvme/${TAG}}
LIMIT=${LIMIT:-}
INDICES=${INDICES:-4,6,20,27,34,36,51,60,68,86}

mkdir -p "$OUTPUT" "$LOG_DIR" "$NVME_DIR"
RUN_LOG=${LOG_DIR}/${TAG}_runner.log
CAPTURE_SERVER_LOG=${LOG_DIR}/${TAG}_capture_server.log
REPLAY_SERVER_LOG=${LOG_DIR}/${TAG}_replay_server.log

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already has a healthy server; refusing to reuse it" >&2
  exit 1
fi

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
cleanup() { stop_server; }
trap cleanup EXIT INT TERM

wait_for_health() {
  local log=$1
  for _ in $(seq 1 180); do
    if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$log" >&2
      return 1
    fi
    sleep 5
  done
  echo "qw3 health timeout" >&2
  tail -100 "$log" >&2
  return 1
}

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args+=(--limit "$LIMIT")
fi

echo "[$(date '+%F %T')] capture start indices=${INDICES} limit=${LIMIT:-all}" | tee -a "$RUN_LOG"
env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$DUMP_FILE" \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 2000000 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 --kvmem-budget 229376 \
    --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-gpu-memory-ratio 0.51 \
    --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir "$NVME_DIR" \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >>"$CAPTURE_SERVER_LOG" 2>&1 &
server_pid=$!
wait_for_health "$CAPTURE_SERVER_LOG"

python3 "$ROOT/scripts/kvmem_eval/capture_longmemeval_selected_retrieval.py" \
  --data "$DATA" --baseline "$BASELINE" --indices "$INDICES" \
  --output-root "$OUTPUT" --dump-file "$DUMP_FILE" \
  --api-base "http://127.0.0.1:${PORT}/v1" --model "$(basename "$MODEL")" \
  --timeout-sec 7200 "${limit_args[@]}" 2>&1 | tee -a "$RUN_LOG"

stop_server
echo "[$(date '+%F %T')] dense selected-text replay start" | tee -a "$RUN_LOG"
env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 262144 --kv-dtype fp16 \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >>"$REPLAY_SERVER_LOG" 2>&1 &
server_pid=$!
wait_for_health "$REPLAY_SERVER_LOG"

DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_longmemeval_selected_text_replay.py" \
    --data "$DATA" --baseline "$BASELINE" --dump-file "$DUMP_FILE" \
    --output-root "$OUTPUT" --api-base "http://127.0.0.1:${PORT}/v1" \
    --model "$(basename "$MODEL")" --context-window 262144 \
    --max-tokens 32768 --thinking-budget 4096 --temperature 0.6 --top-p 0.95 \
    --timeout-sec 7200 "${limit_args[@]}" 2>&1 | tee -a "$RUN_LOG"

stop_server
echo "[$(date '+%F %T')] complete output=${OUTPUT}" | tee -a "$RUN_LOG"
