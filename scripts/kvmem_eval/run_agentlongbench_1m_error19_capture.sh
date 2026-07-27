#!/usr/bin/env bash
# Capture final selected blocks for the 19 DS50 samples where KVMem was wrong
# and at least one frozen baseline was correct.  Retrieval/generation-state
# parameters match the completed 1M KVMem run; generation is capped at one token.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8088}
TAG=${TAG:-agentlongbench_1m_ds50_kvmem_error19_selected_dump_${STAMP}}
DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
IDS_FILE=${IDS_FILE:-$ROOT/scripts/kvmem_eval/agentlongbench_1m_kvmem_error19_ids.txt}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
OUTPUT=${OUTPUT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
DUMP_FILE=${DUMP_FILE:-$OUTPUT/kvmem_retrieval_dump.jsonl}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}
LIMIT=${LIMIT:-}

mkdir -p "$OUTPUT" "$LOG_ROOT"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$DUMP_FILE" \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 1310720 --kv-dtype fp8 \
    --kvmem --kvmem-block-tokens 32 \
    --kvmem-budget 229376 --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-immutable-k --kvmem-gpu-memory-ratio 0.5 \
    --kvmem-cpu-gb 80 \
    --enable-thinking --thinking-budget 8192 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
if [[ "$healthy" -ne 1 ]]; then
  echo "qw3 health timeout; see $SERVER_LOG" >&2
  exit 4
fi

capture_args=(
  "$ROOT/scripts/kvmem_eval/capture_agentlongbench_selected_retrieval.py"
  --ids-file "$IDS_FILE"
  --dataset "$DATA"
  --output-root "$OUTPUT"
  --dump-file "$DUMP_FILE"
  --api-base "http://127.0.0.1:$PORT/v1"
  --model "$(basename "$MODEL")"
  --timeout-sec 7200
  --thinking-budget 8192
  --temperature 0.6
  --top-p 0.95
  --seed 20260722
)
if [[ -n "$LIMIT" ]]; then
  capture_args+=(--limit "$LIMIT")
fi
"$ROOT/.venv/bin/python" "${capture_args[@]}" 2>&1 | tee -a "$RUN_LOG"
