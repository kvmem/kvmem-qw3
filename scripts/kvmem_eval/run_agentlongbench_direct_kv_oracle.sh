#!/usr/bin/env bash
# Controlled direct-KV oracle experiment for the first local-evidence failures.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
TAG=${ORACLE_TAG:-agentlongbench_1m_error3_direct_kv_oracle_chain_${STAMP}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}
DUMP_FILE=${DUMP_FILE:-$OUTPUT_ROOT/kvmem_retrieval_dump.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
IDS_FILE=${IDS_FILE:-$ROOT/scripts/kvmem_eval/agentlongbench_1m_kvmem_error19_ids.txt}
DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
PORT=${PORT:-8088}
LIMIT=${LIMIT:-3}
ORACLE_ONLY=${ORACLE_ONLY:-0}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi
if [[ -e "$DUMP_FILE" ]]; then
  echo "oracle dump already exists: $DUMP_FILE" >&2
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

{
  echo "[oracle-config] tag=$TAG"
  echo "[oracle-config] output_root=$OUTPUT_ROOT"
  echo "[oracle-config] limit=$LIMIT"
  echo "[oracle-config] oracle_only=$ORACLE_ONLY"
  echo "[oracle-config] direct historical KV, 224K context, 32K generation"
} | tee -a "$RUN_LOG"

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_ENABLE_ORACLE=1 \
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

for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
curl -fsS --noproxy '*' \
  "http://127.0.0.1:$PORT/health" >/dev/null

oracle_args=()
if [[ "$ORACLE_ONLY" == "1" ]]; then
  oracle_args+=(--oracle-only)
fi

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem_oracle.py" \
  --dataset "$DATA" \
  --ids-file "$IDS_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --dump-file "$DUMP_FILE" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --limit "$LIMIT" \
  --timeout-sec 7200 \
  "${oracle_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"
