#!/usr/bin/env bash
# Capture and analyze KVMem retrieval for the exact 52 IDs in rag_blocks_52_delivery.
# Generation is capped at one token: selection is finalized after prefill, so this
# preserves retrieval while avoiding a second full answer-generation experiment.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
TAG=${TAG:-agentlongbench_rag52_kvmem32k_overlap_20260718}
DELIVERY=${DELIVERY:-/home/chaidi/kvmem_eval/rag_blocks_52_delivery}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl}
RESULT_BASE=${RESULT_BASE:-/data/chaidi/kvmem_eval/results}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
MODEL=${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}
NVME_DIR=${NVME_DIR:-/data/chaidi/kvmem_eval/nvme/${TAG}}
OUTPUT=${OUTPUT:-${RESULT_BASE}/${TAG}}
DUMP_FILE=${DUMP_FILE:-${OUTPUT}/kvmem_retrieval_dump.jsonl}
SERVER_LOG=${SERVER_LOG:-${LOG_BASE}/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-${LOG_BASE}/${TAG}_runner.log}
LIMIT=${LIMIT:-}
GPU_RATIO=${GPU_RATIO:-0.9}

mkdir -p "$OUTPUT" "$LOG_BASE" "$NVME_DIR"

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already has a healthy server; refusing to reuse it" >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date '+%F %T')] start exact_rag52 kvmem_budget=32768 block=32 mean-k max_tokens=1" \
  | tee -a "$RUN_LOG"
# The current capacity guard needs a higher ceiling to allocate the same
# effective 2048-page pool (1024 selection + 1024 generation reserve) used by
# the original 32K/32K run. This changes only the ceiling, not either budget.
env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_DUMP_SCORES="$DUMP_FILE" \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 262144 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 --kvmem-budget 32768 \
    --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-gpu-memory-ratio "$GPU_RATIO" \
    --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir "$NVME_DIR" \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >>"$SERVER_LOG" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
if ! curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "qw3 health timeout" >&2
  exit 1
fi

capture_args=(
  "$ROOT/scripts/kvmem_eval/capture_agentlongbench_rag52_retrieval.py"
  --delivery "$DELIVERY"
  --dataset "$DATA"
  --output-root "$OUTPUT"
  --dump-file "$DUMP_FILE"
  --api-base "http://127.0.0.1:${PORT}/v1"
  --model "$(basename "$MODEL")"
  --timeout-sec 3600
  --kvmem-gpu-memory-ratio "$GPU_RATIO"
)
if [[ -n "$LIMIT" ]]; then
  capture_args+=(--limit "$LIMIT")
fi
python3 "${capture_args[@]}" 2>&1 | tee -a "$RUN_LOG"

# A limited smoke capture intentionally stops before the exact-52 analysis.
if [[ -n "$LIMIT" ]]; then
  echo "[$(date '+%F %T')] limited capture complete: ${LIMIT}" | tee -a "$RUN_LOG"
  exit 0
fi

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/analyze_agentlongbench_rag52_overlap.py" \
  --delivery "$DELIVERY" \
  --dataset "$DATA" \
  --kvmem-dump "$DUMP_FILE" \
  --output-root "$OUTPUT/analysis" 2>&1 | tee -a "$RUN_LOG"

echo "[$(date '+%F %T')] complete analysis=$OUTPUT/analysis" | tee -a "$RUN_LOG"
