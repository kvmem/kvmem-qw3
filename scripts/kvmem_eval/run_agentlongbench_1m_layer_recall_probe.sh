#!/usr/bin/env bash
# Retrieval-only per-layer diagnostic on eight grounded AgentLongBench-1M
# samples. Generation is capped at one token because the production KVMem
# selection and the layer score dump are complete before decode begins.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${TAG:-agentlongbench_1m_layer_recall_q8_fp8_b512_sb16_k224k_20260728}
PORT=${PORT:-8093}
LIMIT=${LIMIT:-8}
DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
IDS_FILE=${IDS_FILE:-$ROOT/scripts/kvmem_eval/agentlongbench_1m_layer_probe_ids.txt}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
BENCHMARK_REPO=${BENCHMARK_REPO:-/home/chaidi/AgentLongBench_Motivation}
TOKENIZER=${TOKENIZER:-/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_capture.log}
DUMP_FILE=${DUMP_FILE:-$OUTPUT_ROOT/layer_scores.jsonl}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-$OUTPUT_ROOT/layer_recall}

if [[ ! "$LIMIT" =~ ^[1-8]$ ]]; then
  echo "LIMIT must be an integer in [1, 8], got: $LIMIT" >&2
  exit 2
fi
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

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
  QW3_KVMEM_DUMP_LAYER_SCORES="$DUMP_FILE" \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 1310720 --kv-dtype fp8 \
    --kvmem --kvmem-block-tokens 512 \
    --kvmem-budget 229376 --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval \
    --kvmem-retrieval-method sub-block-mean-k \
    --kvmem-subblocks 16 --kvmem-subblock-reduce max \
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
    echo "qw3 exited during startup; see $SERVER_LOG" >&2
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
if [[ "$healthy" -ne 1 ]]; then
  echo "qw3 health timeout; see $SERVER_LOG" >&2
  exit 4
fi

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/capture_agentlongbench_selected_retrieval.py" \
  --ids-file "$IDS_FILE" \
  --dataset "$DATA" \
  --benchmark-repo "$BENCHMARK_REPO" \
  --output-root "$OUTPUT_ROOT/capture" \
  --dump-file "$DUMP_FILE" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --thinking-budget 8192 \
  --timeout-sec 7200 \
  --limit "$LIMIT" \
  2>&1 | tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/analyze_agentlongbench_layer_recall.py" \
  --dataset "$DATA" \
  --layer-dump "$DUMP_FILE" \
  --output-root "$ANALYSIS_ROOT" \
  --tokenizer "$TOKENIZER" \
  --benchmark-repo "$BENCHMARK_REPO" \
  2>&1 | tee -a "$RUN_LOG"

echo "layer recall probe complete: $ANALYSIS_ROOT"
