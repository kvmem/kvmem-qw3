#!/usr/bin/env bash
# Warm-history Full Context control for the three frozen AgentLongBench <=256K
# latency representatives.  All histories fit losslessly in the 256K pool and
# the final query explicitly disables semantic reselection.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_20260803}
PORT=${PORT:-18084}
SERVER_LOG="$OUT_ROOT/alb_le256_full_context_server.log"
CLIENT_LOG="$OUT_ROOT/alb_le256_full_context_client.log"
OUTPUT="$OUT_ROOT/alb_le256_full_context.jsonl"
DATA=/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl
BENCH=/home/chaidi/AgentLongBench_Motivation
MODEL="$ROOT/models/Qwen3.6-27B-Q8_0.gguf"

mkdir -p "$OUT_ROOT"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server" >&2
  exit 2
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
  -u QW3_KVMEM_RECOMPUTE_QUERY \
  -u QW3_KVMEM_IMMUTABLE_SOURCE_K \
  -u QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS \
  -u QW3_KVMEM_RAW_K_NVME_DIR \
  -u QW3_KVMEM_RAW_K_NVME_GIB \
  QW3_KVMEM_TRACE=1 QW3_KVMEM_TIMING=1 \
  QW3_Q8_BF16_MAIN=0 QW3_FATTN_NSPLIT=1 QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
  --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --backend qwen-native --native-kernels cuda \
  --ctx 262144 --batch 2048 --prefill-chunk 2048 --kv-dtype fp16 \
  --native-mtp-speculate --mtp-chain 4 \
  --enable-thinking --thinking-budget 4096 -n 32768 \
  --kvmem --kvmem-block-tokens 32 --kvmem-budget 229376 \
  --kvmem-gen-budget 32768 --kvmem-update-mode step \
  --kvmem-query-conditioned --kvmem-retrieval-method mean-k \
  --kvmem-index-placement gpu --kvmem-sink-blocks 8 \
  --kvmem-recent-blocks 0 --kvmem-gpu-memory-ratio 0.55 \
  --kvmem-cpu-gb 64 --no-kvmem-raw-k-nvme \
  --no-kvmem-recompute-query --no-kvmem-immutable-k \
  --kvmem-opt-stage-out off --kvmem-opt-stage-in on --kvmem-opt-pack off \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 3
  fi
  sleep 2
done
curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null

common=(
  --dataset "$DATA" --benchmark-repo "$BENCH"
  --slice alb_le256_full_context
  --api-base "http://127.0.0.1:$PORT/v1"
  --server-log "$SERVER_LOG" --output "$OUTPUT"
  --active-context-budget 229376 --generation-reserve 32768
  --block-tokens 32 --temperature 0.7 --top-p 0.9
  --final-reselect off --timeout 7200
)

python3 "$ROOT/scripts/kvmem_eval/measure_frozen_kvmem_latency.py" \
  "${common[@]}" \
  --question-id eaf6372f9c37a999d2b88fa4c4a9373d212bb9e817b0c52fdd7d9b2356275a89 \
  2>&1 | tee -a "$CLIENT_LOG"

python3 "$ROOT/scripts/kvmem_eval/measure_frozen_kvmem_latency.py" \
  "${common[@]}" --enable-thinking \
  --question-id 52cdde276a122e3a13f69f3da271afefa59cf69bde72e924112329964f39c848 \
  --question-id febc5e457e02ecea1bc51e9f3955591ee48e573c44e15a435d15c0c474062fcd \
  2>&1 | tee -a "$CLIENT_LOG"
