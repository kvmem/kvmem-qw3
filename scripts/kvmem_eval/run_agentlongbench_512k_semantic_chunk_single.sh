#!/usr/bin/env bash
# One-sample AgentLongBench-512K validation of chunk-level semantic replay.
# Runs beside other servers, uses CPU-only KVMem spill, and never touches NVMe.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-18089}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.50}
GEN_BUDGET=${GEN_BUDGET:-8192}
START_INDEX=${START_INDEX:-0}
LIMIT=${LIMIT:-1}
TAG=${TAG:-agentlongbench_512k_k32_semantic_chunk_q512_adaptive_b32_fp8_mtp4_20260806}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]] || [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_INDEX must be >= 0 and LIMIT must be > 0" >&2
  exit 2
fi
mapfile -t selected_ids < <(
  "$ROOT/.venv/bin/python" - "$DATA" "$START_INDEX" "$LIMIT" <<'PY'
import json
import sys

path, start, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
selected = rows[start:start + limit]
if len(selected) != limit:
    raise SystemExit(
        f"requested [{start},{start + limit}) but dataset has {len(rows)} rows"
    )
for row in selected:
    print(row["stable_sample_id"])
PY
)
selection_args=()
for sample_id in "${selected_ids[@]}"; do
  selection_args+=(--question-id "$sample_id")
done

if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server" >&2
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
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=8 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_Q8_BF16_MAIN=0 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 655360 --kv-dtype fp8 \
    --kvmem --kvmem-block-tokens 32 \
    --kvmem-budget 32768 --kvmem-prefill-budget 32768 \
    --kvmem-gen-budget "$GEN_BUDGET" \
    --kvmem-sink-tokens 512 --kvmem-recent-tokens 4096 \
    --kvmem-method retrieval \
    --kvmem-retrieval-method key-direction-adaptive \
    --kvmem-index-placement gpu --kvmem-index-staging-mb 64 \
    --kvmem-adaptive-score-mode auto \
    --kvmem-adaptive-gain-1to2 0.10 \
    --kvmem-adaptive-gain-2to4 0.06 \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-immutable-k --kvmem-gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --kvmem-cpu-gb 48 --no-kvmem-raw-k-nvme \
    --kvmem-opt-stage-out on --kvmem-opt-stage-in on \
    --kvmem-opt-pack on \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$SERVER_LOG" >&2 || true
    exit 4
  fi
  sleep 2
done
curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null

"$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --dataset "$DATA" --manifest "$MANIFEST" --allow-custom-subset \
  --benchmark-name AgentLongBench-512K-normal100 \
  --output-root "$RESULT_ROOT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --method kvmem_adaptive_k32_b32_semantic_chunk_q512_fp8_mtp4 \
  --temperature 0.6 --top-p 0.95 --max-tokens "$GEN_BUDGET" \
  --context-window 655360 --context-safety-margin 16 \
  --timeout-sec 14400 --max-sample-sec 14400 --attempts 1 \
  --enable-thinking --seed 20260722 \
  --kvmem-retrieval-trace-metadata \
  --kvmem-prefill-window semantic_chunk \
  --kvmem-prefill-semantic-start-tokens 32768 \
  --kvmem-prefill-semantic-query-tokens 512 \
  "${selection_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"
