#!/usr/bin/env bash
# AgentLongBench validation runner for chunk-level semantic replay. Defaults to
# the original 512K setup; wrappers may override the dataset and context size.
# Uses CPU-only KVMem spill and never touches NVMe.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-18089}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.50}
CTX_TOKENS=${CTX_TOKENS:-655360}
BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-512K-normal100}
KVMEM_BUDGET=${KVMEM_BUDGET:-32768}
GEN_BUDGET=${GEN_BUDGET:-8192}
BLOCK_TOKENS=${BLOCK_TOKENS:-32}
# Keep the server-side generation reserve independent from the number of
# answer tokens requested by scorer/performance smoke tests.
REQUEST_MAX_TOKENS=${REQUEST_MAX_TOKENS:-$GEN_BUDGET}
RECENT_TOKENS=${RECENT_TOKENS:-4096}
SEMANTIC_QUERY_TOKENS=${SEMANTIC_QUERY_TOKENS:-512}
SEMANTIC_START_TOKENS=${SEMANTIC_START_TOKENS:-$KVMEM_BUDGET}
THINKING_BUDGET=${THINKING_BUDGET:-4096}
CPU_GB=${CPU_GB:-48}
INDEX_STAGING_MB=${INDEX_STAGING_MB:-64}
ADAPTIVE_BLOCK_STATS_MIB=${ADAPTIVE_BLOCK_STATS_MIB:-512}
QUERY_SCORE_CHUNK=${QUERY_SCORE_CHUNK:-}
START_INDEX=${START_INDEX:-0}
LIMIT=${LIMIT:-1}
SAMPLE_ORDER=${SAMPLE_ORDER:-dataset}
TAG=${TAG:-agentlongbench_512k_k32_semantic_chunk_q512_adaptive_b32_fp8_mtp4_20260806}
METHOD=${METHOD:-kvmem_adaptive_k32_b32_semantic_chunk_q512_fp8_mtp4}
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

if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]] || [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$REQUEST_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$CTX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_INDEX must be >= 0; LIMIT, REQUEST_MAX_TOKENS, and CTX_TOKENS must be > 0" >&2
  exit 2
fi
mapfile -t selected_ids < <(
  "$ROOT/.venv/bin/python" - "$DATA" "$START_INDEX" "$LIMIT" "$SAMPLE_ORDER" <<'PY'
import json
import sys

path, start, limit, sample_order = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
)
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if sample_order == "budget_gap":
    # Ordered by the positive category-level K224-K32 accuracy gap measured
    # on the two complete 512K controls. Stable sort preserves canonical order
    # within each task family.
    priority = {
        "Count Frequency(Env)": 0,       # +50.0 percentage points
        "Count Correctness(Env)": 1,     # +30.4
        "Count Frequency(Tool)": 2,      # +18.2
        "Find Target Offsets(Tool)": 3,  # +8.7
        "Find Duplicates(Tool)": 4,      # -13.6
    }
    rows.sort(key=lambda row: priority.get(row.get("task_type"), 100))
elif sample_order != "dataset":
    raise SystemExit(f"unsupported SAMPLE_ORDER={sample_order!r}")
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

query_score_env=(-u QW3_KVMEM_QUERY_SCORE_CHUNK)
if [[ -n "$QUERY_SCORE_CHUNK" ]]; then
  query_score_env=(QW3_KVMEM_QUERY_SCORE_CHUNK="$QUERY_SCORE_CHUNK")
fi

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  "${query_score_env[@]}" \
  -u QW3_KVMEM_QUERY_SCORE_TOKENS \
  -u QW3_KVMEM_ADAPTIVE_GPU_PACKED \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=8 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_Q8_BF16_MAIN=0 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB="$ADAPTIVE_BLOCK_STATS_MIB" \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx "$CTX_TOKENS" --kv-dtype fp8 \
    --kvmem --kvmem-block-tokens "$BLOCK_TOKENS" \
    --kvmem-budget "$KVMEM_BUDGET" \
    --kvmem-prefill-budget "$KVMEM_BUDGET" \
    --kvmem-gen-budget "$GEN_BUDGET" \
    --kvmem-sink-tokens 512 --kvmem-recent-tokens "$RECENT_TOKENS" \
    --kvmem-method retrieval \
    --kvmem-retrieval-method key-direction-adaptive \
    --kvmem-index-placement gpu \
    --kvmem-index-staging-mb "$INDEX_STAGING_MB" \
    --kvmem-adaptive-score-mode auto \
    --kvmem-adaptive-gain-1to2 0.10 \
    --kvmem-adaptive-gain-2to4 0.06 \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-immutable-k --kvmem-gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --kvmem-cpu-gb "$CPU_GB" --no-kvmem-raw-k-nvme \
    --kvmem-opt-stage-out on --kvmem-opt-stage-in on \
    --kvmem-opt-pack on \
    --enable-thinking --thinking-budget "$THINKING_BUDGET" \
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
  --benchmark-name "$BENCHMARK_NAME" \
  --output-root "$RESULT_ROOT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --method "$METHOD" \
  --temperature 0.6 --top-p 0.95 --max-tokens "$REQUEST_MAX_TOKENS" \
  --context-window "$CTX_TOKENS" --context-safety-margin 16 \
  --timeout-sec 14400 --max-sample-sec 14400 --attempts 1 \
  --enable-thinking --seed 20260722 \
  --kvmem-retrieval-trace-metadata \
  --kvmem-prefill-window semantic_chunk \
  --kvmem-prefill-semantic-start-tokens "$SEMANTIC_START_TOKENS" \
  --kvmem-prefill-semantic-query-tokens "$SEMANTIC_QUERY_TOKENS" \
  "${selection_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"
