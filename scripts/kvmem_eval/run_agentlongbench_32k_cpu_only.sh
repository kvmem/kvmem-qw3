#!/usr/bin/env bash
# Generic AgentLongBench K=32K/G=32K utility runner with CPU-only lower tier.
# Dataset-specific wrappers provide the frozen scorer/KV/thinking parameters.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${TAG:?TAG is required}"
: "${DATA:?DATA is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${EXPECTED:?EXPECTED is required}"

# EXPECTED is also the canonical subset size.  Some source datasets contain
# additional settings after the paper subset (for example DeepseekMillion has
# 100 rows while the cross-method 1M evaluation is the first 50 ki-c rows).
# Passing it to the runner prevents silently evaluating extra rows and only
# discovering the mismatch in the final validation step.
QUESTION_IDS=${QUESTION_IDS:-}
if [[ -n "$QUESTION_IDS" ]]; then
  LIMIT=${LIMIT:-}
else
  LIMIT=${LIMIT:-$EXPECTED}
fi

PORT=${PORT:-8087}
CTX=${CTX:-262144}
CPU_GB=${CPU_GB:-64}
KVMEM_BLOCK_TOKENS=${KVMEM_BLOCK_TOKENS:-32}
KVMEM_BUDGET=${KVMEM_BUDGET:-32768}
KVMEM_PREFILL_BUDGET=${KVMEM_PREFILL_BUDGET:-$KVMEM_BUDGET}
KVMEM_RETRIEVAL_METHOD=${KVMEM_RETRIEVAL_METHOD:-mean-k}
KVMEM_INDEX_PLACEMENT=${KVMEM_INDEX_PLACEMENT:-gpu}
KVMEM_INDEX_STAGING_MB=${KVMEM_INDEX_STAGING_MB:-64}
KVMEM_ADAPTIVE_SCORE_MODE=${KVMEM_ADAPTIVE_SCORE_MODE:-auto}
KVMEM_ADAPTIVE_GAIN_1TO2=${KVMEM_ADAPTIVE_GAIN_1TO2:-0.10}
KVMEM_ADAPTIVE_GAIN_2TO4=${KVMEM_ADAPTIVE_GAIN_2TO4:-0.06}
KVMEM_SINK_TOKENS=${KVMEM_SINK_TOKENS:-}
KVMEM_RECENT_TOKENS=${KVMEM_RECENT_TOKENS:-}
KV_DTYPE=${KV_DTYPE:-fp16}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.5}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
THINKING_BUDGET=${THINKING_BUDGET:-4096}
MAX_TOKENS=${MAX_TOKENS:-32768}
SEED=${SEED-20260722}
RECOMPUTE_QUERY=${RECOMPUTE_QUERY:-on}
IMMUTABLE_K=${IMMUTABLE_K:-on}
IMMUTABLE_REFRESH_TOKENS=${IMMUTABLE_REFRESH_TOKENS:-}
OPT_STAGE_OUT=${OPT_STAGE_OUT:-on}
OPT_STAGE_IN=${OPT_STAGE_IN:-on}
OPT_PACK=${OPT_PACK:-on}
BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-32K-CPU-only}
METHOD=${METHOD:-kvmem_k32k_g32k_cpu_only}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}
KVMEM_DUMP_SCORES=${KVMEM_DUMP_SCORES:-}

for value in "$RECOMPUTE_QUERY" "$IMMUTABLE_K" \
             "$OPT_STAGE_OUT" "$OPT_STAGE_IN" "$OPT_PACK"; do
  if [[ "$value" != on && "$value" != off ]]; then
    echo "feature and optimization switches must be on|off" >&2
    exit 2
  fi
done
if [[ ! "$EXPECTED" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED must be a positive integer" >&2
  exit 2
fi
if [[ -n "$LIMIT" ]]; then
  if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT != EXPECTED )); then
    echo "LIMIT must be a positive integer equal to EXPECTED" >&2
    exit 2
  fi
fi
if [[ -n "$LIMIT" && -n "$QUESTION_IDS" ]]; then
  echo "LIMIT and QUESTION_IDS are mutually exclusive" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

feature_args=()
if [[ "$RECOMPUTE_QUERY" == off ]]; then
  feature_args+=(--no-kvmem-recompute-query)
fi
if [[ "$IMMUTABLE_K" == on ]]; then
  feature_args+=(--kvmem-immutable-k)
else
  feature_args+=(--no-kvmem-immutable-k)
fi

retrieval_args=(
  --kvmem-retrieval-method "$KVMEM_RETRIEVAL_METHOD"
  --kvmem-index-placement "$KVMEM_INDEX_PLACEMENT"
  --kvmem-index-staging-mb "$KVMEM_INDEX_STAGING_MB"
  --kvmem-adaptive-score-mode "$KVMEM_ADAPTIVE_SCORE_MODE"
)
if [[ "$KVMEM_RETRIEVAL_METHOD" == key-direction-adaptive ]]; then
  retrieval_args+=(
    --kvmem-adaptive-gain-1to2 "$KVMEM_ADAPTIVE_GAIN_1TO2"
    --kvmem-adaptive-gain-2to4 "$KVMEM_ADAPTIVE_GAIN_2TO4"
  )
fi

retention_args=(--kvmem-sink-blocks 8 --kvmem-recent-blocks 0)
if [[ -n "$KVMEM_SINK_TOKENS" || -n "$KVMEM_RECENT_TOKENS" ]]; then
  if [[ ! "$KVMEM_SINK_TOKENS" =~ ^[0-9]+$ ]] ||
     [[ ! "$KVMEM_RECENT_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "KVMEM_SINK_TOKENS and KVMEM_RECENT_TOKENS must both be non-negative integers" >&2
    exit 2
  fi
  retention_args=(
    --kvmem-sink-tokens "$KVMEM_SINK_TOKENS"
    --kvmem-recent-tokens "$KVMEM_RECENT_TOKENS"
  )
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server" >&2
  exit 3
fi

server_env=(
  -u QW3_KVMEM_REBUILT_STATE_DIR
  -u QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS
  -u QW3_KVMEM_RAW_K_NVME_DIR
  -u QW3_KVMEM_RAW_K_NVME_GIB
  QW3_KVMEM_TRACE=1
  QW3_KVMEM_TIMING=1
  QW3_Q8_BF16_MAIN=0
  QW3_FATTN_NSPLIT=1
  QW3_PREFILL_FA2_NSPLIT=1
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192
)
if [[ "$RECOMPUTE_QUERY" == on ]]; then
  server_env+=(QW3_KVMEM_RECOMPUTE_QUERY=1)
fi
if [[ "$IMMUTABLE_K" == on ]]; then
  server_env+=(QW3_KVMEM_IMMUTABLE_SOURCE_K=1)
fi
if [[ -n "$IMMUTABLE_REFRESH_TOKENS" ]]; then
  server_env+=(QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS="$IMMUTABLE_REFRESH_TOKENS")
fi
if [[ -n "$KVMEM_DUMP_SCORES" ]]; then
  server_env+=(QW3_KVMEM_DUMP_SCORES="$KVMEM_DUMP_SCORES")
fi

env "${server_env[@]}" "$ROOT/build/qw3" serve \
  --model "$MODEL" --ctx "$CTX" --kv-dtype "$KV_DTYPE" \
  --kvmem --kvmem-block-tokens "$KVMEM_BLOCK_TOKENS" \
  --kvmem-budget "$KVMEM_BUDGET" \
  --kvmem-prefill-budget "$KVMEM_PREFILL_BUDGET" \
  --kvmem-gen-budget 32768 \
  "${retention_args[@]}" \
  --kvmem-method retrieval "${retrieval_args[@]}" \
  --kvmem-update-mode step --kvmem-query-conditioned \
  "${feature_args[@]}" \
  --kvmem-gpu-memory-ratio "$GPU_MEMORY_RATIO" \
  --kvmem-cpu-gb "$CPU_GB" --no-kvmem-raw-k-nvme \
  --kvmem-opt-stage-out "$OPT_STAGE_OUT" \
  --kvmem-opt-stage-in "$OPT_STAGE_IN" --kvmem-opt-pack "$OPT_PACK" \
  --enable-thinking --thinking-budget "$THINKING_BUDGET" \
  --prefill-chunk "$PREFILL_CHUNK" --temp "$TEMP" -n "$MAX_TOKENS" \
  --native-mtp-speculate --mtp-chain 4 \
  --host 127.0.0.1 --port "$PORT" >"$SERVER_LOG" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
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

seed_args=()
if [[ -n "$SEED" ]]; then
  seed_args+=(--seed "$SEED")
fi

sample_args=()
if [[ -n "$LIMIT" ]]; then
  sample_args+=(--limit "$LIMIT")
fi
if [[ -n "$QUESTION_IDS" ]]; then
  IFS=',' read -r -a question_ids <<<"$QUESTION_IDS"
  for question_id in "${question_ids[@]}"; do
    if [[ -z "$question_id" ]]; then
      echo "QUESTION_IDS contains an empty comma-separated ID" >&2
      exit 2
    fi
    sample_args+=(--question-id "$question_id")
  done
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --dataset "$DATA" --manifest "$MANIFEST" --allow-custom-subset \
  --benchmark-name "$BENCHMARK_NAME" --output-root "$RESULT_ROOT" \
  --api-base "http://127.0.0.1:$PORT/v1" --model "$(basename "$MODEL")" \
  --method "$METHOD" --temperature "$TEMP" --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" --context-window "$CTX" \
  --context-safety-margin 16 --timeout-sec 7200 --max-sample-sec 7200 \
  --attempts 3 "${sample_args[@]}" --enable-thinking "${seed_args[@]}" \
  --kvmem-retrieval-trace-metadata 2>&1 | tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" - "$RESULT_ROOT/validation_report.json" "$EXPECTED" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
if (not report.get("passed") or report.get("answers_unique") != expected
        or report.get("eval_unique") != expected):
    raise SystemExit(f"AgentLongBench validation failed: {report}")
print(f"validation passed: {expected} answers and evaluations")
PY
