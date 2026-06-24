#!/usr/bin/env bash
# Full accuracy benchmark: lm_eval official tasks + serving config comparison (~4-8 h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
init_result_dir accuracy
require_binary "$QW3" "qw3"
require_model

TIMEOUT="${ACCURACY_TIMEOUT:-7200}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8080}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
CTX="${CTX:-262144}"
KV_DTYPE="${KV_DTYPE:-fp16}"
KV_POOL_PAGES="${KV_POOL_PAGES:-49152}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_serve() {
  local name="$1"
  shift
  local log_file="${OUT}/serve_${name}.log"

  cleanup
  SERVER_PID=""

  step "start qw3 serve (${name})"
  "$QW3" serve \
    --model "$MODEL" \
    --host "$SERVE_HOST" \
    --port "$SERVE_PORT" \
    --ctx "$CTX" \
    -n 0 \
    --temp 0 \
    --prefill-chunk "$PREFILL_CHUNK" \
    --kv-dtype "$KV_DTYPE" \
    --kv-pool-pages "$KV_POOL_PAGES" \
    "$@" \
    >"$log_file" 2>&1 &
  SERVER_PID=$!

  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -sf "http://${SERVE_HOST}:${SERVE_PORT}/health" >/dev/null 2>&1; then
      echo "serve ready (${name}) pid=${SERVER_PID}"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "serve exited early (${name}); see ${log_file}" >&2
      tail -40 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "serve health check timed out (${name})" >&2
  tail -40 "$log_file" >&2 || true
  return 1
}

run_lm_eval() {
  local name="$1"
  local config="$2"
  shift 2
  local out_path="${OUT}/lm_eval_${name}"

  step "lm_eval ${name}"
  if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "missing venv activate script: ${VENV_ACTIVATE}" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  mkdir -p "$out_path"
  if ! lm_eval run --config "$config" --output_path "$out_path" "$@"; then
    echo "WARN: lm_eval ${name} failed (continuing)" >&2
    return 0
  fi
}

# Production config: continuous batching + paged KV
start_serve prod \
  --continuous-batching \
  --paged-kv \
  --max-active 4 \
  --body-batch

# GSM8K only under three serving configs (thinking mode; ~20 samples each).
run_lm_eval prod_gsm8k benchmark/gsm8k_qwen36_local_official.yaml --limit 20

# No MTP baseline (same production path, explicit label)
start_serve no_mtp \
  --continuous-batching \
  --paged-kv \
  --max-active 4 \
  --body-batch

run_lm_eval no_mtp_gsm8k benchmark/gsm8k_qwen36_local_official.yaml --limit 20

# MTP configuration
start_serve mtp \
  --continuous-batching \
  --paged-kv \
  --max-active 4 \
  --body-batch \
  --mtp-chain 4 \
  --mtp-policy fixed

run_lm_eval mtp_gsm8k benchmark/gsm8k_qwen36_local_official.yaml --limit 20

cleanup

step "summarize"
python3 scripts/summarize_benchmark_results.py \
  --kind accuracy \
  --input-dir "$OUT" \
  --baseline benchmark/baselines/lm_eval \
  --output "${OUT}/summary_accuracy.md"

echo "accuracy benchmark finished -> ${OUT}"
