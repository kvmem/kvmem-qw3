#!/usr/bin/env bash
# Exact selected-normal-KV + rebuilt-DeltaNet-state control on the frozen ten
# LongMemEval-M errors. It first captures the selection with the exact import
# binary/configuration; historical retrieval dumps are intentionally never used.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${TAG:-longmemeval_m_k224k_query_replay_immutable_rebuilt_state10_fixed_20260722}
PORT=${PORT:-8087}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl}
BASELINE=${BASELINE:-/data/chaidi/kvmem_eval/results/longmemeval_m_k224k_query_replay_immutable10_20260722_101213_eval_20260722_021400.jsonl}
CAPTURE_BASELINE=${CAPTURE_BASELINE:-/data/chaidi/kvmem_eval/results/m102_2m_k224k_g32k_bt32_meank_r051_20260719_eval_20260719_063739.jsonl}
INDICES=${INDICES:-4,6,20,27,34,36,51,60,68,86}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
STATE_DIR=${STATE_DIR:-/data/chaidi/kvmem_eval/rebuilt_states/$TAG}
NVME_DIR=${NVME_DIR:-/data/qw3_kvmem_eval_nvme_$TAG}
CAPTURE_ROOT=${CAPTURE_ROOT:-$RESULT_ROOT/capture}
DUMP=${DUMP:-$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl}
IMPORT_DUMP=${IMPORT_DUMP:-$RESULT_ROOT/import_retrieval_dump.jsonl}
CAPTURE_SERVE_LOG=${CAPTURE_SERVE_LOG:-$RESULT_ROOT/capture_serve.log}
SERVE_LOG=${SERVE_LOG:-$RESULT_ROOT/serve.log}
RUN_LOG=${RUN_LOG:-$RESULT_ROOT/run.log}

# The host may define an HTTP(S) proxy globally. All server readiness probes and
# benchmark requests are loopback traffic and must never be sent through it.
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT" "$STATE_DIR" "$NVME_DIR" "$CAPTURE_ROOT"

# A previous experiment may have just released this port. Refuse to kill or
# alter it; wait briefly for its normal shutdown instead.
for _ in $(seq 1 120); do
  if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT still has a healthy server; refusing to disturb it" >&2
  exit 3
fi

export QW3_KVMEM_REBUILT_STATE_DIR="$STATE_DIR"
export QW3_KVMEM_RECOMPUTE_QUERY=1
export QW3_KVMEM_IMMUTABLE_SOURCE_K=1
export QW3_KVMEM_TRACE=1
export QW3_KVMEM_TIMING=1
export QW3_FATTN_NSPLIT=1
export QW3_PREFILL_FA2_NSPLIT=1
export DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro}

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}
cleanup() { stop_server; }
trap cleanup EXIT INT TERM

start_server() {
  local dump_file=$1
  local serve_log=$2
  QW3_KVMEM_DUMP_SCORES="$dump_file" "$ROOT/build/qw3" serve \
    --model "$ROOT/models/Qwen3.6-27B-Q8_0.gguf" \
    --ctx 2000000 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 \
    --kvmem-budget 229376 --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-gpu-memory-ratio 0.51 \
    --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir "$NVME_DIR" \
    --thinking-budget 4096 --prefill-chunk 2048 --temp 0.6 \
    --host 127.0.0.1 --port "$PORT" \
    --kvmem-query-conditioned --enable-thinking \
    --native-mtp-speculate --mtp-chain 4 \
    >"$serve_log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "qw3 server exited during startup; see $serve_log" >&2
      return 4
    fi
    sleep 2
  done
  echo "qw3 health timeout; see $serve_log" >&2
  return 4
}

echo "[$(date '+%F %T')] exact-config retrieval capture" | tee -a "$RUN_LOG"
start_server "$DUMP" "$CAPTURE_SERVE_LOG"
"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/capture_longmemeval_selected_retrieval.py" \
  --data "$DATA" --baseline "$CAPTURE_BASELINE" --indices "$INDICES" \
  --output-root "$CAPTURE_ROOT" --dump-file "$DUMP" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model Qwen3.6-27B-Q8_0.gguf --timeout-sec 7200 \
  --expected-immutable-source-k 1 \
  --expected-block-tokens 32 --expected-budget-blocks 7168 \
  --expected-sink-blocks 8 --expected-recent-blocks 0 \
  --expected-method mean-k \
  2>&1 | tee -a "$RUN_LOG"
stop_server

echo "[$(date '+%F %T')] rebuilt-state export/import" | tee -a "$RUN_LOG"
start_server "$IMPORT_DUMP" "$SERVE_LOG"

run_control() {
  "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/kvmem_eval/run_longmemeval_rebuilt_state.py" \
    --data "$DATA" --dump-file "$DUMP" \
    --state-dir "$STATE_DIR" --output-root "$RESULT_ROOT" \
    --api-base "http://127.0.0.1:$PORT/v1" \
    --model Qwen3.6-27B-Q8_0.gguf \
    --phase all --max-tokens 32768 --thinking-budget 4096 \
    --temperature 0.6 --top-p 0.95 --timeout-sec 7200 \
    --baseline "$BASELINE" \
    --expected-immutable-source-k 1 \
    --expected-block-tokens 32 --expected-budget-blocks 7168 \
    --expected-sink-blocks 8 --expected-recent-blocks 0 \
    --expected-method mean-k "$@"
}

# Runtime smoke first. The full invocation resumes the exported state and
# sample-wise result, so this adds no duplicate model work when it succeeds.
run_control --limit 1 2>&1 | tee -a "$RUN_LOG"
run_control 2>&1 | tee -a "$RUN_LOG"
