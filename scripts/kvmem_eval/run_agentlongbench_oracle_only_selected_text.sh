#!/usr/bin/env bash
# Dense text-reprefill control for the exact blocks selected by the final-query
# AgentLongBench oracle-only direct-KV experiment.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8088}
CAPTURE_TAG=${CAPTURE_TAG:-agentlongbench_1m_error3_final_query_oracle_only_chain_20260727_1358}
TAG=${TAG:-${CAPTURE_TAG}_selected_text_dense_fp8}
DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
CAPTURE_ROOT=${CAPTURE_ROOT:-/data/chaidi/kvmem_eval/results/$CAPTURE_TAG}
KVMEM_EVAL=${KVMEM_EVAL:-$CAPTURE_ROOT/eval.jsonl}
OUTPUT=${OUTPUT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}

mkdir -p "$OUTPUT" "$LOG_ROOT"
if curl -fsS --noproxy '*' \
    "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 262144 --kv-dtype fp8 \
    --enable-thinking --thinking-budget 8192 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 300); do
  if curl -fsS --noproxy '*' \
      "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_selected_replay.py" \
  --dataset "$DATA" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --tokenizer \
    /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 \
  --kvmem-dump "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --kvmem-eval "$KVMEM_EVAL" \
  --output-root "$OUTPUT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --benchmark-name AgentLongBench-1M-oracle-only-selected-text \
  --method kvmem_oracle_only_exact_selected_tokens_dense_text_fp8 \
  --temperature 0.6 --top-p 0.95 \
  --max-tokens 32768 --thinking-budget 8192 \
  --context-window 262144 --context-safety-margin 16 \
  --timeout-sec 7200 --max-sample-sec 7200 --attempts 3 \
  --seed 20260722 \
  2>&1 | tee -a "$RUN_LOG"

eval_count=$(wc -l <"$OUTPUT/eval.jsonl")
if [[ "$eval_count" -ne 3 ]]; then
  echo "selected-text validation failed: expected 3 eval rows, got $eval_count" >&2
  exit 5
fi
echo "selected-text validation passed: 3 evaluations" | tee -a "$RUN_LOG"
