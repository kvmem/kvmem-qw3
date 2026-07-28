#!/usr/bin/env bash
# Dense text-reprefill control for the exact selected source tokens captured by
# run_agentlongbench_1m_error19_capture.sh.  No KVMem retrieval is run here.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8088}
CAPTURE_TAG=${CAPTURE_TAG:-agentlongbench_1m_ds50_kvmem_error19_selected_dump_20260727_103114}
TAG=${TAG:-${CAPTURE_TAG}_selected_text_dense_fp8}
DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
CAPTURE_ROOT=${CAPTURE_ROOT:-/data/chaidi/kvmem_eval/results/$CAPTURE_TAG}
KVMEM_EVAL=${KVMEM_EVAL:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009/eval.jsonl}
OUTPUT=${OUTPUT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}

mkdir -p "$OUTPUT" "$LOG_ROOT"
if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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
  if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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
  --tokenizer /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8 \
  --kvmem-dump "$CAPTURE_ROOT/kvmem_retrieval_dump.jsonl" \
  --kvmem-eval "$KVMEM_EVAL" \
  --output-root "$OUTPUT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --benchmark-name AgentLongBench-1M-DS50-error19-selected-text \
  --method kvmem_k224k_exact_selected_tokens_dense_text_fp8 \
  --temperature 0.6 --top-p 0.95 \
  --max-tokens 32768 --thinking-budget 8192 \
  --context-window 262144 --context-safety-margin 16 \
  --timeout-sec 7200 --max-sample-sec 7200 --attempts 3 \
  --seed 20260722 \
  2>&1 | tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" - "$OUTPUT/validation_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (not report.get("passed") or report.get("answers_unique") != 19
        or report.get("eval_unique") != 19):
    raise SystemExit(f"selected-text validation failed: {report}")
print("selected-text validation passed: 19 answers and 19 evaluations")
PY
