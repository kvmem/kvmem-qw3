#!/usr/bin/env bash
# AgentLongBench long-250 KVMem 64K-budget comparison.
#
# This preserves the completed 32K experiment and changes only:
#   - --kvmem-budget: 32768 -> 65536
#   - QW3_KVMEM_TRACE: disabled -> enabled
#
# All prompt construction, query spans, generation parameters, and grading are
# shared with the completed 32K run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
TAG=${TAG:-agentlongbench_kvmem_64k_b32_lmeparams_thinking4k_trace_20260716}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl}
MANIFEST=${MANIFEST:-/data/chaidi/kvmem_eval/data/agentlongbench_250/manifest.jsonl}
RESULT_BASE=${RESULT_BASE:-/data/chaidi/kvmem_eval/results}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
NVME_DIR=${NVME_DIR:-/data/chaidi/kvmem_eval/nvme/${TAG}}
MODEL=${MODEL:-${ROOT}/models/Qwen3.6-27B-Q8_0.gguf}
SMOKE_OUT=${RESULT_BASE}/${TAG}_smoke8
FULL_OUT=${RESULT_BASE}/${TAG}
SERVER_LOG=${LOG_BASE}/${TAG}_server.log
RUN_LOG=${LOG_BASE}/${TAG}_runner.log
PID_FILE=${LOG_BASE}/${TAG}.pid

mkdir -p "$LOG_BASE" "$NVME_DIR" "$SMOKE_OUT" "$FULL_OUT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

gpu_compute_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

echo "[$(timestamp)] queued tag=${TAG} pid=$$ kvmem_budget=65536 gen_budget=32768 trace=1 enable_thinking=true thinking_budget=4096 temp=0.6 top_p=0.95 max_tokens=32768" | tee -a "$RUN_LOG"
echo $$ >"$PID_FILE"

while true; do
  used=$(gpu_compute_mib)
  if [[ "$used" -lt 8192 ]]; then
    break
  fi
  echo "[$(timestamp)] waiting_for_gpu compute_memory_mib=${used}" | tee -a "$RUN_LOG"
  sleep 30
done

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already has a healthy server; refusing to reuse it" | tee -a "$RUN_LOG"
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

echo "[$(timestamp)] starting_qw3 binary=${ROOT}/build/qw3 kvmem_budget=65536 gen_budget=32768 trace=1" | tee -a "$RUN_LOG"
env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_TRACE=1 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx 262144 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 --kvmem-budget 65536 \
    --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-gpu-memory-ratio 0.5 \
    --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir "$NVME_DIR" \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temp 0.6 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >>"$SERVER_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 180); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[$(timestamp)] qw3 exited during startup" | tee -a "$RUN_LOG"
    tail -100 "$SERVER_LOG" | tee -a "$RUN_LOG"
    exit 1
  fi
  sleep 5
done
if [[ "$healthy" -ne 1 ]]; then
  echo "[$(timestamp)] qw3 health timeout" | tee -a "$RUN_LOG"
  tail -100 "$SERVER_LOG" | tee -a "$RUN_LOG"
  exit 1
fi

common=(
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py"
  --dataset "$DATA"
  --manifest "$MANIFEST"
  --api-base "http://127.0.0.1:${PORT}/v1"
  --model "$(basename "$MODEL")"
  --method kvmem_mean_k_64k_b32
  --temperature 0.6 --top-p 0.95 --max-tokens 32768
  --context-window 262144 --timeout-sec 3600 --max-sample-sec 3600
  --enable-thinking
)

echo "[$(timestamp)] validating_tokenizer_250" | tee -a "$RUN_LOG"
python3 "$ROOT/scripts/kvmem_eval/validate_agentlongbench_tokenizer.py" \
  --dataset "$DATA" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --llama-tokenize /tmp/llama-cpu-build/bin/llama-tokenize \
  --model "$MODEL" \
  --output "$FULL_OUT/tokenizer_validation.json" 2>&1 | tee -a "$RUN_LOG"

echo "[$(timestamp)] starting_smoke8" | tee -a "$RUN_LOG"
python3 "${common[@]}" --limit 8 --output-root "$SMOKE_OUT" 2>&1 | tee -a "$RUN_LOG"

python3 - "$SMOKE_OUT/validation_report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("passed") or report.get("answers_unique") != 8 or report.get("eval_unique") != 8:
    raise SystemExit(f"smoke validation failed: {report}")
print("smoke validation passed: 8 answers and 8 evaluations")
PY

explicit_spans=$(grep -c 'kvmem explicit query span' "$SERVER_LOG" || true)
query_conditioned=$(grep -c 'native kvmem query-conditioned' "$SERVER_LOG" || true)
mean_k_scored=$(grep -c 'query-conditioned softmax-pages' "$SERVER_LOG" || true)
echo "[$(timestamp)] trace_check explicit_spans=${explicit_spans} query_conditioned=${query_conditioned} mean_k_scored=${mean_k_scored}" | tee -a "$RUN_LOG"
if [[ "$explicit_spans" -lt 8 || "$query_conditioned" -lt 8 || "$mean_k_scored" -lt 8 ]]; then
  echo "TRACE validation failed; expected at least 8 query spans and query-conditioned mean-k scoring events" | tee -a "$RUN_LOG"
  exit 1
fi

echo "[$(timestamp)] starting_full250 enable_thinking=true" | tee -a "$RUN_LOG"
python3 "${common[@]}" --output-root "$FULL_OUT" 2>&1 | tee -a "$RUN_LOG"

python3 - "$FULL_OUT/validation_report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("passed") or report.get("answers_unique") != 250 or report.get("eval_unique") != 250:
    raise SystemExit(f"full validation failed: {report}")
print("full validation passed: 250 answers and 250 evaluations")
PY

echo "[$(timestamp)] complete results=${FULL_OUT}" | tee -a "$RUN_LOG"
