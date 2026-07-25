#!/usr/bin/env bash
# AgentLongBench-Long fixed 512K normal100 with ordinary KVMem + MTP-4.
# The default storage profile is CPU-only opt_1: a 640K logical context,
# 224K selected context, 32K generation reserve, and no NVMe tier.  CTX,
# CPU_GB, NVME_GB, NVME_DIR, and KVMEM_OPT_LEVEL remain overridable so the
# earlier 1M + NVMe configuration can still be reproduced without editing
# this launcher. Query replay, immutable source K, and MTP-4 are explicit;
# DeltaNet rebuilt-state capture/seed/export/import is deliberately absent.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PORT=${PORT:-8087}
CTX=${CTX:-655360}
CPU_GB=${CPU_GB:-64}
NVME_GB=${NVME_GB:-0}
KVMEM_OPT_LEVEL=${KVMEM_OPT_LEVEL:-opt_1}
KVMEM_BUDGET=${KVMEM_BUDGET:-229376}
GEN_BUDGET=${GEN_BUDGET:-32768}
KV_DTYPE=${KV_DTYPE:-fp16}
if [[ -z "${PREFILL_CHUNK:-}" ]]; then
  if [[ "$KV_DTYPE" == "fp8" ]]; then
    PREFILL_CHUNK=8192
  else
    PREFILL_CHUNK=2048
  fi
fi
TAG=${TAG:-agentlongbench_512k_normal100_k224k_g32k_b32_qr_immutable_mtp4_fp16_cpu_opt1_20260723}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}
NVME_DIR=${NVME_DIR:-/tmp/qw3_kvmem_eval_nvme/$TAG}
SERVER_LOG=${SERVER_LOG:-$LOG_ROOT/${TAG}_server.log}
RUN_LOG=${RUN_LOG:-$LOG_ROOT/${TAG}_runner.log}
PID_FILE=${PID_FILE:-$LOG_ROOT/${TAG}.pid}
LIMIT=${LIMIT:-}
EXPECTED=${EXPECTED:-${LIMIT:-100}}
QUESTION_IDS=${QUESTION_IDS:-}
METHOD=${METHOD:-kvmem_mean_k_${KVMEM_BUDGET}t_b32_query_replay_immutable_mtp4_${KV_DTYPE}}
BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-512K-normal100}

if [[ "$KV_DTYPE" != "fp16" && "$KV_DTYPE" != "fp8" ]]; then
  echo "KV_DTYPE must be fp16 or fp8, got: $KV_DTYPE" >&2
  exit 2
fi
if [[ ! "$PREFILL_CHUNK" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFILL_CHUNK must be a positive integer, got: $PREFILL_CHUNK" >&2
  exit 2
fi

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
echo $$ >"$PID_FILE"

tier_args=(
  --kvmem-cpu-gb "$CPU_GB"
)
if [[ "$NVME_GB" != "0" && "$NVME_GB" != "0.0" ]]; then
  mkdir -p "$NVME_DIR"
  tier_args+=(
    --kvmem-nvme-gb "$NVME_GB"
    --kvmem-nvme-dir "$NVME_DIR"
  )
fi

limit_args=()
if [[ -n "$LIMIT" && -n "$QUESTION_IDS" ]]; then
  echo "LIMIT and QUESTION_IDS are mutually exclusive" >&2
  exit 2
fi
if [[ -n "$LIMIT" ]]; then
  if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
    echo "LIMIT must be an integer in [1, 100], got: $LIMIT" >&2
    exit 2
  fi
  limit_args+=(--limit "$LIMIT")
fi
if [[ -n "$QUESTION_IDS" ]]; then
  IFS=',' read -r -a question_ids <<<"$QUESTION_IDS"
  for question_id in "${question_ids[@]}"; do
    if [[ -z "$question_id" ]]; then
      echo "QUESTION_IDS contains an empty comma-separated ID" >&2
      exit 2
    fi
    limit_args+=(--question-id "$question_id")
  done
fi
if [[ ! "$EXPECTED" =~ ^[1-9][0-9]*$ ]] || (( EXPECTED > 100 )); then
  echo "EXPECTED must be an integer in [1, 100], got: $EXPECTED" >&2
  exit 2
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

if curl -fsS --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; refusing to disturb it" >&2
  exit 3
fi

# `env -u` is part of the experiment invariant: this run must never seed,
# capture, export, or import a rebuilt DeltaNet state. The evaluator likewise
# sends only kvmem_query_span/context-span metadata.
env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  "$ROOT/build/qw3" serve \
    --model "$MODEL" \
    --ctx "$CTX" --kv-dtype "$KV_DTYPE" \
    --kvmem --kvmem-block-tokens 32 \
    --kvmem-budget "$KVMEM_BUDGET" --kvmem-gen-budget "$GEN_BUDGET" \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-immutable-k --kvmem-gpu-memory-ratio 0.51 \
    --kvmem-optimization-level "$KVMEM_OPT_LEVEL" \
    "${tier_args[@]}" \
    --enable-thinking --thinking-budget 4096 \
    --prefill-chunk "$PREFILL_CHUNK" --temp 0.6 \
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

if ! grep -q 'kvmem_recompute_query=1' "$SERVER_LOG" ||
   ! grep -q 'kvmem_immutable_k=1' "$SERVER_LOG" ||
   ! grep -q 'mtp_chain=4' "$SERVER_LOG" ||
   ! grep -q 'mtp_speculate=1' "$SERVER_LOG"; then
  echo "server did not confirm query replay + immutable K + MTP-4" >&2
  exit 5
fi
if rg -q 'rebuilt-state|REBUILT_STATE' "$SERVER_LOG"; then
  echo "rebuilt-state activity unexpectedly appeared in server log" >&2
  exit 5
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --dataset "$DATA" --manifest "$MANIFEST" --allow-custom-subset \
  --benchmark-name "$BENCHMARK_NAME" \
  --output-root "$RESULT_ROOT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$(basename "$MODEL")" \
  --method "$METHOD" \
  --temperature 0.6 --top-p 0.95 --max-tokens 32768 \
  --context-window "$CTX" --context-safety-margin 16 \
  --timeout-sec 7200 --max-sample-sec 7200 --attempts 3 \
  --enable-thinking --seed 20260722 \
  --kvmem-retrieval-trace-metadata \
  "${limit_args[@]}" \
  2>&1 | tee -a "$RUN_LOG"

"$ROOT/.venv/bin/python" - "$RESULT_ROOT/validation_report.json" "$EXPECTED" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
if (not report.get("passed") or report.get("answers_unique") != expected
        or report.get("eval_unique") != expected):
    raise SystemExit(f"AgentLongBench validation failed: {report}")
print(
    f"AgentLongBench validation passed: {expected} answers "
    f"and {expected} evaluations"
)
PY
