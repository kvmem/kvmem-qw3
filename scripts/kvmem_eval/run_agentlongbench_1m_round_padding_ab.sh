#!/usr/bin/env bash
set -euo pipefail

# Controlled one-sample A/B for token-level newline alignment. Both arms use
# identical B32 whole-round retrieval; only kvmem_round_padding differs.

ROOT=${ROOT:-/home/chaidi/qw3}
PORT=${PORT:-8089}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SAMPLE_ID=${SAMPLE_ID:-af004d2443fcd470b168453ff3553d666b3001456ffa55a21726a6850725af0b}
DATASET=${DATASET:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
BASE_ROOT=${BASE_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_round_padding_ab_${STAMP}}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}

mkdir -p "$BASE_ROOT" "$LOG_ROOT"

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT INT TERM

run_arm() {
  local arm=$1
  local padding=$2
  local output_root="$BASE_ROOT/$arm"
  local server_log="$LOG_ROOT/agentlongbench_1m_round_padding_ab_${STAMP}_${arm}_server.log"
  local runner_log="$LOG_ROOT/agentlongbench_1m_round_padding_ab_${STAMP}_${arm}_runner.log"
  mkdir -p "$output_root"

  env \
    QW3_KVMEM_TRACE=1 \
    QW3_KVMEM_TIMING=1 \
    QW3_KVMEM_DUMP_SCORES="$output_root/retrieval_scores.jsonl" \
    QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
    QW3_Q8_BF16_MAIN=0 \
    QW3_FATTN_NSPLIT=1 \
    QW3_PREFILL_FA2_NSPLIT=1 \
    "$ROOT/build/qw3" serve \
      --model "$ROOT/models/Qwen3.6-27B-Q8_0.gguf" \
      --ctx 1310720 \
      --kv-dtype fp8 \
      --kvmem \
      --kvmem-block-tokens 32 \
      --kvmem-budget 229376 \
      --kvmem-gen-budget 32768 \
      --kvmem-sink-blocks 8 \
      --kvmem-recent-blocks 0 \
      --kvmem-method retrieval \
      --kvmem-retrieval-method sub-block-mean-k \
      --kvmem-subblocks 1 \
      --kvmem-subblock-reduce max \
      --kvmem-round-retrieval \
      --kvmem-update-mode step \
      --kvmem-query-conditioned \
      --kvmem-immutable-k \
      --kvmem-gpu-memory-ratio 0.5 \
      --kvmem-cpu-gb 80 \
      --enable-thinking \
      --thinking-budget 8192 \
      --prefill-chunk 2048 \
      --temp 0.6 \
      --native-mtp-speculate \
      --mtp-chain 4 \
      --host 127.0.0.1 \
      --port "$PORT" \
      >"$server_log" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 600); do
    if curl -fsS --noproxy '*' \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$server_log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  curl -fsS --noproxy '*' \
    "http://127.0.0.1:${PORT}/health" >/dev/null

  local padding_args=()
  if [[ "$padding" -gt 0 ]]; then
    padding_args=(--kvmem-round-padding "$padding")
  fi
  "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
      --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
      --dataset "$DATASET" \
      --manifest "$MANIFEST" \
      --allow-custom-subset \
      --benchmark-name AgentLongBench-1M-round-padding-ab \
      --output-root "$output_root" \
      --api-base "http://127.0.0.1:${PORT}/v1" \
      --model Qwen3.6-27B-Q8_0.gguf \
      --method "kvmem_round_only_b32_${arm}" \
      --temperature 0.6 \
      --top-p 0.95 \
      --max-tokens 32768 \
      --context-window 1310720 \
      --context-safety-margin 16 \
      --timeout-sec 7200 \
      --max-sample-sec 7200 \
      --attempts 1 \
      --enable-thinking \
      --seed 20260722 \
      --kvmem-retrieval-trace-metadata \
      --kvmem-round-only \
      --question-id "$SAMPLE_ID" \
      "${padding_args[@]}" \
      2>&1 | tee "$runner_log"

  cleanup_server
}

# Run the requested padded arm first so its result is available before the
# matched unpadded control finishes.
run_arm padding32 32
run_arm unpadded 0

echo "[complete] A/B results: $BASE_ROOT"
