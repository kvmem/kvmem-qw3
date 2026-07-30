#!/usr/bin/env bash
set -euo pipefail

# Ten-sample controlled A/B for newline-aligned whole-round retrieval.  The
# cohort covers all eight AgentLongBench task types and mixes samples that were
# correct/incorrect in the preserved B1024/SB32-Max run.  Both arms use the
# exact same IDs and server parameters; only request-level round padding differs.

ROOT=${ROOT:-/home/chaidi/qw3}
PORT=${PORT:-8089}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
DATASET=${DATASET:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
BASE_ROOT=${BASE_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_round_padding_ab10_${STAMP}}
LOG_ROOT=${LOG_ROOT:-/data/chaidi/kvmem_eval/logs}

SAMPLE_IDS=(
  # Count Frequency(Tool): prior B1024 result incorrect.
  2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021
  # Find Duplicates(Tool): one prior correct and one prior incorrect.
  3b3f472b43854d64e5dc4a984dea0f993bcefe08630d0295dcc844c14dfbef76
  3c7151b28bf4845c9c4480ed8782a5085550eac3dc840e0d22e0626f3c15d6f6
  # Find Target Offsets(Tool): prior incorrect.
  8176925a596b177f4e8655785ddd744e3bb3e886d512bddc892c223b9548536c
  # Count Correctness(Env): prior correct.
  daea520e8f6883a738e966bde243df4ab0434d5482eddd529cc359d053b1be40
  # Count Frequency(Env): prior incorrect.
  cad128acfc7c410134bcd250e4036a9d6a337abfb4a6674a6a086f55e7ae71cc
  # Find Round with Largest Value(Env): prior correct.
  50e4dad103f3876d3be4b1ab49e648529c49ae3999b70cd682893793ba1b434b
  # Weighted Summation(Env): prior incorrect.
  e9f1123e4c0a0e12ee6c6219b972463d3742faf497400ad3ede728ca9a7beaac
  # Intersection: one prior correct and one prior incorrect.
  3dd137b4815f91d942915dc978de380ec4fb7d48685c1e93bd9f18f8f4b0b824
  460a3ec48d3007d80f68bf8a81b973d806a1350a43b205a2191b4c6c4081117b
)

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
  local log_prefix="$LOG_ROOT/agentlongbench_1m_round_padding_ab10_${STAMP}_${arm}"
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
      >"${log_prefix}_server.log" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 600); do
    if curl -fsS --noproxy '*' \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "${log_prefix}_server.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  curl -fsS --noproxy '*' \
    "http://127.0.0.1:${PORT}/health" >/dev/null

  local question_args=()
  local sample_id
  for sample_id in "${SAMPLE_IDS[@]}"; do
    question_args+=(--question-id "$sample_id")
  done
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
      --benchmark-name AgentLongBench-1M-round-padding-ab10 \
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
      "${question_args[@]}" \
      "${padding_args[@]}" \
      2>&1 | tee "${log_prefix}_runner.log"

  cleanup_server
}

run_arm padding32 32
run_arm unpadded 0

echo "[complete] A/B results: $BASE_ROOT"
