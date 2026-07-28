#!/usr/bin/env bash
# Causal incremental replay over the frozen ten LongMemEval-M errors.
# Historical assistant turns are teacher-forced with max_tokens=0.  After the
# 224K+32K active capacity is full, each session's first user query triggers
# semantic reselection; the rest of that session keeps the selected window.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_error10_20260720/samples.jsonl}
TAG=${TAG:-longmemeval_m_error10_session_ingest_k224k_b32_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8087}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.55}
THINKING_BUDGET=${THINKING_BUDGET:-4096}

JUDGE_ARGS=()
if [[ ${NO_JUDGE:-0} == 1 ]]; then
  JUDGE_ARGS+=(--no-judge)
elif [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  echo "DEEPSEEK_API_KEY is required unless NO_JUDGE=1" >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ -n ${LIMIT:-} ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

INDICES_ARGS=()
if [[ -n ${INDICES:-} ]]; then
  INDICES_ARGS+=(--indices "$INDICES")
fi

env \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
    --tag "$TAG" --data "$DATA" --force \
    --binary "$ROOT/build/qw3" \
    --eval-script "$ROOT/scripts/kvmem_eval/run_longmemeval_session_ingest.py" \
    --port "$PORT" --ctx 2000000 --kv-dtype fp16 \
    --block-tokens 32 --budget 229376 --gen-budget 32768 \
    --sink-blocks 8 --recent-blocks 0 \
    --method retrieval --retrieval-method mean-k --update-mode step \
    --query-conditioned --gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --cpu-gb 64 --nvme-gb 256 \
    --thinking --thinking-budget "$THINKING_BUDGET" \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 \
    --max-tokens 32768 --mtp --mtp-chain 4 \
    --read-timeout 7200 \
    --eval-extra-arg=--active-capacity \
    --eval-extra-arg=262144 \
    "${JUDGE_ARGS[@]}" "${LIMIT_ARGS[@]}" "${INDICES_ARGS[@]}"
