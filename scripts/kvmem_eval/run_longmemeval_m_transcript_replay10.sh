#!/usr/bin/env bash
# Experimental lifecycle replay over the frozen ten LongMemEval-M KVMem errors.
# The source transcript is rendered as real user/assistant turns. Once the
# 224K+32K pool is full, every new user query triggers semantic re-selection;
# historical assistant responses remain teacher-forced prefill. Only the final
# benchmark question is decoded.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_error10_20260720/samples.jsonl}
TAG=${TAG:-longmemeval_m_error10_transcript_replay_k224k_b32_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8087}
KV_DTYPE=${KV_DTYPE:-fp16}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.51}
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
  QW3_KVMEM_RECOMPUTE_QUERY=0 \
  QW3_KVMEM_CLEAN_QUERY=0 \
  QW3_KVMEM_TRACE=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
    --tag "$TAG" --data "$DATA" --force \
    --port "$PORT" --ctx 2000000 --kv-dtype "$KV_DTYPE" \
    --block-tokens 32 --budget 229376 --gen-budget 32768 \
    --sink-blocks 8 --recent-blocks 0 \
    --method retrieval --retrieval-method mean-k --update-mode step \
    --query-conditioned --gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --cpu-gb 64 --nvme-gb 256 \
    --thinking --thinking-budget "$THINKING_BUDGET" \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 \
    --max-tokens 32768 --mtp --mtp-chain 4 \
    --read-timeout 7200 \
    --eval-extra-arg=--transcript-replay \
    "${JUDGE_ARGS[@]}" "${LIMIT_ARGS[@]}" "${INDICES_ARGS[@]}"
