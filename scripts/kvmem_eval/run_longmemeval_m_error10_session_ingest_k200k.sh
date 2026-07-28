#!/usr/bin/env bash
# Session-wise causal replay over the frozen ten LongMemEval-M errors.
#
# The evaluation harness owns dataset session boundaries.  Once the prior
# rendered history reaches 200K context + 32K generation reserve, each new
# dataset session uses its first user message for semantic mean-K retrieval and
# query replay.  The remainder of that session is teacher-forced under the
# selected fixed window.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_error10_20260720/samples.jsonl}
TAG=${TAG:-longmemeval_m_error10_session_ingest_k200k_b32_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-18087}
CTX=${CTX:-2000000}
KVMEM_BUDGET=${KVMEM_BUDGET:-204800}
GEN_BUDGET=${GEN_BUDGET:-32768}
ACTIVE_CAPACITY=${ACTIVE_CAPACITY:-$((KVMEM_BUDGET + GEN_BUDGET))}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.51}
# ctx=2,000,000 with immutable raw-K has a 64.88-GiB theoretical host-K
# ceiling.  Keep a small explicit margin so startup validation passes; cold
# spill beyond the host tier still uses the NVMe tier below.
CPU_GB=${CPU_GB:-66}
NVME_GB=${NVME_GB:-256}
NVME_DIR=${NVME_DIR:-/tmp/qw3_kvmem_eval_nvme_${TAG}}
KVMEM_OPT_LEVEL=${KVMEM_OPT_LEVEL:-opt_1}
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

if (( ACTIVE_CAPACITY != KVMEM_BUDGET + GEN_BUDGET )); then
  echo "warning: ACTIVE_CAPACITY=$ACTIVE_CAPACITY differs from budget sum " \
       "$((KVMEM_BUDGET + GEN_BUDGET))" >&2
fi

mkdir -p "$NVME_DIR"

env \
  -u QW3_KVMEM_REBUILT_STATE_DIR \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_IMMUTABLE_SOURCE_K=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192 \
  DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
    --tag "$TAG" --data "$DATA" --force \
    --binary "$ROOT/build/qw3" \
    --eval-script "$ROOT/scripts/kvmem_eval/run_longmemeval_session_ingest.py" \
    --port "$PORT" --ctx "$CTX" --kv-dtype fp16 \
    --block-tokens 32 \
    --budget "$KVMEM_BUDGET" --gen-budget "$GEN_BUDGET" \
    --sink-blocks 8 --recent-blocks 0 \
    --method retrieval --retrieval-method mean-k --update-mode step \
    --optimization-level "$KVMEM_OPT_LEVEL" \
    --query-conditioned --gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --cpu-gb "$CPU_GB" --nvme-gb "$NVME_GB" --nvme-dir "$NVME_DIR" \
    --server-extra-arg=--kvmem-immutable-k \
    --thinking --thinking-budget "$THINKING_BUDGET" \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 \
    --max-tokens 32768 --mtp --mtp-chain 4 \
    --read-timeout 7200 \
    --eval-extra-arg=--active-capacity \
    --eval-extra-arg="$ACTIVE_CAPACITY" \
    "${JUDGE_ARGS[@]}" "${LIMIT_ARGS[@]}" "${INDICES_ARGS[@]}"
