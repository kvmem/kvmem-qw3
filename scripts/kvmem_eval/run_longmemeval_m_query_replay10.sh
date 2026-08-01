#!/usr/bin/env bash
# Controlled A/B on the ten frozen LongMemEval-M KVMem errors: keep the normal
# 224K mean-K selection, then recompute the final query suffix against that
# fixed window before decoding.  The experimental path is env-gated and OFF in
# every existing launcher.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${TAG:-longmemeval_m_k224k_query_replay10_20260720}
PORT=${PORT:-8087}
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl}
INDICES=${INDICES:-4,6,20,27,34,36,51,60,68,86}
KVMEM_OPT_STAGE_OUT=${KVMEM_OPT_STAGE_OUT:-off}
KVMEM_OPT_STAGE_IN=${KVMEM_OPT_STAGE_IN:-on}
KVMEM_OPT_PACK=${KVMEM_OPT_PACK:-off}
CTX=${CTX:-2000000}
CPU_GB=${CPU_GB:-64}
NVME_GB=${NVME_GB:-256}
NVME_DIR=${NVME_DIR:-/data/qw3_kvmem_eval_nvme_${TAG}}
GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-0.51}

if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  echo "DEEPSEEK_API_KEY is required for the paired accuracy run" >&2
  exit 2
fi

env \
  QW3_KVMEM_RECOMPUTE_QUERY=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
    --tag "$TAG" --data "$DATA" --force \
    --port "$PORT" --ctx "$CTX" --kv-dtype fp16 \
    --block-tokens 32 --budget 229376 --gen-budget 32768 \
    --sink-blocks 8 --recent-blocks 0 \
    --method retrieval --retrieval-method mean-k --update-mode step \
    --kvmem-opt-stage-out "$KVMEM_OPT_STAGE_OUT" \
    --kvmem-opt-stage-in "$KVMEM_OPT_STAGE_IN" \
    --kvmem-opt-pack "$KVMEM_OPT_PACK" \
    --query-conditioned --gpu-memory-ratio "$GPU_MEMORY_RATIO" \
    --cpu-gb "$CPU_GB" --nvme-gb "$NVME_GB" --nvme-dir "$NVME_DIR" \
    --thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 \
    --max-tokens 32768 --mtp --mtp-chain 4 \
    --indices "$INDICES" --read-timeout 7200
