#!/usr/bin/env bash
# Standalone original-KVMem rerun of the frozen LongMemEval-M error-10 subset.
# Query replay and selected-text dense replay are deliberately disabled so this
# remains a clean matched control.  This script does not run unless invoked.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/longmemeval_m_error10_20260720/samples.jsonl}
TAG=${TAG:-longmemeval_m_error10_original_k224k_b32_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-8087}

if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  echo "DEEPSEEK_API_KEY is required for LongMemEval grading" >&2
  exit 2
fi

env \
  QW3_KVMEM_RECOMPUTE_QUERY=0 \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro} \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py" \
    --tag "$TAG" --data "$DATA" --force \
    --port "$PORT" --ctx 2000000 --kv-dtype fp16 \
    --block-tokens 32 --budget 229376 --gen-budget 32768 \
    --sink-blocks 8 --recent-blocks 0 \
    --method retrieval --retrieval-method mean-k --update-mode step \
    --query-conditioned --gpu-memory-ratio 0.51 \
    --cpu-gb 64 --nvme-gb 256 \
    --thinking --thinking-budget 4096 \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 \
    --max-tokens 32768 --mtp --mtp-chain 4 \
    --read-timeout 7200
