#!/usr/bin/env bash
# Fill the long-range >256K rows from the tail while the canonical supervisor
# advances from the head. run_memoryagentbench_baselines.py serializes any
# eventual overlap with a per-method, per-row flock.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
RUNNER=${RUNNER:-$ROOT/scripts/kvmem_eval/run_memoryagentbench_baselines.py}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812}
API_BASE=${API_BASE:-http://127.0.0.1:18122/v1}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-27B-FP8}

COMMON=(
  --workspace "$WORKSPACE"
  --reference-results /data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full
  --min-context-tokens-exclusive 262144
  --context-window 262144
  --generation-reserve 32768
  --strict-final-prompt-tokens 65536
  --sliding-prompt-tokens 65536
  --compact-full-history
  --compact-no-tail
  --source-shared-workspace /data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802
  --rag-dynamic-budget
  --rag-no-metadata
  --rag-top-k 2200
  --rag-block-size 32
  --rag-overlap 8
  --rag-model /data/chaidi/kvmem_eval/models/jinaai/jina-embeddings-v2-small-en
  --tokenizer-dir /home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8
  --embedding-batch-size 256
  --api-base "$API_BASE"
  --model-name "$MODEL_NAME"
  --server-mtp-chain 4
  --prefix-cache-guard-pages 0
  --split Long_Range_Understanding
  --method compact-rag
)

until curl --noproxy '*' -fsS --max-time 2 "${API_BASE%/v1}/health" >/dev/null; do
  sleep 5
done

# Dataset-row indices of the selected >256K Long_Range_Understanding cohort,
# intentionally in reverse order relative to the canonical runner.
for row in 94 93 89 81 78 74 35 28 19 18 15 14 12 10 9 0; do
  echo "[$(date --iso-8601=seconds)] parallel compact-rag row=$row"
  until env PYTHONUNBUFFERED=1 "$PY" "$RUNNER" run "${COMMON[@]}" --row "$row"; do
    echo "[$(date --iso-8601=seconds)] row=$row interrupted; waiting for server retry"
    until curl --noproxy '*' -fsS --max-time 2 "${API_BASE%/v1}/health" >/dev/null; do
      sleep 10
    done
  done
done
