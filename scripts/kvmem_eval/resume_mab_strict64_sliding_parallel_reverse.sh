#!/usr/bin/env bash
# Saturate the second vLLM sequence during the strict-64K Sliding Window pass.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/qw3/.venv/bin/python}
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
  --method sliding-window
)

# Utility must remain phase ordered: do not issue any Sliding Window request
# until every strict Compact+RAG context has a durable row summary.
while [[ $(find "$WORKSPACE/methods/compact-rag/rows" -name row_summary.json 2>/dev/null | wc -l) -lt 30 ]]; do
  sleep 30
done
until curl --noproxy '*' -fsS --max-time 2 "${API_BASE%/v1}/health" >/dev/null; do
  sleep 5
done

mapfile -t WORK < <(
  "$PY" - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / "shared" / "rag"
for path in sorted(root.glob("*.json"), reverse=True):
    row = json.loads(path.read_text(encoding="utf-8"))
    print(f"{row['split']}\t{row['source']}\t{int(row['dataset_row'])}")
PY
)

for item in "${WORK[@]}"; do
  IFS=$'\t' read -r split source row <<<"$item"
  echo "[$(date --iso-8601=seconds)] reverse sliding split=$split source=$source row=$row"
  until env PYTHONUNBUFFERED=1 "$PY" "$RUNNER" run "${COMMON[@]}" \
      --split "$split" --source "$source" --row "$row"; do
    echo "[$(date --iso-8601=seconds)] interrupted; waiting for server retry"
    until curl --noproxy '*' -fsS --max-time 2 "${API_BASE%/v1}/health" >/dev/null; do
      sleep 10
    done
  done
done
