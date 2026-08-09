#!/usr/bin/env bash
# MemoryAgentBench >256K utility run using the same KVMem construction policy
# as the AgentLongBench K64 semantic-chunk experiment. Each of the 30 canonical
# contexts is built once; its questions use independent frozen archive branches.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_k64_semantic_chunk_${STAMP}}
PY=${PY:-/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python}

export QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB=${QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB:-512}
export QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=${QW3_FLASHINFER_PREFILL_WORKSPACE_MIB:-192}
export QW3_FATTN_NSPLIT=${QW3_FATTN_NSPLIT:-1}
export QW3_PREFILL_FA2_NSPLIT=${QW3_PREFILL_FA2_NSPLIT:-1}
# The runner enables compact PERF_TRACE for every final question. Keep the
# verbose per-block trace opt-in because it materially perturbs long-row
# latency and may generate multi-GiB logs.
if [[ ${QW3_KVMEM_TRACE:-0} == 0 ]]; then
  unset QW3_KVMEM_TRACE
fi

"$PY" "$ROOT/scripts/kvmem_eval/run_memoryagentbench_archive.py" \
  --out-dir "$OUT_DIR" \
  --selection-manifest \
    "$ROOT/scripts/kvmem_eval/memoryagentbench_over256k_contexts.jsonl" \
  --archive-storage cpu-only \
  --tmpfs-limit-gib 60 \
  --block-tokens 32 \
  --budget 65536 \
  --gen-budget 32768 \
  --sink-tokens 512 \
  --recent-tokens 0 \
  --gpu-memory-ratio 1.0 \
  --cpu-gb-tmpfs 48 \
  --cpu-gb-ssd 48 \
  --prefill-chunk 2048 \
  --prefill-window semantic_chunk \
  --prefill-semantic-start-tokens 65536 \
  --prefill-semantic-query-tokens 0 \
  --immutable-refresh-tokens 8 \
  --ladder-tokens 262144 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --index-placement gpu \
  --index-staging-mb 64

"$PY" "$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py" \
  --results-dir "$OUT_DIR" --allow-partial
"$PY" "$ROOT/scripts/kvmem_eval/summarize_memoryagentbench_archive.py" \
  --results-dir "$OUT_DIR"

echo "MemoryAgentBench >256K K64 semantic-chunk complete: $OUT_DIR"
