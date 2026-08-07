#!/usr/bin/env bash
# Revised 512K semantic-chunk experiment:
# - 64K historical selection and 32K append/generation reserve;
# - every complete 2048-token chunk is the retrieval query;
# - no fixed recent band, leaving all non-sink slots to semantic selection.
# This wrapper only supplies a new reproducible configuration. It does not
# launch unless invoked explicitly.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-1.0}
export KVMEM_BUDGET=${KVMEM_BUDGET:-65536}
export GEN_BUDGET=${GEN_BUDGET:-32768}
export RECENT_TOKENS=${RECENT_TOKENS:-0}
export SEMANTIC_QUERY_TOKENS=${SEMANTIC_QUERY_TOKENS:-0}
export SEMANTIC_START_TOKENS=${SEMANTIC_START_TOKENS:-65536}
export THINKING_BUDGET=${THINKING_BUDGET:-8192}
export SAMPLE_ORDER=${SAMPLE_ORDER:-budget_gap}
# Keep the established scorer workspace and index-staging ceilings.  Exact
# long-query tiling must optimize within these bounds rather than trading more
# GPU memory for fewer index scans.
export ADAPTIVE_BLOCK_STATS_MIB=${ADAPTIVE_BLOCK_STATS_MIB:-512}
export INDEX_STAGING_MB=${INDEX_STAGING_MB:-64}
export TAG=${TAG:-agentlongbench_512k_k64_g32_semantic_chunk2048_fullquery_recent0_adaptive_b32_fp8_mtp4_gaporder_20260806}
export METHOD=${METHOD:-kvmem_adaptive_k64_g32_b32_semantic_chunk2048_fullquery_recent0_fp8_mtp4}

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_semantic_chunk_single.sh"
