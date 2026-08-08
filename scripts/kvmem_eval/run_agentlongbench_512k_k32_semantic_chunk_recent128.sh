#!/usr/bin/env bash
# Accuracy candidate for 32K semantic-chunk KVMem.  A four-block (128-token)
# recent anchor preserves the rendered tool/dialogue tail while leaving 1004
# of the 1024 budget blocks available after the 512-token sink reservation.
# All other settings match the established full-query K32 chunked control.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-1.0}
export KVMEM_BUDGET=${KVMEM_BUDGET:-32768}
export GEN_BUDGET=${GEN_BUDGET:-32768}
export REQUEST_MAX_TOKENS=${REQUEST_MAX_TOKENS:-32768}
export RECENT_TOKENS=${RECENT_TOKENS:-128}
export SEMANTIC_QUERY_TOKENS=${SEMANTIC_QUERY_TOKENS:-0}
export SEMANTIC_START_TOKENS=${SEMANTIC_START_TOKENS:-32768}
export THINKING_BUDGET=${THINKING_BUDGET:-8192}
export CPU_GB=${CPU_GB:-48}
export SAMPLE_ORDER=${SAMPLE_ORDER:-budget_gap}
export ADAPTIVE_BLOCK_STATS_MIB=${ADAPTIVE_BLOCK_STATS_MIB:-512}
export INDEX_STAGING_MB=${INDEX_STAGING_MB:-64}
export TAG=${TAG:-agentlongbench_512k_k32_g32_semantic_chunk2048_fullquery_recent128_adaptive_b32_fp8_mtp4}
export METHOD=${METHOD:-kvmem_adaptive_k32_g32_b32_semantic_chunk2048_fullquery_recent128_fp8_mtp4}

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_semantic_chunk_single.sh"
