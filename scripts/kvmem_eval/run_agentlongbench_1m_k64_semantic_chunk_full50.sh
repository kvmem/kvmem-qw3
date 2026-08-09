#!/usr/bin/env bash
# AgentLongBench 1M canonical-50 validation using the established K64
# semantic-chunk configuration.  The selected sample set is the first 50 rows
# in canonical dataset order, matching the existing 1M utility baselines.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export PORT=${PORT:-18118}
export GPU_MEMORY_RATIO=${GPU_MEMORY_RATIO:-1.0}
export CTX_TOKENS=${CTX_TOKENS:-1310720}
export BENCHMARK_NAME=${BENCHMARK_NAME:-AgentLongBench-1M-DeepseekMillion50}
export KVMEM_BUDGET=${KVMEM_BUDGET:-65536}
export GEN_BUDGET=${GEN_BUDGET:-32768}
export BLOCK_TOKENS=${BLOCK_TOKENS:-32}
export REQUEST_MAX_TOKENS=${REQUEST_MAX_TOKENS:-32768}
export RECENT_TOKENS=${RECENT_TOKENS:-0}
export SEMANTIC_QUERY_TOKENS=${SEMANTIC_QUERY_TOKENS:-0}
export SEMANTIC_START_TOKENS=${SEMANTIC_START_TOKENS:-65536}
export THINKING_BUDGET=${THINKING_BUDGET:-8192}
export CPU_GB=${CPU_GB:-48}
export ADAPTIVE_BLOCK_STATS_MIB=${ADAPTIVE_BLOCK_STATS_MIB:-512}
export INDEX_STAGING_MB=${INDEX_STAGING_MB:-64}
export START_INDEX=${START_INDEX:-0}
export LIMIT=${LIMIT:-50}
export SAMPLE_ORDER=${SAMPLE_ORDER:-dataset}
export DATA=${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}
export MANIFEST=${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}
export TAG=${TAG:-agentlongbench_1m_k64_g32_semantic_chunk2048_fullquery_recent0_adaptive_b32_fp8_mtp4_full50_${STAMP}}
export METHOD=${METHOD:-kvmem_adaptive_k64_g32_b32_semantic_chunk2048_fullquery_recent0_fp8_mtp4}

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_semantic_chunk_single.sh"
