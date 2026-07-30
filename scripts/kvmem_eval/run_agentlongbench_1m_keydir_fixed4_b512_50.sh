#!/usr/bin/env bash
# Full canonical 50-row AgentLongBench-1M run for hierarchical Fixed-4:
#   512-token retrieval/materialization block
#   16 contiguous 32-token scoring slices per block
#   4 Key-direction prototypes per slice
#   MaxSim across all 64 prototypes when ranking each 512-token block.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export KVMEM_BLOCK_TOKENS=512
export BENCHMARK_NAME=AgentLongBench-1M-keydir-fixed4-B512-S32-50
export TAG=${TAG:-agentlongbench_1m_k224k_g32k_b512_s32_keydir_fixed4_max_full50_${STAMP}}
export METHOD=kvmem_keydir_fixed4_max_k224k_g32k_b512_s32_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k

# An explicitly empty QUESTION_IDS disables the default ten-ID probe cohort.
export QUESTION_IDS=
export LIMIT=50
export EXPECTED=50

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_fixed4_10.sh"
