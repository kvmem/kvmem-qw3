#!/usr/bin/env bash
# AgentLongBench-1M hierarchical Fixed-4 probe:
#   512-token retrieval/materialization block
#   16 contiguous 32-token scoring slices per block
#   4 Key-direction prototypes per slice
#   MaxSim across all 64 prototypes when ranking the 512-token block.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export KVMEM_BLOCK_TOKENS=512
export BENCHMARK_NAME=AgentLongBench-1M-keydir-fixed4-B512-S32-10
export TAG=${TAG:-agentlongbench_1m_k224k_g32k_b512_s32_keydir_fixed4_max_10_${STAMP}}
export METHOD=kvmem_keydir_fixed4_max_k224k_g32k_b512_s32_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_fixed4_10.sh"
