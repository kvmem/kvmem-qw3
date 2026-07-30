#!/usr/bin/env bash
# AgentLongBench-1M Adaptive Multi-Prototype probe:
#   512-token retrieval/materialization block
#   sixteen 32-token scoring slices per block
#   each slice dynamically stores 1, 2, or 4 Key-direction prototypes
#   MaxSim across the packed prototypes when ranking the 512-token block.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export KVMEM_BLOCK_TOKENS=512
export KVMEM_ADAPTIVE_GAIN_1TO2=${KVMEM_ADAPTIVE_GAIN_1TO2:-0.10}
export KVMEM_ADAPTIVE_GAIN_2TO4=${KVMEM_ADAPTIVE_GAIN_2TO4:-0.06}
export BENCHMARK_NAME=AgentLongBench-1M-keydir-adaptive-B512-S32-10
export TAG=${TAG:-agentlongbench_1m_k224k_g32k_b512_s32_keydir_adaptive_max_10_${STAMP}}
export METHOD=kvmem_keydir_adaptive_max_k224k_g32k_b512_s32_g12-${KVMEM_ADAPTIVE_GAIN_1TO2}_g24-${KVMEM_ADAPTIVE_GAIN_2TO4}_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_adaptive_10.sh"
