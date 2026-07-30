#!/usr/bin/env bash
# Full canonical 50-row AgentLongBench-1M Adaptive Multi-Prototype run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export KVMEM_BLOCK_TOKENS=512
export KVMEM_ADAPTIVE_GAIN_1TO2=${KVMEM_ADAPTIVE_GAIN_1TO2:-0.10}
export KVMEM_ADAPTIVE_GAIN_2TO4=${KVMEM_ADAPTIVE_GAIN_2TO4:-0.06}
export BENCHMARK_NAME=AgentLongBench-1M-keydir-adaptive-B512-S32-50
export TAG=${TAG:-agentlongbench_1m_k224k_g32k_b512_s32_keydir_adaptive_max_full50_${STAMP}}
export METHOD=kvmem_keydir_adaptive_max_k224k_g32k_b512_s32_g12-${KVMEM_ADAPTIVE_GAIN_1TO2}_g24-${KVMEM_ADAPTIVE_GAIN_2TO4}_query_replay_immutable_mtp4_kvfp8_idxfp16_qfp16_t06_think8k
export QUESTION_IDS=
export LIMIT=50
export EXPECTED=50

exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_keydir_adaptive_10.sh"
