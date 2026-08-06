#!/usr/bin/env bash
# Canonical AgentLongBench 512K (100 rows) and 1M (50 rows) utility rerun.
# This is the B32 controlled counterpart to
# run_paper_utility_32k_adaptive_alb_long.sh: only the physical/retrieval block
# size and identifying names differ. Both runs use K=32K/G=32K Adaptive
# Multi-Prototype retrieval and a CPU-only lower tier.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE=${OUT_BASE:-/data/chaidi/kvmem_eval/results/paper_utility_alb_long_k32g32_adaptive_b32_cpu_only_${STAMP}}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
mkdir -p "$OUT_BASE" "$LOG_BASE"

run_one() {
  local tag=$1
  local result_root=$2
  local ctx=$3
  local cpu_gb=$4
  local data=$5
  local manifest=$6
  local expected=$7
  local benchmark_name=$8

  env \
    TAG="$tag" RESULT_ROOT="$result_root" \
    PORT=18087 CTX="$ctx" CPU_GB="$cpu_gb" \
    DATA="$data" MANIFEST="$manifest" EXPECTED="$expected" \
    BENCHMARK_NAME="$benchmark_name" \
    METHOD=kvmem_keydir_adaptive_k32k_g32k_b32_sink512_recent0_g12-0.10_g24-0.06_qr_immutable_refresh1_kvfp8_idxqfp16_think8k_cpu_only \
    KVMEM_BLOCK_TOKENS=32 KVMEM_RETRIEVAL_METHOD=key-direction-adaptive \
    KVMEM_ADAPTIVE_GAIN_1TO2=0.10 KVMEM_ADAPTIVE_GAIN_2TO4=0.06 \
    KVMEM_SINK_TOKENS=512 KVMEM_RECENT_TOKENS=0 \
    KVMEM_ADAPTIVE_SCORE_MODE=auto KVMEM_INDEX_PLACEMENT=gpu \
    KVMEM_INDEX_STAGING_MB=64 KV_DTYPE=fp8 \
    THINKING_BUDGET=8192 MAX_TOKENS=32768 TEMP=0.6 TOP_P=0.95 \
    SEED=20260722 \
    RECOMPUTE_QUERY=on IMMUTABLE_K=on IMMUTABLE_REFRESH_TOKENS=1 \
    OPT_STAGE_OUT=on OPT_STAGE_IN=on OPT_PACK=on \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_32k_cpu_only.sh"
}

echo "[$(date --iso-8601=seconds)] START $OUT_BASE"

run_one \
  "paper_utility_alb_512k_k32g32_adaptive_b32_cpu_only_${STAMP}" \
  "$OUT_BASE/agentlongbench_512k" \
  655360 64 \
  /data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl \
  /home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl \
  100 AgentLongBench-512K-K32G32-Adaptive-B32-CPU-only

run_one \
  "paper_utility_alb_1m_k32g32_adaptive_b32_cpu_only_${STAMP}" \
  "$OUT_BASE/agentlongbench_1m" \
  1310720 80 \
  /home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  /home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  50 AgentLongBench-1M-K32G32-Adaptive-B32-CPU-only

echo "[$(date --iso-8601=seconds)] COMPLETE $OUT_BASE"
