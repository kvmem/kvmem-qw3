#!/usr/bin/env bash
# AgentLongBench <=256K and 512K utility rerun using the Adaptive
# Multi-Prototype configuration validated by the AgentLongBench-1M run.
#
# KVMem remains K=32K/G=32K and CPU-only below the GPU tier.  The retrieval
# representation follows the 1M experiment: 512-token blocks, dynamically
# packed 1/2/4 prototypes, FP8 KV with FP16 index/query, query recomputation,
# immutable raw K, and refresh_tokens=1.  The 1M run's 4K/224K sink ratio is
# scaled to one 512-token block for the 32K budget; recent remains disabled.
# The historical plain-Mean-K runner remains unchanged for paired comparison.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE=${OUT_BASE:-/data/chaidi/kvmem_eval/results/paper_utility_k32g32_adaptive_cpu_only_${STAMP}}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
mkdir -p "$OUT_BASE" "$LOG_BASE"

run_alb_le256() {
  env \
    TAG="paper_utility_alb_le256_k32g32_adaptive_cpu_only_${STAMP}" \
    RESULT_ROOT="$OUT_BASE/agentlongbench_le256" \
    PORT=18087 CTX=262144 CPU_GB=64 \
    DATA=/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl \
    MANIFEST=/data/chaidi/kvmem_eval/data/agentlongbench_250/manifest.jsonl \
    EXPECTED=250 BENCHMARK_NAME=AgentLongBench-le256-K32G32-Adaptive-CPU-only \
    METHOD=kvmem_keydir_adaptive_k32k_g32k_b512_sink512_recent0_g12-0.10_g24-0.06_qr_immutable_refresh1_kvfp8_idxqfp16_think8k_cpu_only \
    KVMEM_BLOCK_TOKENS=512 KVMEM_RETRIEVAL_METHOD=key-direction-adaptive \
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

run_alb_512k() {
  env \
    TAG="paper_utility_alb_512k_k32g32_adaptive_cpu_only_${STAMP}" \
    RESULT_ROOT="$OUT_BASE/agentlongbench_512k" \
    PORT=18087 CTX=655360 CPU_GB=64 \
    DATA=/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl \
    MANIFEST=/home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl \
    EXPECTED=100 BENCHMARK_NAME=AgentLongBench-512K-K32G32-Adaptive-CPU-only \
    METHOD=kvmem_keydir_adaptive_k32k_g32k_b512_sink512_recent0_g12-0.10_g24-0.06_qr_immutable_refresh1_kvfp8_idxqfp16_think8k_cpu_only \
    KVMEM_BLOCK_TOKENS=512 KVMEM_RETRIEVAL_METHOD=key-direction-adaptive \
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
run_alb_le256
run_alb_512k
echo "[$(date --iso-8601=seconds)] COMPLETE $OUT_BASE"
