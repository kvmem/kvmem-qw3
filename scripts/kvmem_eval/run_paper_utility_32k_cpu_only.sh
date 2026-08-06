#!/usr/bin/env bash
# Sequential paper utility rerun: K=32K, G=32K, CPU lower tier, no SSD.
# All other model/scorer/sampling choices follow each frozen utility run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE=${OUT_BASE:-/data/chaidi/kvmem_eval/results/paper_utility_k32g32_cpu_only_${STAMP}}
LOG_BASE=${LOG_BASE:-/data/chaidi/kvmem_eval/logs}
mkdir -p "$OUT_BASE" "$LOG_BASE"

run_alb_le256() {
  env \
    TAG="paper_utility_alb_le256_k32g32_cpu_only_${STAMP}" \
    RESULT_ROOT="$OUT_BASE/agentlongbench_le256" \
    PORT=18087 CTX=262144 CPU_GB=64 \
    DATA=/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl \
    MANIFEST=/data/chaidi/kvmem_eval/data/agentlongbench_250/manifest.jsonl \
    EXPECTED=250 BENCHMARK_NAME=AgentLongBench-le256-K32G32-CPU-only \
    METHOD=kvmem_mean_k_k32k_g32k_b32_fp16_think4k_cpu_only \
    KVMEM_BLOCK_TOKENS=32 KVMEM_RETRIEVAL_METHOD=mean-k \
    KVMEM_INDEX_PLACEMENT=gpu KV_DTYPE=fp16 \
    THINKING_BUDGET=4096 MAX_TOKENS=32768 TEMP=0.6 TOP_P=0.95 \
    SEED='' \
    RECOMPUTE_QUERY=off IMMUTABLE_K=off \
    OPT_STAGE_OUT=off OPT_STAGE_IN=on OPT_PACK=off \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_32k_cpu_only.sh"
}

run_alb_512k() {
  env \
    TAG="paper_utility_alb_512k_k32g32_cpu_only_${STAMP}" \
    RESULT_ROOT="$OUT_BASE/agentlongbench_512k" \
    PORT=18087 CTX=655360 CPU_GB=64 \
    DATA=/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl \
    MANIFEST=/home/chaidi/AgentLongBench-Long/results/agentlongbench_512k_normal100/compact_only_normal100/manifest/selected_samples.jsonl \
    EXPECTED=100 BENCHMARK_NAME=AgentLongBench-512K-K32G32-CPU-only \
    METHOD=kvmem_mean_k_k32k_g32k_b32_query_replay_immutable_fp16_think4k_cpu_only \
    KVMEM_BLOCK_TOKENS=32 KVMEM_RETRIEVAL_METHOD=mean-k \
    KVMEM_INDEX_PLACEMENT=gpu KV_DTYPE=fp16 \
    THINKING_BUDGET=4096 MAX_TOKENS=32768 TEMP=0.6 TOP_P=0.95 \
    RECOMPUTE_QUERY=on IMMUTABLE_K=on \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_32k_cpu_only.sh"
}

run_alb_1m() {
  env \
    TAG="paper_utility_alb_1m_k32g32_cpu_only_${STAMP}" \
    RESULT_ROOT="$OUT_BASE/agentlongbench_1m" \
    PORT=18087 CTX=1310720 CPU_GB=80 \
    DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
    MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
    EXPECTED=50 BENCHMARK_NAME=AgentLongBench-1M-K32G32-CPU-only \
    METHOD=kvmem_keydir_fixed4_k32k_g32k_b512_s32_query_replay_immutable_fp8_think8k_cpu_only \
    KVMEM_BLOCK_TOKENS=512 KVMEM_RETRIEVAL_METHOD=key-direction-fixed4 \
    KVMEM_INDEX_PLACEMENT=gpu KV_DTYPE=fp8 \
    THINKING_BUDGET=8192 MAX_TOKENS=32768 TEMP=0.6 TOP_P=0.95 \
    RECOMPUTE_QUERY=on IMMUTABLE_K=on IMMUTABLE_REFRESH_TOKENS=1 \
    "$ROOT/scripts/kvmem_eval/run_agentlongbench_32k_cpu_only.sh"
}

run_memoryagentbench() {
  /home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python \
    "$ROOT/scripts/kvmem_eval/run_memoryagentbench_archive.py" \
    --out-dir "$OUT_BASE/memoryagentbench" \
    --archive-storage cpu-only \
    --tmpfs-root /dev/shm --tmpfs-limit-gib 60 \
    --budget 32768 --gen-budget 32768 \
    --block-tokens 512 --sink-tokens 2048 --recent-tokens 16384 \
    --cpu-gb-tmpfs 16 --gpu-memory-ratio 0.5 \
    --index-placement cpu --index-staging-mb 64 \
    --prefill-chunk 2048 --temperature 0.6 --top-p 0.95 --top-k 20
}

score_memoryagentbench() {
  local results_dir="$OUT_BASE/memoryagentbench"
  local eval_python=/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python

  "$eval_python" \
    "$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py" \
    --results-dir "$results_dir"

  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is required for the 400 special judge records" >&2
    return 2
  fi
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    NO_PROXY= no_proxy= \
    "$eval_python" \
    "$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py" \
    --results-dir "$results_dir" \
    --model deepseek-v4-pro \
    --thinking enabled --infbench-thinking disabled
}

echo "[$(date --iso-8601=seconds)] START $OUT_BASE"
run_alb_le256
run_alb_512k
run_alb_1m
run_memoryagentbench
score_memoryagentbench
echo "[$(date --iso-8601=seconds)] COMPLETE $OUT_BASE"
