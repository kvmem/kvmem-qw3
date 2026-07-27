#!/usr/bin/env bash
# The same three AgentLongBench-1M error samples, using 512-token physical
# retrieval blocks split into 16 x 32-token Mean-K scoring sub-blocks. The
# physical block score is the maximum sub-block score.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"

IDS=(
  af004d2443fcd470b168453ff3553d666b3001456ffa55a21726a6850725af0b
  2167e18164c2bfeffc68feb4ebc9e7a66204e04c167289a9643cfcebc267f021
  8fb000bd3d77601221e94bb66667e16344605c4e2d1fc41666228fe4db853706
)
QUESTION_IDS=$(IFS=,; echo "${IDS[*]}")

exec env \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  TAG="${TAG:-agentlongbench_1m_error3_k224k_g32k_b512_sb16max_fp8_qr_immutable_refresh1_t06_think8k_20260727}" \
  PORT="${PORT:-8088}" \
  CTX=1310720 \
  CPU_GB=80 \
  NVME_GB=0 \
  KVMEM_BUDGET=229376 \
  GEN_BUDGET=32768 \
  KVMEM_BLOCK_TOKENS=512 \
  KVMEM_RETRIEVAL_METHOD=sub-block-mean-k \
  KVMEM_SUBBLOCKS=16 \
  KVMEM_SUBBLOCK_REDUCE=max \
  KV_DTYPE=fp8 \
  PREFILL_CHUNK=2048 \
  GPU_MEMORY_RATIO=0.5 \
  TEMP=0.6 \
  THINKING_BUDGET=8192 \
  DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  QUESTION_IDS="$QUESTION_IDS" \
  EXPECTED=3 \
  BENCHMARK_NAME=AgentLongBench-1M-error3-K224K-B512-SB16Max \
  METHOD=kvmem_subblock_mean_k_k224k_g32k_b512_sb16max_query_replay_immutable_refresh1_mtp4_fp8_t06_think8k \
  KVMEM_OPT_LEVEL= \
  KVMEM_OPTIMIZE_OFF= \
  "$BASE"
