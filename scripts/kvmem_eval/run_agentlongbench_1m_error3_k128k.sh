#!/usr/bin/env bash
# Ordinary Mean-K control on the same three oracle-analysis samples, changing
# only the active KVMem context budget from 224K to 128K.
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
  TAG="${TAG:-agentlongbench_1m_error3_k128k_g32k_b32_meank_fp8_qr_immutable_refresh1_t06_think8k_20260727}" \
  PORT="${PORT:-8088}" \
  CTX=1310720 \
  CPU_GB=80 \
  NVME_GB=0 \
  KVMEM_BUDGET=131072 \
  GEN_BUDGET=32768 \
  KV_DTYPE=fp8 \
  PREFILL_CHUNK=2048 \
  GPU_MEMORY_RATIO=0.5 \
  TEMP=0.6 \
  THINKING_BUDGET=8192 \
  DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  QUESTION_IDS="$QUESTION_IDS" \
  EXPECTED=3 \
  BENCHMARK_NAME=AgentLongBench-1M-error3-K128K \
  METHOD=kvmem_mean_k_k128k_g32k_b32_query_replay_immutable_refresh1_mtp4_fp8_t06_think8k \
  KVMEM_OPT_STAGE_OUT=on \
  KVMEM_OPT_STAGE_IN=on \
  KVMEM_OPT_PACK=on \
  "$BASE"
