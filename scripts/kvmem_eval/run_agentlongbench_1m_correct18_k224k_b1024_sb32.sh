#!/usr/bin/env bash
# Regression evaluation over the 18 samples answered correctly by the canonical
# AgentLongBench-1M KVMem B32/Mean-K run. This measures correct-to-wrong flips
# after changing retrieval grouping to 1024-token physical blocks scored as
# 32 x 32-token Mean-K sub-blocks with max reduction.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"
ID_FILE="$ROOT/scripts/kvmem_eval/agentlongbench_1m_kvmem_correct18_ids.txt"

mapfile -t ids < <(sed '/^[[:space:]]*$/d' "$ID_FILE")
if [[ "${#ids[@]}" -ne 18 ]]; then
  echo "expected 18 sample IDs in $ID_FILE, found ${#ids[@]}" >&2
  exit 2
fi
QUESTION_IDS=$(IFS=,; echo "${ids[*]}")

exec env \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  TAG="${TAG:-agentlongbench_1m_correct18_k224k_g32k_b1024_sb32max_fp8_qrfix_immutable_refresh1_t06_think8k_20260727}" \
  PORT="${PORT:-8089}" \
  CTX=1310720 \
  CPU_GB=80 \
  NVME_GB=0 \
  KVMEM_BUDGET=229376 \
  GEN_BUDGET=32768 \
  KVMEM_BLOCK_TOKENS=1024 \
  KVMEM_RETRIEVAL_METHOD=sub-block-mean-k \
  KVMEM_SUBBLOCKS=32 \
  KVMEM_SUBBLOCK_REDUCE=max \
  KV_DTYPE=fp8 \
  PREFILL_CHUNK=2048 \
  GPU_MEMORY_RATIO=0.5 \
  TEMP=0.6 \
  THINKING_BUDGET=8192 \
  DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  QUESTION_IDS="$QUESTION_IDS" \
  EXPECTED=18 \
  BENCHMARK_NAME=AgentLongBench-1M-canonical-correct18-K224K-B1024-SB32Max \
  METHOD=kvmem_subblock_mean_k_k224k_g32k_b1024_sb32max_query_replay_immutable_refresh1_mtp4_fp8_t06_think8k_regression \
  KVMEM_OPT_LEVEL= \
  KVMEM_OPTIMIZE_OFF= \
  "$BASE"
