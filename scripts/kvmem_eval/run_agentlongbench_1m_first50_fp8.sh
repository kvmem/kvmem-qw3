#!/usr/bin/env bash
# DeepseekMillion fixed-order first-50 wrapper.
#
# Prompt construction, query-span marking, request execution, and official
# evaluation are all delegated to the same reusable AgentLongBench launcher
# used by the 512K experiment.  This file only fixes the 1M dataset and the
# long-context FP8 serving profile.  LIMIT/EXPECTED/TAG/PORT remain
# overridable so a one-sample smoke test and a resumable full run use exactly
# the same path.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"

LIMIT=${LIMIT:-50}
EXPECTED=${EXPECTED:-$LIMIT}
TAG=${TAG:-agentlongbench_1m_first50_k224k_g32k_b32_qr_immutable_mtp4_fp8_cpu64_opt3_20260725}

exec env \
  TAG="$TAG" \
  PORT="${PORT:-8087}" \
  CTX="${CTX:-1200000}" \
  CPU_GB="${CPU_GB:-64}" \
  NVME_GB="${NVME_GB:-0}" \
  KVMEM_OPT_LEVEL="${KVMEM_OPT_LEVEL:-opt_3}" \
  KVMEM_BUDGET="${KVMEM_BUDGET:-229376}" \
  GEN_BUDGET="${GEN_BUDGET:-32768}" \
  KV_DTYPE="${KV_DTYPE:-fp8}" \
  PREFILL_CHUNK="${PREFILL_CHUNK:-8192}" \
  DATA="${DATA:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl}" \
  MANIFEST="${MANIFEST:-/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl}" \
  LIMIT="$LIMIT" \
  EXPECTED="$EXPECTED" \
  BENCHMARK_NAME="${BENCHMARK_NAME:-AgentLongBench-1M-DeepseekMillion-first50}" \
  METHOD="${METHOD:-kvmem_mean_k_224k_b32_query_replay_immutable_mtp4_fp8_opt3}" \
  "$BASE"
