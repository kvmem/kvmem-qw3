#!/usr/bin/env bash
# First-three AgentLongBench 1M error control for per-round reselection.
#
# ROUND_QUERY=first_user preserves the existing turn-ingest policy.
# ROUND_QUERY=whole_round selects and query-replays every complete
# role-preserving round once the prior logical history reaches 224K.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROUND_QUERY=${ROUND_QUERY:-whole_round}
if [[ "$ROUND_QUERY" != "first_user" &&
      "$ROUND_QUERY" != "whole_round" ]]; then
  echo "ROUND_QUERY must be first_user or whole_round" >&2
  exit 2
fi

TAG=${TAG:-agentlongbench_1m_error3_round_${ROUND_QUERY}_k224k_b32_fp8_$(date +%Y%m%d_%H%M%S)}

exec env \
  TAG="$TAG" \
  PORT="${PORT:-18087}" \
  CTX="${CTX:-1310720}" \
  KVMEM_BUDGET=229376 \
  GEN_BUDGET=32768 \
  ACTIVE_CAPACITY=229376 \
  KV_DTYPE=fp8 \
  CPU_GB="${CPU_GB:-64}" \
  NVME_GB="${NVME_GB:-0}" \
  GPU_MEMORY_RATIO="${GPU_MEMORY_RATIO:-0.50}" \
  KVMEM_OPT_LEVEL="${KVMEM_OPT_LEVEL:-opt_3}" \
  IMMUTABLE_REFRESH_TOKENS=1 \
  THINKING_BUDGET=8192 \
  DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  LIMIT=3 \
  EXPECTED=3 \
  ROUND_QUERY="$ROUND_QUERY" \
  BENCHMARK_NAME="AgentLongBench-1M-error3-round-${ROUND_QUERY}" \
  METHOD="kvmem_turn_ingest_${ROUND_QUERY}_k224k_b32_query_replay_immutable_mtp4_fp8" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_turn_ingest.sh"
