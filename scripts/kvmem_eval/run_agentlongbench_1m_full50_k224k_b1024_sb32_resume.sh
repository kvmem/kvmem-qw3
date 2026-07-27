#!/usr/bin/env bash
# Single-process full AgentLongBench-1M B1024/SB32 run.  On its first launch,
# seed the destination with completed rows from the former error19 and
# canonical-correct18 subset runs.  The canonical evaluator then skips those
# IDs and evaluates only missing samples; subsequent launches resume normally.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh"
SEED="$ROOT/scripts/kvmem_eval/seed_agentlongbench_resume.py"
DATA=/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl
MANIFEST=/home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl
TAG=${TAG:-agentlongbench_1m_full50_k224k_g32k_b1024_sb32max_fp8_qrfix_immutable_refresh1_t06_think8k_20260727}
RESULT_ROOT=${RESULT_ROOT:-/data/chaidi/kvmem_eval/results/$TAG}
BENCHMARK_NAME=AgentLongBench-1M-full50-K224K-B1024-SB32Max
METHOD=kvmem_subblock_mean_k_k224k_g32k_b1024_sb32max_query_replay_immutable_refresh1_mtp4_fp8_t06_think8k
ERROR19_ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_1m_error19_k224k_g32k_b1024_sb32max_fp8_qrfix_immutable_refresh1_t06_think8k_20260727
CORRECT18_ROOT=/data/chaidi/kvmem_eval/results/agentlongbench_1m_correct18_k224k_g32k_b1024_sb32max_fp8_qrfix_immutable_refresh1_t06_think8k_20260727
CANONICAL50_EVAL=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_k224k_g32k_b32_meank_kvfp8_idxqfp16_canonical_refresh1_t06_think8k_cpu_all_on_gpuratio050_20260726_235009/eval.jsonl

mapfile -t ids < <(
  "$ROOT/.venv/bin/python" - "$CANONICAL50_EVAL" <<'PY'
import json
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["stable_sample_id"])
PY
)
if [[ "${#ids[@]}" -ne 50 ]]; then
  echo "expected 50 canonical sample IDs, found ${#ids[@]}" >&2
  exit 2
fi
QUESTION_IDS=$(IFS=,; echo "${ids[*]}")

if [[ ! -e "$RESULT_ROOT" ]]; then
  "$ROOT/.venv/bin/python" "$SEED" \
    --dataset "$DATA" \
    --selected-id-source "$CANONICAL50_EVAL" \
    --output-root "$RESULT_ROOT" \
    --source-root "$ERROR19_ROOT" \
    --source-root "$CORRECT18_ROOT" \
    --benchmark-name "$BENCHMARK_NAME" \
    --method "$METHOD"
elif [[ ! -e "$RESULT_ROOT/run_config.json" &&
        ! -e "$RESULT_ROOT/resume_seed_provenance.json" ]]; then
  echo "refusing an unrecognized non-empty resume root: $RESULT_ROOT" >&2
  exit 3
fi

exec env \
  QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
  TAG="$TAG" \
  RESULT_ROOT="$RESULT_ROOT" \
  PORT="${PORT:-8088}" \
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
  DATA="$DATA" \
  MANIFEST="$MANIFEST" \
  QUESTION_IDS="$QUESTION_IDS" \
  LIMIT= \
  EXPECTED=50 \
  BENCHMARK_NAME="$BENCHMARK_NAME" \
  METHOD="$METHOD" \
  KVMEM_OPT_LEVEL= \
  KVMEM_OPTIMIZE_OFF= \
  "$BASE"
