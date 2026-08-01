#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/beam_10m}
OUT_DIR=${OUT_DIR:-/data/chaidi/kvmem_eval/results}
TAG=${TAG:-beam10m_c1_q20_k224k_b256}
NVME_DIR=${NVME_DIR:-/tmp/qw3_kvmem_eval_nvme_${TAG}}
CONVERSATIONS=${CONVERSATIONS:-1}
QUESTION_LIMIT=${QUESTION_LIMIT:-20}
# Conversation 1 renders to 11,723,318 Qwen tokens. Keep a 12M server
# context so the complete benchmark sample (plus a short probing question)
# fits; set CTX=10000000 explicitly for strict 10M capacity-only runs.
CTX=${CTX:-12000000}
KV_DTYPE=${KV_DTYPE:-fp8}
CPU_GB=${CPU_GB:-96}
# FP8 + local-position MTP at ctx=12M needs about 194.57 GiB for
# position-free raw-K and 225.44 GiB for the authoritative V arena.
# 430 GiB leaves roughly 10 GiB above the fixed-capacity requirement.
NVME_GB=${NVME_GB:-430}
MAX_TOKENS=${MAX_TOKENS:-4096}
PORT=${PORT:-8086}

args=(
  python3 "$ROOT/scripts/kvmem_eval/run_kvmem_eval.py"
  --tag "$TAG"
  --data "$DATA"
  --eval-script "$ROOT/scripts/kvmem_eval/run_beam_10m.py"
  --out-dir "$OUT_DIR"
  --port "$PORT"
  --ctx "$CTX"
  --kv-dtype "$KV_DTYPE"
  --block-tokens 256
  --budget 224000
  --gen-budget 32768
  --sink-blocks 1
  --retrieval-method mean-k
  --kvmem-opt-stage-out on
  --kvmem-opt-stage-in on
  --kvmem-opt-pack on
  --gpu-memory-ratio 0.49
  --cpu-gb "$CPU_GB"
  --nvme-gb "$NVME_GB"
  --nvme-dir "$NVME_DIR"
  --max-tokens "$MAX_TOKENS"
  --temperature 0
  --top-p 1
  --read-timeout 7200
  --health-timeout 900
  --server-extra-arg=--kvmem-prefix-cache
  --server-extra-arg=--kvmem-raw-k-nvme
  --server-extra-arg=--kvmem-index-placement
  --server-extra-arg=cpu
  --server-extra-arg=--kvmem-index-staging-mb
  --server-extra-arg=64
  --eval-extra-arg=--conversation-ids
  --eval-extra-arg="$CONVERSATIONS"
  --limit "$QUESTION_LIMIT"
  --no-judge
)

if [[ ${NO_THINKING:-0} == 1 ]]; then
  args+=(--no-thinking)
fi
if [[ ${NO_MTP:-0} == 1 ]]; then
  args+=(--no-mtp)
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  args+=(--dry-run)
fi

QW3_KVMEM_PREFIX_CACHE_TRACE=1 \
QW3_KVMEM_TRACE=${QW3_KVMEM_TRACE:-1} \
QW3_Q8_BF16_MAIN=0 \
"${args[@]}"

if [[ ${DRY_RUN:-0} != 1 ]]; then
  hits=$(grep -c 'kvmem prefix-cache HIT' "$OUT_DIR/${TAG}_serve.log" || true)
  if (( hits < QUESTION_LIMIT )); then
    echo "BEAM warm-prefix verification failed: hits=$hits expected_at_least=$QUESTION_LIMIT" >&2
    exit 1
  fi
  echo "BEAM warm-prefix verification passed: hits=$hits"
fi
