#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-/data/chaidi/kvmem_eval/data/beam_10m}
OUT_DIR=${OUT_DIR:-/data/chaidi/kvmem_eval/results}
TAG=${TAG:-beam10m_c1_q20_k32g32_adaptive_b512_cpu64_frozen}
NVME_DIR=${NVME_DIR:-/tmp/qw3_kvmem_eval_nvme_${TAG}}
CONVERSATIONS=${CONVERSATIONS:-1}
QUESTION_LIMIT=${QUESTION_LIMIT:-20}
QUESTION_INDICES=${QUESTION_INDICES:-}
# Conversation 1 renders to 11,723,318 Qwen tokens. Keep a 12M server
# context so the complete benchmark sample (plus a short probing question)
# fits; set CTX=10000000 explicitly for strict 10M capacity-only runs.
CTX=${CTX:-12000000}
KV_DTYPE=${KV_DTYPE:-fp8}
CPU_GB=${CPU_GB:-64}
# FP8 + local-position MTP at ctx=12M needs about 194.57 GiB for
# position-free raw-K and 225.44 GiB for the authoritative V arena.
# 430 GiB leaves roughly 10 GiB above the fixed-capacity requirement.
NVME_GB=${NVME_GB:-430}
THINKING_BUDGET=${THINKING_BUDGET:-8192}
MAX_TOKENS=${MAX_TOKENS:-32768}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
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
  --block-tokens 512
  --budget 32768
  --gen-budget 32768
  --sink-blocks 1
  --retrieval-method key-direction-adaptive
  --adaptive-gain-1to2 0.10
  --adaptive-gain-2to4 0.06
  --kvmem-opt-stage-out on
  --kvmem-opt-stage-in on
  --kvmem-opt-pack on
  --gpu-memory-ratio 0.49
  --cpu-gb "$CPU_GB"
  --nvme-gb "$NVME_GB"
  --nvme-dir "$NVME_DIR"
  --thinking-budget "$THINKING_BUDGET"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  # A cold 11.7M-token history ingest took about 2.27 h on this host.  The
  # frozen queries are short, but the shared client timeout must also cover
  # that one-time build.
  --read-timeout 21600
  --health-timeout 900
  --server-extra-arg=--kvmem-raw-k-nvme
  --server-extra-arg=--kvmem-index-placement
  --server-extra-arg=cpu
  --server-extra-arg=--kvmem-index-staging-mb
  --server-extra-arg=64
  --server-extra-arg=--kvmem-adaptive-score-mode
  --server-extra-arg=auto
  --eval-extra-arg=--conversation-ids
  --eval-extra-arg="$CONVERSATIONS"
  --limit "$QUESTION_LIMIT"
  --no-judge
)

if [[ -n "$QUESTION_INDICES" ]]; then
  args+=(--indices "$QUESTION_INDICES")
fi

if [[ ${NO_THINKING:-0} == 1 ]]; then
  args+=(--no-thinking)
fi
if [[ ${NO_MTP:-0} == 1 ]]; then
  args+=(--no-mtp)
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  args+=(--dry-run)
fi

QW3_KVMEM_TRACE=${QW3_KVMEM_TRACE:-1} \
QW3_Q8_BF16_MAIN=0 \
"${args[@]}"

if [[ ${DRY_RUN:-0} != 1 ]]; then
  saves=$(grep -c 'native kvmem local-cache SAVE' "$OUT_DIR/${TAG}_serve.log" || true)
  restores=$(grep -c 'native kvmem local-cache RESTORE' "$OUT_DIR/${TAG}_serve.log" || true)
  expected_restores=$((2 * QUESTION_LIMIT))
  if (( saves < 1 || restores < expected_restores )); then
    echo "BEAM frozen-cache verification failed: saves=$saves restores=$restores expected_restores_at_least=$expected_restores" >&2
    exit 1
  fi
  echo "BEAM frozen-cache verification passed: saves=$saves restores=$restores"
fi
