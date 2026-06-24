#!/usr/bin/env bash
# MTP chain=5 compatibility study: CB and paged-KV impact at concurrency=1.
#
# Matrix:
#   - legacy_mtp (CB off) x {plain, paged-kv}
#   - continuous_mtp (CB on) x max_active in {1,2,4} x {plain, paged-kv}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
require_binary "$QW3" "qw3"
require_model

CHAIN="${MTP_STUDY_CHAIN:-5}"
CTX="${MTP_STUDY_CTX:-4096}"
MAX_TOKENS="${MTP_STUDY_MAX_TOKENS:-256}"
PROMPT_REPEAT="${MTP_STUDY_PROMPT_REPEAT:-32}"
KV_DTYPE="${KV_DTYPE:-fp16}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
KV_POOL_PAGES="${KV_POOL_PAGES:-49152}"
TIMEOUT="${EFFICIENCY_TIMEOUT:-3600}"

OUT="${RESULT_ROOT:-${REPO_ROOT}/benchmark/results/$(git_sha_short)_$(date +%Y%m%d_%H%M%S)}/mtp5_cb_pagedkv_study"
mkdir -p "$OUT"
echo "results -> ${OUT}"

COMMON=(
  --qw3 "$QW3"
  --model "$MODEL"
  --modes 'legacy_mtp continuous_mtp'
  --concurrency-levels '1'
  --max-active-levels '1 2 4'
  --ctx "$CTX"
  --max-tokens "$MAX_TOKENS"
  --prompt-repeat "$PROMPT_REPEAT"
  --chain "$CHAIN"
  --kv-dtype "$KV_DTYPE"
  --prefill-chunk "$PREFILL_CHUNK"
  --kv-pool-pages "$KV_POOL_PAGES"
  --timeout "$TIMEOUT"
)

step "plain KV (no paged-kv)"
python3 scripts/mtp_throughput_benchmark.py \
  "${COMMON[@]}" \
  --out-json "${OUT}/plain_kv.json"

step "paged KV"
python3 scripts/mtp_throughput_benchmark.py \
  "${COMMON[@]}" \
  --paged-kv \
  --body-batch \
  --out-json "${OUT}/paged_kv.json"

step "summarize"
python3 scripts/summarize_mtp5_cb_pagedkv_study.py \
  --input-dir "$OUT" \
  --output "${OUT}/summary.md"

echo "mtp5 cb/paged-kv study finished -> ${OUT}"
