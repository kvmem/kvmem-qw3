#!/usr/bin/env bash
# MTP + continuous batching concurrency sweep:
#   max_active in {1,2,4,8}, concurrency sweeps 1..max_active each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
require_binary "$QW3" "qw3"
require_model

CHAIN="${MTP_SWEEP_CHAIN:-5}"
CTX="${MTP_SWEEP_CTX:-4096}"
MAX_TOKENS="${MTP_SWEEP_MAX_TOKENS:-256}"
PROMPT_REPEAT="${MTP_SWEEP_PROMPT_REPEAT:-32}"
KV_DTYPE="${KV_DTYPE:-fp16}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
KV_POOL_PAGES="${KV_POOL_PAGES:-49152}"
MAX_ACTIVE_LEVELS="${MTP_SWEEP_MAX_ACTIVE:-1 2 4 8}"
TIMEOUT="${EFFICIENCY_TIMEOUT:-3600}"

OUT="${RESULT_ROOT:-${REPO_ROOT}/benchmark/results/$(git_sha_short)_$(date +%Y%m%d_%H%M%S)}/mtp_cb_concurrency_sweep"
mkdir -p "$OUT"
echo "results -> ${OUT}"

step "MTP chain=${CHAIN}: concurrency sweep per max_active"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --modes continuous_mtp \
  --max-active-levels "$MAX_ACTIVE_LEVELS" \
  --concurrency-up-to-max-active \
  --reuse-server \
  --ctx "$CTX" \
  --max-tokens "$MAX_TOKENS" \
  --prompt-repeat "$PROMPT_REPEAT" \
  --chain "$CHAIN" \
  --kv-dtype "$KV_DTYPE" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --kv-pool-pages "$KV_POOL_PAGES" \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/sweep.json"

step "summarize"
python3 scripts/summarize_mtp_cb_concurrency_sweep.py \
  --input "${OUT}/sweep.json" \
  --output "${OUT}/summary.md"

echo "mtp cb concurrency sweep finished -> ${OUT}"
