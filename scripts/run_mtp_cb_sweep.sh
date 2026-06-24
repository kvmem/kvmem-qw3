#!/usr/bin/env bash
# Focused MTP x continuous-batching sweep:
#   - MTP chain in {2,3,4,5}
#   - concurrency in {1,2,4,8}
#   - per chain: CB on (continuous_mtp) vs CB off (legacy_mtp)
#   - plus a no-MTP baseline (continuous vs legacy)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
require_binary "$QW3" "qw3"
require_model

CTX="${MTP_SWEEP_CTX:-4096}"
MAX_TOKENS="${MTP_SWEEP_MAX_TOKENS:-256}"
PROMPT_REPEAT="${MTP_SWEEP_PROMPT_REPEAT:-32}"
KV_DTYPE="${KV_DTYPE:-fp16}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
CHAINS="${MTP_SWEEP_CHAINS:-2 3 4 5}"
CONC="${MTP_SWEEP_CONC:-1 2 4 8}"
MAX_ACTIVE="${MTP_SWEEP_MAX_ACTIVE:-8}"
TIMEOUT="${EFFICIENCY_TIMEOUT:-3600}"

OUT="${RESULT_ROOT:-${REPO_ROOT}/benchmark/results/$(git_sha_short)_$(date +%Y%m%d_%H%M%S)}/mtp_cb_sweep"
mkdir -p "$OUT"
echo "results -> ${OUT}"

step "No-MTP baseline (continuous vs legacy)"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" --model "$MODEL" \
  --modes 'legacy continuous' \
  --concurrency-levels "$CONC" \
  --ctx "$CTX" --max-tokens "$MAX_TOKENS" --prompt-repeat "$PROMPT_REPEAT" \
  --chain 0 --kv-dtype "$KV_DTYPE" --prefill-chunk "$PREFILL_CHUNK" \
  --max-active "$MAX_ACTIVE" --timeout "$TIMEOUT" \
  --out-json "${OUT}/sweep_baseline_nomtp.json" \
  || echo "WARN: baseline had failures (continuing)"

for chain in $CHAINS; do
  step "MTP chain=${chain}: CB on (continuous_mtp) vs CB off (legacy_mtp)"
  python3 scripts/mtp_throughput_benchmark.py \
    --qw3 "$QW3" --model "$MODEL" \
    --modes 'legacy_mtp continuous_mtp' \
    --concurrency-levels "$CONC" \
    --ctx "$CTX" --max-tokens "$MAX_TOKENS" --prompt-repeat "$PROMPT_REPEAT" \
    --chain "$chain" --kv-dtype "$KV_DTYPE" --prefill-chunk "$PREFILL_CHUNK" \
    --max-active "$MAX_ACTIVE" --timeout "$TIMEOUT" \
    --out-json "${OUT}/sweep_chain${chain}.json" \
    || echo "WARN: chain=${chain} had failures (continuing)"
done

step "summarize"
python3 scripts/summarize_mtp_cb_sweep.py --input-dir "$OUT" --output "${OUT}/summary_mtp_cb_sweep.md"
echo "mtp/cb sweep finished -> ${OUT}"
