#!/usr/bin/env bash
# Resume efficiency benchmark from Block C (Block A/B already completed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"
load_manifest
OUT="${1:?usage: $0 <efficiency_output_dir>}"
require_binary "$QW3" "qw3"
require_model
: "${LLAMA_COMPLETION:?}"
TIMEOUT="${EFFICIENCY_TIMEOUT:-3600}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
CTX="${CTX:-262144}"
KV_DTYPE="${KV_DTYPE:-fp16}"

step "Block C: MTP continuous batching (4K)"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" --model "$MODEL" \
  --modes 'continuous continuous_mtp' \
  --concurrency-levels '1 2 4' \
  --ctx 4096 --max-tokens 128 --prompt-repeat 32 \
  --chain 4 --kv-dtype "$KV_DTYPE" --max-active 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_cb_4k.json" || true

for spec in \
  "32768:longctx_32k_nomtp:" \
  "32768:longctx_32k_mtp4:--mtp-chain 4" \
  "65536:longctx_65536_mtp4:--mtp-chain 4" \
  "131072:longctx_131072_mtp4:--mtp-chain 4"
do
  IFS=: read -r target variant extra <<<"$spec"
  step "Block C: long-context spot ${variant}"
  # shellcheck disable=SC2086
  python3 scripts/long_prompt_sweep.py \
    --qw3 "$QW3" --llama "$LLAMA_COMPLETION" --model "$MODEL" \
    --prompt-tokens "$target" --trials 2 -n 1024 -c "$CTX" \
    --prefill-chunk "$PREFILL_CHUNK" --paged-kv --variant "$variant" \
    --timeout "$TIMEOUT" $extra \
    --json "${OUT}/mtp_longctx_${variant}.json" || true
done

step "Block D: FP8 KV spot"
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" --model "$MODEL" \
  --modes 'continuous continuous_mtp' \
  --concurrency-levels '4' \
  --ctx 4096 --max-tokens 128 --prompt-repeat 32 \
  --chain 4 --kv-dtype fp8 --max-active 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_cb_4k_fp8.json" || true

step "summarize"
python3 scripts/summarize_benchmark_results.py \
  --kind efficiency --input-dir "$OUT" \
  --output "${OUT}/summary_efficiency.md"

echo "efficiency resume finished -> ${OUT}"
