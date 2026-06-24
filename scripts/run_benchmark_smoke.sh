#!/usr/bin/env bash
# Smoke benchmark: correctness gate + short throughput checks (~30 min).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_benchmark_common.sh"

load_manifest
init_result_dir smoke
require_binary "$QW3" "qw3"
require_model

TIMEOUT="${SMOKE_TIMEOUT:-900}"
STATUS=0

run_step() {
  if ! "$@"; then
    echo "FAILED: $*" >&2
    STATUS=1
  fi
}

step "ctest"
if [[ -d build ]]; then
  run_step ctest --test-dir build --output-on-failure
else
  echo "skip ctest: build/ not found" >&2
fi

step "paged_kv_regression"
run_step python3 scripts/paged_kv_regression.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --prompts 'boundary_24_words code' \
  --page-sizes '16 64' \
  --kv-dtypes fp16 \
  --max-tokens 16 \
  --ctx 4096 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/paged_kv_regression.json"

step "continuous_batching_regression"
run_step python3 scripts/continuous_batching_regression.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --prompts 'capital math cuda chinese' \
  --max-tokens 8 \
  --ctx 1024 \
  --prefill-chunk 64 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/continuous_batching_regression.json"

step "mtp_continuous_regression"
run_step python3 scripts/mtp_continuous_regression.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --chain 4 \
  --kv-dtype fp16 \
  --max-tokens 32 \
  --ctx 4096 \
  --prefill-chunk 512 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_continuous_regression.json"

step "continuous_prefill_interleave_regression"
run_step python3 scripts/continuous_prefill_interleave_regression.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --ctx 4096 \
  --prefill-chunk 256 \
  --long-words 3000 \
  --max-tokens 4 \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/continuous_prefill_interleave.json"

step "kvmem_e2e_regression"
if [[ "${SKIP_KVMEM_E2E:-0}" == "1" ]]; then
  echo "skip kvmem_e2e_regression: SKIP_KVMEM_E2E=1"
else
  run_step python3 scripts/kvmem_e2e_regression.py \
    --qw3 "$QW3" \
    --model "$MODEL" \
    --ctx 2048 \
    --max-tokens 8 \
    --timeout "$TIMEOUT" \
    --out-json "${OUT}/kvmem_e2e_regression.json"
fi

step "continuous_batching_benchmark (smoke)"
run_step python3 scripts/continuous_batching_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --prompts 'capital math' \
  --max-tokens 32 \
  --ctx 2048 \
  --prefill-chunk 512 \
  --max-active 2 \
  --variants 'plain continuous body recurrent' \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/cb_smoke.json"

step "mtp_throughput_benchmark (smoke)"
run_step python3 scripts/mtp_throughput_benchmark.py \
  --qw3 "$QW3" \
  --model "$MODEL" \
  --ctx 4096 \
  --max-tokens 32 \
  --prompt-repeat 8 \
  --chain 4 \
  --kv-dtype fp16 \
  --concurrency-levels '1 2' \
  --modes 'continuous continuous_mtp' \
  --timeout "$TIMEOUT" \
  --out-json "${OUT}/mtp_smoke.json"

{
  echo "# Smoke Benchmark Summary"
  echo ""
  echo "- finished_at: $(date -Iseconds)"
  echo "- exit_status: ${STATUS}"
  echo "- output_dir: ${OUT}"
} >"${OUT}/summary_smoke.md"

echo
if [[ "$STATUS" -eq 0 ]]; then
  echo "smoke benchmark PASSED -> ${OUT}"
else
  echo "smoke benchmark FAILED -> ${OUT}" >&2
fi
exit "$STATUS"
