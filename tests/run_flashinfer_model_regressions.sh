#!/usr/bin/env bash
# Optional full-model regression and timing sweep. Run without sanitizers and
# without other GPU jobs. Every process loads once, resets between requests,
# then checks four byte-identical greedy outputs per prompt/context.
# Usage: bash tests/run_flashinfer_model_regressions.sh BUILD MODEL LOG_DIRECTORY
set -euo pipefail
test_build=$(realpath "${1:?build directory required}")
test_model=$(realpath "${2:?local Q8_0 GGUF required}")
test_logs=${3:?log directory required}
mkdir -p "$test_logs"
test_failed=0
run_case() {
  local test_name=$1 test_dtype=$2 test_chunk=$3
  shift 3
  local test_status=0
  local test_log="$test_logs/$test_name.log"
  local test_command=(env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QW3_KV_DTYPE="$test_dtype"
    "$@" "$test_build/qw3-flashinfer-model-repeat" "$test_model" "$test_chunk")
  printf '%q ' "${test_command[@]}" > "$test_log.command"
  printf '\n' >> "$test_log.command"
  timeout 600 "${test_command[@]}" > "$test_log" 2>&1 || test_status=$?
  printf '%s exit=%s\n' "$test_name" "$test_status" | tee "$test_log.status"
  if ((test_status != 0)); then
    test_failed=1
    tail -10 "$test_log"
  fi
}
for test_dtype in fp16 fp8; do
  for test_chunk in 256 1024 2048 0; do
    run_case "$test_dtype-chunk$test_chunk" "$test_dtype" "$test_chunk" \
      QW3_PREFILL_ATTN=flashinfer QW3_DECODE_ATTN=flashinfer QW3_FLASHINFER_PLAN_CACHE=1
  done
done
run_case fp16-cache-off fp16 1024 QW3_FLASHINFER_PLAN_CACHE=0
run_case fp16-native fp16 1024 QW3_PREFILL_ATTN=mma-gqa-v2 QW3_DECODE_ATTN=native
run_case fp32-native fp32 1024 QW3_PREFILL_ATTN=mma-gqa-v2 QW3_DECODE_ATTN=native
exit "$test_failed"
