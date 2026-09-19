#!/usr/bin/env bash
# Run on an idle CUDA device. Each process starts with clean dispatch settings.
# Usage: bash tests/run_native_attention_sanitizers.sh BUILD/qw3-native-attention-oracle LOG_DIR
set -euo pipefail
test_binary=$(realpath "${1:?oracle binary required}")
test_logs=${2:?log directory required}
test_sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
test_binary_args=()
mkdir -p "$test_logs"
test_failed=0
run_case() {
  local test_name=$1 test_kernel=$2
  shift 2
  for test_tool in memcheck racecheck; do
    local test_log="$test_logs/$test_name-$test_tool.log"
    local test_status=0
    timeout 180 env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      QW3_PREFILL_ATTN="$test_kernel" QW3_PREFILL_ATTN_MIN_BATCH=1 QW3_DECODE_ATTN=native "$@" \
      "$test_sanitizer" --tool "$test_tool" --error-exitcode 99 --print-limit 12 \
      "$test_binary" "${test_binary_args[@]}" > "$test_log" 2>&1 || test_status=$?
    printf '%s %s exit=%s\n' "$test_name" "$test_tool" "$test_status" | tee "$test_log.status"
    if ((test_status != 0)); then
      test_failed=1
      tail -12 "$test_log"
    fi
  done
}
run_case v2_default mma-gqa-v2
run_case v2_nsplit1 mma-gqa-v2 QW3_PREFILL_FA2_NSPLIT=1
run_case v2_unsupported_nsplit4 mma-gqa-v2 QW3_PREFILL_FA2_NSPLIT=4
run_case v2_br8_bc64_unpadded mma-gqa-v2 QW3_PREFILL_FA2_BR=8 QW3_PREFILL_FA2_BC=64 QW3_PREFILL_FA2_KPAD=0 QW3_PREFILL_FA2_SPAD=0
run_case v2_br32 mma-gqa-v2 QW3_PREFILL_FA2_BR=32
run_case v2_sync_k mma-gqa-v2 QW3_PREFILL_FA2_KCPASYNC=0
run_case v2_sync_v mma-gqa-v2 QW3_PREFILL_FA2_VCPASYNC=0
run_case v2_sync_kv mma-gqa-v2 QW3_PREFILL_FA2_KCPASYNC=0 QW3_PREFILL_FA2_VCPASYNC=0
run_case v2_fp16_acc mma-gqa-v2 QW3_PREFILL_FA2_FP16_O=1
for test_kernel in mma mma-pipe mma-gqa mma-gqa-v3 mma-gqa-v4 mma-gqa-v5 vec; do
  run_case "$test_kernel" "$test_kernel"
done
run_case v1_nsplit4 mma-gqa QW3_PREFILL_FA2_NSPLIT=4
run_case v4_nsplit2 mma-gqa-v4 QW3_PREFILL_FA2_NSPLIT=2
run_case v5_nsplit2 mma-gqa-v5 QW3_PREFILL_FA2_NSPLIT=2
test_binary_args=(--fp32)
run_case fp32 mma-gqa-v2
exit "$test_failed"
