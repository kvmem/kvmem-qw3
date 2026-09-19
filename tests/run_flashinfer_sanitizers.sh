#!/usr/bin/env bash
# Usage: bash tests/run_flashinfer_sanitizers.sh BUILD_DIRECTORY LOG_DIRECTORY
# Run on an idle GPU; performance measurements must run separately.
set -euo pipefail
test_build=$(realpath "${1:?build directory required}")
test_logs=${2:?log directory required}
test_sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
mkdir -p "$test_logs"
test_failed=0
for test_case in decode prefill queued plan-cache; do
  test_args=()
  test_binary="$test_build/qw3-flashinfer-decode-gqa"
  case "$test_case" in
    prefill) test_args=(--prefill) ;;
    queued) test_args=(--queued) ;;
    plan-cache) test_binary="$test_build/qw3-flashinfer-plan-cache" ;;
  esac
  for test_tool in memcheck racecheck; do
    if [[ "$test_case" == queued && "$test_tool" == racecheck ]]; then
      test_args=(--queued-small)
    fi
    test_log="$test_logs/$test_case-$test_tool.log"
    test_status=0
    timeout 300 env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      QW3_PREFILL_ATTN=flashinfer QW3_DECODE_ATTN=flashinfer \
      QW3_FLASHINFER_PLAN_CACHE=1 \
      "$test_sanitizer" --tool "$test_tool" --error-exitcode 99 --print-limit 8 \
      "$test_binary" "${test_args[@]}" > "$test_log" 2>&1 || test_status=$?
    printf '%s %s exit=%s\n' "$test_case" "$test_tool" "$test_status" | tee "$test_log.status"
    if ((test_status != 0)); then
      test_failed=1
      tail -12 "$test_log"
    fi
  done
done
exit "$test_failed"
