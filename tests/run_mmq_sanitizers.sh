#!/usr/bin/env bash
# Usage: bash tests/run_mmq_sanitizers.sh BUILD/qw3-mmq-parity LOG_DIR
set -euo pipefail
test_binary=$(realpath "${1:?MMQ parity binary required}")
test_logs=${2:?log directory required}
test_sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
mkdir -p "$test_logs"
test_failed=0
for test_version in 2 3 4 5 6 7 8 auto; do
  for test_tool in memcheck racecheck; do
    test_log="$test_logs/v$test_version-$test_tool.log"
    test_status=0
    timeout 180 env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QW3_MMQ_VERSION="$test_version" \
      "$test_sanitizer" --tool "$test_tool" --error-exitcode 99 --print-limit 12 \
      "$test_binary" --small > "$test_log" 2>&1 || test_status=$?
    printf 'MMQ %s %s exit=%s\n' "$test_version" "$test_tool" "$test_status" | tee "$test_log.status"
    if ((test_status != 0)); then
      test_failed=1
      tail -12 "$test_log"
    fi
  done
done
exit "$test_failed"
