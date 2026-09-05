#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
TEST_META="${OSWORLD_TEST_META:-${OSWORLD_ROOT}/evaluation_examples/test_qw3_smoke.json}"
EXPERIMENT_ID="${QW3_OSWORLD_EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_ROOT="${QW3_OSWORLD_EXPERIMENT_ROOT:-${QW3_ROOT}/benchmark/osworld_v2/compaction_ab/${EXPERIMENT_ID}}"
MAX_STEPS="${QW3_OSWORLD_MAX_STEPS:-8}"
HISTORY_N="${QW3_OSWORLD_AB_HISTORY_N:-100}"
COMPACT_IMAGE_MAX="${QW3_OSWORLD_AB_COMPACT_IMAGE_MAX:-4}"
NO_COMPACT_IMAGE_MAX="${QW3_OSWORLD_AB_NO_COMPACT_IMAGE_MAX:-64}"
FOLD_SIZE="${QW3_OSWORLD_AB_FOLD_SIZE:-2}"

if (( MAX_STEPS > HISTORY_N )); then
  echo "MAX_STEPS must not exceed HISTORY_N for the screenshot-only A/B." >&2
  exit 2
fi
if (( NO_COMPACT_IMAGE_MAX < MAX_STEPS )); then
  echo "NO_COMPACT_IMAGE_MAX must be at least MAX_STEPS." >&2
  exit 2
fi
if (( FOLD_SIZE < 1 || FOLD_SIZE > COMPACT_IMAGE_MAX )); then
  echo "FOLD_SIZE must be between 1 and COMPACT_IMAGE_MAX." >&2
  exit 2
fi

expected_sequence() {
  local image_max="$1"
  local active=0
  local step
  local values=()
  for (( step = 1; step <= MAX_STEPS; step++ )); do
    (( active += 1 ))
    if (( active > image_max )); then
      (( active -= FOLD_SIZE ))
    fi
    values+=("${active}")
  done
  local IFS=,
  echo "${values[*]}"
}

expected_events() {
  local image_max="$1"
  local active=0
  local step
  local values=()
  for (( step = 1; step <= MAX_STEPS; step++ )); do
    (( active += 1 ))
    if (( active > image_max )); then
      values+=("${step}")
      (( active -= FOLD_SIZE ))
    fi
  done
  local IFS=,
  echo "${values[*]}"
}

# Keep every acquisition path proxy-free. The run should use only already
# pinned local assets plus the benchmark's hosted task website.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${EXPERIMENT_ROOT}"

QW3_OSWORLD_RESULT_ROOT="${EXPERIMENT_ROOT}" \
QW3_OSWORLD_RUN_LABEL="compact" \
QW3_OSWORLD_RUN_ID="trace" \
QW3_OSWORLD_EVAL_PROFILE="latency-ab" \
QW3_OSWORLD_MAX_STEPS="${MAX_STEPS}" \
QW3_OSWORLD_HISTORY_N="${HISTORY_N}" \
QW3_OSWORLD_IMAGE_MAX="${COMPACT_IMAGE_MAX}" \
QW3_OSWORLD_FOLD_SIZE="${FOLD_SIZE}" \
QW3_OSWORLD_TRACE_GUEST=1 \
OSWORLD_TEST_META="${TEST_META}" \
  "${SCRIPT_DIR}/run_qw3_eval.sh"

QW3_OSWORLD_RESULT_ROOT="${EXPERIMENT_ROOT}" \
QW3_OSWORLD_RUN_LABEL="no-compact" \
QW3_OSWORLD_RUN_ID="trace" \
QW3_OSWORLD_EVAL_PROFILE="latency-ab" \
QW3_OSWORLD_MAX_STEPS="${MAX_STEPS}" \
QW3_OSWORLD_HISTORY_N="${HISTORY_N}" \
QW3_OSWORLD_IMAGE_MAX="${NO_COMPACT_IMAGE_MAX}" \
QW3_OSWORLD_FOLD_SIZE="${FOLD_SIZE}" \
QW3_OSWORLD_TRACE_GUEST=1 \
OSWORLD_TEST_META="${TEST_META}" \
  "${SCRIPT_DIR}/run_qw3_eval.sh"

"${OSWORLD_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/verify_trace_bundle.py" \
  --run-dir "${EXPERIMENT_ROOT}/compact-trace" \
  --expected-steps "${MAX_STEPS}" \
  --expected-active-images "$(expected_sequence "${COMPACT_IMAGE_MAX}")" \
  --expected-compaction-steps "$(expected_events "${COMPACT_IMAGE_MAX}")"

"${OSWORLD_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/verify_trace_bundle.py" \
  --run-dir "${EXPERIMENT_ROOT}/no-compact-trace" \
  --expected-steps "${MAX_STEPS}" \
  --expected-active-images "$(expected_sequence "${NO_COMPACT_IMAGE_MAX}")" \
  --expected-compaction-steps "$(expected_events "${NO_COMPACT_IMAGE_MAX}")"

"${OSWORLD_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/analyze_compaction.py" \
  --compact-run "${EXPERIMENT_ROOT}/compact-trace" \
  --no-compact-run "${EXPERIMENT_ROOT}/no-compact-trace" \
  --output-dir "${EXPERIMENT_ROOT}/analysis"

echo "OSWorld compaction A/B experiment: ${EXPERIMENT_ROOT}"
