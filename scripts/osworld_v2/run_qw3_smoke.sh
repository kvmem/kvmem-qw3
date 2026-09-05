#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"

export OSWORLD_ROOT
export OSWORLD_TEST_META="${OSWORLD_TEST_META:-${OSWORLD_ROOT}/evaluation_examples/test_qw3_smoke.json}"
export QW3_OSWORLD_RUN_LABEL="${QW3_OSWORLD_RUN_LABEL:-smoke}"
export QW3_OSWORLD_MAX_STEPS="${QW3_OSWORLD_MAX_STEPS:-15}"
export QW3_OSWORLD_EVAL_PROFILE="${QW3_OSWORLD_EVAL_PROFILE:-diagnostic-smoke-15}"

exec "${SCRIPT_DIR}/run_qw3_eval.sh"
