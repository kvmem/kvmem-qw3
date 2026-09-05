#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"

export OSWORLD_ROOT
export OSWORLD_TEST_META="${OSWORLD_TEST_META:-${OSWORLD_ROOT}/evaluation_examples/test_qw3_no_proxy.json}"
export QW3_OSWORLD_RUN_LABEL="${QW3_OSWORLD_RUN_LABEL:-no-proxy}"
export QW3_OSWORLD_MAX_STEPS="${QW3_OSWORLD_MAX_STEPS:-200}"

# Reuse the full-suite GitLab reachability preflight; this 100-task subset
# contains both GitLab tasks but excludes every proxy=true task.
exec "${SCRIPT_DIR}/run_qw3_full.sh"
