#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
SETUP_EXPORTS="${OSWORLD_SETUP_EXPORTS:-${OSWORLD_ROOT}/cache/osworld-setup-exports.sh}"
GITLAB_EXPORTS="${OSWORLD_GITLAB_EXPORTS:-${QW3_ROOT}/third_party/OSWorld-GitLab/.env.osworld.local}"
TEST_META="${OSWORLD_TEST_META:-${OSWORLD_ROOT}/evaluation_examples/test_v2.json}"
PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-${OSWORLD_ROOT}/evaluation_examples/settings/proxy/dataimpulse.json}"

if [[ -f "${SETUP_EXPORTS}" ]]; then
  # shellcheck disable=SC1090
  source "${SETUP_EXPORTS}"
fi
if [[ -f "${GITLAB_EXPORTS}" ]]; then
  # shellcheck disable=SC1090
  source "${GITLAB_EXPORTS}"
fi

if [[ -z "${GITLAB_URL:-}" || -z "${GITLAB_PRIVATE_TOKEN:-}" ]]; then
  echo "Full OSWorld V2 requires a reachable private GitLab and token." >&2
  exit 78
fi

[[ -f "${TEST_META}" ]] || {
  echo "Missing OSWorld task list: ${TEST_META}" >&2
  exit 78
}

requires_proxy=0
for task_id in 036 037 050 055 056 062 075 098; do
  if grep -Eq "\"${task_id}\"" "${TEST_META}"; then
    requires_proxy=1
    break
  fi
done
if [[ "${requires_proxy}" -eq 1 ]]; then
  if [[ ! -s "${PROXY_CONFIG_FILE}" ]] || \
      grep -Eq 'your_username|your_password|<[^>]+>' "${PROXY_CONFIG_FILE}"; then
    echo "Selected tasks include proxy=true entries, but ${PROXY_CONFIG_FILE} still contains placeholders." >&2
    exit 78
  fi
fi
export PROXY_CONFIG_FILE

curl --noproxy '*' --fail --silent --show-error \
  --header "PRIVATE-TOKEN: ${GITLAB_PRIVATE_TOKEN}" \
  "${GITLAB_URL}/api/v4/version" >/dev/null || {
    echo "Configured GitLab is not reachable or its token is invalid." >&2
    exit 78
  }

export OSWORLD_ROOT
export OSWORLD_TEST_META="${TEST_META}"
export QW3_OSWORLD_RUN_LABEL="${QW3_OSWORLD_RUN_LABEL:-full}"
export QW3_OSWORLD_MAX_STEPS="${QW3_OSWORLD_MAX_STEPS:-200}"

exec "${SCRIPT_DIR}/run_qw3_eval.sh"
