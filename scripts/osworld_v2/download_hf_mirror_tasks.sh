#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
TARGET_DIR="${OSWORLD_ROOT}/evaluation_examples/task_class"
EXPECTED_ENDPOINT="https://hf-mirror.com"
EXPECTED_COMMIT="3736efa55d9d5dc78f57e873ef78886663e41200"
EXPECTED_MANIFEST_SHA256="42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-${EXPECTED_ENDPOINT}}"
[[ "${HF_ENDPOINT%/}" = "${EXPECTED_ENDPOINT}" ]] || {
  echo "HF_ENDPOINT must be ${EXPECTED_ENDPOINT}, got ${HF_ENDPOINT}." >&2
  exit 1
}

mkdir -p "${OSWORLD_ROOT}/cache" "${TARGET_DIR}"
TEMP_ROOT="$(mktemp -d "${OSWORLD_ROOT}/cache/osworld-tasks-git.XXXXXX")"
cleanup() {
  rm -rf -- "${TEMP_ROOT}"
}
trap cleanup EXIT

export HF_PYTHON="${OSWORLD_ROOT}/.venv/bin/python"
export GIT_ASKPASS="${SCRIPT_DIR}/hf_git_askpass.sh"
export GIT_TERMINAL_PROMPT=0
export GIT_LFS_SKIP_SMUDGE=1

git \
  -c http.proxy= \
  -c https.proxy= \
  -c advice.detachedHead=false \
  clone \
  --depth 1 \
  --branch v2026.08.08 \
  "${EXPECTED_ENDPOINT}/datasets/xlangai/osworld_v2_tasks" \
  "${TEMP_ROOT}/repo"

actual_commit="$(git -C "${TEMP_ROOT}/repo" rev-parse HEAD)"
[[ "${actual_commit}" = "${EXPECTED_COMMIT}" ]] || {
  echo "Task repository commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}." >&2
  exit 1
}

"${SCRIPT_DIR}/verify_task_hashes.py" \
  --manifest "${TEMP_ROOT}/repo/manifests/task_hashes.json" \
  --task-dir "${TEMP_ROOT}/repo" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --expected-task-count 108

install -m 0644 "${TEMP_ROOT}/repo"/task_*.py "${TARGET_DIR}/"

"${SCRIPT_DIR}/verify_task_hashes.py" \
  --manifest "${TEMP_ROOT}/repo/manifests/task_hashes.json" \
  --task-dir "${TARGET_DIR}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --expected-task-count 108

echo "hf-mirror Git task download complete: commit=${EXPECTED_COMMIT}, tasks=108"
