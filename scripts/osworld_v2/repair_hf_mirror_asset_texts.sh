#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
ASSET_DIR="${OSWORLD_ROOT}/cache/osworld_v2_assets"
ASSET_MANIFEST="${OSWORLD_ROOT}/cache/osworld_v2_asset_manifest.json"
EXPECTED_ENDPOINT="https://hf-mirror.com"
EXPECTED_COMMIT="acad110ef3136405f95434b54862bf9066176c2a"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-${EXPECTED_ENDPOINT}}"
[[ "${HF_ENDPOINT%/}" = "${EXPECTED_ENDPOINT}" ]] || {
  echo "HF_ENDPOINT must be ${EXPECTED_ENDPOINT}, got ${HF_ENDPOINT}." >&2
  exit 1
}

[[ -d "${ASSET_DIR}" ]] || {
  echo "Missing asset directory: ${ASSET_DIR}" >&2
  exit 1
}
mkdir -p "${OSWORLD_ROOT}/cache"
TEMP_ROOT="$(mktemp -d "${OSWORLD_ROOT}/cache/osworld-assets-git.XXXXXX")"
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
  "${EXPECTED_ENDPOINT}/datasets/xlangai/osworld_v2_assets_gated" \
  "${TEMP_ROOT}/repo"

actual_commit="$(git -C "${TEMP_ROOT}/repo" rev-parse HEAD)"
[[ "${actual_commit}" = "${EXPECTED_COMMIT}" ]] || {
  echo "Asset repository commit mismatch: ${actual_commit} != ${EXPECTED_COMMIT}." >&2
  exit 1
}

installed=0
while IFS= read -r -d '' name; do
  source_path="${TEMP_ROOT}/repo/${name}"
  if grep -q '^version https://git-lfs.github.com/spec/v1$' "${source_path}"; then
    continue
  fi
  install -D -m 0644 "${source_path}" "${ASSET_DIR}/${name}"
  installed=$((installed + 1))
done < <(git -C "${TEMP_ROOT}/repo" ls-files -z)
echo "canonical_non_lfs_assets_installed=${installed}"

"${OSWORLD_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/verify_assets.py" \
  --asset-dir "${ASSET_DIR}" \
  --manifest "${ASSET_MANIFEST}" \
  --git-checkout "${TEMP_ROOT}/repo" \
  --expected-commit "${EXPECTED_COMMIT}"
