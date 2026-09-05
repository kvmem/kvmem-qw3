#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
RELEASE="osworld-v2-2026.08.08"
TASK_MANIFEST_DIR="${OSWORLD_ROOT}/cache/osworld_v2_task_manifest"
TASK_MANIFEST="${TASK_MANIFEST_DIR}/manifests/task_hashes.json"
TASK_MANIFEST_SHA256="42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915"
VM_DIR="${OSWORLD_ROOT}/docker_vm_data"
VM_ZIP="${VM_DIR}/osworld-v2-ubuntu-x86.qcow2.zip"
VM_IMAGE="${VM_DIR}/osworld-v2-ubuntu-x86.qcow2"
VM_SHA256="eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa"
VM_URL="https://hf-mirror.com/datasets/xlangai/v2-image/resolve/main/osworld-v2-ubuntu-x86.before-20260716T053042Z-repo-update.qcow2.zip"
RUNTIME_REF="happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9"

# Downloads must bypass inherited HTTP(S) proxies. Hugging Face content is
# fetched through the mirror requested for this machine.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="${OSWORLD_ROOT}/cache/uv"

cd "${OSWORLD_ROOT}"
test "$(git describe --tags --exact-match)" = "v2026.08.08"

uv run hf download xlangai/osworld_v2_tasks manifests/task_hashes.json \
  --repo-type dataset \
  --revision v2026.08.08 \
  --local-dir "${TASK_MANIFEST_DIR}"

"${QW3_ROOT}/scripts/osworld_v2/download_hf_mirror_tasks.sh"

"${QW3_ROOT}/scripts/osworld_v2/verify_task_hashes.py" \
  --manifest "${TASK_MANIFEST}" \
  --task-dir "${OSWORLD_ROOT}/evaluation_examples/task_class" \
  --expected-manifest-sha256 "${TASK_MANIFEST_SHA256}" \
  --expected-task-count 108

uv run scripts/tools/download_osworld_v2_assets.py \
  --benchmark-release "${RELEASE}" \
  --target-dir cache/osworld_v2_assets \
  --clean

"${QW3_ROOT}/scripts/osworld_v2/repair_hf_mirror_asset_texts.sh"

env \
  WEBSITE_HOST_SUFFIX="${WEBSITE_HOST_SUFFIX:-site.hku.icu}" \
  GITLAB_URL="${GITLAB_URL:-http://gitlab.static-audit.invalid}" \
  GITLAB_PRIVATE_TOKEN="${GITLAB_PRIVATE_TOKEN:-static-audit-only}" \
  "${OSWORLD_ROOT}/.venv/bin/python" \
  "${QW3_ROOT}/scripts/osworld_v2/audit_tasks.py" \
  --osworld-root "${OSWORLD_ROOT}"

mkdir -p "${VM_DIR}"
if [[ ! -f "${VM_ZIP}" ]] || ! printf '%s  %s\n' "${VM_SHA256}" "${VM_ZIP}" | sha256sum --check --status; then
  ionice -c3 nice -n 19 curl -fL \
    --retry 8 \
    --retry-all-errors \
    --connect-timeout 30 \
    --continue-at - \
    --limit-rate 25M \
    --output "${VM_ZIP}" \
    "${VM_URL}"
fi
printf '%s  %s\n' "${VM_SHA256}" "${VM_ZIP}" | sha256sum --check

if [[ ! -f "${VM_IMAGE}" ]]; then
  unzip -q "${VM_ZIP}" -d "${VM_DIR}"
fi
test -s "${VM_IMAGE}"

# The release manifest names the runtime container without a tag. Record and
# retag the resolved immutable digest because the provider invokes `latest`.
ionice -c3 nice -n 19 docker pull "${RUNTIME_REF}"
docker tag "${RUNTIME_REF}" happysixd/osworld-docker:latest

echo "Pinned OSWorld V2 tasks, assets, and Docker VM image are ready."
