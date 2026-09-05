#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSWORLD_ROOT="${OSWORLD_ROOT:-${QW3_ROOT}/third_party/OSWorld-V2}"
TASK_MANIFEST="${OSWORLD_ROOT}/cache/osworld_v2_task_manifest/manifests/task_hashes.json"
TASK_MANIFEST_SHA256="42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915"
ASSET_MANIFEST="${OSWORLD_ROOT}/cache/osworld_v2_asset_manifest.json"
ASSET_COMMIT="acad110ef3136405f95434b54862bf9066176c2a"
VM_ZIP="${OSWORLD_ROOT}/docker_vm_data/osworld-v2-ubuntu-x86.qcow2.zip"
VM_IMAGE="${OSWORLD_ROOT}/docker_vm_data/osworld-v2-ubuntu-x86.qcow2"
VM_SHA256="eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa"
RUNTIME_REF="happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9"
GITLAB_REF="gitlab/gitlab-ce@sha256:9c17686c3cd1b72f52282543d2dcf7c96add29b4698ec367bfb1455be8f13ce0"
GITLAB_RUNNER_REF="gitlab/gitlab-runner@sha256:9d3b2b8be462e18176b82eb40a5f54a5ee6eaa2b4f46dcc362c4801aef90030f"
DOCKER_CLI_REF="docker@sha256:3f4743208d2338c934d7b8bcfbe1bb54c0b2355c510ad5e0f31c0c4a54bd704e"
export UV_CACHE_DIR="${OSWORLD_ROOT}/cache/uv"

cd "${OSWORLD_ROOT}"
test "$(git describe --tags --exact-match)" = "v2026.08.08"
uv run python -c "import desktop_env; print('desktop_env ok')"
uv run python -c "import desktop_env.desktop_env; print('DesktopEnv import ok')"
uv run python -c "from desktop_env.providers.docker.provider import DockerProvider; print('docker provider import ok')"

task_count="$(find evaluation_examples/task_class -maxdepth 1 -name 'task_*.py' | wc -l)"
test "${task_count}" -eq 108
"${QW3_ROOT}/scripts/osworld_v2/verify_task_hashes.py" \
  --manifest "${TASK_MANIFEST}" \
  --task-dir "${OSWORLD_ROOT}/evaluation_examples/task_class" \
  --expected-manifest-sha256 "${TASK_MANIFEST_SHA256}" \
  --expected-task-count 108
test -d cache/osworld_v2_assets
"${OSWORLD_ROOT}/.venv/bin/python" "${QW3_ROOT}/scripts/osworld_v2/verify_assets.py" \
  --asset-dir "${OSWORLD_ROOT}/cache/osworld_v2_assets" \
  --manifest "${ASSET_MANIFEST}" \
  --expected-commit "${ASSET_COMMIT}"
test -s cache/qw3_task_audit.json
printf '%s  %s\n' "${VM_SHA256}" "${VM_ZIP}" | sha256sum --check
test -s "${VM_IMAGE}"

docker info >/dev/null
docker compose version
docker image inspect "${RUNTIME_REF}" >/dev/null
docker image inspect "${GITLAB_REF}" >/dev/null
docker image inspect "${GITLAB_RUNNER_REF}" >/dev/null
docker image inspect "${DOCKER_CLI_REF}" >/dev/null
test "$(docker image inspect happysixd/osworld-docker:latest --format '{{.Id}}')" = \
  "$(docker image inspect "${RUNTIME_REF}" --format '{{.Id}}')"
test -e /dev/kvm
test -x "${QW3_ROOT}/build/qw3"
test -s "${QW3_ROOT}/models/Qwen3.8-27B-Q8_0.gguf"
test -f "${QW3_ROOT}/models/Qwen3.8-27B-BF16/model.safetensors.index.json"

echo "OSWorld V2 local setup verified: tasks=${task_count}, provider=docker, release=v2026.08.08"
