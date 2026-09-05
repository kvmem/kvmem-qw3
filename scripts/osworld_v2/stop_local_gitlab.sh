#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GITLAB_ROOT="${OSWORLD_GITLAB_ROOT:-${QW3_ROOT}/third_party/OSWorld-GitLab}"

docker compose \
  --project-name osworld-gitlab \
  --file "${GITLAB_ROOT}/docker-compose.yml" \
  --file "${SCRIPT_DIR}/gitlab-compose.private.yml" \
  down

echo "OSWorld GitLab containers stopped; persistent volumes and credentials retained."
