#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QW3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GITLAB_ROOT="${OSWORLD_GITLAB_ROOT:-${QW3_ROOT}/third_party/OSWorld-GitLab}"
BASE_COMPOSE="${GITLAB_ROOT}/docker-compose.yml"
PRIVATE_COMPOSE="${SCRIPT_DIR}/gitlab-compose.private.yml"
SECRET_EXPORTS="${OSWORLD_GITLAB_EXPORTS:-${GITLAB_ROOT}/.env.osworld.local}"
NETWORK_NAME="${OSWORLD_GITLAB_NETWORK:-osworld-services}"
NETWORK_SUBNET="${OSWORLD_GITLAB_SUBNET:-172.30.0.0/24}"
NETWORK_GATEWAY="${OSWORLD_GITLAB_GATEWAY:-172.30.0.1}"
GITLAB_HOSTNAME="${OSWORLD_GITLAB_HOSTNAME:-gitlab.${NETWORK_GATEWAY}.sslip.io}"
EXPECTED_GITLAB_COMMIT="8655d651722f4254e59e813de9f68a6732ea525c"

# Any image pull triggered by Docker must not inherit shell proxy variables.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

die() {
  echo "$*" >&2
  exit 1
}

if pgrep -f 'run_dense256_pass4.py' >/dev/null || \
    pgrep -f 'ordinary_only_batch.py.*dense256_claude_auto_compact' >/dev/null || \
    [[ -n "$(docker ps --filter label=com.qw3.deepswe.run-owner --quiet 2>/dev/null)" ]]; then
  die "DeepSWE main/worker/container is still active; refusing to start GitLab."
fi

[[ "$(git -C "${GITLAB_ROOT}" rev-parse HEAD)" = "${EXPECTED_GITLAB_COMMIT}" ]] || \
  die "OSWorld-GitLab checkout is not pinned to ${EXPECTED_GITLAB_COMMIT}."
docker info >/dev/null

if docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1; then
  actual_subnet="$(docker network inspect "${NETWORK_NAME}" --format '{{(index .IPAM.Config 0).Subnet}}')"
  actual_gateway="$(docker network inspect "${NETWORK_NAME}" --format '{{(index .IPAM.Config 0).Gateway}}')"
  [[ "${actual_subnet}" = "${NETWORK_SUBNET}" ]] || \
    die "${NETWORK_NAME} subnet is ${actual_subnet}, expected ${NETWORK_SUBNET}."
  [[ "${actual_gateway}" = "${NETWORK_GATEWAY}" ]] || \
    die "${NETWORK_NAME} gateway is ${actual_gateway}, expected ${NETWORK_GATEWAY}."
else
  docker network create \
    --driver bridge \
    --subnet "${NETWORK_SUBNET}" \
    --gateway "${NETWORK_GATEWAY}" \
    "${NETWORK_NAME}" >/dev/null
fi

if [[ ! -s "${SECRET_EXPORTS}" ]]; then
  mkdir -p "$(dirname -- "${SECRET_EXPORTS}")"
  umask 077
  generated_token="glpat-$(openssl rand -hex 32)"
  {
    printf '# Generated OSWorld local GitLab credentials; keep this file private.\n'
    printf 'export GITLAB_URL=%q\n' "http://${GITLAB_HOSTNAME}"
    printf 'export GITLAB_PRIVATE_TOKEN=%q\n' "${generated_token}"
  } >"${SECRET_EXPORTS}"
  unset generated_token
fi
chmod 600 "${SECRET_EXPORTS}"
# shellcheck disable=SC1090
source "${SECRET_EXPORTS}"

[[ "${GITLAB_URL:-}" = "http://${GITLAB_HOSTNAME}" ]] || \
  die "${SECRET_EXPORTS} must set GITLAB_URL=http://${GITLAB_HOSTNAME}."
[[ -n "${GITLAB_PRIVATE_TOKEN:-}" ]] || \
  die "${SECRET_EXPORTS} does not set GITLAB_PRIVATE_TOKEN."
export GITLAB_URL GITLAB_PRIVATE_TOKEN OSWORLD_GITLAB_GATEWAY="${NETWORK_GATEWAY}"

compose=(
  docker compose
  --project-name osworld-gitlab
  --file "${BASE_COMPOSE}"
  --file "${PRIVATE_COMPOSE}"
)

"${compose[@]}" config --quiet
"${compose[@]}" up --detach

ready=0
for _ in $(seq 1 120); do
  # GitLab 18.7 can report a healthy container while the legacy
  # /-/readiness route is not exposed by this omnibus configuration.  The
  # authenticated version endpoint exercises workhorse, Rails, the database,
  # and the token installed by gitlab-init-token, and is also the endpoint the
  # evaluation preflight uses immediately below.
  if curl --noproxy '*' --fail --silent --max-time 5 \
      --header "PRIVATE-TOKEN: ${GITLAB_PRIVATE_TOKEN}" \
      "${GITLAB_URL}/api/v4/version" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
[[ "${ready}" -eq 1 ]] || die "GitLab did not become ready within 20 minutes."

token_ready=0
for _ in $(seq 1 120); do
  init_state="$(docker inspect gitlab-init-token --format '{{.State.Status}} {{.State.ExitCode}}' 2>/dev/null || true)"
  if [[ "${init_state}" = "exited 0" ]]; then
    token_ready=1
    break
  fi
  if [[ "${init_state}" = exited\ * && "${init_state}" != "exited 0" ]]; then
    docker logs --tail 100 gitlab-init-token >&2
    die "GitLab token initialization failed: ${init_state}."
  fi
  sleep 5
done
[[ "${token_ready}" -eq 1 ]] || die "GitLab token initialization timed out."

if ! "${compose[@]}" exec -T gitlab-runner \
    test -s /etc/gitlab-runner/config.toml; then
  runner_response="$(curl --noproxy '*' --fail-with-body --silent --show-error \
    --request POST \
    --header "PRIVATE-TOKEN: ${GITLAB_PRIVATE_TOKEN}" \
    --data-urlencode 'runner_type=instance_type' \
    --data-urlencode 'description=osworld-docker-runner' \
    --data-urlencode 'tag_list=osworld,docker' \
    --data-urlencode 'run_untagged=true' \
    --data-urlencode 'locked=false' \
    --data-urlencode 'paused=false' \
    "${GITLAB_URL}/api/v4/user/runners")"
  runner_token="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])' <<<"${runner_response}")"
  unset runner_response
  "${compose[@]}" exec -T \
    -e RUNNER_TOKEN="${runner_token}" \
    gitlab-runner \
    gitlab-runner register \
      --non-interactive \
      --url "${GITLAB_URL}" \
      --token "${runner_token}" \
      --executor docker \
      --docker-image alpine:3.20 \
      --description osworld-docker-runner
  unset runner_token
fi

"${compose[@]}" exec -T gitlab-runner gitlab-runner verify
curl --noproxy '*' --fail --silent --show-error \
  --header "PRIVATE-TOKEN: ${GITLAB_PRIVATE_TOKEN}" \
  "${GITLAB_URL}/api/v4/version" >/dev/null

echo "OSWorld local GitLab is ready at ${GITLAB_URL}."
echo "Credentials are stored in ${SECRET_EXPORTS} (mode 0600)."
