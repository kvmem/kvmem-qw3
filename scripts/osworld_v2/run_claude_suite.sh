#!/usr/bin/env bash
set -euo pipefail
qw3_suite_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
set -a
source "$qw3_suite_root/third_party/OSWorld-V2/cache/osworld-setup-exports.sh"
source "$qw3_suite_root/third_party/OSWorld-GitLab/.env.osworld.local"
set +a
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
exec "$qw3_suite_root/third_party/OSWorld-V2/.venv/bin/python" "$qw3_suite_root/scripts/osworld_v2/run_claude_suite.py" "$@"
