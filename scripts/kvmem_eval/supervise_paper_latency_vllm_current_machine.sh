#!/usr/bin/env bash
# Deprecated compatibility entry point. Formal latency is native-qw3 only.
set -euo pipefail
ROOT=${ROOT:-/home/chaidi/qw3}
echo "warning: vLLM latency entry point is deprecated; running native qw3" >&2
exec "$ROOT/scripts/kvmem_eval/supervise_paper_latency_qw3_current_machine.sh" "$@"
