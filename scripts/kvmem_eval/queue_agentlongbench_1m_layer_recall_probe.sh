#!/usr/bin/env bash
# Wait for the currently running long benchmark to release the only GPU, then
# launch the per-layer retrieval probe without perturbing that benchmark.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WAIT_PID=${WAIT_PID:-3828280}
POLL_SEC=${POLL_SEC:-60}

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$POLL_SEC"
done

# Give the prior CUDA context a short grace period to release all allocations.
sleep 30
exec "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_layer_recall_probe.sh"
