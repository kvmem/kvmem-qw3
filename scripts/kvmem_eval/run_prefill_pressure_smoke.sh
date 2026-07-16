#!/usr/bin/env bash
# Targeted multi-turn smoke test for the deterministic prefill-pressure policy.
set -euo pipefail

cd /home/chaidi/qw3

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

QW3_BIN=${QW3_BIN:-build/qw3}
PORT=${PORT:-18087}
MODEL=${MODEL:-models/Qwen3.6-27B-Q8_0.gguf}
OUT_DIR=${OUT_DIR:-/tmp/qw3_prefill_pressure_smoke}
SERVE_LOG="$OUT_DIR/serve.log"
RUN_LOG="$OUT_DIR/run.log"
mkdir -p "$OUT_DIR"

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_PREFIX_CACHE_TRACE=1 \
  "$QW3_BIN" serve \
    --model "$MODEL" \
    --ctx 8192 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 --kvmem-budget 512 \
    --kvmem-gen-budget 256 \
    --kvmem-sink-blocks 2 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-prefix-cache \
    --kvmem-gpu-memory-ratio 0.5 --kvmem-cpu-gb 1 \
    --prefill-chunk 256 --temp 0 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >"$SERVE_LOG" 2>&1 &
server_pid=$!

healthy=0
for _ in $(seq 1 180); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
    healthy=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 2
done
if [[ "$healthy" -ne 1 ]]; then
  tail -100 "$SERVE_LOG" >&2 || true
  exit 1
fi

python3 scripts/kvmem_eval/verify_prefill_pressure_multiturn.py \
  --base-url "http://127.0.0.1:${PORT}/v1" >"$RUN_LOG" 2>&1

hit_count=$(grep -c 'kvmem prefix-cache HIT' "$SERVE_LOG" || true)
pressure_count=$(grep -c '\[bs-prefill-pressure\]' "$SERVE_LOG" || true)
if (( hit_count < 2 )); then
  echo "expected at least two prefix-cache hits, got $hit_count" >&2
  exit 1
fi
if (( pressure_count < 2 )); then
  echo "expected at least two prefill-pressure selections, got $pressure_count" >&2
  exit 1
fi

echo "prefix_cache_hits=$hit_count pressure_selections=$pressure_count"
grep '\[bs-prefill-pressure\]' "$SERVE_LOG" | tail -20
cat "$RUN_LOG"
