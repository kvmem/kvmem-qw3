#!/usr/bin/env bash
# Re-run the existing 30-sample LongMemEval-S multi-turn diagnostic with the
# last-user query contract.  The one-shot reference already exists as
# /data/chaidi/kvmem_eval/results/warm_query_30_oneshot.jsonl.
set -euo pipefail

cd /home/chaidi/qw3

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

PORT=${PORT:-8087}
DATA=/data/chaidi/kvmem_eval/data/longmemeval_s.json
RESULTS=/data/chaidi/kvmem_eval/results
INDICES=2,8,22,24,57,79,83,108,117,134,149,151,154,159,219,269,280,295,310,359,372,374,376,402,412,450,465,483,497,498
TAG=${TAG:-warm_query_30_last_user}
OUT="$RESULTS/${TAG}.jsonl"
SERVELOG="$RESULTS/${TAG}_serve.log"
RUNLOG="$RESULTS/${TAG}_run.log"
NVME_DIR="/data/chaidi/kvmem_eval/nvme/${TAG}"

mkdir -p "$RESULTS" "$NVME_DIR"

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env \
  QW3_FATTN_NSPLIT=1 \
  QW3_PREFILL_FA2_NSPLIT=1 \
  QW3_KVMEM_TIMING=1 \
  QW3_KVMEM_TRACE=1 \
  QW3_KVMEM_PREFIX_CACHE_TRACE=1 \
  build/qw3 serve \
    --model models/Qwen3.6-27B-Q8_0.gguf \
    --ctx 262144 --kv-dtype fp16 \
    --kvmem --kvmem-block-tokens 32 --kvmem-budget 32768 \
    --kvmem-gen-budget 32768 \
    --kvmem-sink-blocks 8 --kvmem-recent-blocks 0 \
    --kvmem-method retrieval --kvmem-retrieval-method mean-k \
    --kvmem-update-mode step --kvmem-query-conditioned \
    --kvmem-prefix-cache \
    --kvmem-gpu-memory-ratio 0.5 \
    --kvmem-cpu-gb 64.0 --kvmem-nvme-gb 256.0 \
    --kvmem-nvme-dir "$NVME_DIR" \
    --thinking-budget 4096 --prefill-chunk 2048 \
    --enable-thinking --temp 0.0 \
    --native-mtp-speculate --mtp-chain 4 \
    --host 127.0.0.1 --port "$PORT" >"$SERVELOG" 2>&1 &
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
  sleep 5
done
if [[ "$healthy" -ne 1 ]]; then
  tail -80 "$SERVELOG" >&2 || true
  exit 1
fi

python3 scripts/kvmem_eval/reproduce_warm_query.py \
  --data "$DATA" \
  --indices "$INDICES" \
  --mode warm-query \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --max-tokens 8192 \
  --warm-max-tokens 64 \
  --out "$OUT" >"$RUNLOG" 2>&1

echo "result=$OUT"
echo "run_log=$RUNLOG"
echo "serve_log=$SERVELOG"
