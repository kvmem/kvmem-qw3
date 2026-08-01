#!/usr/bin/env bash
set -euo pipefail

# One-sample integration/accuracy smoke for the opt-in logical round retrieval
# path. It intentionally uses a separate port and output root so it can coexist
# with the current fixed-block experiment.

ROOT=${ROOT:-/home/chaidi/qw3}
PORT=${PORT:-8089}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_round_only_smoke_${STAMP}}
SERVER_LOG=${SERVER_LOG:-/data/chaidi/kvmem_eval/logs/agentlongbench_1m_round_only_smoke_${STAMP}_server.log}
RUNNER_LOG=${RUNNER_LOG:-/data/chaidi/kvmem_eval/logs/agentlongbench_1m_round_only_smoke_${STAMP}_runner.log}

mkdir -p "$(dirname "$SERVER_LOG")" "$OUTPUT_ROOT"

QW3_KVMEM_TRACE=1 \
QW3_KVMEM_DUMP_SCORES="$OUTPUT_ROOT/retrieval_scores.jsonl" \
QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
QW3_Q8_BF16_MAIN=0 \
"$ROOT/build/qw3" serve \
  --model "$ROOT/models/Qwen3.6-27B-Q8_0.gguf" \
  --ctx 1310720 \
  --kv-dtype fp8 \
  --kvmem \
  --kvmem-block-tokens 512 \
  --kvmem-budget 229376 \
  --kvmem-gen-budget 32768 \
  --kvmem-sink-blocks 8 \
  --kvmem-recent-blocks 0 \
  --kvmem-method retrieval \
  --kvmem-retrieval-method sub-block-mean-k \
  --kvmem-subblocks 16 \
  --kvmem-subblock-reduce max \
  --kvmem-round-retrieval \
  --kvmem-update-mode step \
  --kvmem-query-conditioned \
  --kvmem-immutable-k \
  --kvmem-opt-stage-out off \
  --kvmem-opt-stage-in on \
  --kvmem-opt-pack on \
  --kvmem-gpu-memory-ratio 0.5 \
  --kvmem-cpu-gb 40 \
  --enable-thinking \
  --thinking-budget 8192 \
  --prefill-chunk 2048 \
  --temp 0.6 \
  --native-mtp-speculate \
  --mtp-chain 4 \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -100 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

"$ROOT/.venv/bin/python" \
  "$ROOT/scripts/kvmem_eval/run_agentlongbench_kvmem.py" \
  --benchmark-repo /home/chaidi/AgentLongBench_Motivation \
  --dataset /home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl \
  --manifest /home/chaidi/AgentLongBench-Long/DeepseekMillion/manifest.jsonl \
  --allow-custom-subset \
  --benchmark-name AgentLongBench-1M-round-only-smoke \
  --output-root "$OUTPUT_ROOT" \
  --api-base "http://127.0.0.1:${PORT}/v1" \
  --model Qwen3.6-27B-Q8_0.gguf \
  --method kvmem_round_only_k224k_g32k_b512_sb16max \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 32768 \
  --context-window 1310720 \
  --context-safety-margin 16 \
  --timeout-sec 7200 \
  --max-sample-sec 7200 \
  --attempts 1 \
  --enable-thinking \
  --seed 20260722 \
  --kvmem-retrieval-trace-metadata \
  --kvmem-round-only \
  --limit 1 \
  2>&1 | tee "$RUNNER_LOG"
