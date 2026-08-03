#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
INPUT=${INPUT:-$ROOT/results/kvmem_archive_beam10m_c1/history.qwen-chat.txt}
ARCHIVE=${ARCHIVE:-/home/chaidi/kca/beam10m_c1_9998336_b128_20260802}
LOG=${LOG:-$ROOT/results/kvmem_archive_beam10m_c1/build_9998336_b128.log}

# 10,000,000 is snapped down by the archive contract to the largest complete
# 2,048-token raw-K chunk: 9,998,336 durable tokens.  The context capacity
# leaves 32K generation reserve plus another 32K for a framed query.
exec env QW3_Q8_BF16_MAIN=0 \
  "$ROOT/build/qw3" archive build \
  --model "$MODEL" \
  --kvmem-archive "$ARCHIVE" \
  --archive-input "$INPUT" \
  --archive-tokens 10000000 \
  --archive-ladder-tokens 1048576 \
  --ctx 10065536 \
  --kvmem-block-tokens 128 \
  --kvmem-budget 204800 \
  --kvmem-gen-budget 32768 \
  --kvmem-sink-tokens 2048 \
  --kvmem-recent-tokens 16384 \
  --kvmem-cpu-gb 32 \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-index-placement cpu \
  --kvmem-index-staging-mb 64 \
  --kvmem-query-conditioned \
  --kvmem-retrieval-method mean-k \
  --kvmem-opt-stage-out on \
  --kvmem-opt-stage-in on \
  --kvmem-opt-pack on \
  --prefill-chunk 2048 \
  >"$LOG" 2>&1
