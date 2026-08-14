#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
INPUT=${INPUT:-$ROOT/results/kvmem_archive_beam10m_c1/history.qwen-chat.txt}
OUT_DIR=${OUT_DIR:-$ROOT/results/adaptive_scorer_ab_10m}
ARCHIVE=${ARCHIVE:-/home/chaidi/kca/beam10m_c1_9998336_b64_scorer_ab_20260812}
LOG=${LOG:-$OUT_DIR/archive_build.log}
ARCHIVE_TOKENS=${ARCHIVE_TOKENS:-10000000}
ARCHIVE_LADDER_TOKENS=${ARCHIVE_LADDER_TOKENS:-1048576}
CTX=${CTX:-10065536}

mkdir -p "$OUT_DIR"

# The builder snaps 10,000,000 tokens down to the largest complete 2,048-token
# raw-K chunk (9,998,336 durable tokens).  The scorer comparison uses 64-token
# blocks to match the workspace-scalability benchmark.
exec env QW3_Q8_BF16_MAIN=0 \
  "$ROOT/build/qw3" archive build \
  --model "$MODEL" \
  --kvmem-archive "$ARCHIVE" \
  --archive-input "$INPUT" \
  --archive-tokens "$ARCHIVE_TOKENS" \
  --archive-ladder-tokens "$ARCHIVE_LADDER_TOKENS" \
  --ctx "$CTX" \
  --kvmem-block-tokens 64 \
  --kvmem-budget 65536 \
  --kvmem-gen-budget 32768 \
  --kvmem-sink-tokens 512 \
  --kvmem-recent-tokens 0 \
  --kvmem-cpu-gb 32 \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-index-placement cpu \
  --kvmem-index-staging-mb 64 \
  --kvmem-query-conditioned \
  --kvmem-retrieval-method key-direction-adaptive \
  --kvmem-adaptive-score-mode tiled-two-pass \
  --kvmem-opt-stage-out on \
  --kvmem-opt-stage-in on \
  --kvmem-opt-pack on \
  --prefill-chunk 2048 \
  >"$LOG" 2>&1
