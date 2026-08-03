#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802}
KVMEM_RESULT=${KVMEM_RESULT:-/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
RUNNER="$ROOT/scripts/kvmem_eval/run_memoryagentbench_baselines.py"
SCORER="$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py"
BINARY=${BINARY:-$ROOT/build/qw3}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
PORT=${PORT:-18090}
EXPECTED_CONTEXTS=${EXPECTED_CONTEXTS:-146}

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/run"

wait_for_kvmem() {
  while true; do
    local completed
    completed=$(find "$KVMEM_RESULT/rows" -name row_summary.json -type f 2>/dev/null | wc -l)
    if [[ "$completed" -eq "$EXPECTED_CONTEXTS" ]]; then
      echo "KVMem generation completed: $completed/$EXPECTED_CONTEXTS"
      return
    fi
    echo "Waiting for KVMem generation: $completed/$EXPECTED_CONTEXTS"
    sleep 60
  done
}

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" || true
  fi
  server_pid=""
}
trap stop_server EXIT

start_server() {
  local method=$1
  local log="$WORKSPACE/logs/server_${method}.log"
  stop_server
  QW3_Q8_BF16_MAIN=0 QW3_PREFIX_CACHE_TRACE=1 \
    "$BINARY" serve \
      --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --ctx 262144 --kv-dtype fp8 \
      --prefill-chunk 2048 \
      --continuous-batching --prefix-cache \
      --max-active 1 --max-pending 8 \
      --mtp-chain 4 --mtp-batched-draft --mtp-paged-prefix \
      -n 25000 \
      >"$log" 2>&1 &
  server_pid=$!
  echo "$server_pid" >"$WORKSPACE/run/server_${method}.pid"
  for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$log"
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return
    fi
    sleep 2
  done
  echo "QW3 server did not become healthy for $method" >&2
  return 1
}

run_method() {
  local method=$1
  local client_log="$WORKSPACE/logs/${method}.log"
  local server_log="$WORKSPACE/logs/server_${method}.log"
  start_server "$method"

  # Two-question gate validates generation, result schema, and the explicitly
  # primed dense prefix-cache path before committing to all 3,671 questions.
  "$PY" "$RUNNER" run \
    --workspace "$WORKSPACE" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --method "$method" --max-contexts 1 --question-limit 2 \
    >>"$client_log" 2>&1
  local hits
  hits=$(grep -c "prefix_cache hit" "$server_log" || true)
  if [[ "$hits" -lt 2 ]]; then
    echo "$method smoke did not produce two lossless prefix-cache hits" >&2
    tail -100 "$server_log" >&2
    return 1
  fi

  "$PY" "$RUNNER" run \
    --workspace "$WORKSPACE" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --method "$method" \
    >>"$client_log" 2>&1
  stop_server

  "$PY" "$SCORER" \
    --results-dir "$WORKSPACE/methods/$method" \
    >"$WORKSPACE/logs/score_${method}.log" 2>&1
}

wait_for_kvmem

"$PY" "$RUNNER" audit --workspace "$WORKSPACE" \
  >"$WORKSPACE/logs/audit.log" 2>&1
"$PY" "$RUNNER" prepare --workspace "$WORKSPACE" \
  >"$WORKSPACE/logs/prepare.log" 2>&1

# The pinned Jina implementation uses GPU only in this phase.  QW3 is not
# running yet, so embedding and inference never contend for GPU memory.
PYTHONNOUSERSITE=1 \
HF_HOME=/data/chaidi/kvmem_eval/hf-cache/jina8k-pinned \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$RUNNER" prepare-rag --workspace "$WORKSPACE" \
  >"$WORKSPACE/logs/prepare_rag.log" 2>&1

run_method compact-only
run_method compact-rag
run_method sliding-window

"$PY" "$RUNNER" status --workspace "$WORKSPACE" \
  >"$WORKSPACE/final_status.json"
echo "MemoryAgentBench baselines completed: $WORKSPACE"
