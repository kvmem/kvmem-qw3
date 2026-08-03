#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802}
KVMEM_RESULT=${KVMEM_RESULT:-/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
SCORER_PY=${SCORER_PY:-/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python}
RUNNER="$ROOT/scripts/kvmem_eval/run_memoryagentbench_baselines.py"
SCORER="$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py"
JUDGE="$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py"
BINARY=${BINARY:-$ROOT/build/qw3}
MODEL=${MODEL:-$ROOT/models/Qwen3.6-27B-Q8_0.gguf}
PORT=${PORT:-18090}
START_METHOD=${START_METHOD:-compact-only}

# "Above 256K" means above the model's exact 262,144-token native limit,
# measured with the same Qwen tokenizer and rendered prompt as the KVMem run.
SELECTION=(
  --reference-results "$KVMEM_RESULT"
  --min-context-tokens-exclusive 262144
)

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/run"

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
  local log="$WORKSPACE/logs/server_${method}_over256k.log"
  stop_server
  QW3_Q8_BF16_MAIN=0 QW3_PREFIX_CACHE_TRACE=1 \
  QW3_PREFIX_CACHE_MAX_PAGES=${QW3_PREFIX_CACHE_MAX_PAGES:-8192} \
  QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES=1 \
    "$BINARY" serve \
      --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --ctx 262144 --kv-dtype fp8 \
      --prefill-chunk 2048 \
      --continuous-batching --prefix-cache \
      --max-active 1 --max-pending 8 \
      --mtp-chain 0 \
      -n 25000 \
      >"$log" 2>&1 &
  server_pid=$!
  echo "$server_pid" >"$WORKSPACE/run/server_${method}_over256k.pid"
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
  local client_log="$WORKSPACE/logs/${method}_over256k.log"
  local server_log="$WORKSPACE/logs/server_${method}_over256k.log"
  start_server "$method"

  "$PY" "$RUNNER" run \
    --workspace "$WORKSPACE" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --server-mtp-chain 0 \
    --prefix-cache-guard-pages 1 \
    --method "$method" --max-contexts 1 --question-limit 2 \
    "${SELECTION[@]}" \
    >>"$client_log" 2>&1
  local smoke_requests
  local hits
  smoke_requests=$(grep -c "chat(stream)" "$server_log" || true)
  hits=$(grep -c "prefix_cache hit" "$server_log" || true)
  # On a resumable restart the first selected row may already have its
  # row_summary sentinel, in which case the smoke command intentionally makes
  # no request.  Only enforce the hit gate when the smoke actually ran.
  if [[ "$smoke_requests" -ge 2 && "$hits" -lt 2 ]]; then
    echo "$method smoke did not produce two lossless prefix-cache hits" >&2
    tail -100 "$server_log" >&2
    return 1
  elif [[ "$smoke_requests" -le 1 ]]; then
    echo "$method smoke questions skipped: first selected row already has the requested answers" \
      >>"$client_log"
  fi

  "$PY" "$RUNNER" run \
    --workspace "$WORKSPACE" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --server-mtp-chain 0 \
    --prefix-cache-guard-pages 1 \
    --method "$method" \
    "${SELECTION[@]}" \
    >>"$client_log" 2>&1
  stop_server

  "$SCORER_PY" "$SCORER" \
    --results-dir "$WORKSPACE/methods/$method" --allow-partial \
    >"$WORKSPACE/logs/score_${method}_over256k.log" 2>&1

  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    "$PY" "$JUDGE" \
      --results-dir "$WORKSPACE/methods/$method" --allow-partial \
      >"$WORKSPACE/logs/judge_${method}_over256k.log" 2>&1
  else
    echo "DEEPSEEK_API_KEY is unset; API judge deferred for $method" \
      >"$WORKSPACE/logs/judge_${method}_over256k.pending"
  fi
}

"$PY" "$RUNNER" audit --workspace "$WORKSPACE" "${SELECTION[@]}" \
  >"$WORKSPACE/logs/audit_over256k.log" 2>&1
"$PY" "$RUNNER" prepare --workspace "$WORKSPACE" "${SELECTION[@]}" \
  >"$WORKSPACE/logs/prepare_over256k.log" 2>&1

# RAG encodes every long row once before QW3 starts, so embeddings and model
# inference never compete for GPU memory.
PYTHONNOUSERSITE=1 \
HF_HOME=/data/chaidi/kvmem_eval/hf-cache/jina8k-pinned \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$RUNNER" prepare-rag --workspace "$WORKSPACE" \
  "${SELECTION[@]}" \
  >"$WORKSPACE/logs/prepare_rag_over256k.log" 2>&1

case "$START_METHOD" in
  compact-only)
    run_method compact-only
    run_method compact-rag
    run_method sliding-window
    ;;
  compact-rag)
    run_method compact-rag
    run_method sliding-window
    ;;
  sliding-window)
    run_method sliding-window
    ;;
  *)
    echo "invalid START_METHOD=$START_METHOD" >&2
    exit 2
    ;;
esac

"$PY" "$RUNNER" status --workspace "$WORKSPACE" "${SELECTION[@]}" \
  >"$WORKSPACE/status_over256k.json"
echo "MemoryAgentBench >256K baselines completed: $WORKSPACE"
