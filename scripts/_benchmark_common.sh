#!/usr/bin/env bash
# Shared helpers for run_benchmark_{smoke,efficiency,accuracy}.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

load_manifest() {
  local manifest="${REPO_ROOT}/benchmark/manifest.env"
  if [[ ! -f "$manifest" ]]; then
    echo "missing ${manifest}; copy benchmark/manifest.env.example first" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$manifest"
  : "${QW3:?set QW3 in benchmark/manifest.env}"
  : "${MODEL:?set MODEL in benchmark/manifest.env}"
}

git_sha_short() {
  git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit"
}

init_result_dir() {
  local kind="$1"
  local root="${RESULT_ROOT:-${REPO_ROOT}/benchmark/results/$(git_sha_short)_$(date +%Y%m%d_%H%M%S)}"
  OUT="${root}/${kind}"
  mkdir -p "$OUT"
  {
    echo "kind=${kind}"
    echo "started_at=$(date -Iseconds)"
    echo "git_sha=$(git_sha_short)"
    echo "qw3=${QW3}"
    echo "model=${MODEL}"
    echo "ctx=${CTX:-}"
    echo "prefill_chunk=${PREFILL_CHUNK:-}"
    echo "kv_dtype=${KV_DTYPE:-}"
  } >"${OUT}/manifest.txt"
  echo "results -> ${OUT}"
}

step() {
  echo
  echo "==> $*"
  echo
}

require_binary() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "${label} not found: ${path}" >&2
    exit 1
  fi
}

require_model() {
  if [[ ! -f "$MODEL" ]]; then
    echo "model not found: ${MODEL}" >&2
    exit 1
  fi
}
