#!/usr/bin/env bash
# Convenience wrapper: sync the comparison script to the remote, run it
# against the standard model + default prompts, and stream the output back.
#
# Usage:
#   bash scripts/run_compare.sh            # uses defaults (n=32, ctx=4096)
#   bash scripts/run_compare.sh -n 64      # forward extra args to the python runner
#   bash scripts/run_compare.sh --long-only -n 64 --token-diff
set -eu

REMOTE_HOST="${REMOTE_HOST:-Pro6000-Remote}"
REMOTE_DIR="${REMOTE_DIR:-/home/chaidi/qw3}"
LLAMA_BIN="${LLAMA_BIN:-/tmp/llama.cpp/build-cuda/bin/llama-completion}"

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Rsync the script in case the user has edited it locally.
rsync -az "${script_dir}/compare_with_llama_cpp.py" \
    "${REMOTE_HOST}:${REMOTE_DIR}/scripts/compare_with_llama_cpp.py"

# Run it remotely. Pass any extra CLI args through verbatim. We pipe through
# `cat` so colour / line-buffering quirks don't truncate output.
ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_DIR}' && LLAMA_CLI='${LLAMA_BIN}' python3 scripts/compare_with_llama_cpp.py $*" \
    | cat
