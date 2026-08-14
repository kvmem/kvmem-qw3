#!/usr/bin/env bash
set -euo pipefail

# Remeasure DeepSeek compaction wall time for the frozen AgentLongBench 1M
# P25/P50/P75 latency cohort.  The API key is accepted only through the
# process environment and is copied to a mode-0600 tmpfs file because the
# upstream runner resolves --api-key-file with pathlib (and therefore cannot
# consume a short-lived process-substitution fd reliably).

REPO=${ALB_LONG_REPO:-/home/chaidi/AgentLongBench-Long}
PYTHON=${PYTHON:-/home/chaidi/qw3/.venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_current_20260812/alb1m_deepseek_compaction_current}
TOKENIZER=${TOKENIZER:-/data/chaidi/kvmem_eval/models/deepseek-ai/DeepSeek-V4-Pro-tokenizer-b5968e9190ef611bbf34a7229255be88a0e937c1}
RUNNER="$REPO/script/deepseekMillion/run_one_round_compact_1m.py"

SAMPLE_IDS=(
  94f775dc6576cb65cd30b031541fe8b47f1ba6b96825443a01308c6a633fbcca
  5052ef703e33f939469245773a16570d87729aba959421457045fbf12fcbf73e
  7a825f6dde6347489a4c6d0a02aeb60520ed5680a857578f74ad4884053aa018
)

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required" >&2
  exit 2
fi
if [[ ! -f "$TOKENIZER/tokenizer.json" || ! -f "$TOKENIZER/tokenizer_config.json" ]]; then
  echo "DeepSeek tokenizer snapshot is incomplete: $TOKENIZER" >&2
  exit 2
fi

umask 077
KEY_FILE=$(mktemp /dev/shm/qw3-deepseek-key.XXXXXX)
cleanup() {
  rm -f "$KEY_FILE"
}
trap cleanup EXIT INT TERM
printf '%s' "$DEEPSEEK_API_KEY" >"$KEY_FILE"
unset DEEPSEEK_API_KEY

QUESTION_ARGS=()
for sample_id in "${SAMPLE_IDS[@]}"; do
  QUESTION_ARGS+=(--question-id "$sample_id")
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"$PYTHON" "$RUNNER" compact-only \
  --output-root "$OUTPUT_ROOT" \
  --api-key-file "$KEY_FILE" \
  --tokenizer-path "$TOKENIZER" \
  --tokenizer-label 'huggingface:deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1' \
  --timeout-sec 1800 \
  --max-call-sec 1800 \
  --attempts 3 \
  --continue-on-error \
  "${QUESTION_ARGS[@]}"
