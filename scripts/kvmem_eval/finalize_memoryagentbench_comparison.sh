#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802}
KVMEM_RESULT=${KVMEM_RESULT:-/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full}
PY=${PY:-python3}
JUDGE="$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py"
COMPARE="$ROOT/scripts/kvmem_eval/compare_memoryagentbench_methods.py"
AUDIT="$ROOT/scripts/kvmem_eval/audit_memoryagentbench_comparison.py"
UTILITY="$ROOT/scripts/kvmem_eval/update_utility_evaluation.py"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required to finish uncached baseline judgments" >&2
  exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

for root in \
  "$KVMEM_RESULT" \
  "$WORKSPACE/methods/compact-only" \
  "$WORKSPACE/methods/compact-rag" \
  "$WORKSPACE/methods/sliding-window"; do
  "$PY" "$JUDGE" --results-dir "$root" \
    >"$root/judge_complete.log" 2>&1
done

COMMON=(
  --kvmem-results "$KVMEM_RESULT"
  --method Compact "$WORKSPACE/methods/compact-only"
  --method Compact+RAG "$WORKSPACE/methods/compact-rag"
  --method Window32K "$WORKSPACE/methods/sliding-window"
)

"$PY" "$COMPARE" "${COMMON[@]}" \
  --min-context-tokens-exclusive 262144 \
  --output "$WORKSPACE/comparison_over256k.json" \
  >"$WORKSPACE/logs/comparison_over256k.log"
"$PY" "$COMPARE" "${COMMON[@]}" \
  --max-context-tokens-inclusive 262144 \
  --output "$WORKSPACE/comparison_under256k.json" \
  >"$WORKSPACE/logs/comparison_under256k.log"
"$PY" "$COMPARE" "${COMMON[@]}" \
  --output "$WORKSPACE/comparison_full.json" \
  >"$WORKSPACE/logs/comparison_full.log"

"$PY" "$AUDIT" \
  --kvmem-results "$KVMEM_RESULT" \
  --workspace "$WORKSPACE" \
  --output "$WORKSPACE/completion_audit.json" \
  >"$WORKSPACE/logs/completion_audit.log"

"$PY" "$UTILITY" >"$WORKSPACE/logs/update_utility_evaluation.log" 2>&1
"$PY" "$UTILITY" --check \
  >>"$WORKSPACE/logs/update_utility_evaluation.log" 2>&1

echo "MemoryAgentBench judgments and comparisons completed: $WORKSPACE"
