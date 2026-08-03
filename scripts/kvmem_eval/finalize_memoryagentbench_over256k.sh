#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802}
KVMEM_RESULT=${KVMEM_RESULT:-/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
SCORER_PY=${SCORER_PY:-/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python}
SCORER="$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py"
JUDGE="$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py"
COMPARE="$ROOT/scripts/kvmem_eval/compare_memoryagentbench_methods.py"
AUDIT="$ROOT/scripts/kvmem_eval/audit_memoryagentbench_comparison.py"

if [[ ! -f "$WORKSPACE/status_over256k.json" ]]; then
  echo "missing completed >256K generation status" >&2
  exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required for uncached baseline judgments" >&2
  exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# Refuse to report a comparison unless every method contains exactly the same
# 30 long contexts and 1,316 questions selected from the KVMem reference.
"$PY" - "$WORKSPACE/status_over256k.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = json.loads(path.read_text(encoding="utf-8"))
if status.get("contexts_total") != 30:
    raise SystemExit(f"unexpected selected contexts: {status.get('contexts_total')}")
for method, values in status.get("methods", {}).items():
    got = (values.get("contexts_completed"), values.get("questions_written"))
    if got != (30, 1316):
        raise SystemExit(f"incomplete {method}: {got} != (30, 1316)")
PY

for method in compact-only compact-rag sliding-window; do
  method_root="$WORKSPACE/methods/$method"
  if ! "$PY" - "$method_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = root / "official_local_summary.json"
if not summary.is_file():
    raise SystemExit(1)
try:
    if json.loads(summary.read_text()).get("questions_scored") != 1316:
        raise SystemExit(1)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
summary_mtime = summary.stat().st_mtime_ns
if any(
    path.stat().st_mtime_ns > summary_mtime
    for path in (root / "rows").glob("*/results.jsonl")
):
    raise SystemExit(1)
PY
  then
    "$SCORER_PY" "$SCORER" --results-dir "$method_root" --allow-partial \
      >"$WORKSPACE/logs/score_${method}_over256k_complete.log" 2>&1
  fi
  "$PY" "$JUDGE" --results-dir "$method_root" --allow-partial \
    >"$WORKSPACE/logs/judge_${method}_over256k_complete.log" 2>&1
done

# This is normally a cache-only integrity rebuild: the completed KVMem run has
# all 3,671 deterministic scores and all 400 special judgments already.
"$PY" "$JUDGE" --results-dir "$KVMEM_RESULT" \
  >"$WORKSPACE/logs/judge_kvmem_complete.log" 2>&1

"$PY" "$COMPARE" \
  --kvmem-results "$KVMEM_RESULT" \
  --method Compact "$WORKSPACE/methods/compact-only" \
  --method Compact+RAG "$WORKSPACE/methods/compact-rag" \
  --method Window32K "$WORKSPACE/methods/sliding-window" \
  --min-context-tokens-exclusive 262144 \
  --output "$WORKSPACE/comparison_over256k.json" \
  >"$WORKSPACE/logs/comparison_over256k.log"

"$PY" "$AUDIT" \
  --kvmem-results "$KVMEM_RESULT" \
  --workspace "$WORKSPACE" \
  --phase over256k \
  --output "$WORKSPACE/completion_audit_over256k.json" \
  >"$WORKSPACE/logs/completion_audit_over256k.log"

echo "MemoryAgentBench >256K judgments/comparison completed: $WORKSPACE"
