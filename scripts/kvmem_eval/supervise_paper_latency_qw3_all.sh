#!/usr/bin/env bash
# Complete formal latency pass.  Both ordinary baselines and KVMem are served
# by the same native qw3 binary/model; no vLLM artifact is accepted.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
OUT=${OUT:-/data/chaidi/kvmem_eval/results/paper_pre_answer_latency_qw3_current_20260812}
LOG=${LOG:-$OUT/logs/all_qw3_supervisor.log}
mkdir -p "$OUT/logs"
exec >>"$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] checking plain-qw3 baselines"
if python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
    --root "$OUT" --stage baseline --output "$OUT/baseline_validation.json" \
    >/dev/null 2>&1; then
  echo "[$(date --iso-8601=seconds)] reusing validated plain-qw3 baselines"
else
  echo "[$(date --iso-8601=seconds)] starting plain-qw3 baselines"
  "$ROOT/scripts/kvmem_eval/supervise_paper_latency_qw3_current_machine.sh"
  python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
    --root "$OUT" --stage baseline --output "$OUT/baseline_validation.json"
fi

echo "[$(date --iso-8601=seconds)] starting qw3 KVMem cells"
OUT="$OUT" "$ROOT/scripts/kvmem_eval/supervise_paper_latency_kvmem_long_current_machine.sh"

python3 "$ROOT/scripts/kvmem_eval/validate_paper_latency_current.py" \
  --root "$OUT" --stage all --output "$OUT/final_all_validation.json"
python3 "$ROOT/scripts/kvmem_eval/summarize_paper_pre_answer_latency_current.py" \
  --root "$OUT" --json-output "$OUT/summary.json" \
  --markdown-output "$OUT/summary.md" --latex-output "$OUT/latency_rows.tex"
echo "[$(date --iso-8601=seconds)] completed all formal qw3 latency cells"
