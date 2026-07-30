#!/usr/bin/env bash
set -euo pipefail

# Wait for the active ab10 run, validate both arms, then launch the disjoint
# remaining39 cohort.  A failed/incomplete first stage deliberately prevents
# the long continuation from starting.

ROOT=${ROOT:-/home/chaidi/qw3}
WAIT_SESSION=${WAIT_SESSION:-round_pad_ab10_174236}
AB10_ROOT=${AB10_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_round_padding_ab10_20260728_174236}

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

"$ROOT/.venv/bin/python" - "$AB10_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for arm in ("padding32", "unpadded"):
    report_path = root / arm / "validation_report.json"
    if not report_path.exists():
        raise SystemExit(f"{arm}: missing validation report")
    report = json.loads(report_path.read_text())
    if (
        report.get("passed") is not True
        or report.get("expected_total") != 10
        or report.get("answers_unique") != 10
        or report.get("eval_unique") != 10
    ):
        raise SystemExit(f"{arm}: incomplete/invalid ab10 result: {report}")
print("ab10 validation passed; starting disjoint remaining39 A/B", flush=True)
PY

exec env STAMP="$(date +%Y%m%d_%H%M%S)" \
  bash "$ROOT/scripts/kvmem_eval/run_agentlongbench_1m_round_padding_remaining39.sh"
