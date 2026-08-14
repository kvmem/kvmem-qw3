#!/usr/bin/env bash
# Continue the controlled paper baseline queue only after the strict MAB job
# has produced the complete >256K cohort for both methods.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
MAB_ROOT=${MAB_ROOT:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812}
MAB_SESSION=${MAB_SESSION:-vllm022_mab_strict64_restart}
MAB_LOG=${MAB_LOG:-/data/chaidi/kvmem_eval/logs/vllm022_mab_strict64_restart_20260812.log}
LOG=${LOG:-/data/chaidi/kvmem_eval/logs/paper_remaining_queue_20260812.log}
SUCCESS_MARKER=${SUCCESS_MARKER:-/data/chaidi/kvmem_eval/results/paper_utility_sliding_complete_20260812.json}
ALB512_ROOT=${ALB512_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_512k_sliding_window_cap64k_vllm022_fp8_mtp4_20260812}
ALB1M_ROOT=${ALB1M_ROOT:-/data/chaidi/kvmem_eval/results/agentlongbench_1m_sliding_window_cap100k_vllm022_fp8_mtp4_20260812}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

validate_mab() {
python3 - "$MAB_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_contexts = 30
expected_questions = 1316
for method in ("compact-rag", "sliding-window"):
    rows_root = root / "methods" / method / "rows"
    result_paths = sorted(rows_root.glob("*/results.jsonl"))
    completed_rows = sorted(rows_root.glob("*/row_summary.json"))
    records = []
    for path in result_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    identities = {
        (row.get("split"), row.get("source"), row.get("dataset_row"),
         row.get("question_index"))
        for row in records
    }
    if len(completed_rows) != expected_contexts:
        raise SystemExit(
            f"{method}: completed contexts={len(completed_rows)}, "
            f"expected={expected_contexts}"
        )
    if len(records) != expected_questions or len(identities) != expected_questions:
        raise SystemExit(
            f"{method}: records={len(records)} unique={len(identities)}, "
            f"expected={expected_questions}"
        )
    if any(row.get("finish_reason") not in {"stop", "length"} for row in records):
        raise SystemExit(f"{method}: unexpected terminal finish reason")
    print(f"{method}: contexts={len(completed_rows)} questions={len(records)} PASS")
PY
}

echo "[$(date --iso-8601=seconds)] waiting for complete strict MAB outputs"
until validate_mab; do
  if ! tmux has-session -t "$MAB_SESSION" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] restarting missing MAB supervisor $MAB_SESSION"
    tmux new-session -d -s "$MAB_SESSION" \
      "cd $ROOT && VLLM_ENV=/home/chaidi/qw3/.venv bash scripts/kvmem_eval/supervise_mab_over256k_compact_rag64k_vllm.sh >>$MAB_LOG 2>&1"
  fi
  sleep 60
done

for auxiliary in mab_sliding_v510_main mab_sliding_v510_reverse; do
  tmux kill-session -t "$auxiliary" 2>/dev/null || true
done
if tmux has-session -t "$MAB_SESSION" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] strict MAB generation complete; stopping its server"
  tmux kill-session -t "$MAB_SESSION"
fi
echo "[$(date --iso-8601=seconds)] MAB rows complete; waiting for its vLLM to release GPU"
while pgrep -f '/bin/vllm serve .*--port 18122' >/dev/null 2>&1; do
  sleep 10
done

echo "[$(date --iso-8601=seconds)] starting controlled AgentLongBench sliding runs"
until "$ROOT/scripts/kvmem_eval/supervise_alb_sliding_updated_vllm.sh"; do
  echo "[$(date --iso-8601=seconds)] AgentLongBench vLLM pass interrupted; resuming in 30s"
  sleep 30
done

# Write the public completion marker only after the full configuration audit:
# exact sample counts, unique IDs, method-specific prompt caps, no RAG metadata,
# no raw tail, and the required DeepSeek question-blind 1M summary provenance.
until python3 "$ROOT/scripts/kvmem_eval/validate_paper_utility_current.py" \
    --stage all --require-special-judge --mab "$MAB_ROOT" \
    --alb512-sliding "$ALB512_ROOT" --alb1m-sliding "$ALB1M_ROOT" \
    --output "$SUCCESS_MARKER"; do
  echo "[$(date --iso-8601=seconds)] utility artifacts not fully judged yet; retrying in 30s"
  sleep 30
done
