#!/usr/bin/env bash
# Resume-safe official/local and DeepSeek special scoring for the strict 64K
# MemoryAgentBench >256K comparison.  It intentionally starts Compact+RAG
# judging as soon as that method is complete, so network-only grading overlaps
# the subsequent Sliding Window GPU run.
set -euo pipefail

ROOT=${ROOT:-/home/chaidi/qw3}
PY=${PY:-/home/chaidi/kvmem-efficiency-bench/.venv-rag/bin/python}
WORKSPACE=${WORKSPACE:-/data/chaidi/kvmem_eval/results/memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812}
SCORER=${SCORER:-$ROOT/scripts/kvmem_eval/score_memoryagentbench_official_local.py}
JUDGE=${JUDGE:-$ROOT/scripts/kvmem_eval/judge_memoryagentbench_special.py}
LOG=${LOG:-$WORKSPACE/logs/special_judge_supervisor.log}

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY must be provided in the supervisor environment" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

method_complete() {
  local method=$1
  "$PY" - "$WORKSPACE/methods/$method" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = list((root / "rows").glob("*/row_summary.json"))
records = []
for path in (root / "rows").glob("*/results.jsonl"):
    records.extend(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
ids = {
    (row.get("split"), row.get("source"), row.get("dataset_row"),
     row.get("question_index"))
    for row in records
}
if len(summaries) != 30 or len(records) != 1316 or len(ids) != 1316:
    raise SystemExit(1)
if any(row.get("finish_reason") not in {"stop", "length"} for row in records):
    raise SystemExit(1)
PY
}

judge_method() {
  local method=$1
  local results="$WORKSPACE/methods/$method"
  echo "[$(date --iso-8601=seconds)] scoring $method"
  "$PY" "$SCORER" --results-dir "$results" --allow-partial \
    >"$WORKSPACE/logs/score_${method}.log" 2>&1
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    "$PY" "$JUDGE" --results-dir "$results" --allow-partial \
      >"$WORKSPACE/logs/judge_${method}.log" 2>&1
  "$PY" - "$results" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
special = json.loads((root / "special_judge_summary.json").read_text())
final = json.loads((root / "final_summary.json").read_text())
if special.get("judge_model") != "deepseek-v4-pro":
    raise SystemExit("unexpected judge model")
if special.get("completed") != 316 or special.get("expected") != 316:
    raise SystemExit(
        f"incomplete special judge: {special.get('completed')}/"
        f"{special.get('expected')}"
    )
if not isinstance(final.get("overall_macro_category_mean"), (int, float)):
    raise SystemExit("missing final overall score")
print(
    f"validated questions={final.get('questions')} "
    f"overall={100.0 * final['overall_macro_category_mean']:.4f}%"
)
PY
  echo "[$(date --iso-8601=seconds)] completed $method special judge"
}

judge_partial() {
  local method=$1
  local results="$WORKSPACE/methods/$method"
  echo "[$(date --iso-8601=seconds)] incrementally judging available $method rows"
  # The judge is resume-safe by its stable per-question key.  Keep exactly one
  # supervisor writing these files, so generation can continue while API-only
  # grading catches up without two processes appending the same judgment.  The
  # CPU-heavy official/local scorer is intentionally deferred until generation
  # is complete; it is not needed to cache API judgments and otherwise competes
  # with tokenization and retrieval preparation in the active utility run.
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    "$PY" "$JUDGE" --results-dir "$results" --allow-partial \
      >>"$WORKSPACE/logs/judge_${method}.log" 2>&1
}

for method in compact-rag sliding-window; do
  echo "[$(date --iso-8601=seconds)] waiting for complete $method results"
  until method_complete "$method"; do
    judge_partial "$method"
    sleep 60
  done
  judge_method "$method"
done

echo "[$(date --iso-8601=seconds)] all strict64 special judges complete"
