#!/usr/bin/env bash
# Preserve the in-flight second legacy capture, then switch to the interleaved
# two-resident-server pipeline as soon as its complete dump is durable.
set -euo pipefail

DUMP=/data/chaidi/kvmem_eval/results/agentlongbench_1m_ds50_kvmem_error19_selected_dump_20260727_103114/kvmem_retrieval_dump.jsonl
while true; do
  count=$(
    /home/chaidi/qw3/.venv/bin/python - "$DUMP" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(sum(
        1
        for line in path.open(encoding="utf-8")
        if line.strip() and json.loads(line).get("type") == "meta"
    ))
PY
  )
  if (( count >= 2 )); then
    break
  fi
  sleep 1
done

# Allow the capture client to append its small manifest row.  It may begin
# tokenizing sample three during this grace period; stopping the unit is safe
# because no third dump can have been committed in three seconds.
sleep 3
systemctl --user stop \
  qw3-agentlongbench-1m-error19-capture-20260727.service || true
for _ in $(seq 1 60); do
  if ! curl -fsS --noproxy '*' \
    http://127.0.0.1:8088/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
exec /bin/bash \
  /home/chaidi/qw3/scripts/kvmem_eval/run_agentlongbench_1m_error19_interleaved.sh
