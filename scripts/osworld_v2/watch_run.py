#!/usr/bin/env python3
"""Maintain a compact status file for one already-running OSWorld evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # the producer may be midway through its final append
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    while True:
        record = json.loads(args.metadata.read_text())
        if record.get("status", "").startswith("stopped"):
            return
        run = Path(record["run_dir"])
        tasks = list((run / "osworld").glob("**/agent_step_trace.jsonl"))
        rows = read_rows(tasks[0]) if tasks else []
        timings = read_rows(tasks[0].with_name("step_timing.jsonl")) if tasks else []
        events = read_rows(tasks[0].with_name("history_compaction.jsonl")) if tasks else []
        log_path = run / "qw3_server.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        record.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      completed_predict_steps=sum(r.get("status") == "success" for r in rows),
                      completed_steps=len(timings),
                      completed_model_requests=log.count("native continuous_batch:"),
                      prefix_hits=log.count("prefix_cache hit id="),
                      capacity_rejections=log.count("admission rejected:"),
                      summary_compactions=sum(e["status"] == "success" for e in events),
                      summary_failures=sum(e["status"] == "error" for e in events),
                      image_folding_events=sum(r.get("compaction", {}).get("triggered", False) for r in rows))
        if rows:
            last = rows[-1]
            record["last_prompt_tokens"] = last.get("api", {}).get("usage", {}).get("prompt_tokens")
            record["last_active_image_count"] = last.get("compaction", {}).get("active_image_count")
            ttfts = [r.get("api", {}).get("server_timing", {}).get("server_ttft_sec") for r in rows[-20:]]
            ttfts = [v for v in ttfts if isinstance(v, (int, float))]
            if ttfts:
                record["recent20_mean_ttft_sec"] = statistics.mean(ttfts)
        if timings:
            record["recent20_mean_step_sec"] = statistics.mean(r["turn_wall_time_ms"] / 1000 for r in timings[-20:])
        results = list((run / "osworld").glob("**/result.txt"))
        if results:
            try:
                record.update(status="completed", score=float(results[0].read_text().strip()), result_file=str(results[0]))
            except ValueError:
                pass
        else:
            try:
                alive = Path(f"/proc/{record['pid']}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
            except FileNotFoundError:
                alive = False
            if not alive:
                record.update(status="infrastructure_failure", score=None,
                              reason="runner exited without result.txt; do not score as model failure")
        temporary = args.metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.metadata)
        if record["status"] != "running":
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
