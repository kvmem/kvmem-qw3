"""Summarize retained live VM trials, including failures and action timings."""
import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values) - 1) * percent / 100
    lower = int(offset)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (offset - lower)


def summarize(run):
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] == "running":
        raise ValueError("The suite is still running; wait for its terminal summary")
    rows = []
    for trial in summary["trials"]:
        directory = Path(trial["directory"])
        row = {key: trial.get(key) for key in ("kvmem", "case", "repetition", "passed", "elapsed_seconds", "error")}
        resources = directory / "resources.jsonl"
        if resources.exists():
            samples = [json.loads(line) for line in resources.read_text().splitlines()]
            row["sampled_gpu_peak_bytes"] = max((s.get("gpu_memory_bytes", 0) for s in samples), default=0)
            row["process_rss_high_water_bytes"] = max((s.get("peak_rss_bytes", 0) for s in samples), default=0)
        database = directory / "app/desktop/desktop.sqlite"
        if database.exists():
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
                observations = dict(db.execute("SELECT id,created FROM observations"))
                actions = list(db.execute("SELECT request,status,result,created FROM actions"))
            screenshot_action, receipt = [], []
            for request, status, result, created in actions:
                if status != "completed":
                    continue
                before = json.loads(request)["observation_id"]
                after = json.loads(result)["observation"]["observation_id"]
                if before in observations:
                    screenshot_action.append(created - observations[before])
                if after in observations:
                    receipt.append(observations[after] - created)
            row["completed_actions"] = sum(a[1] == "completed" for a in actions)
            row["rejected_actions"] = sum(a[1] == "rejected" for a in actions)
            row["unresolved_actions"] = sum(a[1] in ("uncertain", "dispatching") for a in actions)
            for key, samples in (("screenshot_to_action", screenshot_action), ("action_receipt", receipt)):
                for percent in (50, 95):
                    row[f"{key}_p{percent}_seconds"] = percentile(samples, percent)
        log = directory / "engine/engine.log"
        if log.exists():
            text = log.read_text()
            timings = list(map(float, re.findall(r"response_ttft_ms=([\d.]+)", text)))
            row["inference_requests"] = len(timings)
            row["server_visible_output_p50_seconds"] = percentile(timings, 50) / 1000 if timings else None
            row["server_visible_output_p95_seconds"] = percentile(timings, 95) / 1000 if timings else None
            row["max_prompt_tokens"] = max(map(int, re.findall(r"native generate: prompt_tokens=(\d+)", text)), default=0)
            draft = re.findall(r"native dflash_spec_summary: .*?drafted=(\d+) accepted=(\d+).*?fallback=(\w+)", text)
            row["dflash_drafted"] = sum(int(d[0]) for d in draft)
            row["dflash_accepted"] = sum(int(d[1]) for d in draft)
            row["dflash_fallback_requests"] = sum(d[2] == "true" for d in draft)
        rows.append(row)
    aggregates = []
    for policy, case in sorted({(r["kvmem"], r["case"]) for r in rows}):
        trials = [r for r in rows if r["kvmem"] == policy and r["case"] == case]
        passed = [r for r in trials if r["passed"]]
        aggregates.append({"kvmem": policy, "case": case, "passed": len(passed), "trials": len(trials),
                           "successful_task_seconds_median": statistics.median(r["elapsed_seconds"] for r in passed) if passed else None,
                           "sampled_gpu_peak_bytes": max(r.get("sampled_gpu_peak_bytes", 0) for r in trials)})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (run / "trials.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (run / "aggregate.json").write_text(json.dumps(aggregates, indent=2))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    summarize(parser.parse_args().run_dir)
