#!/usr/bin/env python3
"""Freeze a task order from the official site's published per-task scores.

The current public showcase covers only nine tasks from v2026.06.24.
Its scores are a scheduling proxy, not v2026.08.08 success probabilities.
No task instructions, solutions, or trajectories are passed to the agent.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def js_json(source, name):
    match = re.search(r"\bvar\s+" + re.escape(name) + r"\s*=\s*", source)
    if not match:
        raise ValueError(f"Missing official data variable {name}")
    return json.JSONDecoder().raw_decode(source[match.end():])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase-js", required=True, type=Path)
    parser.add_argument("--eligible-tasks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = args.showcase_js.read_bytes()
    source = raw.decode()
    version = js_json(source, "TASK_VERSION")
    models = js_json(source, "MODEL_RUNS")
    runs = js_json(source, "GENERATED_RUNS")
    eligible = json.loads(args.eligible_tasks.read_text())["tasks"]
    if len(eligible) != len(set(eligible)) or not all(re.fullmatch(r"\d{3}", t) for t in eligible):
        raise ValueError("Eligible task IDs must be unique three-digit strings")
    rows = []
    for task in eligible:
        records = []
        for model in models:
            item = runs.get(model["modelId"], {}).get(task)
            if item is None:
                continue
            score = item["score"]
            if item.get("taskVersion") != version:
                raise ValueError("Refusing to mix task versions")
            if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid score for {task}/{model['modelId']}")
            records.append({"model_config": model["modelId"], "score": score,
                            "source_archive": model["sourceArchive"]})
        n = len(records)
        passes = sum(r["score"] == 1.0 for r in records)
        rows.append({"task_id": task, "observations": n, "full_passes": passes,
                     "pass_rate": passes / n if n else None,
                     "mean_partial_score": sum(r["score"] for r in records) / n if n else None,
                     "score_version": version if n else None, "records": records})
    original_order = {task: i for i, task in enumerate(eligible)}
    rows.sort(key=lambda r: (r["pass_rate"] is None,
                            -(r["pass_rate"] or 0), -(r["mean_partial_score"] or 0),
                            original_order[r["task_id"]]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://osworld-v2.xlang.ai/static/js/taskShowcaseData.js",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "score_version": version, "evaluation_version": "v2026.08.08",
        "policy": "Known pass rate descending; mean partial score descending for ties; unknown tasks retain eligible order at the end.",
        "pass_definition": "score == 1.0; each published model/configuration counts once",
        "limitations": [
            "Old-version, selected-showcase results are scheduling proxies, not current-version or Qwen27B pass probabilities.",
            "Six configurations represent five model families; Sonnet 4.6 medium and max are separate observations.",
            "Unknown scores are null, never failures; aggregate leaderboard scores cannot rank individual tasks.",
            "The full official trajectory download requires additional access; the current account lacks that access.",
        ],
        "ranked_known_tasks": sum(r["observations"] > 0 for r in rows),
        "task_count": len(rows), "tasks": rows,
    }
    (args.output_dir / "public_task_ranking.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "test_public_ranked.json").write_text(json.dumps({"tasks": [r["task_id"] for r in rows]}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "records"} for r in rows if r["observations"]], indent=2))


if __name__ == "__main__":
    main()
