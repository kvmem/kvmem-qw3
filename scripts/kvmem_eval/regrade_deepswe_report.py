#!/usr/bin/env python3
"""Regrade an existing guided DeepSWE report from hidden-verifier metrics."""

from __future__ import annotations

import argparse
import json
import pathlib

from run_deepswe_guided_smoke import server_evidence, verifier_grade


def regrade(document: dict) -> dict:
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("report has no tasks list")
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("report task is not an object")
        grade = verifier_grade(task.get("metrics"))
        task["verifier_grade"] = grade
        artifact_dir = task.get("artifact_dir")
        if isinstance(artifact_dir, str):
            task["server_evidence"] = server_evidence(
                pathlib.Path(artifact_dir)
            )
        task["passed"] = (
            task.get("arm_status") == 0
            and task.get("verifier_status") == 0
            and grade["passed"]
        )
    document["schema"] = "qw3.deepswe_guided_kvmem_smoke.v2"
    document["passed"] = all(task["passed"] for task in tasks)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    path = pathlib.Path(args.report).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    regrade(document)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(path),
        "passed": document["passed"],
        "tasks": [
            {
                "task_id": task.get("task_id"),
                "passed": task["passed"],
                "verifier_grade": task["verifier_grade"],
            }
            for task in document["tasks"]
        ],
    }, indent=2, ensure_ascii=False))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
