#!/usr/bin/env python3
"""Audit the gated OSWorld V2 task set and rank safe smoke-test candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_strings(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--osworld-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "third_party" / "OSWorld-V2",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.osworld_root.resolve()
    output = args.output or root / "cache" / "qw3_task_audit.json"
    task_dir = root / "evaluation_examples" / "task_class"
    task_files = sorted(task_dir.glob("task_*.py"))
    if len(task_files) != 108:
        raise SystemExit(f"expected 108 task classes, found {len(task_files)}")

    # Several task modules validate these settings at import time even though
    # the audit never contacts the services. Keep the audit self-contained and
    # use non-secret placeholders only for static task loading.
    os.environ.setdefault("WEBSITE_HOST_SUFFIX", "site.hku.icu")
    os.environ.setdefault("GITLAB_URL", "http://gitlab.audit.invalid")
    os.environ.setdefault("GITLAB_PRIVATE_TOKEN", "audit-placeholder")
    os.environ.setdefault(
        "OSWORLD_FILE_BASE_URL", str(root / "cache" / "osworld_v2_assets")
    )

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(task_dir))
    from task_loader import load_task_from_file

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in task_files:
        task_id = path.stem.removeprefix("task_")
        source = path.read_text(encoding="utf-8")
        try:
            task = load_task_from_file(str(path))
            data = task.to_dict() if hasattr(task, "to_dict") else dict(task)
        except Exception as exc:  # keep auditing the rest of the pinned set
            errors.append({"id": task_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        text = "\n".join(walk_strings(data)) + "\n" + source
        upper = text.upper()
        evaluator_text = "\n".join(walk_strings(data.get("evaluator")))
        uses_llm_evaluator = bool(
            re.search(r"MODEL[_ -]?EVAL|GENERATE_(?:TEXT|CHAT)|LLM_METRIC", source.upper())
            or re.search(r"MODEL[_ -]?EVAL|LLM", evaluator_text.upper())
        )
        uses_gitlab = "GITLAB" in upper
        # Email addresses ending in gmail.com are ordinary task data. Count
        # only explicit Google credential/configuration dependencies here.
        uses_google_credentials = bool(
            re.search(
                r"SETTINGS/GOOGLE(?:DRIVE)?|GOOGLE_(?:ACCOUNT|CREDENTIALS)|"
                r"CLIENT_SECRETS\.JSON|GOOGLE_APPLICATION_CREDENTIALS",
                upper,
            )
        )
        uses_lark_credentials = "SETTINGS/LARK" in upper or "BENCH.XLANG.AI" in upper
        proxy = bool(data.get("proxy", False))
        user_simulator = bool(data.get("user_simulator"))
        related_apps = list(data.get("related_apps") or [])
        external_dependencies = [
            name
            for name, required in (
                ("proxy", proxy),
                ("gitlab", uses_gitlab),
                ("google_credentials", uses_google_credentials),
                ("lark_credentials", uses_lark_credentials),
                ("user_simulator", user_simulator),
                ("llm_evaluator", uses_llm_evaluator),
            )
            if required
        ]
        records.append(
            {
                "id": task_id,
                "instruction": str(data.get("instruction", "")),
                "related_apps": related_apps,
                "proxy": proxy,
                "external_dependencies": external_dependencies,
                "smoke_candidate": not external_dependencies and len(related_apps) <= 1,
            }
        )

    expected_ids = {f"{index:03d}" for index in range(1, 109)}
    loaded_ids = {record["id"] for record in records}
    smoke = [record for record in records if record["smoke_candidate"]]
    report = {
        "release": "v2026.08.08",
        "task_count": len(task_files),
        "loaded_count": len(records),
        "missing_ids": sorted(expected_ids - loaded_ids),
        "load_errors": errors,
        "dependency_counts": {
            dependency: sum(
                dependency in record["external_dependencies"] for record in records
            )
            for dependency in (
                "proxy",
                "gitlab",
                "google_credentials",
                "lark_credentials",
                "user_simulator",
                "llm_evaluator",
            )
        },
        "smoke_candidates": smoke,
        "tasks": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"audit={output}")
    print(f"loaded={len(records)}/108 errors={len(errors)} smoke_candidates={len(smoke)}")
    print("dependency_counts=" + json.dumps(report["dependency_counts"], sort_keys=True))
    if errors or loaded_ids != expected_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
