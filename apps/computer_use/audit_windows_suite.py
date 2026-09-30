"""Audit a terminal three-task Windows on/off suite against retained evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3

from windows_task import CASES

TASKS = ("notepad", "browser", "cross_app")


def audit(run, repetitions):
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] == "running":
        raise RuntimeError("The suite is live; wait for its process to exit")
    native = summary.get("kind") == "windows_native_real_model_snapshot_trials"
    if native and summary.get("restart"):
        raise ValueError("Crash/resume runs are separate from the repeated core-task matrix")
    if native:
        for source, digest in summary["helper_sha256"].items():
            assert hashlib.sha256((run / Path(source).name).read_bytes()).hexdigest() == digest
    expected = set(itertools.product(("on", "off"), TASKS, range(repetitions)))
    keys = [(t["kvmem"], t["case"], t["repetition"]) for t in summary["trials"]]
    checks = []
    binary_hashes, launcher_hashes = set(), set()
    source_commits = set()
    for trial in summary["trials"]:
        item = {key: trial[key] for key in ("kvmem", "case", "repetition")}
        item["passed"] = False
        try:
            assert trial["passed"], trial.get("error", "Trial did not pass")
            case = CASES[trial["case"]]
            assert summary["task_definitions"][trial["case"]] == case, "Task definition changed"
            directory = Path(trial["directory"])
            client = directory / "guest" if native else directory
            task = json.loads((client / "task/result.json").read_text())
            app = json.loads((client / "app/task.json").read_text())
            assert task["passed"] and app["status"] == "completed" and app["turn_status"] == "completed"
            assert task["task_id"] == app["id"]
            assert trial["initial_health"]["paused"] and task["state"]["desktop_paused"]
            if native:
                result = trial["native_result"]
                qualification = trial["native_contracts"]
                assert result["passed"] and result["driver_exit_code"] == 0 and result["shutdown_exit_code"] == 0
                assert result["run_id"] == trial["run_id"] and result["task"] == task
                assert qualification["passed"] and qualification["process_exit_code"] == 0
                assert result["bundle_sha256"] == qualification["bundle_sha256"] == summary["bundle_sha256"]
                assert (client / "native-ui.png").stat().st_size > 0
                assert task["state"]["observation"]["desktop_id"] == trial["initial_health"]["desktop_id"]
                samples = [json.loads(line) for line in (client / "client-resources.jsonl").read_text(encoding="utf-8-sig").splitlines()]
                assert samples and all("error" not in sample for sample in samples)
                names = {p["name"].lower() for sample in samples for p in sample["processes"]}
                assert "python" in names and "codex" in names, "Missing native client process samples"
            else:
                assert trial["final_health"]["paused"]
                assert trial["initial_health"]["desktop_id"] == trial["final_health"]["desktop_id"]
            actual = trial["verified_file"]
            assert actual == task["verified_file"] and app["files"] == [actual]
            assert actual["name"] == case["filename"]
            raw = actual["content"].encode("utf-8")
            assert len(raw) == actual["size"] and hashlib.sha256(raw).hexdigest() == actual["sha256"]
            if "json" in case:
                assert json.loads(actual["content"]) == case["json"]
            else:
                assert actual["content"] == case["content"]
            with sqlite3.connect((client / "app/desktop/desktop.sqlite").as_uri() + "?mode=ro", uri=True) as db:
                rows = list(db.execute("SELECT task,request,status FROM actions"))
            assert rows and all(row[0] == app["id"] for row in rows)
            assert not any(row[2] in ("dispatching", "uncertain") for row in rows)
            actions = [json.loads(row[1])["action"] for row in rows if row[2] == "completed"]
            if trial["case"] == "cross_app":
                assert {"kind": "key", "key": "CTRL+C"} in actions and {"kind": "key", "key": "CTRL+V"} in actions
                assert not any(a.get("kind") == "type" and case["content"] in a.get("text", "") for a in actions)
            manifest = json.loads((directory / "engine/manifest.json").read_text())
            command = manifest["command"]
            assert command[command.index("--speculative-method") + 1] == summary["method"] == "dflash2"
            assert "--mtp-chain" not in command and "--dflash-model" in command
            assert command[command.index("--ctx") + 1] == "1048576"
            assert command[command.index("--kv-dtype") + 1] == "fp8"
            assert ("--kvmem" in command) == (trial["kvmem"] == "on")
            assert manifest["prefix_reuse"] == (trial["kvmem"] == "on")
            log = (directory / "engine/engine.log").read_text()
            speculation = re.findall(r"native dflash_spec_summary: enabled=true .*?drafted=(\d+) accepted=(\d+).*?fallback=(\w+)", log)
            assert speculation and sum(int(s[0]) for s in speculation) > 0
            assert sum(int(s[1]) for s in speculation) > 0, "No DFlash draft was accepted"
            assert all(s[2] == "false" for s in speculation), "DFlash fallback occurred"
            binary_hashes.add(manifest["binary_sha256"])
            launcher_hashes.add(manifest["launcher_sha256"])
            source_commits.add(manifest["source_commit"])
            item.update(passed=True, completed_actions=len(actions), file_sha256=actual["sha256"],
                        dflash_drafted=sum(int(s[0]) for s in speculation),
                        dflash_accepted=sum(int(s[1]) for s in speculation), dflash_fallback_requests=0)
        except (AssertionError, KeyError, ValueError, OSError, sqlite3.Error) as exc:
            item["error"] = repr(exc)
        checks.append(item)
    matrix_complete = set(keys) == expected and len(keys) == len(expected)
    result = {"kind": "retained_windows_suite_evidence_audit", "checks": checks,
              "client_platform": "windows" if native else "linux",
              "matrix_complete": matrix_complete, "expected_trials": len(expected),
              "binary_hashes": sorted(binary_hashes), "launcher_hashes": sorted(launcher_hashes),
              "recorded_commits": sorted(source_commits),
              "passed": summary["status"] == "completed" and matrix_complete and all(c["passed"] for c in checks)
                        and len(binary_hashes) == len(launcher_hashes) == 1}
    (run / "evidence-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("passed", "matrix_complete", "expected_trials")}))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.run_dir.resolve(), args.repetitions) else 1)
