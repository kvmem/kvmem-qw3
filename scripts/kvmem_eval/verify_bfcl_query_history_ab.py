#!/usr/bin/env python3
"""Independently validate saved BFCL A/B artifacts; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output_dir.resolve()
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "artifacts/bfcl_long_context_audit_20261015/source"
sys.path.insert(0, str(SOURCE / "upstream"))
sys.path.insert(0, str(REPO / "scripts/kvmem_eval"))
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from run_bfcl_query_history_ab import input_identity
from summarize_bfcl_query_history_ab import selections


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan = json.loads((ROOT / "plan.json").read_text())
assert len(plan["ids"]) == len(set(plan["ids"]))
assert plan["groups"] == ["off", "on"]
completed_paths = {}
infrastructure_attempts = []
for path in ROOT.glob("sample_*/*/result.json"):
    result = json.loads(path.read_text())
    if result["status"] != "completed":
        infrastructure_attempts.append(str(path))
        continue
    key = result["id"], result["group"]
    assert key not in completed_paths, key
    completed_paths[key] = path.parent
assert len(completed_paths) == len(plan["ids"]) * len(plan["groups"])
assert sha(Path(plan["binary"])) == plan["binary_sha256"]
assert sha(ROOT / "runner_snapshot.py") == json.loads((ROOT / "pre_run_selection.json").read_text())["runner_sha256"]
model = json.loads((ROOT / "model_identity.json").read_text())
assert Path(model["path"]).stat().st_size == model["bytes"]
assert Path(model["path"]).stat().st_mtime_ns == model["mtime_ns"]
for path, expected in json.loads((ROOT / "source_hashes_before.json").read_text()).items():
    assert sha(REPO / path) == expected, path
for item in json.loads((ROOT / "bfcl_source_manifest.json").read_text())["files"]:
    assert sha(Path(item["path"])) == item["sha256"], item["path"]
rows = {r["id"]: r for r in jsonl(SOURCE / "BFCL_v4_multi_turn_long_context.json")}
gold = {r["id"]: r for r in jsonl(SOURCE / "upstream/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json")}
checks = []
all_commands = set()
all_environments = set()
pair_orders = []
for sequence, index in enumerate(plan["ids"]):
    sid = f"multi_turn_long_context_{index}"
    commands = {}
    starts = {}
    for group in plan["groups"]:
        directory = completed_paths[sid, group]
        result = json.loads((directory / "result.json").read_text())
        starts[group] = json.loads((directory / "process.json").read_text())["started_unix"]
        assert result["status"] == "completed"
        calls = json.loads((directory / "decoded_calls.json").read_text())
        assert len(calls) == len(rows[sid]["question"]) == result["query_count"]
        assert 0 < len(result["turns"]) <= result["query_count"]
        if len(result["turns"]) < result["query_count"]:
            assert result["turns"][-1]["step_limit_reached"] or result["turns"][-1].get("context_limit_reached")
        if any(t.get("context_limit_reached") for t in result["turns"]):
            limit = json.loads((directory / "context_limit.json").read_text())
            assert "prompt exceeds KV context" in limit["error"]
            assert "request_limit=131072" in limit["error"]
            assert limit["model_failure_no_retry"]
        regrade = multi_turn_checker(calls, gold[sid]["ground_truth"], rows[sid],
            "multi_turn_long_context", f"independent_artifact_check_{group}_{index}")
        assert regrade["valid"] == result["official_grade"]["valid"]
        requests = jsonl(directory / "requests.jsonl")
        assert len(requests) == result["requests"]
        assert sum(r["usage"]["completion_tokens"] for r in requests) == result["completion_tokens"]
        commits = []
        for r in requests:
            payload, usage = r["request"], r["usage"]
            assert input_identity(payload["messages"], payload["tools"]) == r["input_digest"]
            assert 0 < usage["completion_tokens"] <= plan["max_tokens"]
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert usage["prompt_tokens"] < 131072
            commits.extend(r["query_history_commits"])
            assert len(r["query_history_commits"]) == (1 if group == "on" else 0)
        rounds = [int(re.search(r"committed_round=(\d+)", c).group(1)) for c in commits]
        if group == "on":
            assert rounds == list(range(1, len(requests) + 1))
            assert all("observed=1" in c for c in commits)
        snapshots = selections(directory / "retrieval.jsonl")
        for snapshot in snapshots.values():
            meta = snapshot["meta"]
            assert meta["budget_blocks"] == 1024 and meta["block_tokens"] == 32
            assert len(snapshot["selected"]) == len(set(snapshot["selected"])) == meta["selected"]
            assert meta["selected"] <= 1024
            assert meta["query_ready"] and meta["index_ready"]
            assert meta["scorer_used"] == "mean-k"
        launch = json.loads((directory / "launch.json").read_text())
        command = launch["command"]
        normalized = list(command)
        normalized[normalized.index("--port") + 1] = "<PORT>"
        normalized[normalized.index("--kvmem-query-history") + 1] = "<SWITCH>"
        commands[group] = normalized
        all_commands.add(tuple(normalized))
        environment = dict(launch["environment"])
        environment.pop("QW3_KVMEM_DUMP_SCORES", None)
        all_environments.add(json.dumps(environment, sort_keys=True))
        checks.append({"id": sid, "group": group, "official_score_reproduced": True,
            "query_count": result["query_count"], "request_count": len(requests),
            "query_history_enabled": group == "on", "history_lifecycle_matches_switch": True,
            "selection_snapshots": len(snapshots)})
    assert commands["off"] == commands["on"], sid
    expected_order = plan["groups"] if sequence % 2 == 0 else list(reversed(plan["groups"]))
    assert sorted(starts, key=starts.get) == expected_order, (sid, starts, expected_order)
    pair_orders.append({"id": sid, "order": expected_order})
summary = json.loads((ROOT / "summary.json").read_text())
assert len(all_commands) == len(all_environments) == 1
assert summary["all_planned_episodes_completed"]
expected_queries = sum(len(rows[f"multi_turn_long_context_{i}"]["question"]) for i in plan["ids"])
for group in plan["groups"]:
    completed_turns = sum(len(json.loads((p / "result.json").read_text())["turns"])
                          for (sid, g), p in completed_paths.items() if g == group)
    assert summary["groups"][group]["turns_completed"] == completed_turns <= expected_queries
amendment_path = ROOT / "protocol_amendment.json"
if amendment_path.exists():
    amendment = json.loads(amendment_path.read_text())
    assert sha(Path(amendment["runner_snapshot"])) == amendment["runner_sha256"]
    assert amendment["no_model_failure_rerun"] and amendment["inference_parameters_unchanged"]
reuse_path = ROOT / "reuse_manifest.json"
if reuse_path.exists():
    reuse = json.loads(reuse_path.read_text())
    assert set(reuse["reused_ids"]) | set(reuse["new_ids"]) == set(plan["ids"])
    assert not set(reuse["reused_ids"]) & set(reuse["new_ids"])
    for entry in reuse["files"]:
        assert sha(Path(entry["source"])) == sha(Path(entry["copy"])) == entry["sha256"], entry["source"]
validation = {"passed": True, "episodes": checks, "only_history_switch_changes_between_pair_commands": True,
    "all_runs_share_common_command_parameters_and_environment": True,
    "pair_order_alternates_as_planned": True, "pair_orders": pair_orders,
    "reused_artifact_hashes_verified": reuse_path.exists(),
    "infrastructure_attempts_preserved": infrastructure_attempts,
    "planned_queries_per_group": expected_queries,
    "all_saved_official_scores_independently_reproduced": True,
    "model_stat_unchanged": True, "engine_and_bfcl_source_hashes_unchanged": True,
    "token_usage_consistent": True, "history_rounds_contiguous_and_positive": True,
    "selection_snapshots_within_32k_budget": True}
(ROOT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": True, "episodes": len(checks), "planned_queries_per_group": expected_queries}, indent=2))
