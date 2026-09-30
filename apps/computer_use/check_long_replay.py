"""Verify matched historical-image answers and actual above-budget retrieval.

Run after compare_replay.py has ended. Transport completion alone is not a
correct answer. This checker covers the frozen positive/negative visual corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path

from long_history_probe import EXPECTED, answer


def check(run):
    summary = json.loads((run / "summary.json").read_text())
    assert summary["prefix_reuse"] is False
    corpus = Path(summary["corpus"])
    cases = {case["name"]: case for case in summary["cases"]}
    assert set(cases) == {"present_image", "absent_image"}
    bodies = {}
    for name, case in cases.items():
        raw = (corpus / case["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == case["sha256"], "Frozen corpus changed"
        bodies[name] = json.loads(raw)
    checks = []
    for profile in summary["profiles"]:
        label = f"{profile['method']}-kvmem-{profile['kvmem']}"
        directory = run / label
        if profile["status"] != "completed":
            checks.append({"profile": label, "passed": False, "error": profile.get("error", profile["status"])})
            continue
        manifest = json.loads((directory / "engine/manifest.json").read_text())
        assert manifest["prefix_reuse"] is False
        metadata = []
        if profile["kvmem"] == "on":
            metadata = [json.loads(line) for line in (directory / "engine/retrieval.jsonl").read_text().splitlines()]
            metadata = [row for row in metadata if row.get("type") == "meta"]
        counts = {}
        for trial in profile["runs"]:
            name = trial["case"]
            key = (name, trial["repetition"])
            counts[key] = counts.get(key, 0) + 1
            item = {"profile": label, "case": name, "repetition": trial["repetition"], "passed": False,
                    "elapsed_seconds": trial["elapsed_seconds"], "prompt_tokens": trial.get("usage", {}).get("prompt_tokens")}
            try:
                assert trial["status"] == "completed", trial.get("error", trial["status"])
                expected = EXPECTED if name == "present_image" else dict.fromkeys(EXPECTED)
                parsed = answer(trial["text"])
                item["answer"] = parsed
                assert parsed == expected, "Historical visual fields differ"
                assert not trial["tool_calls"], "Unexpected tool call in answer-only diagnostic"
                assert trial["usage"]["prompt_tokens"] == cases[name]["prompt_tokens"] > 131072
                if profile["kvmem"] == "on":
                    candidates = [m for m in metadata if m["trace_tag"] == bodies[name]["kvmem_trace_tag"]]
                    assert len(candidates) > trial["repetition"], "Missing per-request retrieval evidence"
                    meta = candidates[trial["repetition"]]
                    item["retrieval"] = meta
                    assert meta["prompt_tokens"] == trial["usage"]["prompt_tokens"]
                    assert meta["selected"] < meta["block_count"], "No sparse selection occurred"
                    assert meta["scorer_requested"] == meta["scorer_used"] == "sub-block-mean-k-max"
                    assert meta["scorer_fallback_reason"] == "none"
                    assert meta["query_ready"] and meta["index_ready"]
                    assert meta["query_expected"] == meta["query_captured"]
                    assert meta["qc_total_blocks"] == meta["qc_captured_blocks"]
                item["passed"] = True
            except (AssertionError, KeyError, ValueError) as exc:
                item["error"] = repr(exc)
            checks.append(item)
        repetitions = {t["repetition"] for t in profile["runs"]}
        paired = bool(repetitions) and all(counts.get((name, repetition)) == 1
                                           for name in cases for repetition in repetitions)
        if not paired:
            checks.append({"profile": label, "passed": False, "error": "Positive/negative repetitions are missing or duplicated"})
    result = {"kind": "frozen_long_visual_output_and_retrieval_checks", "checks": checks,
              "passed": bool(checks) and all(item["passed"] for item in checks)}
    (run / "visual-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"run": str(run), "passed": result["passed"], "passed_checks": sum(c["passed"] for c in checks),
                      "checks": len(checks)}))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    results = [check(path) for path in parser.parse_args().run_dirs]
    raise SystemExit(0 if all(results) else 1)
