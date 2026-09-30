#!/usr/bin/env python3
"""Freeze or audit answer keys for the already-frozen DFlash public inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datasets import load_dataset

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--workload", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--verify-existing", action="store_true")
a = p.parse_args()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
m = json.loads((a.workload / "manifest.json").read_text())
assert sha(a.reference) == m["benchmark_source_sha256"]
spec = importlib.util.spec_from_file_location("official", a.reference)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
if not a.verify_existing:
    a.output.mkdir()
audit = a.output / "provenance.json"
assert not audit.exists()
result = {"status": "answer_keys_verified_against_pinned_cache",
          "input_manifest_sha256": sha(a.workload / "manifest.json"),
          "script_sha256": sha(Path(__file__)), "datasets": {}}
for name, item in m["datasets"].items():
    source = a.workload / item["file"]
    assert sha(source) == item["sha256"]
    ds = load_dataset(*item["load_args"], revision=item["revision"], **item["load_kwargs"])
    assert len(ds) == item["source_rows"]
    caches = [Path(record["filename"]) for record in ds.cache_files]
    # Offline datasets may fall back to a different cached revision. Reject it.
    assert caches and all(item["revision"] in path.parts for path in caches)
    rows = []
    for row in map(json.loads, source.read_text().splitlines()):
        reference = ds[row["source_index"]]
        formatted = official.DATASETS[name]["format"](reference)
        question = formatted[0] if official.DATASETS[name].get("multi_turn") else formatted
        assert row["messages"] == [{"role": "user", "content": question}]
        rows.append({"ordinal": row["ordinal"], "source_index": row["source_index"],
                     "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                     "reference": reference})
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path = a.output / (name + ".jsonl")
    if a.verify_existing:
        assert path.read_text() == contents, f"answer key changed: {name}"
    else:
        path.write_text(contents)
    result["datasets"][name] = {"file": path.name, "sha256": sha(path),
                                "revision": item["revision"], "rows": len(rows),
                                "cache_sha256": {str(path): sha(path) for path in caches}}
    print(name, len(rows), "verified", flush=True)
audit.write_text(json.dumps(result, indent=2) + "\n")
