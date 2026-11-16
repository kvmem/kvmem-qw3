#!/usr/bin/env python3
"""Describe changed selected blocks for matched-input BFCL A/B requests."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer


def frames(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "meta":
            current = row["trace_tag"]
            result[current] = {}
        elif current is not None and row.get("sel"):
            result[current][row["p0"], row["nt"]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    root = args.output_dir
    maps = {}
    records = []
    comparisons = [json.loads(s) for s in (root / "matched_input_selection_comparison.jsonl").read_text().splitlines()]
    for comparison in comparisons:
        if not comparison["selected_block_sets_changed"]:
            continue
        index = comparison["id"].rsplit("_", 1)[1]
        for group in ("off", "on"):
            key = index, group
            if key not in maps:
                paths = list((root / f"sample_{index}").glob(group + "*/result.json"))
                completed = [p.parent for p in paths if json.loads(p.read_text())["status"] == "completed"
                    and json.loads(p.read_text())["group"] == group]
                assert len(completed) == 1
                maps[key] = frames(completed[0] / "retrieval.jsonl")
            label = f"bfcl-{index}-{group}-t{comparison['turn_index']}-s{comparison['step']}"
            frame = maps[key][label]
            for p0, nt in comparison[f"only_{group}_selected_blocks"]:
                row = frame[p0, nt]
                text = tokenizer.decode(row["tok"], skip_special_tokens=False)
                numeric = bool(re.fullmatch(r"[0-9eE+.,\s\[\]-]+", text)) and text.count(",") >= 2
                records.append({"id": comparison["id"], "turn_index": comparison["turn_index"],
                    "step": comparison["step"], "group": group, "p0": p0, "nt": nt,
                    "text": text, "numeric_array_fragment_heuristic": numeric})
    summary = {"scope": "Changed selected block occurrences, only for identical normalized input and available full-selection snapshots. Counts are per occurrence, not unique physical blocks. Numeric array detection is a text heuristic, not proof that a block is irrelevant; snapshots do not cover every intermediate pressure window.",
        "comparison_sha256": hashlib.sha256((root / "matched_input_selection_comparison.jsonl").read_bytes()).hexdigest(),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "matched_requests_with_changed_snapshots": sum(c["selected_block_sets_changed"] is True for c in comparisons),
        "changed_block_occurrences": len(records),
        "numeric_array_fragment_occurrences": sum(r["numeric_array_fragment_heuristic"] for r in records),
        "other_block_occurrences": sum(not r["numeric_array_fragment_heuristic"] for r in records),
        "groups": {g: {"changed_block_occurrences": sum(r["group"] == g for r in records),
            "numeric_array_fragment_occurrences": sum(r["group"] == g and r["numeric_array_fragment_heuristic"] for r in records)}
            for g in ("off", "on")}}
    (root / "changed_blocks_diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (root / "changed_block_texts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
