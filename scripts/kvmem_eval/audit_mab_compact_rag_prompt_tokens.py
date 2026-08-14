#!/usr/bin/env python3
"""Recount persisted MAB Compact+RAG prompts with the serving tokenizer.

The original RAG chunks remain frozen.  This audit reconstructs the exact
selected source text for every persisted answer, verifies the shared-prefix
hash, and recounts the complete Qwen chat prompt with the current
``TokenCounter`` implementation.  Run it with the same Python environment as
vLLM so Unicode pre-tokenization is identical to serving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_memoryagentbench_baselines as mab  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    method_root = workspace / "methods" / "compact-rag"
    counter = mab.TokenCounter(
        Path(
            "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
            "Qwen3.6-27B-FP8"
        )
    )
    rows: list[dict] = []
    for result_path in sorted((method_root / "rows").glob("*/results.jsonl")):
        row_name = result_path.parent.name
        records = load_jsonl(result_path)
        compact = load_json(workspace / "shared" / "compact" / f"{row_name}.json")
        rag = load_json(workspace / "shared" / "rag" / f"{row_name}.json")
        prepared_dir = workspace / "prepared" / "rows" / row_name
        manifest = load_json(prepared_dir / "prepare_manifest.json")
        prefix = Path(manifest["archive_prefix"]).read_text(encoding="utf-8")
        prepare_module = mab.load_module(
            mab.PREPARE_SCRIPT, "_mab_prompt_token_audit_prepare"
        )
        context = mab.extract_memorized_context(prefix, prepare_module.SYSTEM_MESSAGE)
        if mab.sha256_text(context) != compact.get("context_sha256"):
            raise RuntimeError(f"{row_name}: compact context hash mismatch")
        common = mab.compact_summary_only_content(str(compact.get("summary") or ""))

        # Persisted answers record the exact aligned common-prefix hash.  New
        # tokenization can choose different newline padding, so recover the
        # historical padding by checking the small bounded newline domain.
        expected_aligned = {str(record["aligned_shared_context_sha256"]) for record in records}
        if len(expected_aligned) != 1:
            raise RuntimeError(f"{row_name}: multiple aligned shared-prefix hashes")
        expected_hash = next(iter(expected_aligned))
        aligned = None
        for newline_count in range(4097):
            candidate = common + ("\n" * newline_count)
            if mab.sha256_text(candidate) == expected_hash:
                aligned = candidate
                break
        if aligned is None:
            raise RuntimeError(f"{row_name}: cannot reconstruct aligned shared prefix")

        retrievals = {
            int(item["question_index"]): item for item in rag["retrievals"]
        }
        questions = load_json(prepared_dir / "questions.json")
        for record in records:
            question_index = int(record["question_index"])
            retrieval = retrievals[question_index]
            selected_count = int(record["retrieval"]["selected_count"])
            selected = sorted(
                (
                    block for block in retrieval["retrieved_blocks"]
                    if int(block["rank"]) <= selected_count
                ),
                key=lambda block: int(block["chunk_index"]),
            )
            selected_ids = [str(block["block_id"]) for block in selected]
            if selected_ids != list(record["retrieval"]["retrieved_block_ids"]):
                raise RuntimeError(
                    f"{row_name} q={question_index}: selected block IDs differ"
                )
            variable = mab.rag_question_suffix(
                selected, questions[question_index], include_metadata=False
            )
            prompt = mab.qwen_chat_no_thinking(
                prepare_module.SYSTEM_MESSAGE, aligned + "\n" + variable
            )
            serving_tokens = counter.count(prompt)
            persisted_tokens = int(record["prompt_tokens_local"])
            rows.append({
                "row": row_name,
                "split": record["split"],
                "source": record["source"],
                "dataset_row": int(record["dataset_row"]),
                "question_index": question_index,
                "qa_pair_id": record.get("qa_pair_id"),
                "persisted_prompt_tokens": persisted_tokens,
                "serving_prompt_tokens": serving_tokens,
                "delta_tokens": serving_tokens - persisted_tokens,
                "over_cap": serving_tokens > args.cap,
            })
        print(f"audited {row_name}: {len(records)} prompts", flush=True)

    over = [row for row in rows if row["over_cap"]]
    deltas = [int(row["delta_tokens"]) for row in rows]
    report = {
        "schema_version": 1,
        "workspace": str(workspace),
        "method": "compact-rag",
        "cap": args.cap,
        "prompt_tokenizer": counter.snapshot(),
        "questions": len(rows),
        "over_cap_questions": len(over),
        "over_cap_rows": len({row["row"] for row in over}),
        "max_serving_prompt_tokens": max(
            int(row["serving_prompt_tokens"]) for row in rows
        ),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "delta_mean": sum(deltas) / len(deltas),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
