#!/usr/bin/env python3
"""Require qw3 token counts to match the archived AgentLongBench counts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-tokenize", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def load_worker(repo: Path):
    source = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("agentlongbench_fullcontext", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import canonical worker: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qw3_tokens(api_base: str, prompt: str) -> list[int]:
    url = api_base.removesuffix("/v1").rstrip("/") + "/tokenize"
    request = urllib.request.Request(
        url,
        data=json.dumps({"content": prompt, "return_tokens": True}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or int(body.get("count", -1)) != len(tokens):
        raise RuntimeError(f"unexpected qw3 /tokenize response: {body.keys()}")
    return [int(token) for token in tokens]


def llama_tokens(executable: Path, model: Path, prompt: str) -> list[int]:
    command = [
        str(executable), "--model", str(model), "--ids", "--stdin",
        "--no-escape", "--log-disable", "--no-bos",
    ]
    proc = subprocess.run(
        command,
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed ({proc.returncode}): "
            + proc.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    result = json.loads(proc.stdout)
    if not isinstance(result, list):
        raise RuntimeError("llama-tokenize did not return an ID array")
    return [int(token) for token in result]


def main() -> None:
    args = parse_args()
    canonical = load_worker(args.benchmark_repo)
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    parity_mismatches = []
    archived_count_mismatches = []
    counts = []
    for index, row in enumerate(rows, start=1):
        prompt = canonical.full_context_prompt(row)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash != row.get("prompt_sha256"):
            raise RuntimeError(
                f"canonical prompt hash mismatch at sample {index}: "
                f"{prompt_hash} != {row.get('prompt_sha256')}"
            )
        qw3 = qw3_tokens(args.api_base, prompt)
        llama = llama_tokens(args.llama_tokenize, args.model, prompt)
        counts.append(len(qw3))
        if qw3 != llama:
            first_difference = next(
                (position for position, pair in enumerate(zip(qw3, llama)) if pair[0] != pair[1]),
                min(len(qw3), len(llama)),
            )
            parity_mismatches.append({
                "index": index,
                "stable_sample_id": row["stable_sample_id"],
                "qw3_count": len(qw3),
                "llama_count": len(llama),
                "first_difference": first_difference,
            })
        archived = int(row["prompt_tokens"])
        if len(qw3) != archived:
            archived_count_mismatches.append({
                "index": index,
                "stable_sample_id": row["stable_sample_id"],
                "archived": archived,
                "current": len(qw3),
                "delta": len(qw3) - archived,
            })
        if index % 25 == 0 or index == len(rows):
            print(
                f"tokenizer validation: {index}/{len(rows)}, "
                f"llama_id_mismatches={len(parity_mismatches)}",
                flush=True,
            )

    report = {
        "passed": (
            not parity_mismatches
            and not archived_count_mismatches
            and len(rows) == 250
        ),
        "samples": len(rows),
        "llama_id_matched": len(rows) - len(parity_mismatches),
        "llama_id_mismatches": parity_mismatches,
        "archived_count_note": "Exact count parity with the canonical archived prompts is required.",
        "archived_count_matched": len(rows) - len(archived_count_mismatches),
        "archived_count_mismatches": archived_count_mismatches,
        "actual_min": min(counts) if counts else None,
        "actual_max": max(counts) if counts else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(
            "tokenizer validation failed: "
            f"{len(parity_mismatches)}/{len(rows)} llama ID mismatches, "
            f"{len(archived_count_mismatches)}/{len(rows)} archived-count mismatches"
        )
    print(
        "tokenizer validation passed: 250/250 prompt hashes, archived counts, "
        "and llama.cpp token IDs are exact",
        flush=True,
    )


if __name__ == "__main__":
    main()
