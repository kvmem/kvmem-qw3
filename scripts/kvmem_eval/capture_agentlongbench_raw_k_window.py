#!/usr/bin/env python3
"""Prefill an AgentLongBench prefix through a requested raw-K token window."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHAT_PREFIX = "<|im_start|>user\n"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sample(path: Path, one_based_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source, start=1):
            if index == one_based_index:
                return json.loads(line)
    raise IndexError(f"sample {one_based_index} is absent from {path}")


def token_end_byte(
    inspect: Path, model: Path, text: str, token_end: int
) -> int:
    process = subprocess.Popen(
        [str(inspect), "--tokenize-pieces-stdin", str(model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(text.encode("utf-8"))
    process.stdin.close()
    result = None
    for raw_line in process.stdout:
        fields = raw_line.rstrip(b"\n").split(b"\t", 4)
        if len(fields) != 5:
            process.kill()
            raise RuntimeError(f"malformed tokenizer row: {raw_line[:160]!r}")
        index = int(fields[0])
        if index == token_end - 1:
            result = int(fields[3])
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"qw3-inspect failed: {stderr[-2000:]}")
    if result is None:
        raise RuntimeError(f"prompt is shorter than token {token_end}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--token-end", type=int, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl"
        ),
    )
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=Path("/home/chaidi/AgentLongBench_Motivation"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/Qwen3.6-27B-Q8_0.gguf",
    )
    parser.add_argument("--inspect", type=Path, default=ROOT / "build/qw3-inspect")
    args = parser.parse_args()

    canonical = load_module(
        args.benchmark_repo /
        "fullcontext/run_agentlongbench_fullcontext_worker.py",
        "agentlongbench_fullcontext",
    )
    sample = load_sample(args.dataset, args.sample_index)
    canonical_prompt = canonical.full_context_prompt(sample)
    actual_prompt = CHAT_PREFIX + canonical_prompt
    byte_end = token_end_byte(
        args.inspect, args.model, actual_prompt, args.token_end
    )
    prefix_bytes = len(CHAT_PREFIX.encode("utf-8"))
    if byte_end <= prefix_bytes:
        raise RuntimeError("requested token end is inside the chat wrapper")
    content = actual_prompt.encode("utf-8")[
        prefix_bytes:byte_end
    ].decode("utf-8")

    url = args.api_base.removesuffix("/v1").rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model.name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=7200) as response:
        body = json.loads(response.read().decode("utf-8"))
    print(
        json.dumps(
            {
                "sample_index": args.sample_index,
                "stable_sample_id": sample.get("stable_sample_id"),
                "requested_token_end": args.token_end,
                "content_bytes": len(content.encode("utf-8")),
                "response": body,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
