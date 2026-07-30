#!/usr/bin/env python3
"""Cluster tokens inside one KVMem block by raw content-Key direction.

The raw input is produced by the env-gated QW3_KVMEM_DUMP_RAW_K diagnostic.
No punctuation or message boundary participates in clustering.  A token-token
similarity is the mean cosine across every (normal-attention layer, KV head);
deterministic k-medoids then exposes one to four directions in the block.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    "/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl"
)
DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_MODEL = ROOT / "models/Qwen3.6-27B-Q8_0.gguf"
DEFAULT_INSPECT = ROOT / "build/qw3-inspect"
CHAT_PREFIX = "<|im_start|>user\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-k", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True, help="one-based")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--inspect", type=Path, default=DEFAULT_INSPECT)
    parser.add_argument("--max-prototypes", type=int, default=4)
    parser.add_argument(
        "--min-relative-gain",
        type=float,
        default=0.10,
        help="minimum marginal distortion reduction, as a fraction of D(K=1)",
    )
    return parser.parse_args()


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


def token_pieces(
    inspect: Path,
    model: Path,
    text: str,
    token_begin: int,
    token_end: int,
) -> list[bytes]:
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
    pieces: list[bytes] = []
    for raw_line in process.stdout:
        fields = raw_line.rstrip(b"\n").split(b"\t", 4)
        if len(fields) != 5:
            process.kill()
            raise RuntimeError(f"malformed tokenizer row: {raw_line[:160]!r}")
        index = int(fields[0])
        if token_begin <= index < token_end:
            pieces.append(bytes.fromhex(fields[4].decode()))
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"qw3-inspect failed: {stderr[-2000:]}")
    if len(pieces) != token_end - token_begin:
        raise RuntimeError(
            f"tokenizer returned {len(pieces)} requested pieces, "
            f"expected {token_end-token_begin}"
        )
    return pieces


def unpack_values(raw: bytes, dtype: str) -> tuple[float, ...]:
    if dtype == "float16":
        item_size = 2
        format_code = "e"
    elif dtype == "float32":
        item_size = 4
        format_code = "f"
    else:
        raise ValueError(f"unsupported raw-K dtype: {dtype}")
    if len(raw) % item_size:
        raise ValueError("raw-K dump has a partial element")
    return struct.unpack(f"<{len(raw)//item_size}{format_code}", raw)


def pairwise_key_similarity(
    values: tuple[float, ...],
    n_layers: int,
    n_tokens: int,
    n_heads: int,
    head_dim: int,
) -> list[list[float]]:
    per_token = n_heads * head_dim
    per_layer = n_tokens * per_token
    norms = [
        [[0.0 for _ in range(n_heads)] for _ in range(n_tokens)]
        for _ in range(n_layers)
    ]
    for layer in range(n_layers):
        layer_base = layer * per_layer
        for token in range(n_tokens):
            token_base = layer_base + token * per_token
            for head in range(n_heads):
                begin = token_base + head * head_dim
                norm2 = 0.0
                for dim in range(head_dim):
                    value = values[begin + dim]
                    norm2 += value * value
                norms[layer][token][head] = math.sqrt(max(norm2, 1e-30))

    similarity = [[1.0 if i == j else 0.0 for j in range(n_tokens)]
                  for i in range(n_tokens)]
    units = n_layers * n_heads
    for left in range(n_tokens):
        for right in range(left):
            total = 0.0
            for layer in range(n_layers):
                layer_base = layer * per_layer
                left_base = layer_base + left * per_token
                right_base = layer_base + right * per_token
                for head in range(n_heads):
                    left_head = left_base + head * head_dim
                    right_head = right_base + head * head_dim
                    dot = 0.0
                    for dim in range(head_dim):
                        dot += (
                            values[left_head + dim] *
                            values[right_head + dim]
                        )
                    total += dot / (
                        norms[layer][left][head] *
                        norms[layer][right][head]
                    )
            score = total / units
            similarity[left][right] = score
            similarity[right][left] = score
    return similarity


def assign(similarity: list[list[float]], medoids: list[int]) -> list[int]:
    return [
        max(range(len(medoids)), key=lambda cluster: similarity[token][medoids[cluster]])
        for token in range(len(similarity))
    ]


def kmedoids(
    similarity: list[list[float]], k: int
) -> tuple[list[int], list[int]]:
    n = len(similarity)
    first = max(range(n), key=lambda token: sum(similarity[token]))
    medoids = [first]
    while len(medoids) < k:
        candidate = min(
            (token for token in range(n) if token not in medoids),
            key=lambda token: max(similarity[token][medoid] for medoid in medoids),
        )
        medoids.append(candidate)
    for _ in range(50):
        assignments = assign(similarity, medoids)
        updated: list[int] = []
        for cluster in range(k):
            members = [
                token for token, owner in enumerate(assignments)
                if owner == cluster
            ]
            if not members:
                updated.append(medoids[cluster])
                continue
            updated.append(
                max(
                    members,
                    key=lambda candidate: sum(
                        similarity[candidate][member] for member in members
                    ),
                )
            )
        if updated == medoids:
            break
        medoids = updated
    return medoids, assign(similarity, medoids)


def distortion(
    similarity: list[list[float]],
    medoids: list[int],
    assignments: list[int],
) -> float:
    return sum(
        1.0 - similarity[token][medoids[assignments[token]]]
        for token in range(len(similarity))
    ) / len(similarity)


def silhouette(
    similarity: list[list[float]], assignments: list[int], k: int
) -> float:
    if k <= 1:
        return 0.0
    clusters = [
        [token for token, owner in enumerate(assignments) if owner == cluster]
        for cluster in range(k)
    ]
    values = []
    for token, owner in enumerate(assignments):
        own = [member for member in clusters[owner] if member != token]
        if not own:
            values.append(0.0)
            continue
        within = sum(1.0 - similarity[token][member] for member in own) / len(own)
        nearest = min(
            sum(1.0 - similarity[token][member] for member in members) /
            len(members)
            for cluster, members in enumerate(clusters)
            if cluster != owner and members
        )
        values.append(
            (nearest - within) / max(nearest, within, 1e-12)
        )
    return sum(values) / len(values)


def escaped(data: bytes, limit: int = 160) -> str:
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def member_fragments(members: list[int], pieces: list[bytes]) -> list[str]:
    fragments: list[str] = []
    start = members[0]
    previous = start
    for token in members[1:] + [10**9]:
        if token != previous + 1:
            data = b"".join(pieces[start : previous + 1])
            fragments.append(f"[{start},{previous+1}) {escaped(data)}")
            start = token
        previous = token
    return fragments


def main() -> None:
    args = parse_args()
    meta = json.loads(
        Path(str(args.raw_k) + ".json").read_text(encoding="utf-8")
    )
    begin = int(meta["token_begin"])
    end = int(meta["token_end"])
    n_tokens = end - begin
    n_layers = int(meta["n_layers"])
    n_heads = int(meta["n_kv_heads"])
    head_dim = int(meta["head_dim"])
    values = unpack_values(args.raw_k.read_bytes(), str(meta["dtype"]))
    expected = n_layers * n_tokens * n_heads * head_dim
    if len(values) != expected:
        raise RuntimeError(
            f"raw-K shape mismatch: {len(values)} values, expected {expected}"
        )

    canonical = load_module(
        args.benchmark_repo /
        "fullcontext/run_agentlongbench_fullcontext_worker.py",
        "agentlongbench_fullcontext",
    )
    sample = load_sample(args.dataset, args.sample_index)
    prompt = CHAT_PREFIX + canonical.full_context_prompt(sample)
    pieces = token_pieces(args.inspect, args.model, prompt, begin, end)

    similarity = pairwise_key_similarity(
        values, n_layers, n_tokens, n_heads, head_dim
    )
    results = []
    for k in range(1, min(args.max_prototypes, n_tokens) + 1):
        medoids, assignments = kmedoids(similarity, k)
        results.append(
            {
                "k": k,
                "medoids": medoids,
                "assignments": assignments,
                "distortion": distortion(similarity, medoids, assignments),
                "silhouette": silhouette(similarity, assignments, k),
            }
        )

    chosen = 1
    baseline = results[0]["distortion"]
    for index in range(1, len(results)):
        previous = results[index - 1]["distortion"]
        current = results[index]["distortion"]
        relative_gain = (previous - current) / max(baseline, 1e-12)
        sizes = [
            results[index]["assignments"].count(cluster)
            for cluster in range(results[index]["k"])
        ]
        if relative_gain < args.min_relative_gain or min(sizes) < 2:
            break
        chosen = results[index]["k"]

    print(f"sample_index: {args.sample_index}")
    print(f"stable_sample_id: {sample.get('stable_sample_id')}")
    print(f"question: {sample.get('question')}")
    print(
        f"raw content-K: tokens=[{begin},{end}), layers={n_layers}, "
        f"kv_heads={n_heads}, head_dim={head_dim}, dtype={meta['dtype']}"
    )
    print(f"block text: {escaped(b''.join(pieces), limit=500)}")
    print("\nprototype sweep:")
    previous = None
    for result in results:
        gain = (
            0.0 if previous is None else
            (previous - result["distortion"]) / max(baseline, 1e-12)
        )
        sizes = [
            result["assignments"].count(cluster)
            for cluster in range(result["k"])
        ]
        print(
            f"  K={result['k']}: distortion={result['distortion']:.6f}, "
            f"marginal_gain/D1={gain:.2%}, "
            f"silhouette={result['silhouette']:.4f}, sizes={sizes}"
        )
        previous = result["distortion"]
    print(
        f"\nchosen prototypes: K={chosen} "
        f"(min_relative_gain={args.min_relative_gain:.2f}, min_cluster_size=2)"
    )
    selected = results[chosen - 1]
    for cluster in range(chosen):
        members = [
            token for token, owner in enumerate(selected["assignments"])
            if owner == cluster
        ]
        medoid = selected["medoids"][cluster]
        print(
            f"  prototype {cluster}: size={len(members)}, "
            f"medoid=local{medoid}/global{begin+medoid}"
        )
        for fragment in member_fragments(members, pieces):
            print(f"    {fragment}")

    adjacent = sorted(
        (
            (1.0 - similarity[token - 1][token], token)
            for token in range(1, n_tokens)
        ),
        reverse=True,
    )[:6]
    print("\nlargest adjacent Key-direction changes (diagnostic only):")
    for distance, token in adjacent:
        left = escaped(pieces[token - 1], 60)
        right = escaped(pieces[token], 60)
        print(
            f"  local boundary {token-1}|{token}: "
            f"distance={distance:.6f}, {left!r} | {right!r}"
        )


if __name__ == "__main__":
    main()
