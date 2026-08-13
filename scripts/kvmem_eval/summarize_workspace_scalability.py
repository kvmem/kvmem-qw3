#!/usr/bin/env python3
"""Summarize the six-point KVMem workspace-scalability table."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GIB = float(1 << 30)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def label(tokens: int) -> str:
    names = {
        262144: "256K",
        524288: "512K",
        1048576: "1M",
        2097152: "2M",
        4194304: "4M",
        9998336: "10M",
    }
    return names.get(tokens, str(tokens))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prefill-warmup", type=int, default=2)
    ap.add_argument("--query-warmup", type=int, default=1)
    ap.add_argument(
        "--expected-index", choices=("adaptive", "mean-k"), default="adaptive"
    )
    args = ap.parse_args()

    data: dict[str, Any] = json.loads(args.input.read_text())
    resources = {int(row["turn"]): row for row in data.get("resources", [])}
    prefill: dict[int, list[dict[str, Any]]] = defaultdict(list)
    queries: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("prefill_probes", []):
        prefill[int(row["turn"])].append(row)
    for row in data.get("repeated_queries", []):
        queries[int(row["turn"])].append(row)

    expected = [262144, 524288, 1048576, 2097152, 4194304, 9998336]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for turn, workspace in enumerate(expected):
        resource = resources.get(turn)
        if resource is None:
            errors.append(f"turn {turn}: missing resource snapshot")
            continue
        if int(resource["workspace_tokens"]) != workspace:
            errors.append(
                f"turn {turn}: workspace={resource['workspace_tokens']} expected={workspace}"
            )
        expected_adaptive = 1 if args.expected_index == "adaptive" else 0
        if int(resource["index_cpu"]) != 1 or int(
            resource["index_adaptive"]
        ) != expected_adaptive:
            errors.append(
                f"turn {turn}: index is not CPU {args.expected_index}"
            )

        p_rows = sorted(prefill.get(turn, []), key=lambda row: int(row["probe"]))
        q_rows = sorted(queries.get(turn, []), key=lambda row: int(row["query"]))
        p_measured = p_rows[args.prefill_warmup :]
        q_measured = q_rows[args.query_warmup :]
        if not p_measured:
            errors.append(f"turn {turn}: no measured prefill probes")
            continue
        if not q_measured:
            errors.append(f"turn {turn}: no measured query probes")
            continue

        prefill_tps = [float(row["prefill_tps"]) for row in p_measured]
        ttft_ms = [
            float(row["first_token_ms"])
            for row in q_measured
            if row.get("first_token_ms") is not None
            and float(row["first_token_ms"]) >= 0.0
        ]
        retrieval_ms = [float(row["score_ms"]) for row in q_measured]
        if len(ttft_ms) != len(q_measured):
            errors.append(
                f"turn {turn}: missing direct first_token_ms for one or more query probes"
            )
            continue
        decode_tps = [
            1000.0 * int(row["decoded"]) / float(row["decode_ms"])
            for row in q_measured
            if int(row["decoded"]) > 0 and float(row["decode_ms"]) > 0
        ]
        if not decode_tps:
            errors.append(f"turn {turn}: no valid decode probes")
            continue

        rows.append(
            {
                "workspace": label(workspace),
                "workspace_tokens": workspace,
                "gpu_state_gib": resource["gpu_kv_capacity_bytes"] / GIB,
                "host_state_gib": (
                    resource["cpu_kv_bytes"] + resource["index_logical_bytes"]
                ) / GIB,
                "nvme_state_gib": resource["nvme_kv_bytes"] / GIB,
                "ttft_ms_p50": statistics.median(ttft_ms),
                "ttft_ms_p95": percentile(ttft_ms, 0.95),
                "gpu_kv_pool_gib": resource["gpu_kv_capacity_bytes"] / GIB,
                "gpu_kv_used_gib": resource["gpu_kv_used_bytes"] / GIB,
                "retrieval_index_gib": resource["index_logical_bytes"] / GIB,
                "retrieval_latency_ms_p50": statistics.median(retrieval_ms),
                "retrieval_latency_ms_p95": percentile(retrieval_ms, 0.95),
                "host_nvme_kv_gib": resource["persistent_authority_bytes"] / GIB,
                "external_physical_gib": resource["external_physical_bytes"] / GIB,
                "prefill_tps_p50": statistics.median(prefill_tps),
                "prefill_tps_p05": percentile(prefill_tps, 0.05),
                "decode_tps_p50": statistics.median(decode_tps),
                "decode_tps_p05": percentile(decode_tps, 0.05),
                "index_prototypes": int(resource["index_prototypes"]),
                "indexed_blocks": int(resource["indexed_blocks"]),
                "prefill_samples": len(prefill_tps),
                "query_samples": len(retrieval_ms),
            }
        )

    result = {
        "schema": "qw3.workspace_scalability.v2",
        "source": str(args.input),
        "errors": errors,
        "rows": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("| Workspace | GPU State (GiB) | Host State (GiB) | NVMe State (GiB) | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['workspace']} | {row['gpu_state_gib']:.3f} | "
            f"{row['host_state_gib']:.3f} | "
            f"{row['nvme_state_gib']:.3f} | "
            f"{row['ttft_ms_p50'] / 1000.0:.3f} | "
            f"{row['prefill_tps_p50']:.1f} | {row['decode_tps_p50']:.2f} |"
        )
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
