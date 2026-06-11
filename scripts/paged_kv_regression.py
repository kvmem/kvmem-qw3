#!/usr/bin/env python3
"""Paged-KV correctness and efficiency regression runner.

The script is intentionally dependency-free. It drives the local `qw3` CLI,
checks that generated text is invariant across page sizes, optionally compares
against a baseline binary, and records prefill/decode throughput parsed from
the native backend summary line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


GENERATE_RE = re.compile(
    r"native generate:\s+"
    r"prompt_tokens=(?P<prompt_tokens>\d+)\s+"
    r"prefill=(?P<prefill_s>[0-9.]+)s"
    r"(?:\s+\((?P<prefill_tps>[0-9.]+)\s+tok/s\))?\s+"
    r"decoded=(?P<decoded>\d+)\s+"
    r"decode=(?P<decode_s>[0-9.]+)s"
    r"(?:\s+\((?P<decode_tps>[0-9.]+)\s+tok/s\))?"
)


@dataclass
class RunResult:
    name: str
    binary: str
    command: List[str]
    prompt_name: str
    prompt_chars: int
    page_size: Optional[int]
    alloc_mode: Optional[str]
    kv_dtype: str
    max_tokens: int
    ctx_size: int
    returncode: int
    elapsed_s: float
    stdout: str
    stderr: str
    prompt_tokens: Optional[int] = None
    decoded: Optional[int] = None
    prefill_s: Optional[float] = None
    decode_s: Optional[float] = None
    prefill_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


@dataclass
class Comparison:
    kind: str
    prompt_name: str
    kv_dtype: str
    lhs_name: str
    rhs_name: str
    lhs_page_size: Optional[int]
    rhs_page_size: Optional[int]
    lhs_alloc_mode: Optional[str]
    rhs_alloc_mode: Optional[str]
    equal_stdout: bool
    lhs_prefix_chars: int
    rhs_prefix_chars: int
    common_prefix_chars: int


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.replace(",", " ").split():
        if not item:
            continue
        v = int(item)
        if v <= 0:
            raise argparse.ArgumentTypeError("list values must be positive integers")
        out.append(v)
    return out


def parse_str_list(raw: str) -> List[str]:
    return [x for x in raw.replace(",", " ").split() if x]


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def prompt_catalog() -> Dict[str, str]:
    boundary_words = [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "twenty one", "twenty two", "twenty three", "twenty four",
    ]
    code_prompt = (
        "Write a compact CUDA kernel checklist covering occupancy, memory "
        "coalescing, shared memory bank conflicts, and launch overhead."
    )
    chinese_prompt = "请用三句话解释分页 KV 缓存为什么适合连续批处理。"
    repeated = " ".join(["alpha beta gamma delta"] * 128)
    longish = " ".join(["token"] * 2300)
    return {
        "short": "Hello",
        "boundary_24_words": " ".join(boundary_words),
        "code": code_prompt,
        "chinese": chinese_prompt,
        "repeated_512_words": repeated,
        "longish_2300_words": longish,
    }


def select_prompts(names: Sequence[str]) -> Dict[str, str]:
    catalog = prompt_catalog()
    if not names:
        return {
            "short": catalog["short"],
            "boundary_24_words": catalog["boundary_24_words"],
            "code": catalog["code"],
            "chinese": catalog["chinese"],
        }
    unknown = [n for n in names if n not in catalog]
    if unknown:
        raise SystemExit(
            "unknown prompt(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(sorted(catalog))
        )
    return {n: catalog[n] for n in names}


def parse_metrics(text: str) -> Dict[str, Optional[float]]:
    matches = list(GENERATE_RE.finditer(text))
    if not matches:
        return {}
    m = matches[-1]
    out: Dict[str, Optional[float]] = {}
    out["prompt_tokens"] = int(m.group("prompt_tokens"))
    out["decoded"] = int(m.group("decoded"))
    out["prefill_s"] = float(m.group("prefill_s"))
    out["decode_s"] = float(m.group("decode_s"))
    out["prefill_tps"] = (
        float(m.group("prefill_tps")) if m.group("prefill_tps") else None
    )
    out["decode_tps"] = (
        float(m.group("decode_tps")) if m.group("decode_tps") else None
    )
    return out


def run_qw3(
    *,
    name: str,
    binary: Path,
    model: Path,
    prompt_name: str,
    prompt: str,
    page_size: Optional[int],
    alloc_mode: Optional[str],
    kv_dtype: str,
    max_tokens: int,
    ctx_size: int,
    timeout_s: int,
    prefill_chunk: Optional[int],
    extra_args: Sequence[str],
    extra_env: Dict[str, str],
) -> RunResult:
    env = os.environ.copy()
    env.update(extra_env)
    if page_size is None:
        env.pop("QW3_PAGED_KV_PAGE_SIZE", None)
    else:
        env["QW3_PAGED_KV_PAGE_SIZE"] = str(page_size)
    if alloc_mode is None:
        env.pop("QW3_PAGED_KV_ALLOC", None)
    else:
        env["QW3_PAGED_KV_ALLOC"] = alloc_mode

    cmd = [
        str(binary),
        "--backend",
        "qwen-native",
        "--model",
        str(model),
        "--native-heavy",
        "--raw",
        "--kv-dtype",
        kv_dtype,
        "--temp",
        "0",
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "-c",
        str(ctx_size),
    ]
    if prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(prefill_chunk)])
    cmd.extend(extra_args)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - start
        result = RunResult(
            name=name,
            binary=str(binary),
            command=cmd,
            prompt_name=prompt_name,
            prompt_chars=len(prompt),
            page_size=page_size,
            alloc_mode=alloc_mode,
            kv_dtype=kv_dtype,
            max_tokens=max_tokens,
            ctx_size=ctx_size,
            returncode=proc.returncode,
            elapsed_s=elapsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if proc.returncode != 0:
            result.error = f"process exited with code {proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        result = RunResult(
            name=name,
            binary=str(binary),
            command=cmd,
            prompt_name=prompt_name,
            prompt_chars=len(prompt),
            page_size=page_size,
            alloc_mode=alloc_mode,
            kv_dtype=kv_dtype,
            max_tokens=max_tokens,
            ctx_size=ctx_size,
            returncode=124,
            elapsed_s=elapsed,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"timeout after {timeout_s}s",
        )

    metrics = parse_metrics(result.stdout + "\n" + result.stderr)
    for key, value in metrics.items():
        setattr(result, key, value)
    return result


def compare_outputs(
    kind: str,
    lhs: RunResult,
    rhs: RunResult,
) -> Comparison:
    cp = common_prefix_len(lhs.stdout, rhs.stdout)
    return Comparison(
        kind=kind,
        prompt_name=lhs.prompt_name,
        kv_dtype=lhs.kv_dtype,
        lhs_name=lhs.name,
        rhs_name=rhs.name,
        lhs_page_size=lhs.page_size,
        rhs_page_size=rhs.page_size,
        lhs_alloc_mode=lhs.alloc_mode,
        rhs_alloc_mode=rhs.alloc_mode,
        equal_stdout=lhs.stdout == rhs.stdout,
        lhs_prefix_chars=len(lhs.stdout),
        rhs_prefix_chars=len(rhs.stdout),
        common_prefix_chars=cp,
    )


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def print_result_line(result: RunResult) -> None:
    status = "ok" if result.ok else "FAIL"
    page = "-" if result.page_size is None else str(result.page_size)
    alloc = "-" if result.alloc_mode is None else result.alloc_mode
    prefill = "-" if result.prefill_tps is None else f"{result.prefill_tps:.2f}"
    decode = "-" if result.decode_tps is None else f"{result.decode_tps:.2f}"
    ptok = "-" if result.prompt_tokens is None else str(result.prompt_tokens)
    decoded = "-" if result.decoded is None else str(result.decoded)
    print(
        f"{status:4s} {result.name:14s} prompt={result.prompt_name:18s} "
        f"page={page:>4s} alloc={alloc:12s} kv={result.kv_dtype:4s} "
        f"prompt_tokens={ptok:>5s} decoded={decoded:>4s} "
        f"prefill_tps={prefill:>9s} decode_tps={decode:>9s}"
    )
    if result.error:
        print(f"     error: {result.error}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run paged-KV correctness and efficiency checks for qw3."
    )
    ap.add_argument("--qw3", default="./build/qw3", help="paged/current qw3 binary")
    ap.add_argument("--baseline-qw3", help="optional baseline qw3 binary")
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--out-json", default="/tmp/qw3_paged_kv_regression.json")
    ap.add_argument(
        "--page-sizes",
        type=parse_int_list,
        default=parse_int_list("1 7 16 64"),
        help="paged KV page sizes to test, e.g. '1 7 16 64'",
    )
    ap.add_argument(
        "--alloc-modes",
        type=parse_str_list,
        default=parse_str_list("identity"),
        help="paged KV physical allocation modes: identity reverse evens-first",
    )
    ap.add_argument(
        "--kv-dtypes",
        type=parse_str_list,
        default=parse_str_list("fp16"),
        help="KV dtypes to test, e.g. 'fp16 q8 fp8'",
    )
    ap.add_argument(
        "--prompts",
        type=parse_str_list,
        default=parse_str_list("short boundary_24_words code chinese"),
        help=(
            "prompt names from the built-in catalog; empty string is not supported. "
            "Available: short boundary_24_words code chinese repeated_512_words longish_2300_words"
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--prefill-chunk", type=int)
    ap.add_argument(
        "--perf-repeat",
        type=int,
        default=1,
        help="repeat each current-binary case N times for throughput averaging",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra argument passed to qw3; repeat for multiple args",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="extra environment setting KEY=VALUE; repeat for multiple vars",
    )
    args = ap.parse_args(argv)

    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.ctx <= 0:
        raise SystemExit("--ctx must be positive")
    if args.perf_repeat <= 0:
        raise SystemExit("--perf-repeat must be positive")
    valid_alloc_modes = {"identity", "reverse", "evens-first"}
    bad_alloc_modes = [m for m in args.alloc_modes if m not in valid_alloc_modes]
    if bad_alloc_modes:
        raise SystemExit(
            "invalid --alloc-modes value(s): "
            + ", ".join(bad_alloc_modes)
            + "; want identity reverse evens-first"
        )

    qw3 = Path(args.qw3)
    baseline_qw3 = Path(args.baseline_qw3) if args.baseline_qw3 else None
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"current qw3 binary not found: {qw3}")
    if baseline_qw3 and not baseline_qw3.exists():
        raise SystemExit(f"baseline qw3 binary not found: {baseline_qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    extra_env: Dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        extra_env[key] = value

    prompts = select_prompts(args.prompts)
    results: List[RunResult] = []
    comparisons: List[Comparison] = []

    print(
        f"running paged KV regression: prompts={len(prompts)} "
        f"page_sizes={args.page_sizes} alloc_modes={args.alloc_modes} "
        f"kv_dtypes={args.kv_dtypes}"
    )

    baseline_by_key: Dict[tuple, RunResult] = {}
    if baseline_qw3:
        for kv_dtype in args.kv_dtypes:
            for prompt_name, prompt in prompts.items():
                result = run_qw3(
                    name="baseline",
                    binary=baseline_qw3,
                    model=model,
                    prompt_name=prompt_name,
                    prompt=prompt,
                    page_size=None,
                    alloc_mode=None,
                    kv_dtype=kv_dtype,
                    max_tokens=args.max_tokens,
                    ctx_size=args.ctx,
                    timeout_s=args.timeout,
                    prefill_chunk=args.prefill_chunk,
                    extra_args=args.extra_arg,
                    extra_env=extra_env,
                )
                results.append(result)
                baseline_by_key[(prompt_name, kv_dtype)] = result
                print_result_line(result)

    current_by_key: Dict[tuple, List[RunResult]] = {}
    for kv_dtype in args.kv_dtypes:
        for prompt_name, prompt in prompts.items():
            for page_size in args.page_sizes:
                for alloc_mode in args.alloc_modes:
                    for repeat_idx in range(args.perf_repeat):
                        name = f"paged#{repeat_idx + 1}" if args.perf_repeat > 1 else "paged"
                        result = run_qw3(
                            name=name,
                            binary=qw3,
                            model=model,
                            prompt_name=prompt_name,
                            prompt=prompt,
                            page_size=page_size,
                            alloc_mode=alloc_mode,
                            kv_dtype=kv_dtype,
                            max_tokens=args.max_tokens,
                            ctx_size=args.ctx,
                            timeout_s=args.timeout,
                            prefill_chunk=args.prefill_chunk,
                            extra_args=args.extra_arg,
                            extra_env=extra_env,
                        )
                        results.append(result)
                        current_by_key.setdefault(
                            (prompt_name, kv_dtype, page_size, alloc_mode), []
                        ).append(result)
                        print_result_line(result)

    # Correctness: page-size invariance against the first page size's first run.
    for kv_dtype in args.kv_dtypes:
        for prompt_name in prompts:
            reference = current_by_key[
                (prompt_name, kv_dtype, args.page_sizes[0], args.alloc_modes[0])
            ][0]
            for alloc_mode in args.alloc_modes:
                for page_size in args.page_sizes:
                    if page_size == args.page_sizes[0] and alloc_mode == args.alloc_modes[0]:
                        continue
                    candidate = current_by_key[(prompt_name, kv_dtype, page_size, alloc_mode)][0]
                    if reference.ok and candidate.ok:
                        comparisons.append(compare_outputs("paged_invariance", reference, candidate))

            if baseline_qw3:
                baseline = baseline_by_key[(prompt_name, kv_dtype)]
                for page_size in args.page_sizes:
                    for alloc_mode in args.alloc_modes:
                        candidate = current_by_key[(prompt_name, kv_dtype, page_size, alloc_mode)][0]
                        if baseline.ok and candidate.ok:
                            comparisons.append(compare_outputs("baseline", baseline, candidate))

    failed_runs = [r for r in results if not r.ok]
    failed_comparisons = [c for c in comparisons if not c.equal_stdout]

    by_page: Dict[str, Dict[str, Optional[float]]] = {}
    by_page_alloc: Dict[str, Dict[str, Optional[float]]] = {}
    for page_size in args.page_sizes:
        page_results = [
            r for r in results
            if r.name.startswith("paged") and r.page_size == page_size and r.ok
        ]
        by_page[str(page_size)] = {
            "mean_prefill_tps": mean(r.prefill_tps for r in page_results),
            "mean_decode_tps": mean(r.decode_tps for r in page_results),
            "runs": len(page_results),
        }
        for alloc_mode in args.alloc_modes:
            alloc_results = [
                r for r in page_results
                if r.alloc_mode == alloc_mode
            ]
            by_page_alloc[f"{page_size}:{alloc_mode}"] = {
                "mean_prefill_tps": mean(r.prefill_tps for r in alloc_results),
                "mean_decode_tps": mean(r.decode_tps for r in alloc_results),
                "runs": len(alloc_results),
            }

    summary = {
        "config": {
            "qw3": str(qw3),
            "baseline_qw3": str(baseline_qw3) if baseline_qw3 else None,
            "model": str(model),
            "page_sizes": args.page_sizes,
            "alloc_modes": args.alloc_modes,
            "kv_dtypes": args.kv_dtypes,
            "prompts": list(prompts.keys()),
            "max_tokens": args.max_tokens,
            "ctx": args.ctx,
            "prefill_chunk": args.prefill_chunk,
            "perf_repeat": args.perf_repeat,
        },
        "status": {
            "runs": len(results),
            "failed_runs": len(failed_runs),
            "comparisons": len(comparisons),
            "failed_comparisons": len(failed_comparisons),
        },
        "throughput_by_page_size": by_page,
        "throughput_by_page_size_and_alloc": by_page_alloc,
        "results": [asdict(r) for r in results],
        "comparisons": [asdict(c) for c in comparisons],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote summary: {out_path}")
    if by_page:
        print("throughput by page size:")
        for page_size, stats in by_page.items():
            prefill = stats["mean_prefill_tps"]
            decode = stats["mean_decode_tps"]
            prefill_s = "-" if prefill is None else f"{prefill:.2f}"
            decode_s = "-" if decode is None else f"{decode:.2f}"
            print(
                f"  page={page_size:>4s} runs={stats['runs']:>3d} "
                f"mean_prefill_tps={prefill_s:>9s} mean_decode_tps={decode_s:>9s}"
            )
    if by_page_alloc:
        print("throughput by page size and alloc:")
        for key, stats in by_page_alloc.items():
            prefill = stats["mean_prefill_tps"]
            decode = stats["mean_decode_tps"]
            prefill_s = "-" if prefill is None else f"{prefill:.2f}"
            decode_s = "-" if decode is None else f"{decode:.2f}"
            print(
                f"  {key:>20s} runs={stats['runs']:>3d} "
                f"mean_prefill_tps={prefill_s:>9s} mean_decode_tps={decode_s:>9s}"
            )

    if failed_runs:
        print(f"failed runs: {len(failed_runs)}", file=sys.stderr)
        for r in failed_runs[:10]:
            print(
                f"  {r.name} prompt={r.prompt_name} page={r.page_size} "
                f"alloc={r.alloc_mode} kv={r.kv_dtype}: {r.error}",
                file=sys.stderr,
            )
    if failed_comparisons:
        print(f"failed output comparisons: {len(failed_comparisons)}", file=sys.stderr)
        for c in failed_comparisons[:10]:
            print(
                f"  {c.kind} prompt={c.prompt_name} kv={c.kv_dtype} "
                f"{c.lhs_name}/page={c.lhs_page_size}/alloc={c.lhs_alloc_mode} vs "
                f"{c.rhs_name}/page={c.rhs_page_size}/alloc={c.rhs_alloc_mode}: "
                f"common_prefix={c.common_prefix_chars}",
                file=sys.stderr,
            )

    return 1 if failed_runs or failed_comparisons else 0


if __name__ == "__main__":
    raise SystemExit(main())
