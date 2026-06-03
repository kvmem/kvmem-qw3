#!/usr/bin/env python3
"""Probe native MTP draft acceptance and speculative decoding.

By default the script runs qw3 with --native-mtp-trace, parses the MTP
draft/verify summary, and emits a compact table plus optional JSON. With
--mtp-speculate it instead enables the experimental speculative path and parses
the real accept/reject summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from long_prompt_sweep import make_prompt as make_long_prompt
from qw3_subprocess import run_captured


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Hello! Briefly tell me what FlashAttention is.",
    "请简要解释什么是注意力机制",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""',
]

_CODING_PASSAGE = r'''
You are reviewing a Python service that schedules GPU benchmark jobs. The
service must avoid loading two 27B models at the same time, keep benchmark
results reproducible, and emit compact JSON summaries for later analysis.

```python
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BenchmarkJob:
    name: str
    command: list[str]
    env: dict[str, str]
    timeout_s: float
    output_json: Path

class JobError(RuntimeError):
    pass

class GpuLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

def run_job(job: BenchmarkJob) -> dict:
    env = os.environ.copy()
    env.update(job.env)
    start = time.perf_counter()
    proc = subprocess.run(
        job.command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=job.timeout_s,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise JobError(proc.stdout[-4000:])
    data = json.loads(job.output_json.read_text())
    data["elapsed_s"] = elapsed
    return data
```

The next version should add cooperative locking, structured retry handling,
per-context aggregation, and a small CLI that can run adaptive and fixed MTP
depths in a serial order. Keep the implementation simple and avoid broad
dependencies.

'''


def make_coding_prompt(target_tokens: int) -> str:
    """Build a coding-task prompt of approximately `target_tokens` tokens."""
    base_tokens = 430
    per_repeat = 390
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    body = _CODING_PASSAGE * n_repeats
    return (
        "Complete the following coding task. Produce concrete Python code, "
        "explain the important design choices briefly, and include focused "
        "tests at the end.\n\n"
        f"{body}\n"
        "Now write the improved implementation:\n"
    )


_CREATIVE_PASSAGE = '''
Story brief:

The city of Ardent Vale is built around a dry riverbed that rings like glass
when the wind passes through it. Every year, the bellmakers lower bronze chimes
into the empty channel and listen for a tone that predicts the next season.
This year the riverbed answers with a human voice. It speaks the name of Mara
Venn, a mapmaker who has spent ten years drawing borders for a kingdom that no
longer trusts maps.

Mara carries three things: a compass that points toward unfinished promises, a
letter from her missing brother, and a page torn from an atlas showing a coast
that should not exist. She is practical, skeptical, and tired of prophecies
that arrive after the damage is already done.

The scene should balance lyrical description with concrete action. Keep the
emotional stakes visible through behavior rather than exposition. Avoid parody,
avoid summary, and avoid explaining the themes directly. The prose should feel
polished but not ornate.

Excerpt:

At noon the bells went down into the riverbed. Men in leather gloves guided the
ropes while the children of the vale leaned over the railings, each of them
quiet for once, each waiting for winter to reveal itself in a sound. The bronze
chimes sank through dust instead of water. They vanished into the white cut of
the channel, and for a breath the city held still.

Then the riverbed spoke.

Not thunder. Not music. A voice, low enough that the stone bridges shivered.
"Mara Venn," it said.

The mapmaker looked up from the square where she had been bargaining for ink.
The seller's hand tightened around the bottle. Across the stalls, people
turned, and every face became a border Mara did not know how to cross.

'''


def make_creative_prompt(target_tokens: int) -> str:
    """Build a creative-writing prompt of approximately `target_tokens` tokens."""
    base_tokens = 520
    per_repeat = 465
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    body = _CREATIVE_PASSAGE * n_repeats
    return (
        "Continue the following literary fantasy scene. Write in third person, "
        "maintain continuity with all details, and generate vivid prose with "
        "dialogue and forward motion. Do not outline; write the scene itself.\n\n"
        f"{body}\n"
        "Continuation:\n"
    )


_QW3_LINE = re.compile(
    r"\[qw3\] native generate: prompt_tokens=(\d+) "
    r"prefill=([0-9.]+)s \(([0-9.]+) tok/s\) "
    r"decoded=(\d+) decode=([0-9.]+)s \(([0-9.]+) tok/s\)"
)
_MTP_SUMMARY = re.compile(
    r"\[qw3\] native mtp_summary: drafts=(\d+) verified=(\d+) "
    r"accepted=(\d+) acceptance=([0-9.]+) mtp_ops=(\d+)"
)
_MTP_CHAIN_OFFSET = re.compile(
    r"\[qw3\] native mtp_chain_offset: step=(\d+) verified=(\d+) "
    r"accepted=(\d+) acceptance=([0-9.]+)"
)
_MTP_SPEC_SUMMARY = re.compile(
    r"\[qw3\] native mtp_spec_summary: enabled=(true|false) batches=(\d+) "
    r"drafted=(\d+) accepted=(\d+) rejected=(\d+) rollbacks=(\d+) "
    r"(?:adaptive=(true|false) promotions=(\d+) )?"
    r"(?:reject_budget=([0-9]+|off) fallback=(true|false) )?"
    r"acceptance=([0-9.]+) mtp_ops=(\d+) prefix_tokens=(\d+) prefix_ops=(\d+)"
)
_MTP_SPEC_TIMINGS = re.compile(
    r"draft_s=([0-9.]+)s snapshot_s=([0-9.]+)s verify_s=([0-9.]+)s "
    r"restore_s=([0-9.]+)s replay_s=([0-9.]+)s plain_s=([0-9.]+)s"
    r"(?: prefix_s=([0-9.]+)s)?"
)
_MTP_ACCEPT_HIST = re.compile(r"\[qw3\] native mtp_accept_hist:((?: len\d+=\d+)+)")
_MTP_ACCEPT_HIST_ITEM = re.compile(r"len(\d+)=(\d+)")
_MTP_PREFIX1_REUSE = re.compile(r"\bprefix1_reuse=(\d+)")
_MTP_STATE_CKPT_REUSE = re.compile(r"\bstate_ckpt_reuse=(\d+)")
_MTP_STATE_CKPT_COUNT = re.compile(r"\bstate_ckpt_count=(\d+)")


def parse_env_assignments(items: Iterable[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--qw3-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"--qw3-env key is empty in: {item}")
        env[key] = value
    return env


def write_prompt_temp(prompt: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="qw3_mtp_prompt_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(prompt)
    return Path(path)


def remove_prompt_temp(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_raw_log(args, index: int, text: str) -> None:
    if not args.raw_log_dir:
        return
    out_dir = Path(args.raw_log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"prompt_{index:03d}.log").write_text(text, encoding="utf-8")


@dataclass
class ProbeResult:
    index: int
    ok: bool
    prompt_preview: str
    prompt_tokens: int = 0
    prefill_tok_s: float = 0.0
    decoded_tokens: int = 0
    decode_tok_s: float = 0.0
    drafts: int = 0
    verified: int = 0
    accepted: int = 0
    acceptance: float = 0.0
    mtp_ops: int = 0
    spec_batches: int = 0
    spec_rejected: int = 0
    spec_rollbacks: int = 0
    spec_adaptive: bool = False
    spec_promotions: int = 0
    spec_fallback: bool = False
    spec_reject_budget: str = ""
    spec_draft_s: float = 0.0
    spec_snapshot_s: float = 0.0
    spec_verify_s: float = 0.0
    spec_restore_s: float = 0.0
    spec_replay_s: float = 0.0
    spec_plain_s: float = 0.0
    spec_prefix_s: float = 0.0
    spec_prefix1_reuse: int = 0
    spec_state_ckpt_reuse: int = 0
    spec_state_ckpt_count: int = 0
    prefix_tokens: int = 0
    prefix_ops: int = 0
    chain_verified: list[int] = None
    chain_accepted: list[int] = None
    chain_acceptance: list[float] = None
    accept_len_hist: list[int] = None
    error: str = ""


def load_prompts(path: Optional[str]) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def parse_prompt_targets(value: str) -> list[int]:
    targets = []
    for item in value.split():
        try:
            target = int(item)
        except ValueError as exc:
            raise SystemExit(f"--prompt-tokens must be integers, got: {item}") from exc
        if target <= 0:
            raise SystemExit("--prompt-tokens values must be positive")
        targets.append(target)
    if not targets:
        raise SystemExit("--prompt-tokens cannot be empty")
    return targets


def run_probe(args, prompt: str, index: int) -> ProbeResult:
    prompt_path = write_prompt_temp(prompt)
    cmd = [
        args.qw3,
        "--backend", "qwen-native",
        "--native-heavy",
        "--native-mtp-chain", str(args.mtp_chain),
        "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", args.model,
        "--raw",
        "-c", str(args.ctx),
        "-n", str(args.n),
        "--prompt-file", str(prompt_path),
    ]
    if args.baseline:
        pass
    elif args.mtp_speculate:
        cmd.insert(4, "--native-mtp-speculate")
    else:
        cmd.insert(4, "--native-mtp-trace")
    if args.mtp_prefix:
        cmd.insert(5, "--native-mtp-prefix")
    env = os.environ.copy()
    env.update(parse_env_assignments(args.qw3_env))
    try:
        proc = run_captured(cmd, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return ProbeResult(index=index, ok=False, prompt_preview=prompt[:80], error="timeout")
    except OSError as exc:
        return ProbeResult(index=index, ok=False, prompt_preview=prompt[:80], error=f"spawn failed: {exc}")
    finally:
        remove_prompt_temp(prompt_path)

    out = proc.stdout + proc.stderr
    write_raw_log(args, index, out)
    if proc.returncode != 0:
        return ProbeResult(index=index, ok=False, prompt_preview=prompt[:80],
                           error=f"exit {proc.returncode}: {out.strip()[-240:]}")

    timing = _QW3_LINE.search(out)
    summary = None
    if args.mtp_speculate:
        summary = _MTP_SPEC_SUMMARY.search(out)
    elif not args.baseline:
        summary = _MTP_SUMMARY.search(out)
    chain_offsets = _MTP_CHAIN_OFFSET.findall(out)
    if not timing or (not args.baseline and not summary):
        return ProbeResult(index=index, ok=False, prompt_preview=prompt[:80],
                           error=f"parse failed; tail={out.strip()[-400:]}")
    if args.baseline:
        return ProbeResult(
            index=index,
            ok=True,
            prompt_preview=prompt[:80],
            prompt_tokens=int(timing.group(1)),
            prefill_tok_s=float(timing.group(3)),
            decoded_tokens=int(timing.group(4)),
            decode_tok_s=float(timing.group(6)),
            chain_verified=[0] * args.mtp_chain,
            chain_accepted=[0] * args.mtp_chain,
            chain_acceptance=[0.0] * args.mtp_chain,
        )
    chain_verified = [0] * args.mtp_chain
    chain_accepted = [0] * args.mtp_chain
    chain_acceptance = [0.0] * args.mtp_chain
    for step_s, verified_s, accepted_s, acceptance_s in chain_offsets:
        step = int(step_s)
        if 1 <= step <= args.mtp_chain:
            chain_verified[step - 1] = int(verified_s)
            chain_accepted[step - 1] = int(accepted_s)
            chain_acceptance[step - 1] = float(acceptance_s)

    if args.mtp_speculate:
        drafted = int(summary.group(3))
        accepted = int(summary.group(4))
        spec_timings = _MTP_SPEC_TIMINGS.search(out)
        accept_hist = [0] * (args.mtp_chain + 1)
        if hist_match := _MTP_ACCEPT_HIST.search(out):
            for idx_s, value_s in _MTP_ACCEPT_HIST_ITEM.findall(hist_match.group(1)):
                idx = int(idx_s)
                if 0 <= idx < len(accept_hist):
                    accept_hist[idx] = int(value_s)
        prefix1_reuse = 0
        if prefix1_match := _MTP_PREFIX1_REUSE.search(out):
            prefix1_reuse = int(prefix1_match.group(1))
        state_ckpt_reuse = 0
        if state_ckpt_reuse_match := _MTP_STATE_CKPT_REUSE.search(out):
            state_ckpt_reuse = int(state_ckpt_reuse_match.group(1))
        state_ckpt_count = 0
        if state_ckpt_count_match := _MTP_STATE_CKPT_COUNT.search(out):
            state_ckpt_count = int(state_ckpt_count_match.group(1))
        timing_values = [0.0] * 7
        if spec_timings:
            timing_values = [
                float(spec_timings.group(i) or 0.0)
                for i in range(1, 8)
            ]
        return ProbeResult(
            index=index,
            ok=True,
            prompt_preview=prompt[:80],
            prompt_tokens=int(timing.group(1)),
            prefill_tok_s=float(timing.group(3)),
            decoded_tokens=int(timing.group(4)),
            decode_tok_s=float(timing.group(6)),
            drafts=drafted,
            verified=drafted,
            accepted=accepted,
            acceptance=float(summary.group(11)),
            mtp_ops=int(summary.group(12)),
            spec_batches=int(summary.group(2)),
            spec_rejected=int(summary.group(5)),
            spec_rollbacks=int(summary.group(6)),
            spec_adaptive=summary.group(7) == "true",
            spec_promotions=int(summary.group(8) or 0),
            spec_reject_budget=summary.group(9) or "",
            spec_fallback=summary.group(10) == "true",
            spec_draft_s=timing_values[0],
            spec_snapshot_s=timing_values[1],
            spec_verify_s=timing_values[2],
            spec_restore_s=timing_values[3],
            spec_replay_s=timing_values[4],
            spec_plain_s=timing_values[5],
            spec_prefix_s=timing_values[6],
            spec_prefix1_reuse=prefix1_reuse,
            spec_state_ckpt_reuse=state_ckpt_reuse,
            spec_state_ckpt_count=state_ckpt_count,
            prefix_tokens=int(summary.group(13)),
            prefix_ops=int(summary.group(14)),
            chain_verified=chain_verified,
            chain_accepted=chain_accepted,
            chain_acceptance=chain_acceptance,
            accept_len_hist=accept_hist,
        )

    return ProbeResult(
        index=index,
        ok=True,
        prompt_preview=prompt[:80],
        prompt_tokens=int(timing.group(1)),
        prefill_tok_s=float(timing.group(3)),
        decoded_tokens=int(timing.group(4)),
        decode_tok_s=float(timing.group(6)),
        drafts=int(summary.group(1)),
        verified=int(summary.group(2)),
        accepted=int(summary.group(3)),
        acceptance=float(summary.group(4)),
        mtp_ops=int(summary.group(5)),
        chain_verified=chain_verified,
        chain_accepted=chain_accepted,
        chain_acceptance=chain_acceptance,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qw3", default=os.environ.get("QW3", "./build/qw3"))
    parser.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-tokens",
                        help="Generate long sweep-style prompts for the requested token targets")
    parser.add_argument("--prompt-kind", choices=["default", "coding", "creative"], default="default",
                        help="Prompt generator to use with --prompt-tokens")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-n", type=int, default=32)
    parser.add_argument("-c", "--ctx", type=int, default=4096)
    parser.add_argument("--mtp-chain", type=int, default=1)
    parser.add_argument("--mtp-prefix", action="store_true",
                        help="Populate the diagnostic MTP prefix KV cache before drafting")
    parser.add_argument("--mtp-speculate", action="store_true",
                        help="Run the experimental MTP speculative decode path")
    parser.add_argument("--baseline", action="store_true",
                        help="Run plain qwen-native greedy decode without MTP tracing")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--qw3-env", action="append", default=[])
    parser.add_argument("--raw-log-dir",
                        help="Write raw qw3 stdout/stderr for each prompt to this directory")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.mtp_chain < 1:
        raise SystemExit("--mtp-chain must be >= 1")
    if args.baseline and args.mtp_speculate:
        raise SystemExit("--baseline and --mtp-speculate are mutually exclusive")
    if args.prompts and args.prompt_tokens:
        raise SystemExit("--prompts and --prompt-tokens are mutually exclusive")

    if args.prompt_tokens:
        if args.prompt_kind == "coding":
            make_prompt = make_coding_prompt
        elif args.prompt_kind == "creative":
            make_prompt = make_creative_prompt
        else:
            make_prompt = make_long_prompt
        prompts = [make_prompt(t) for t in parse_prompt_targets(args.prompt_tokens)]
    else:
        prompts = load_prompts(args.prompts)
    if args.limit > 0:
        prompts = prompts[:args.limit]

    results = [run_probe(args, prompt, i) for i, prompt in enumerate(prompts)]
    ok = [r for r in results if r.ok]
    total_verified = sum(r.verified for r in ok)
    total_accepted = sum(r.accepted for r in ok)
    total_rate = (total_accepted / total_verified) if total_verified else 0.0
    chain_verified = [0] * args.mtp_chain
    chain_accepted = [0] * args.mtp_chain
    for r in ok:
        for i in range(args.mtp_chain):
            if r.chain_verified and i < len(r.chain_verified):
                chain_verified[i] += r.chain_verified[i]
            if r.chain_accepted and i < len(r.chain_accepted):
                chain_accepted[i] += r.chain_accepted[i]
    accept_len_hist = [0] * (args.mtp_chain + 1)
    for r in ok:
        if r.accept_len_hist:
            for i, value in enumerate(r.accept_len_hist[:len(accept_len_hist)]):
                accept_len_hist[i] += value
    total_prompt_tokens = sum(r.prompt_tokens for r in ok)
    total_decoded_tokens = sum(r.decoded_tokens for r in ok)
    prefill_wall = sum(
        (r.prompt_tokens / r.prefill_tok_s) for r in ok if r.prefill_tok_s > 0
    )
    decode_wall = sum(
        (r.decoded_tokens / r.decode_tok_s) for r in ok if r.decode_tok_s > 0
    )
    aggregate_prefill_tok_s = (
        total_prompt_tokens / prefill_wall if prefill_wall > 0 else 0.0
    )
    aggregate_decode_tok_s = (
        total_decoded_tokens / decode_wall if decode_wall > 0 else 0.0
    )

    print("idx ok prompt_tokens prefill_tok_s decoded drafts verified accepted accept_rate decode_tok_s")
    for r in results:
        if r.ok:
            print(f"{r.index:3d} yes {r.prompt_tokens:13d} {r.prefill_tok_s:13.2f} "
                  f"{r.decoded_tokens:7d} "
                  f"{r.drafts:6d} {r.verified:8d} {r.accepted:8d} "
                  f"{r.acceptance:11.4f} {r.decode_tok_s:12.2f}")
        else:
            print(f"{r.index:3d} no  error={r.error}")
    print(f"TOTAL verified={total_verified} accepted={total_accepted} "
          f"acceptance={total_rate:.4f} prefill_tok_s={aggregate_prefill_tok_s:.2f} "
          f"decode_tok_s={aggregate_decode_tok_s:.2f}")
    if args.mtp_chain > 1:
        for i in range(args.mtp_chain):
            rate = (chain_accepted[i] / chain_verified[i]) if chain_verified[i] else 0.0
            print(f"CHAIN step={i + 1} verified={chain_verified[i]} "
                  f"accepted={chain_accepted[i]} acceptance={rate:.4f}")
    if args.mtp_speculate and any(accept_len_hist):
        hist = " ".join(
            f"len{i}={value}" for i, value in enumerate(accept_len_hist)
        )
        batches = sum(accept_len_hist)
        committed = sum((i + 1) * value for i, value in enumerate(accept_len_hist))
        avg_committed = (committed / batches) if batches else 0.0
        print(f"ACCEPT_HIST {hist} avg_committed_per_batch={avg_committed:.3f}")

    if args.json:
        payload = {
            "args": vars(args),
            "summary": {
                "ok": len(ok),
                "total": len(results),
                "verified": total_verified,
                "accepted": total_accepted,
                "acceptance": total_rate,
                "prompt_tokens": total_prompt_tokens,
                "decoded_tokens": total_decoded_tokens,
                "aggregate_prefill_tok_s": aggregate_prefill_tok_s,
                "aggregate_decode_tok_s": aggregate_decode_tok_s,
                "chain_verified": chain_verified,
                "chain_accepted": chain_accepted,
                "chain_acceptance": [
                    (chain_accepted[i] / chain_verified[i]) if chain_verified[i] else 0.0
                    for i in range(args.mtp_chain)
                ],
                "accept_len_hist": accept_len_hist,
                "avg_committed_per_batch": (
                    sum((i + 1) * value for i, value in enumerate(accept_len_hist)) /
                    sum(accept_len_hist)
                    if sum(accept_len_hist) else 0.0
                ),
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")

    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
