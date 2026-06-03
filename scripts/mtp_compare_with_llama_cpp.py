#!/usr/bin/env python3
"""Compare qw3 native MTP speculative decoding against llama.cpp draft-MTP.

This is intentionally a thin orchestration layer over the existing probes:

- `mtp_acceptance_probe.py` runs qw3 with `--native-mtp-speculate`.
- `probe_llama_mtp_server.py` runs `llama-server --spec-type draft-mtp`.

llama.cpp is restarted for each prompt length by default so prompt-eval timing
does not get contaminated by server-side prefix-cache reuse.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from qw3_subprocess import run_captured


DEFAULT_QW3_ENV = []
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(value: str, flag: str) -> list[int]:
    out: list[int] = []
    for item in value.replace(",", " ").split():
        try:
            parsed = int(item)
        except ValueError as exc:
            raise SystemExit(f"{flag} values must be integers, got: {item}") from exc
        if parsed <= 0:
            raise SystemExit(f"{flag} values must be positive")
        out.append(parsed)
    if not out:
        raise SystemExit(f"{flag} cannot be empty")
    return out


def run_json(cmd: list[str], json_path: Path, timeout: float) -> dict:
    print("+ " + shlex.join(cmd), flush=True)
    proc = run_captured(cmd, timeout=timeout, cwd=REPO_ROOT)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n",
              file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {shlex.join(cmd)}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def aggregate_qw3(results: Iterable[dict], key: str) -> float:
    rows = [r for r in results if r.get("ok")]
    if key == "prefill":
        tokens = sum(int(r["prompt_tokens"]) for r in rows)
        wall = sum(float(r["prompt_tokens"]) / float(r["prefill_tok_s"])
                   for r in rows if float(r.get("prefill_tok_s") or 0.0) > 0.0)
    elif key == "decode":
        tokens = sum(int(r["decoded_tokens"]) for r in rows)
        wall = sum(float(r["decoded_tokens"]) / float(r["decode_tok_s"])
                   for r in rows if float(r.get("decode_tok_s") or 0.0) > 0.0)
    else:
        raise ValueError(key)
    return (tokens / wall) if wall > 0.0 else 0.0


def aggregate_llama(results: Iterable[dict], key: str) -> float:
    rows = [r for r in results if r.get("ok")]
    if key == "prefill":
        tokens = sum(int(r["prompt_tokens"]) for r in rows)
        wall = sum(float(r["prompt_tokens"]) / float(r["prompt_tok_s"])
                   for r in rows if float(r.get("prompt_tok_s") or 0.0) > 0.0)
    elif key == "decode":
        tokens = sum(int(r["decoded_tokens"]) for r in rows)
        wall = sum(float(r["decoded_tokens"]) / float(r["decode_tok_s"])
                   for r in rows if float(r.get("decode_tok_s") or 0.0) > 0.0)
    else:
        raise ValueError(key)
    return (tokens / wall) if wall > 0.0 else 0.0


def summarize_chain(chain: int, qw3_payload: dict, llama_payload: dict) -> dict:
    q_rows = qw3_payload["results"]
    l_rows = llama_payload["results"]
    q_prefill = aggregate_qw3(q_rows, "prefill")
    l_prefill = aggregate_llama(l_rows, "prefill")
    q_decode = aggregate_qw3(q_rows, "decode")
    l_decode = aggregate_llama(l_rows, "decode")
    q_verified = sum(int(r.get("verified") or 0) for r in q_rows if r.get("ok"))
    q_accepted = sum(int(r.get("accepted") or 0) for r in q_rows if r.get("ok"))
    l_draft = sum(int(r.get("draft_tokens") or 0) for r in l_rows if r.get("ok"))
    l_accepted = sum(int(r.get("draft_accepted") or 0) for r in l_rows if r.get("ok"))
    return {
        "mtp_chain": chain,
        "ok": (
            qw3_payload["summary"]["ok"] == qw3_payload["summary"]["total"] and
            llama_payload["summary"]["ok"] == llama_payload["summary"]["total"] and
            bool(llama_payload.get("mtp_initialized"))
        ),
        "qw3_prefill_tok_s": q_prefill,
        "llama_prefill_tok_s": l_prefill,
        "prefill_ratio": (q_prefill / l_prefill) if l_prefill > 0.0 else 0.0,
        "qw3_decode_tok_s": q_decode,
        "llama_decode_tok_s": l_decode,
        "decode_ratio": (q_decode / l_decode) if l_decode > 0.0 else 0.0,
        "qw3_acceptance": (q_accepted / q_verified) if q_verified else 0.0,
        "llama_acceptance": (l_accepted / l_draft) if l_draft else 0.0,
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in q_rows if r.get("ok")),
        "decoded_tokens": sum(int(r.get("decoded_tokens") or 0) for r in q_rows if r.get("ok")),
        "per_prompt": [
            {
                "target_index": i,
                "qw3_prompt_tokens": q.get("prompt_tokens", 0),
                "llama_prompt_tokens": l.get("prompt_tokens", 0),
                "qw3_prefill_tok_s": q.get("prefill_tok_s", 0.0),
                "llama_prefill_tok_s": l.get("prompt_tok_s", 0.0),
                "qw3_decode_tok_s": q.get("decode_tok_s", 0.0),
                "llama_decode_tok_s": l.get("decode_tok_s", 0.0),
                "qw3_acceptance": q.get("acceptance", 0.0),
                "llama_acceptance": l.get("draft_acceptance", 0.0),
            }
            for i, (q, l) in enumerate(zip(q_rows, l_rows))
        ],
    }


def print_summary(rows: list[dict]) -> None:
    print()
    print("=" * 118)
    print("MTP qw3 vs llama.cpp, weighted tok/s")
    print("=" * 118)
    print(f"{'mtp':>4} | {'qw3 prefill':>11} {'llama prefill':>13} {'pref ratio':>10} | "
          f"{'qw3 decode':>10} {'llama decode':>12} {'dec ratio':>9} | "
          f"{'qw3 acc':>7} {'llama acc':>9}")
    print("-" * 118)
    for row in rows:
        print(
            f"{row['mtp_chain']:>4d} | "
            f"{row['qw3_prefill_tok_s']:>11.2f} {row['llama_prefill_tok_s']:>13.2f} "
            f"{row['prefill_ratio'] * 100:>9.1f}% | "
            f"{row['qw3_decode_tok_s']:>10.2f} {row['llama_decode_tok_s']:>12.2f} "
            f"{row['decode_ratio'] * 100:>8.1f}% | "
            f"{row['qw3_acceptance'] * 100:>6.1f}% {row['llama_acceptance'] * 100:>8.1f}%"
        )
    print("=" * 118)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qw3", default="./build/qw3")
    parser.add_argument("--llama-server",
                        default="/home/chaidi/qw3/llama.cpp/build-cuda/bin/llama-server")
    parser.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--prompt-tokens", default="4096 8192")
    parser.add_argument("--mtp-chains", default="2")
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("-c", "--ctx", type=int, default=32768)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--qw3-env", action="append", default=[])
    parser.add_argument("--no-default-qw3-env", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()

    chains = parse_int_list(args.mtp_chains, "--mtp-chains")
    prompt_targets = parse_int_list(args.prompt_tokens, "--prompt-tokens")
    qw3_env = ([] if args.no_default_qw3_env else list(DEFAULT_QW3_ENV)) + args.qw3_env

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summaries: list[dict] = []
    artifacts: dict[str, dict[str, str]] = {}
    tmpdir = Path(tempfile.mkdtemp(prefix="qw3_mtp_compare_", dir="/tmp"))
    for chain in chains:
        qw3_json = tmpdir / f"qw3_mtp{chain}_{timestamp}.json"
        llama_json = tmpdir / f"llama_mtp{chain}_{timestamp}.json"
        qw3_cmd = [
            sys.executable, "scripts/mtp_acceptance_probe.py",
            "--qw3", args.qw3,
            "--model", args.model,
            "--prompt-tokens", args.prompt_tokens,
            "--mtp-chain", str(chain),
            "--mtp-speculate",
            "-n", str(args.n),
            "-c", str(args.ctx),
            "--timeout", str(args.timeout),
            "--json", str(qw3_json),
        ]
        for item in qw3_env:
            qw3_cmd.extend(["--qw3-env", item])
        llama_cmd = [
            sys.executable, "scripts/probe_llama_mtp_server.py",
            "--server", args.llama_server,
            "--model", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "-c", str(args.ctx),
            "-n", str(args.n),
            "--spec-draft-n-max", str(chain),
            "--prompt-tokens", args.prompt_tokens,
            "--restart-per-prompt",
            "--startup-timeout", str(args.startup_timeout),
            "--timeout", str(args.timeout),
            "--json", str(llama_json),
        ]

        qw3_child_timeout = len(prompt_targets) * (args.timeout + 30.0)
        llama_child_timeout = len(prompt_targets) * (args.timeout + args.startup_timeout + 60.0)
        qw3_payload = run_json(qw3_cmd, qw3_json, qw3_child_timeout)
        llama_payload = run_json(llama_cmd, llama_json, llama_child_timeout)
        summary = summarize_chain(chain, qw3_payload, llama_payload)
        summaries.append(summary)
        artifacts[str(chain)] = {
            "qw3_json": str(qw3_json),
            "llama_json": str(llama_json),
        }

    print_summary(summaries)

    payload = {
        "args": vars(args),
        "default_qw3_env": [] if args.no_default_qw3_env else DEFAULT_QW3_ENV,
        "summaries": summaries,
        "artifacts": artifacts,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    print(f"child artifacts kept in {tmpdir}")

    return 0 if all(row["ok"] for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
