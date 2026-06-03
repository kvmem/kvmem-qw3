#!/usr/bin/env python3
"""Smoke-test llama.cpp server-side MTP speculative decoding.

The `llama-completion` example does not currently expose the MTP speculative
path. This script starts `llama-server --spec-type draft-mtp`, sends one or
more local OpenAI-compatible completion requests, reports draft acceptance
stats, and then stops the server.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from long_prompt_sweep import make_prompt as make_long_prompt


DEFAULT_PROMPT = "Write one short sentence about CUDA."
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Hello! Briefly tell me what FlashAttention is.",
    "请简要解释什么是注意力机制",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""',
]


@dataclass
class LlamaMtpProbe:
    index: int
    ok: bool
    mtp_initialized: bool
    prompt_preview: str = ""
    text_preview: str = ""
    prompt_tokens: int = 0
    decoded_tokens: int = 0
    prompt_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    draft_tokens: int = 0
    draft_accepted: int = 0
    draft_acceptance: float = 0.0
    error: str = ""


def read_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_PROMPT
    return Path(path).read_text(encoding="utf-8")


def load_prompts(args) -> list[str]:
    if args.prompt_tokens:
        targets = [int(x) for x in re.split(r"[,\s]+", args.prompt_tokens) if x]
        if not targets:
            raise SystemExit("--prompt-tokens cannot be empty")
        if any(target <= 0 for target in targets):
            raise SystemExit("--prompt-tokens values must be positive")
        return [make_long_prompt(target) for target in targets]
    if args.default_prompts:
        return list(DEFAULT_PROMPTS)
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.rstrip("\n")]
        if not prompts:
            raise SystemExit(f"no prompts found in {args.prompts}")
        return prompts
    return [read_prompt(args.prompt_file)]


def request_json(url: str, payload: dict | None = None, timeout: float = 2.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_until_ready(proc: subprocess.Popen, base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = "server did not become ready"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            request_json(f"{base_url}/v1/models", timeout=1.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.2)
    raise TimeoutError(last_error)


def stop_process_group(proc: subprocess.Popen) -> str:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
    return out or ""


def start_server(args) -> subprocess.Popen:
    cmd = [
        args.server,
        "-m", args.model,
        "--spec-type", "draft-mtp",
        "-ngl", "all",
        "-c", str(args.ctx),
        "-n", str(args.n),
        "--host", args.host,
        "--port", str(args.port),
        "--no-webui",
        "--no-warmup",
    ]
    if args.spec_draft_n_max is not None:
        cmd.extend(["--spec-draft-n-max", str(args.spec_draft_n_max)])
    if args.spec_draft_n_min is not None:
        cmd.extend(["--spec-draft-n-min", str(args.spec_draft_n_min)])
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def run_completion(args, base_url: str, prompt: str, index: int) -> LlamaMtpProbe:
    try:
        payload = {
            "model": Path(args.model).name,
            "prompt": prompt,
            "max_tokens": args.n,
            "temperature": 0,
        }
        response = request_json(f"{base_url}/v1/completions", payload, args.timeout)
    except Exception as exc:
        return LlamaMtpProbe(index=index, ok=False, mtp_initialized=False,
                             prompt_preview=prompt[:80], error=str(exc))

    timings = response.get("timings") or {}
    text = ""
    choices = response.get("choices") or []
    if choices:
        text = choices[0].get("text") or ""
    draft_n = int(timings.get("draft_n") or 0)
    draft_accepted = int(timings.get("draft_n_accepted") or 0)
    return LlamaMtpProbe(
        index=index,
        ok=True,
        mtp_initialized=False,
        prompt_preview=prompt[:80],
        text_preview=text[:120],
        prompt_tokens=int(timings.get("prompt_n") or 0),
        decoded_tokens=int(timings.get("predicted_n") or 0),
        prompt_tok_s=float(timings.get("prompt_per_second") or 0.0),
        decode_tok_s=float(timings.get("predicted_per_second") or 0.0),
        draft_tokens=draft_n,
        draft_accepted=draft_accepted,
        draft_acceptance=(draft_accepted / draft_n) if draft_n else 0.0,
    )


def run_probe(args) -> tuple[list[LlamaMtpProbe], str, bool]:
    prompts = load_prompts(args)
    if args.restart_per_prompt and len(prompts) > 1:
        results: list[LlamaMtpProbe] = []
        logs_parts: list[str] = []
        all_mtp_initialized = True
        for prompt_index, prompt in enumerate(prompts):
            try:
                proc = start_server(args)
            except OSError as exc:
                results.append(
                    LlamaMtpProbe(index=prompt_index, ok=False,
                                  mtp_initialized=False,
                                  prompt_preview=prompt[:80],
                                  error=f"spawn failed: {exc}")
                )
                all_mtp_initialized = False
                continue
            base_url = f"http://{args.host}:{args.port}"
            logs = ""
            mtp_initialized = False
            try:
                wait_until_ready(proc, base_url, args.startup_timeout)
                result = run_completion(args, base_url, prompt, prompt_index)
                results.append(result)
            except Exception as exc:
                results.append(
                    LlamaMtpProbe(index=prompt_index, ok=False,
                                  mtp_initialized=False,
                                  prompt_preview=prompt[:80],
                                  error=str(exc))
                )
            finally:
                logs = stop_process_group(proc)
                logs_parts.append(logs)
                mtp_initialized = (
                    "common_speculative_impl_draft_mtp" in logs and
                    "speculative decoding context initialized" in logs
                )
                results[-1].mtp_initialized = mtp_initialized
                all_mtp_initialized = all_mtp_initialized and mtp_initialized
                time.sleep(0.5)
        return results, "\n".join(logs_parts), all_mtp_initialized

    try:
        proc = start_server(args)
    except OSError as exc:
        return [
            LlamaMtpProbe(index=0, ok=False, mtp_initialized=False,
                          error=f"spawn failed: {exc}")
        ], "", False
    base_url = f"http://{args.host}:{args.port}"
    logs = ""
    try:
        wait_until_ready(proc, base_url, args.startup_timeout)
        results = [
            run_completion(args, base_url, prompt, i)
            for i, prompt in enumerate(prompts)
        ]
    except Exception as exc:
        results = [LlamaMtpProbe(index=0, ok=False, mtp_initialized=False, error=str(exc))]
    finally:
        logs = stop_process_group(proc)

    mtp_initialized = (
        "common_speculative_impl_draft_mtp" in logs and
        "speculative decoding context initialized" in logs
    )
    for result in results:
        result.mtp_initialized = mtp_initialized
        if not result.ok and mtp_initialized and not result.error:
            result.error = "request failed after MTP initialization"
    return results, logs, mtp_initialized


def summarize(results: list[LlamaMtpProbe]) -> dict:
    ok = [r for r in results if r.ok]
    draft_tokens = sum(r.draft_tokens for r in ok)
    draft_accepted = sum(r.draft_accepted for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    decoded_tokens = sum(r.decoded_tokens for r in ok)
    prompt_wall = sum(
        (r.prompt_tokens / r.prompt_tok_s) for r in ok if r.prompt_tok_s > 0
    )
    decode_wall = sum(
        (r.decoded_tokens / r.decode_tok_s) for r in ok if r.decode_tok_s > 0
    )
    return {
        "ok": len(ok),
        "total": len(results),
        "prompt_tokens": prompt_tokens,
        "decoded_tokens": decoded_tokens,
        "draft_tokens": draft_tokens,
        "draft_accepted": draft_accepted,
        "draft_acceptance": (draft_accepted / draft_tokens) if draft_tokens else 0.0,
        "aggregate_prompt_tok_s": (
            prompt_tokens / prompt_wall if prompt_wall > 0 else 0.0
        ),
        "aggregate_decode_tok_s": (
            decoded_tokens / decode_wall if decode_wall > 0 else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        default=os.environ.get("LLAMA_SERVER", "/home/chaidi/qw3/llama.cpp/build-cuda/bin/llama-server"),
    )
    parser.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("-c", "--ctx", type=int, default=1024)
    parser.add_argument("-n", type=int, default=16)
    parser.add_argument("--spec-draft-n-max", type=int)
    parser.add_argument("--spec-draft-n-min", type=int)
    parser.add_argument("--default-prompts", action="store_true")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-tokens",
                        help="Generate long sweep-style prompts for the requested token targets")
    parser.add_argument("--restart-per-prompt", action="store_true",
                        help="Restart llama-server for each prompt to avoid prefix-cache reuse "
                             "contaminating prompt eval timing")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json")
    args = parser.parse_args()

    results, logs, mtp_initialized = run_probe(args)
    summary = summarize(results)
    print("idx ok mtp_initialized prompt_tokens prompt_tok_s decoded draft accepted accept_rate decode_tok_s")
    for result in results:
        if result.ok:
            print(
                f"{result.index:3d} yes {str(result.mtp_initialized).lower():>15} "
                f"{result.prompt_tokens:13d} {result.prompt_tok_s:12.2f} "
                f"{result.decoded_tokens:7d} "
                f"{result.draft_tokens:5d} {result.draft_accepted:8d} "
                f"{result.draft_acceptance:10.3f} {result.decode_tok_s:12.2f}"
            )
        else:
            print(f"{result.index:3d} no  {str(result.mtp_initialized).lower():>15} error={result.error}")
            if result.error:
                print(f"error[{result.index}]: {result.error}", file=sys.stderr)
    print(
        f"TOTAL ok={summary['ok']}/{summary['total']} "
        f"draft={summary['draft_tokens']} accepted={summary['draft_accepted']} "
        f"acceptance={summary['draft_acceptance']:.4f} "
        f"prompt_tok_s={summary['aggregate_prompt_tok_s']:.2f} "
        f"decode_tok_s={summary['aggregate_decode_tok_s']:.2f}"
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"args": vars(args), "summary": summary,
                        "results": [asdict(r) for r in results],
                        "mtp_initialized": mtp_initialized,
                        "server_log_tail": logs[-4000:]},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["ok"] == summary["total"] and mtp_initialized else 1


if __name__ == "__main__":
    raise SystemExit(main())
