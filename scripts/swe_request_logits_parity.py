#!/usr/bin/env python3
"""Run logits-level baseline/KVMem parity on captured OpenAI chat requests."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def trim_ascii_ws(s: str) -> str:
    return s.strip(" \t\r\n")


def render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(item["text"])
        return "".join(out)
    return dump_json(content)


def render_tool_call(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    if not isinstance(fn, dict):
        return ""
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return ""
    args: Any = {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        args = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else {"arguments": raw}
        except json.JSONDecodeError:
            args = {"arguments": raw}
    out = f"<tool_call>\n<function={name}>\n"
    if isinstance(args, dict):
        for key, value in args.items():
            out += f"<parameter={key}>\n"
            out += value if isinstance(value, str) else dump_json(value)
            out += "\n</parameter>\n"
    out += "</function>\n</tool_call>"
    return out


def split_reasoning(text: str) -> tuple[str, str]:
    open_tag = "<think>"
    close_tag = "</think>"
    start = text.find(open_tag)
    if start < 0:
        return "", text
    reasoning_start = start + len(open_tag)
    end = text.find(close_tag, reasoning_start)
    if end < 0:
        reasoning = text[reasoning_start:]
        if reasoning.startswith("\n"):
            reasoning = reasoning[1:]
        return reasoning, ""
    reasoning = text[reasoning_start:end]
    content = text[end + len(close_tag):]
    if reasoning.startswith("\n"):
        reasoning = reasoning[1:]
    if reasoning.endswith("\n"):
        reasoning = reasoning[:-1]
    while content.startswith(("\n", "\r")):
        content = content[1:]
    return reasoning, content


def is_tool_response_content(content: str) -> bool:
    trimmed = trim_ascii_ws(content)
    return trimmed.startswith("<tool_response>") and trimmed.endswith("</tool_response>")


def last_query_index_for_template(messages: list[Any]) -> int:
    if not messages:
        return 0
    multi_step_tool = True
    last_query = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        item = messages[i]
        if not isinstance(item, dict):
            continue
        if multi_step_tool and item.get("role") == "user":
            content = trim_ascii_ws(render_content(item.get("content")))
            if not is_tool_response_content(content):
                multi_step_tool = False
                last_query = i
    return last_query


def response_usage_prompt_tokens(row: dict[str, Any]) -> int | None:
    body = row.get("response_body")
    if isinstance(body, dict):
        usage = body.get("usage")
    elif isinstance(body, str):
        try:
            usage = json.loads(body).get("usage")
        except Exception:
            usage = None
    else:
        usage = None
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        return usage["prompt_tokens"]
    return None


def render_messages(messages: Any,
                    tools: Any,
                    enable_thinking: bool,
                    forced_tool_name: str = "") -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")

    num_sys = 0
    merged_system = ""
    if messages and isinstance(messages[0], dict):
        first_role = messages[0].get("role", "")
        if first_role in {"system", "developer"}:
            merged_system = trim_ascii_ws(render_content(messages[0].get("content")))
            num_sys = 1
            if len(messages) > 1 and isinstance(messages[1], dict):
                second_role = messages[1].get("role", "")
                if second_role in {"system", "developer"}:
                    second = trim_ascii_ws(render_content(messages[1].get("content")))
                    merged_system += "\n" + second
                    num_sys = 2

    prompt = ""
    if isinstance(tools, list) and tools:
        prompt += "<|im_start|>system\n"
        prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for tool in tools:
            prompt += "\n" + dump_json(tool)
        prompt += "\n</tools>"
        prompt += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        prompt += "<tool_call>\n<function=example_function_name>\n"
        prompt += "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        prompt += "<parameter=example_parameter_2>\n"
        prompt += "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        prompt += "</parameter>\n</function>\n</tool_call>\n\n"
        prompt += "<IMPORTANT>\n"
        prompt += "Reminder:\n"
        prompt += "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        prompt += "- Required parameters MUST be specified\n"
        prompt += "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        if forced_tool_name:
            prompt += f"- You MUST call the function named `{forced_tool_name}`\n"
        prompt += "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        prompt += "</IMPORTANT>"
        if merged_system:
            prompt += "\n\n" + merged_system
        prompt += "<|im_end|>\n"
    elif merged_system:
        prompt += "<|im_start|>system\n" + merged_system + "<|im_end|>\n"

    last_query_index = last_query_index_for_template(messages)
    for i, item in enumerate(messages):
        if not isinstance(item, dict) or i < num_sys:
            continue
        role = item.get("role", "")
        if role in {"system", "developer"}:
            continue
        content = trim_ascii_ws(render_content(item.get("content")))
        if role == "user":
            prompt += "<|im_start|>user\n" + content + "<|im_end|>\n"
        elif role == "assistant":
            assistant_content = content
            reasoning_content = ""
            if isinstance(item.get("reasoning_content"), str):
                reasoning_content = trim_ascii_ws(item["reasoning_content"])
            else:
                split_reason, split_content = split_reasoning(assistant_content)
                if split_reason or split_content != assistant_content:
                    reasoning_content = trim_ascii_ws(split_reason)
                    assistant_content = split_content
            prompt += "<|im_start|>assistant\n"
            if i > last_query_index:
                prompt += "<think>\n" + reasoning_content + "\n</think>\n\n"
            prompt += assistant_content
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                first = True
                for call in tool_calls:
                    rendered = render_tool_call(call)
                    if not rendered:
                        continue
                    if first:
                        if assistant_content:
                            prompt += "\n\n"
                    else:
                        prompt += "\n"
                    prompt += rendered
                    first = False
            prompt += "<|im_end|>\n"
        elif role == "tool":
            prev_tool = (
                i > 0 and isinstance(messages[i - 1], dict)
                and messages[i - 1].get("role") == "tool"
            )
            next_tool = (
                i + 1 < len(messages) and isinstance(messages[i + 1], dict)
                and messages[i + 1].get("role") == "tool"
            )
            if not prev_tool:
                prompt += "<|im_start|>user"
            prompt += "\n<tool_response>\n" + content + "\n</tool_response>"
            if not next_tool:
                prompt += "<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    if enable_thinking:
        prompt += "<think>\n"
    else:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def load_posts(capture: Path) -> list[dict[str, Any]]:
    posts = []
    with capture.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") == "POST" and row.get("request_body") is not None:
                posts.append(row)
    return posts


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class PhaseCompare:
    steps_compared: int
    argmax_equal: bool
    top_ids_equal: bool
    first_argmax_diff_step: int | None
    first_top_diff_step: int | None
    first_argmax_diff_margin: float | None
    max_common_top_logit_abs_diff: float


@dataclass
class CaseSummary:
    request_index: int
    prompt_tokens: int
    captured_server_prompt_tokens: int | None
    prompt_matches_captured_server: bool | None
    baseline_stdout: str
    kvmem_stdout: str
    prompt_tokens_equal: bool
    prefill: PhaseCompare
    decode: PhaseCompare
    baseline_elapsed_s: float
    kvmem_elapsed_s: float


def run_cli(cmd: list[str], stdout_path: Path, stderr_path: Path, timeout_s: int) -> float:
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, text=True, stdout=out, stderr=err, timeout=timeout_s)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return elapsed


def compare_dumps(base_dump: Path, kvmem_dump: Path) -> dict[str, Any]:
    b_rows = load_jsonl(base_dump)
    k_rows = load_jsonl(kvmem_dump)
    b_prompt = b_rows[0]
    k_prompt = k_rows[0]
    prompt_equal = b_prompt.get("tokens") == k_prompt.get("tokens")

    def phase_compare(phase: str) -> PhaseCompare:
        b_steps = [r for r in b_rows if r.get("event") == "step" and r.get("phase") == phase]
        k_steps = [r for r in k_rows if r.get("event") == "step" and r.get("phase") == phase]
        first_argmax = None
        first_top = None
        first_margin = None
        max_abs = 0.0
        for i, (b, k) in enumerate(zip(b_steps, k_steps)):
            if first_argmax is None and b.get("argmax_token") != k.get("argmax_token"):
                first_argmax = i
                top = b.get("top", [])
                if isinstance(top, list) and len(top) >= 2:
                    first_margin = abs(float(top[0].get("logit", 0.0)) -
                                       float(top[1].get("logit", 0.0)))
            b_top_ids = [x.get("id") for x in b.get("top", [])]
            k_top_ids = [x.get("id") for x in k.get("top", [])]
            if first_top is None and b_top_ids != k_top_ids:
                first_top = i
            b_logits = {x.get("id"): x.get("logit") for x in b.get("top", [])}
            k_logits = {x.get("id"): x.get("logit") for x in k.get("top", [])}
            for tid in set(b_logits) & set(k_logits):
                max_abs = max(max_abs, abs(float(b_logits[tid]) - float(k_logits[tid])))
        return PhaseCompare(
            steps_compared=min(len(b_steps), len(k_steps)),
            argmax_equal=first_argmax is None and len(b_steps) == len(k_steps),
            top_ids_equal=first_top is None and len(b_steps) == len(k_steps),
            first_argmax_diff_step=first_argmax,
            first_top_diff_step=first_top,
            first_argmax_diff_margin=first_margin,
            max_common_top_logit_abs_diff=max_abs,
        )

    return {
        "prompt_tokens": len(b_prompt.get("tokens", [])),
        "prompt_tokens_equal": prompt_equal,
        "prefill": phase_compare("prefill"),
        "decode": phase_compare("decode"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Logits parity on captured SWE/OpenAI requests.")
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--indices", default="0", help="Comma-separated POST indices.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--dump-top-k", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--kvmem-budget", type=int, default=65536)
    ap.add_argument("--kvmem-block-tokens", type=int, default=128)
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument(
        "--require-captured-prompt-match",
        action="store_true",
        help="Fail if the locally rendered prompt token count differs from the captured server usage.",
    )
    args = ap.parse_args()

    posts = load_posts(args.capture)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[CaseSummary] = []
    for idx in indices:
        body = posts[idx]["request_body"]
        tool_choice = body.get("tool_choice")
        forced = ""
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                forced = fn["name"]
        prompt = render_messages(
            body["messages"],
            None if tool_choice == "none" else body.get("tools"),
            bool(body.get("enable_thinking", False)),
            forced,
        )
        case_dir = args.out_dir / f"request_{idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = case_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        common = [
            args.qw3,
            "--backend", "qwen-native",
            "--model", args.model,
            "--native-heavy",
            "--raw",
            "--kv-dtype", args.kv_dtype,
            "--temp", "0",
            "--top-k", "1",
            "--seed", "1",
            "--prefill-chunk", str(args.prefill_chunk),
            "-c", str(args.ctx),
            "-n", str(args.max_tokens),
            "--prompt-file", str(prompt_path),
            "--dump-logits-top-k", str(args.dump_top_k),
        ]
        base_dump = case_dir / "baseline_logits.jsonl"
        kvmem_dump = case_dir / "kvmem_logits.jsonl"
        base_elapsed = run_cli(
            common + ["--dump-logits", str(base_dump)],
            case_dir / "baseline_stdout.txt",
            case_dir / "baseline_stderr.txt",
            args.timeout,
        )
        kvmem_elapsed = run_cli(
            common + [
                "--dump-logits", str(kvmem_dump),
                "--kvmem",
                "--kvmem-block-tokens", str(args.kvmem_block_tokens),
                "--kvmem-budget", str(args.kvmem_budget),
                "--kvmem-update-mode", "step",
                "--kvmem-method", "retrieval",
                "--kvmem-retrieval-method", "mean_attention",
            ],
            case_dir / "kvmem_stdout.txt",
            case_dir / "kvmem_stderr.txt",
            args.timeout,
        )
        cmp = compare_dumps(base_dump, kvmem_dump)
        captured_prompt_tokens = response_usage_prompt_tokens(posts[idx])
        summary = CaseSummary(
            request_index=idx,
            captured_server_prompt_tokens=captured_prompt_tokens,
            prompt_matches_captured_server=(
                None if captured_prompt_tokens is None
                else captured_prompt_tokens == cmp["prompt_tokens"]
            ),
            baseline_stdout=(case_dir / "baseline_stdout.txt").read_text(encoding="utf-8", errors="replace"),
            kvmem_stdout=(case_dir / "kvmem_stdout.txt").read_text(encoding="utf-8", errors="replace"),
            baseline_elapsed_s=base_elapsed,
            kvmem_elapsed_s=kvmem_elapsed,
            **cmp,
        )
        summaries.append(summary)
        print(
            f"request {idx}: prompt_tokens={summary.prompt_tokens} "
            f"captured_prompt_tokens={captured_prompt_tokens} "
            f"decode_argmax_equal={summary.decode.argmax_equal} "
            f"decode_top_ids_equal={summary.decode.top_ids_equal} "
            f"decode_max_abs={summary.decode.max_common_top_logit_abs_diff:.6g}"
        )

    out = {
        "ok": all(
            s.prompt_tokens_equal and
            (not args.require_captured_prompt_match or s.prompt_matches_captured_server is not False) and
            s.decode.argmax_equal
            for s in summaries
        ),
        "summaries": [asdict(s) for s in summaries],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
