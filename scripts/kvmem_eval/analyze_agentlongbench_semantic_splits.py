#!/usr/bin/env python3
"""Inspect message/punctuation-aware score segments on real AgentLongBench prompts.

This is an offline analysis tool.  It does not change KVMem selection or KV
storage.  It uses qw3's GGUF tokenizer, maps the benchmark's flattened message
boundaries to exact token positions, and then splits selected 32-token scoring
blocks using:

  * hard boundaries at message starts/ends;
  * role-aware soft punctuation boundaries;
  * a midpoint split for any remaining fragment longer than 20 tokens.

The output is intentionally human-readable so proposed prototype boundaries can
be reviewed before they are implemented in the runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    "/home/chaidi/AgentLongBench-Long/DeepseekMillion/samples.jsonl"
)
DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_MODEL = ROOT / "models/Qwen3.6-27B-Q8_0.gguf"
DEFAULT_INSPECT = ROOT / "build/qw3-inspect"
CHAT_PREFIX = "<|im_start|>user\n"


@dataclass(frozen=True)
class MessageSpan:
    raw_index: int
    role: str
    begin: int
    content_begin: int
    end: int
    content: str


@dataclass(frozen=True)
class TokenPiece:
    index: int
    token_id: int
    begin: int
    end: int
    value: bytes


@dataclass(frozen=True)
class Boundary:
    token: int
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--inspect", type=Path, default=DEFAULT_INSPECT)
    parser.add_argument("--sample-index", type=int, required=True, help="one-based")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument(
        "--target",
        default="",
        help="optional target text; the closest preceding occurrence is inspected",
    )
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--physical-page-tokens", type=int, default=16)
    parser.add_argument("--max-segment-tokens", type=int, default=20)
    parser.add_argument("--min-segment-tokens", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=4)
    return parser.parse_args()


def load_canonical(repo: Path) -> Any:
    source = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("agentlongbench_fullcontext", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import canonical worker: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sample(dataset: Path, one_based_index: int) -> dict[str, Any]:
    if one_based_index < 1:
        raise ValueError("--sample-index must be positive")
    with dataset.open(encoding="utf-8") as source:
        for index, line in enumerate(source, start=1):
            if index == one_based_index:
                return json.loads(line)
    raise IndexError(f"sample {one_based_index} is not present in {dataset}")


def prompt_message_spans(
    canonical: Any, sample: dict[str, Any], prompt: str
) -> list[MessageSpan]:
    marker = "Full conversation history:\n"
    history_begin = len(prompt[: prompt.index(marker) + len(marker)].encode("utf-8"))
    history_end = len(prompt[: prompt.index("\n\nQuestion:\n", history_begin)].encode("utf-8"))
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []

    cursor = history_begin
    result: list[MessageSpan] = []
    first = True
    for raw_index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "system":
            continue
        begin = cursor
        if not first:
            cursor += 1  # canonical "\n".join separator, assigned to this message
        rendered_begin = cursor
        rendered = canonical.message_to_text(message)
        rendered_bytes = rendered.encode("utf-8")
        cursor += len(rendered_bytes)
        content = (
            str(message.get("content", ""))
            if isinstance(message, dict)
            else str(message)
        )
        role = (
            str(message.get("role", "unknown"))
            if isinstance(message, dict)
            else "unknown"
        )
        content_offset = rendered_bytes.find(content.encode("utf-8"))
        if content_offset < 0:
            raise RuntimeError(
                f"raw message {raw_index} content is absent from canonical rendering"
            )
        result.append(
            MessageSpan(
                raw_index,
                role,
                begin,
                rendered_begin + content_offset,
                cursor,
                content,
            )
        )
        first = False
    if cursor != history_end:
        raise RuntimeError(
            f"message rendering covers bytes [{history_begin},{cursor}), "
            f"expected [{history_begin},{history_end})"
        )
    return result


def stream_relevant_token_pieces(
    inspect: Path,
    model: Path,
    text: str,
    retain_begin: int,
    retain_end: int,
) -> tuple[list[TokenPiece], int]:
    command = [str(inspect), "--tokenize-pieces-stdin", str(model)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(text.encode("utf-8"))
    process.stdin.close()

    retained: list[TokenPiece] = []
    token_count = 0
    for raw_line in process.stdout:
        fields = raw_line.rstrip(b"\n").split(b"\t", 4)
        if len(fields) != 5:
            process.kill()
            raise RuntimeError(f"malformed qw3-inspect token row: {raw_line[:200]!r}")
        index, token_id, begin, end = map(int, fields[:4])
        token_count = index + 1
        if end > retain_begin and begin < retain_end:
            retained.append(
                TokenPiece(index, token_id, begin, end, bytes.fromhex(fields[4].decode()))
            )
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"qw3-inspect failed with code {return_code}: {stderr[-2000:]}"
        )
    return retained, token_count


def byte_boundary_to_token(
    pieces: list[TokenPiece], byte_offset: int
) -> tuple[int, bool]:
    for piece in pieces:
        if byte_offset == piece.begin:
            return piece.index, False
        if byte_offset == piece.end:
            return piece.index + 1, False
        if piece.begin < byte_offset < piece.end:
            distance_to_begin = byte_offset - piece.begin
            distance_to_end = piece.end - byte_offset
            if distance_to_begin <= distance_to_end:
                return piece.index, True
            return piece.index + 1, True
    raise RuntimeError(f"byte boundary {byte_offset} is outside retained token pieces")


def role_at_token(
    token: TokenPiece, messages: list[MessageSpan], prefix_bytes: int
) -> tuple[str, int | None]:
    midpoint = (token.begin + token.end) // 2 - prefix_bytes
    for message in messages:
        if message.begin <= midpoint < message.end:
            return message.role, message.raw_index
    return "wrapper", None


def message_at_token(
    token: TokenPiece, messages: list[MessageSpan], prefix_bytes: int
) -> MessageSpan | None:
    midpoint = (token.begin + token.end) // 2 - prefix_bytes
    for message in messages:
        if message.begin <= midpoint < message.end:
            return message
    return None


def punctuation_kind(piece: bytes, role: str) -> str | None:
    text = piece.decode("utf-8", errors="ignore").rstrip()
    if not text:
        return "newline" if b"\n" in piece else None
    if "\n" in text:
        return "newline"
    if role == "tool":
        if text.endswith((",", "，")):
            return "tool-comma"
        if text.endswith(("]", "}")):
            return "tool-close"
        return None
    if text.endswith((".", "?", "!", ";", "。", "？", "！", "；")):
        return "sentence"
    return None


def choose_soft_boundaries(
    begin: int,
    end: int,
    candidates: dict[int, str],
    max_segment_tokens: int,
    min_segment_tokens: int,
    available: int,
) -> list[Boundary]:
    chosen: list[Boundary] = []
    segments = [(begin, end)]
    priority = {"newline": 0, "sentence": 1, "tool-close": 2, "tool-comma": 3}
    while available > 0:
        oversized = [
            (segment_end - segment_begin, segment_begin, segment_end)
            for segment_begin, segment_end in segments
            if segment_end - segment_begin > max_segment_tokens
        ]
        if not oversized:
            break
        _, segment_begin, segment_end = max(oversized)
        midpoint = (segment_begin + segment_end) / 2.0
        valid = [
            position
            for position in candidates
            if segment_begin + min_segment_tokens
            <= position
            <= segment_end - min_segment_tokens
        ]
        if not valid:
            break
        position = min(
            valid,
            key=lambda item: (
                abs(item - midpoint),
                priority.get(candidates[item], 9),
            ),
        )
        chosen.append(Boundary(position, candidates[position]))
        segments.remove((segment_begin, segment_end))
        segments.extend(((segment_begin, position), (position, segment_end)))
        available -= 1
    return chosen


def segment_block(
    block_begin: int,
    block_end: int,
    pieces_by_index: dict[int, TokenPiece],
    messages: list[MessageSpan],
    prefix_bytes: int,
    max_segment_tokens: int,
    min_segment_tokens: int,
    max_segments: int,
) -> list[Boundary]:
    hard_positions: dict[int, str] = {
        block_begin: "block-start",
        block_end: "block-end",
    }
    for message in messages:
        for byte_position, kind in (
            (message.begin + prefix_bytes, "message-start"),
            (message.end + prefix_bytes, "message-end"),
        ):
            try:
                token_position, rounded = byte_boundary_to_token(
                    list(pieces_by_index.values()), byte_position
                )
            except RuntimeError:
                continue
            if block_begin < token_position < block_end:
                suffix = "-rounded" if rounded else ""
                hard_positions[token_position] = kind + suffix

    soft_candidates: dict[int, str] = {}
    for token_index in range(block_begin, block_end - 1):
        piece = pieces_by_index[token_index]
        message = message_at_token(piece, messages, prefix_bytes)
        role = message.role if message is not None else "wrapper"
        # The newline after "[user]", "[assistant]", or "[tool ...]" is a
        # serialization detail, not a semantic sentence boundary.
        if (
            message is not None
            and piece.end <= prefix_bytes + message.content_begin
        ):
            continue
        kind = punctuation_kind(piece.value, role)
        if kind is not None:
            soft_candidates[token_index + 1] = kind

    hard = sorted(hard_positions)
    soft: list[Boundary] = []
    available = max(0, max_segments - (len(hard) - 1))
    for begin, end in zip(hard, hard[1:]):
        if available <= 0:
            break
        additions = choose_soft_boundaries(
            begin,
            end,
            soft_candidates,
            max_segment_tokens,
            min_segment_tokens,
            available,
        )
        soft.extend(additions)
        available -= len(additions)
    return sorted(
        [Boundary(position, kind) for position, kind in hard_positions.items()] + soft,
        key=lambda boundary: boundary.token,
    )


def escaped(data: bytes, limit: int = 300) -> str:
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def print_block(
    title: str,
    anchor_token: int,
    block_tokens: int,
    page_tokens: int,
    pieces: list[TokenPiece],
    messages: list[MessageSpan],
    prefix_bytes: int,
    args: argparse.Namespace,
) -> None:
    block_begin = anchor_token // block_tokens * block_tokens
    block_end = block_begin + block_tokens
    pieces_by_index = {piece.index: piece for piece in pieces}
    missing = [
        index for index in range(block_begin, block_end) if index not in pieces_by_index
    ]
    if missing:
        raise RuntimeError(
            f"retained range does not contain block [{block_begin},{block_end}); "
            f"missing {missing[:5]}"
        )
    boundaries = segment_block(
        block_begin,
        block_end,
        pieces_by_index,
        messages,
        prefix_bytes,
        args.max_segment_tokens,
        args.min_segment_tokens,
        args.max_segments,
    )

    raw = b"".join(pieces_by_index[index].value for index in range(block_begin, block_end))
    page_ids = sorted({index // page_tokens for index in range(block_begin, block_end)})
    print(f"\n=== {title} ===")
    print(
        f"KVMem score block: tokens [{block_begin},{block_end}) "
        f"({block_tokens} tokens), physical pages={page_ids}"
    )
    print(f"unsplit: {escaped(raw)}")
    print("dynamic segments:")
    for ordinal, (left, right) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        data = b"".join(
            pieces_by_index[index].value for index in range(left.token, right.token)
        )
        roles = []
        raw_indices = []
        for index in range(left.token, right.token):
            role, raw_index = role_at_token(
                pieces_by_index[index], messages, prefix_bytes
            )
            if role not in roles:
                roles.append(role)
            if raw_index is not None and raw_index not in raw_indices:
                raw_indices.append(raw_index)
        print(
            f"  {ordinal}. [{left.token},{right.token}) "
            f"len={right.token-left.token:2d} "
            f"start={left.kind:<18} roles={'+'.join(roles)} "
            f"raw_messages={raw_indices}: {escaped(data)}"
        )


def main() -> None:
    args = parse_args()
    canonical = load_canonical(args.benchmark_repo)
    sample = load_sample(args.dataset, args.sample_index)
    prompt = canonical.full_context_prompt(sample)
    messages = prompt_message_spans(canonical, sample, prompt)
    prefix_bytes = len(CHAT_PREFIX.encode("utf-8"))

    round_prefix = f"Round {args.round}:"
    round_messages = [
        message
        for message in messages
        if message.role == "user" and message.content.startswith(round_prefix)
    ]
    if len(round_messages) != 1:
        raise RuntimeError(
            f"expected one user message starting {round_prefix!r}, "
            f"found {len(round_messages)}"
        )
    round_message = round_messages[0]

    target_message: MessageSpan | None = None
    target_byte: int | None = None
    if args.target:
        candidates = [
            message
            for message in messages
            if message.begin < round_message.begin and args.target in message.content
        ]
        if not candidates:
            raise RuntimeError(
                f"{args.target!r} does not occur before {round_prefix!r}"
            )
        target_message = candidates[-1]
        prompt_bytes = prompt.encode("utf-8")
        local_byte = prompt_bytes[
            target_message.begin : target_message.end
        ].rfind(args.target.encode("utf-8"))
        target_byte = prefix_bytes + target_message.begin + local_byte

    interesting_bytes = [
        prefix_bytes + round_message.begin,
        prefix_bytes + round_message.end,
    ]
    previous_messages = [
        message for message in messages if message.end <= round_message.begin
    ][-2:]
    for message in previous_messages:
        interesting_bytes.extend(
            (prefix_bytes + message.begin, prefix_bytes + message.end)
        )
    if target_byte is not None:
        interesting_bytes.extend(
            (target_byte, target_byte + len(args.target.encode("utf-8")))
        )
    retain_begin = max(0, min(interesting_bytes) - 8192)
    retain_end = max(interesting_bytes) + 8192

    actual_prompt = CHAT_PREFIX + prompt
    pieces, total_tokens = stream_relevant_token_pieces(
        args.inspect, args.model, actual_prompt, retain_begin, retain_end
    )
    round_token, round_rounded = byte_boundary_to_token(
        pieces, prefix_bytes + round_message.begin
    )

    print(f"sample_index: {args.sample_index}")
    print(f"stable_sample_id: {sample.get('stable_sample_id')}")
    print(f"question: {sample.get('question')}")
    print(f"answer: {sample.get('answer')}")
    print(f"actual prompt tokens (chat prefix + canonical prompt): {total_tokens}")
    print(
        f"round user raw_message={round_message.raw_index}, "
        f"byte={prefix_bytes+round_message.begin}, token={round_token}, "
        f"rounded={round_rounded}"
    )

    print_block(
        f"block crossing the Round {args.round} user-message boundary",
        round_token,
        args.block_tokens,
        args.physical_page_tokens,
        pieces,
        messages,
        prefix_bytes,
        args,
    )

    if target_byte is not None and target_message is not None:
        target_token, target_rounded = byte_boundary_to_token(pieces, target_byte)
        print(
            f"\ntarget={args.target!r}, closest preceding raw_message="
            f"{target_message.raw_index} ({target_message.role}), "
            f"byte={target_byte}, token={target_token}, rounded={target_rounded}"
        )
        print_block(
            f"block containing target {args.target!r}",
            target_token,
            args.block_tokens,
            args.physical_page_tokens,
            pieces,
            messages,
            prefix_bytes,
            args,
        )


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(1)
