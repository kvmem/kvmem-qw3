#!/usr/bin/env python3
"""Replay captured OpenAI-compatible requests and compare model responses."""

from __future__ import annotations

import argparse
import difflib
import http.client
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class ReplayResult:
    index: int
    method: str
    path: str
    status: int
    elapsed_ms: float
    request_prompt_chars: int
    reference_status: int | None
    exact_raw_match: bool
    semantic_match: bool
    first_diff: int | None
    reference_text_preview: str
    replay_text_preview: str
    reference_text: str
    replay_text: str
    replay_response_raw: str
    error: str | None = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def combine_path(base: str, path: str) -> str:
    parsed = urlparse(base)
    if parsed.path and parsed.path != "/":
        prefix = parsed.path.rstrip("/")
        if not path.startswith(prefix + "/") and path != prefix:
            return prefix + path
    return path


def post(base_url: str, method: str, path: str, body_obj: Any, timeout_s: float) -> tuple[int, bytes, float, str | None]:
    parsed = urlparse(base_url)
    req_path = combine_path(base_url, path)
    body = dump_json(body_obj).encode("utf-8") if body_obj is not None else b""
    headers = {"Content-Type": "application/json"}
    start = time.monotonic()
    try:
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout_s)
        conn.request(method, req_path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
        conn.close()
        return status, raw, (time.monotonic() - start) * 1000.0, None
    except Exception as exc:  # noqa: BLE001
        return 0, b"", (time.monotonic() - start) * 1000.0, str(exc)


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise argparse.ArgumentTypeError("--set expects key=json_value")
    key, raw = item.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("--set key cannot be empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def apply_overrides(body: Any, overrides: list[tuple[str, Any]]) -> Any:
    if not overrides or not isinstance(body, dict):
        return body
    out = json.loads(json.dumps(body, ensure_ascii=False))
    for key, value in overrides:
        out[key] = value
    return out


def first_diff(a: str, b: str) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def extract_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        ch = choices[0]
        if isinstance(ch, dict):
            msg = ch.get("message")
            if isinstance(msg, dict):
                parts = []
                for key in ("reasoning_content", "content"):
                    val = msg.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                if msg.get("tool_calls") is not None:
                    parts.append(json.dumps(msg.get("tool_calls"), ensure_ascii=False, sort_keys=True))
                return "\n".join(parts)
            if isinstance(ch.get("text"), str):
                return ch["text"]
            delta = ch.get("delta")
            if isinstance(delta, dict):
                return json.dumps(delta, ensure_ascii=False, sort_keys=True)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def prompt_chars(body: Any) -> int:
    if not isinstance(body, dict):
        return 0
    if isinstance(body.get("prompt"), str):
        return len(body["prompt"])
    total = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return total


def preview(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... <truncated {len(text) - limit} chars>"


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay requests captured by openai_proxy_capture.py.")
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--base-url", required=True, help="Target base URL, e.g. http://127.0.0.1:8091")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-post", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--preview-chars", type=int, default=512)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=parse_override,
        help="Override request JSON field, e.g. --set top_k=1 --set seed=1.",
    )
    ap.add_argument(
        "--reference-json",
        type=Path,
        help="Use replay_response_raw from a prior replay as the comparison reference.",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.capture)
    prior_refs: list[dict[str, Any]] | None = None
    if args.reference_json:
        prior = json.loads(args.reference_json.read_text(encoding="utf-8"))
        prior_refs = prior.get("results") or []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if args.only_post and row.get("method") != "POST":
            continue
        if row.get("request_body") is None:
            continue
        selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break

    results: list[ReplayResult] = []
    for idx, row in enumerate(selected):
        method = row.get("method", "POST")
        path = row.get("path", "/v1/chat/completions")
        req = apply_overrides(row.get("request_body"), args.overrides)
        if prior_refs is not None:
            ref_item = prior_refs[idx] if idx < len(prior_refs) else {}
            ref_raw = ref_item.get("replay_response_raw") or ""
            ref_status = ref_item.get("status")
        else:
            ref_raw = row.get("response_raw") or ""
            ref_status = row.get("response_status")
        status, raw, elapsed_ms, error = post(args.base_url, method, path, req, args.timeout)
        replay_raw = raw.decode("utf-8", "replace")
        exact = error is None and status == ref_status and replay_raw == ref_raw
        try:
            ref_obj = json.loads(ref_raw) if ref_raw else None
        except Exception:
            ref_obj = None
        try:
            replay_obj = json.loads(replay_raw) if replay_raw else None
        except Exception:
            replay_obj = None
        ref_text = extract_text(ref_obj)
        replay_text = extract_text(replay_obj)
        semantic = error is None and status == ref_status and ref_text == replay_text
        diff = first_diff(ref_text, replay_text)
        results.append(
            ReplayResult(
                index=idx,
                method=method,
                path=path,
                status=status,
                elapsed_ms=elapsed_ms,
                request_prompt_chars=prompt_chars(req),
                reference_status=ref_status,
                exact_raw_match=exact,
                semantic_match=semantic,
                first_diff=diff,
                reference_text_preview=preview(ref_text, args.preview_chars),
                replay_text_preview=preview(replay_text, args.preview_chars),
                reference_text=ref_text,
                replay_text=replay_text,
                replay_response_raw=replay_raw,
                error=error,
            )
        )
        print(
            f"{idx:04d} status={status} ref={ref_status} "
            f"semantic={'ok' if semantic else 'DIFF'} elapsed={elapsed_ms:.1f}ms "
            f"prompt_chars={results[-1].request_prompt_chars}"
        )

    semantic_matches = sum(1 for r in results if r.semantic_match)
    exact_matches = sum(1 for r in results if r.exact_raw_match)
    summary = {
        "capture": str(args.capture),
        "base_url": args.base_url,
        "count": len(results),
        "semantic_matches": semantic_matches,
        "exact_raw_matches": exact_matches,
        "all_semantic_match": semantic_matches == len(results),
        "all_exact_raw_match": exact_matches == len(results),
        "results": [asdict(r) for r in results],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"replay summary: semantic {semantic_matches}/{len(results)}, "
        f"exact {exact_matches}/{len(results)} -> {args.out_json}"
    )
    return 0 if semantic_matches == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
