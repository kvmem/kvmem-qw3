#!/usr/bin/env python3
"""OpenAI-compatible HTTP client for the qw3 serve KVMem evaluation.

qw3 is the isolated inference service: this harness touches it *only* through the
OpenAI-compatible chat-completions endpoint (no eval logic lives in qw3 C++). We
send one request at a time (concurrency = 1, matching the motivation study and
avoiding tier/GPU contention), stream the response to measure time-to-first-token,
and keep the model's <think> reasoning separate from the final answer.

The qw3 server (src/qw3_server.cpp) emits reasoning and answer as two distinct SSE
delta fields:

    delta.reasoning_content  -> chain-of-thought (the <think>...</think> block)
    delta.content            -> the answer the judge should grade

So `ChatResult.answer` holds the post-thinking content only; `ChatResult.reasoning`
holds the chain-of-thought (kept for logging/debugging, never sent to the judge).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ChatResult:
    answer: str                      # post-thinking content (graded by the judge)
    reasoning: str                   # chain-of-thought; may be empty
    ttft_s: float | None             # wall-clock seconds to first streamed token
    latency_s: float                 # wall-clock seconds for the whole request
    finish_reason: str               # "stop" | "length" | "tool_calls" | ...
    prompt_tokens: int | None
    completion_tokens: int | None
    first_part: str                  # "reasoning" | "content" | "" (which arrived first)
    error: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class Qw3Client:
    """Sequential OpenAI-compatible client for a single qw3 serve endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "dummy",
        model: str = "Qwen3.6-27B-Q8_0.gguf",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
        connect_timeout: float = 30.0,
        read_timeout: float = 3600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._session = requests.Session()

    # -- health ----------------------------------------------------------------

    def health(self) -> bool:
        """True if the server answers /health (base_url is .../v1, /health is at root)."""
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            r = self._session.get(f"{root}/health", timeout=(self.connect_timeout, 10.0))
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -- chat ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], **overrides: Any) -> ChatResult:
        """Stream one chat completion. Returns answer + reasoning + timing.

        `overrides` may set temperature / max_tokens / enable_thinking per call.
        Network/HTTP errors are captured into ChatResult.error rather than raised,
        so a single bad sample doesn't abort a 102-sample run.
        """
        payload = {
            "model": overrides.get("model", self.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "enable_thinking": overrides.get("enable_thinking", self.enable_thinking),
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        t0 = time.monotonic()
        ttft: float | None = None
        first_part = ""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        finish_reason = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                return ChatResult(
                    answer="", reasoning="", ttft_s=None,
                    latency_s=time.monotonic() - t0, finish_reason="error",
                    prompt_tokens=None, completion_tokens=None, first_part="",
                    error=f"HTTP {resp.status_code}: {body}",
                )

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # usage may ride on a trailing chunk (choices empty)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                rc = delta.get("reasoning_content")
                if rc:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                        first_part = first_part or "reasoning"
                    reasoning_parts.append(rc)

                cc = delta.get("content")
                if cc:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                        first_part = first_part or "content"
                    content_parts.append(cc)

                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
        except requests.RequestException as exc:
            return ChatResult(
                answer="".join(content_parts), reasoning="".join(reasoning_parts),
                ttft_s=ttft, latency_s=time.monotonic() - t0,
                finish_reason="error", prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, first_part=first_part,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ChatResult(
            answer="".join(content_parts).strip(),
            reasoning="".join(reasoning_parts).strip(),
            ttft_s=ttft,
            latency_s=time.monotonic() - t0,
            finish_reason=finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            first_part=first_part,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the qw3 serve endpoint.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--prompt", default="In one sentence, what is a KV cache?")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    cli = Qw3Client(base_url=args.base_url, max_tokens=args.max_tokens,
                    enable_thinking=not args.no_thinking)
    print(f"health: {cli.health()}")
    res = cli.chat([{"role": "user", "content": args.prompt}])
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print(f"--- reasoning ({len(res.reasoning)} chars) ---\n{res.reasoning[:400]}")
        print(f"--- answer ---\n{res.answer}")
        print(f"ttft={res.ttft_s:.3f}s latency={res.latency_s:.3f}s "
              f"finish={res.finish_reason} first={res.first_part} "
              f"prompt_tok={res.prompt_tokens} completion_tok={res.completion_tokens}")
