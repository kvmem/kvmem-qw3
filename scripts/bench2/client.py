"""Wall-clock timing client for any OpenAI-compatible server (stdlib only).

Two measurement methods, both pure HTTP — no stderr, no model reload:

  measure_stream(): POST /v1/chat/completions with stream=true and time the
      SSE chunks. The first *content* chunk lands after prefill, so
      ttft_s == prefill latency for the prompt; the gaps between subsequent
      content chunks give the decode inter-token latency, and
      decode_tok_s == (n_tok - 1) / (t_last - t_first). This is the preferred
      method (one request yields both prefill and decode timing).

  measure_two_point(): for servers whose /v1/completions does not stream, fire
      two non-streamed raw completions — max_tokens=1 (≈ prefill + 1 token) and
      max_tokens=M — and isolate decode from the wall-clock difference:
          prefill_s ≈ t(max_tokens=1)
          decode_tok_s ≈ (M - 1) / (t(M) - t(1))
      Greedy (temperature=0, seed=0) so repeated calls are deterministic.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class Timing:
    ok: bool = True
    error: str = ""
    prompt_tokens: int = 0          # echoed if server reports usage; else 0
    decoded_tokens: int = 0
    prefill_s: float = 0.0          # wall-clock to first decoded token (TTFT)
    decode_s: float = 0.0           # wall-clock for the decode-only span
    prefill_tok_s: float = 0.0      # prompt_tokens / prefill_s (0 if unknown)
    decode_tok_s: float = 0.0       # decoded / decode_s
    ttft_s: float = 0.0             # == prefill_s
    method: str = ""                # "stream" | "two_point"


def _models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def first_model_id(base_url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(_models_url(base_url), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    data = out.get("data", [])
    if not data:
        raise RuntimeError(f"no models reported at {base_url}")
    return data[0]["id"]


def wait_ready(base_url: str, timeout: float = 240.0) -> str:
    """Block until /v1/models answers; return the first model id."""
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            return first_model_id(base_url, timeout=2.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_err = exc
            time.sleep(0.3)
    raise TimeoutError(f"server at {base_url} not ready: {last_err}")


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure_stream(base_url: str, model: str, prompt: str, n_decode: int,
                   prompt_tokens_hint: int = 0, timeout: float = 1200.0) -> Timing:
    """Stream /v1/chat/completions and time the SSE chunks.

    ttft_s is the wall-clock to the first content chunk (prefill latency);
    decode_tok_s is (n-1)/(t_last - t_first) over the content chunks.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_decode,
        "temperature": 0.0, "top_p": 1.0, "seed": 0,
        "enable_thinking": False,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_last = t_start
    n_tok = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" not in delta or delta["content"] == "":
                    continue  # skip the leading role-only chunk
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tok += 1
    except Exception as exc:  # noqa: BLE001
        return Timing(ok=False, error=f"stream request failed: {exc}", method="stream")
    if t_first is None or n_tok == 0:
        return Timing(ok=False, error="no content chunks received", method="stream")
    prefill_s = t_first - t_start
    decode_s = t_last - t_first
    decode_tok_s = (n_tok - 1) / decode_s if (n_tok > 1 and decode_s > 0) else 0.0
    prefill_tok_s = prompt_tokens_hint / prefill_s if (prompt_tokens_hint and prefill_s > 0) else 0.0
    return Timing(
        ok=True, prompt_tokens=prompt_tokens_hint, decoded_tokens=n_tok,
        prefill_s=prefill_s, decode_s=decode_s,
        prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s,
        ttft_s=prefill_s, method="stream")


def measure_two_point(base_url: str, model: str, prompt: str, n_decode: int,
                      prompt_tokens_hint: int = 0, timeout: float = 1200.0) -> Timing:
    """Non-streamed two-point timing on /v1/completions (raw prompt).

    t1 = wall-clock for max_tokens=1 (prefill + 1 decoded token).
    tM = wall-clock for max_tokens=n_decode.
    prefill_s ≈ t1; decode_tok_s ≈ (n_decode - 1) / (tM - t1).
    """
    url = base_url.rstrip("/") + "/completions"
    base = {"model": model, "prompt": prompt,
            "temperature": 0.0, "top_p": 1.0, "seed": 0}

    def _timed(max_tokens: int) -> float:
        t0 = time.perf_counter()
        _post_json(url, {**base, "max_tokens": max_tokens}, timeout)
        return time.perf_counter() - t0

    try:
        t1 = _timed(1)
        tM = _timed(n_decode) if n_decode > 1 else t1
    except Exception as exc:  # noqa: BLE001
        return Timing(ok=False, error=f"request failed: {exc}", method="two_point")
    decode_s = max(tM - t1, 0.0)
    decode_tok_s = (n_decode - 1) / decode_s if (n_decode > 1 and decode_s > 0) else 0.0
    prefill_tok_s = prompt_tokens_hint / t1 if (prompt_tokens_hint and t1 > 0) else 0.0
    return Timing(
        ok=True, prompt_tokens=prompt_tokens_hint, decoded_tokens=n_decode,
        prefill_s=t1, decode_s=decode_s,
        prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s,
        ttft_s=t1, method="two_point")
