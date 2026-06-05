"""Tiny stdlib-only OpenAI-compatible client for the qw3 serve API.

No third-party deps (urllib only) so the eval scripts stay dependency-free. The
qw3 server loads the model once and serves many requests, so these helpers just
POST JSON and parse the response — no per-call model reload.

  from oai_client import chat, complete
  txt = chat("http://127.0.0.1:8080/v1", "model-id",
             [{"role": "user", "content": "2+2?"}], max_tokens=64)
  txt = complete("http://127.0.0.1:8080/v1", "model-id", "raw prompt", max_tokens=64)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(base_url: str, model: str, messages: list, max_tokens: int = 256,
         temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
         stop=None, enable_thinking: bool = False, timeout: float = 600.0) -> str:
    """POST /v1/chat/completions; return choices[0].message.content."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "enable_thinking": enable_thinking,
    }
    if stop:
        payload["stop"] = stop
    out = _post(base_url.rstrip("/") + "/chat/completions", payload, timeout)
    return out["choices"][0]["message"]["content"]


def complete(base_url: str, model: str, prompt: str, max_tokens: int = 256,
             temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
             stop=None, timeout: float = 600.0) -> str:
    """POST /v1/completions (raw prompt, no chat template); return choices[0].text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    if stop:
        payload["stop"] = stop
    out = _post(base_url.rstrip("/") + "/completions", payload, timeout)
    return out["choices"][0]["text"]


def list_models(base_url: str, timeout: float = 30.0) -> list:
    req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return [m["id"] for m in out.get("data", [])]
