"""Benchmark configuration: grid definition + engine/model paths.

Two presets:
  --comprehensive (default): the full 3D sweep.
  --quick: a small smoke grid for CI / sanity.

CRITICAL: the model MUST be the MTP-enabled GGUF. There are two distinct
Qwen3.6-27B-Q8_0.gguf files on this box; only the one in qw3/models/ carries
`qwen35.nextn_predict_layers=1` + the 15 MTP head tensors. The other (AgentSys
default) hard-fails llama.cpp's `--spec-type draft-mtp`. See DEFAULT_MODEL below.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

# The MTP-enabled GGUF (866 tensors, nextn_predict_layers=1). Both engines use it.
DEFAULT_MODEL = os.environ.get(
    "QW3_BENCH_MODEL", "/home/chaidi/qw3/models/Qwen3.6-27B-Q8_0.gguf")
DEFAULT_QW3 = os.environ.get("QW3", "./build/qw3")
DEFAULT_LLAMA_SERVER = os.environ.get(
    "LLAMA_SERVER", "/tmp/llama.cpp/build-cuda/bin/llama-server")


@dataclass
class BenchConfig:
    model: str = DEFAULT_MODEL
    qw3: str = DEFAULT_QW3
    llama_server: str = DEFAULT_LLAMA_SERVER

    prompt_tokens: List[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384, 32768])
    n_decode: List[int] = field(default_factory=lambda: [64, 256, 1024])
    mtp_chain: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    trials: int = 3

    # Engines / modes to run.
    engines: List[str] = field(default_factory=lambda: ["qw3", "llama"])
    run_plain: bool = True
    run_mtp: bool = True

    # llama-server.
    llama_host: str = "127.0.0.1"
    llama_port: int = 18099

    # Per-invocation timeout (seconds). Scaled by prompt+decode at runtime.
    base_timeout_s: float = 600.0
    startup_timeout_s: float = 240.0

    # ctx headroom over prompt+decode.
    ctx_headroom: int = 2048

    def ctx_for(self, prompt_tokens: int, n_decode: int) -> int:
        return max(4096, prompt_tokens + n_decode + self.ctx_headroom)

    def timeout_for(self, prompt_tokens: int, n_decode: int) -> float:
        # Long prompts + long decode need more wall-clock; scale loosely.
        return self.base_timeout_s + prompt_tokens / 200.0 + n_decode * 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def comprehensive(cls) -> "BenchConfig":
        return cls()

    @classmethod
    def quick(cls) -> "BenchConfig":
        return cls(
            prompt_tokens=[512, 4096],
            n_decode=[64, 256],
            mtp_chain=[2],
            trials=2,
        )


def host_label() -> str:
    return socket.gethostname()
