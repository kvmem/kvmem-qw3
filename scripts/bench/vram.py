"""VRAM polling for the benchmark suite.

Thin wrapper over scripts/memory_sweep.py's MemoryPoller (50 ms nvidia-smi
polling) so the bench package does not re-implement it. Adds run-with-polling
helpers and an idle-wait so peak readings are not contaminated by a previous
cell's teardown.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from memory_sweep import MemoryPoller, _smi_query  # type: ignore  # noqa: E402


def gpu_used_mib() -> Optional[int]:
    return _smi_query()


def wait_for_idle(threshold_mib: int = 2000, max_wait_s: float = 30.0) -> int:
    """Block until GPU memory drops near idle. Returns observed baseline MiB."""
    deadline = time.time() + max_wait_s
    last = _smi_query() or 0
    while time.time() < deadline:
        cur = _smi_query() or last
        if cur < threshold_mib:
            return cur
        last = cur
        time.sleep(0.5)
    return last


def run_with_polling(cmd: List[str], env: Mapping[str, str], timeout: float,
                     stdin_devnull: bool = False
                     ) -> Tuple[subprocess.CompletedProcess, int]:
    """Run cmd to completion while sampling peak GPU memory at 50 ms."""
    poller = MemoryPoller(interval_s=0.05)
    poller.start()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=dict(env), timeout=timeout,
            stdin=subprocess.DEVNULL if stdin_devnull else None,
        )
    finally:
        poller.stop()
        poller.join(timeout=1.0)
    return proc, poller.peak_mib


class BackgroundVramPoller:
    """Context manager exposing peak MiB for a long-running external process.

    Used by the llama-server runner, where the engine process outlives a single
    request; we sample around the request window.
    """

    def __init__(self, interval_s: float = 0.05):
        self._poller = MemoryPoller(interval_s=interval_s)

    def __enter__(self) -> "BackgroundVramPoller":
        self._poller.start()
        return self

    def __exit__(self, *exc) -> None:
        self._poller.stop()
        self._poller.join(timeout=1.0)

    @property
    def peak_mib(self) -> int:
        return self._poller.peak_mib
