"""Self-contained helpers for the clean bench layer: prompt builder + VRAM.

Kept dependency-free (no import of the legacy sweep scripts) so the bench2
package stands alone.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional

# A neutral public-domain passage (Matthew 1 genealogy) repeated to hit a target
# token count. Calibrated against the Qwen 3.6 tokenizer: the wrapper + 1 repeat
# is ~285 tokens, each extra repeat ~270.
_PASSAGE = (
    "The book of the generation of Jesus Christ, the son of David, the son of "
    "Abraham. Abraham begat Isaac; and Isaac begat Jacob; and Jacob begat Judas "
    "and his brethren; And Judas begat Phares and Zara of Thamar; and Phares "
    "begat Esrom; and Esrom begat Aram; And Aram begat Aminadab; and Aminadab "
    "begat Naasson; and Naasson begat Salmon; And Salmon begat Booz of Rachab; "
    "and Booz begat Obed of Ruth; and Obed begat Jesse; And Jesse begat David "
    "the king; and David the king begat Solomon of her that had been the wife "
    "of Urias; And Solomon begat Roboam; and Roboam begat Abia; and Abia begat "
    "Asa; And Asa begat Josaphat; and Josaphat begat Joram; and Joram begat "
    "Ozias; And Ozias begat Joatham; and Joatham begat Achaz; and Achaz begat "
    "Ezekias; And Ezekias begat Manasses; and Manasses begat Amon; and Amon "
    "begat Josias; And Josias begat Jechonias and his brethren, about the time "
    "they were carried away to Babylon. "
)


def make_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately `target_tokens` Qwen tokens."""
    base_tokens, per_repeat = 285, 270
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    body = _PASSAGE * n_repeats
    return f"Summarize the following passage in one paragraph.\n\n{body}\n\nSummary:"


def smi_used_mib(gpu: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True, timeout=2.0)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class VramPoller(threading.Thread):
    """Sample memory.used at a fixed cadence; expose the peak observed."""

    def __init__(self, interval_s: float = 0.05, gpu: int = 0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu = gpu
        self._stop_evt = threading.Event()
        self.peak_mib = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            v = smi_used_mib(self.gpu)
            if v is not None and v > self.peak_mib:
                self.peak_mib = v
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()

    def __enter__(self) -> "VramPoller":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
        self.join(timeout=1.0)
