"""Run the utility (correctness) phase: passkey retrieval + GSM8K via qw3 serve.

Launches ONE server per sweep at cfg.kv_dtype, fires all eval prompts over HTTP,
returns a UtilityResult (see schema.py). The server is torn down when done.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent / "scripts") not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from oai_client import chat, complete  # type: ignore  # noqa: E402
from kv_q8_utility import make_passkey_prompt  # type: ignore  # noqa: E402
from kv_q8_gsm8k import gold_answer, parse_pred, make_messages  # type: ignore  # noqa: E402

from .config import BenchConfig


def _wait_health(host: str, port: int, timeout: float = 300.0) -> bool:
    import urllib.request
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2.0)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    return False


def run_utility(cfg: BenchConfig) -> "UtilityResult":  # noqa: F821
    """Launch a qw3 serve, run passkey + GSM8K, return results."""
    from .schema import UtilityResult, PasskeyCell

    base_url = f"http://{cfg.util_host}:{cfg.util_port}/v1"
    cmd = [
        cfg.qw3, "serve",
        "--model", cfg.model,
        "--backend", "qwen-native",
        "--port", str(cfg.util_port),
        "--host", cfg.util_host,
        "-c", str(cfg.util_ctx()),
        "--kv-dtype", cfg.kv_dtype,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=os.environ.copy())
    try:
        if not _wait_health(cfg.util_host, cfg.util_port, timeout=300.0):
            proc.kill()
            return UtilityResult(ok=False, error="utility server never became healthy")

        # --- Passkey ---
        rng = random.Random(1234)
        passkey_cells: List[PasskeyCell] = []
        for length in cfg.passkey_lens:
            for depth in cfg.passkey_depths:
                hits = 0
                for _ in range(cfg.passkey_trials):
                    secret = str(rng.randint(10000, 99999))
                    prompt = make_passkey_prompt(length, depth, secret)
                    try:
                        gen = complete(base_url, "qw3", prompt,
                                       max_tokens=cfg.passkey_decode,
                                       temperature=0.0, seed=0, timeout=900.0)
                    except Exception:  # noqa: BLE001
                        gen = ""
                    hits += int(secret in gen)
                passkey_cells.append(PasskeyCell(
                    length=length, depth=depth,
                    hits=hits, trials=cfg.passkey_trials))

        # --- GSM8K ---
        gsm8k_correct = 0
        gsm8k_n = 0
        gsm8k_error: Optional[str] = None
        if cfg.gsm8k_n > 0:
            data_path = Path(cfg.gsm8k_data)
            if not data_path.exists():
                gsm8k_error = f"gsm8k data not found: {cfg.gsm8k_data}"
            else:
                rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
                rng2 = random.Random(0)
                rng2.shuffle(rows)
                rows = rows[:cfg.gsm8k_n]
                gsm8k_n = len(rows)
                for row in rows:
                    gold = gold_answer(row["answer"])
                    msgs = make_messages(row["question"])
                    try:
                        gen = chat(base_url, "qw3", msgs,
                                   max_tokens=cfg.gsm8k_decode,
                                   temperature=0.0, seed=0, timeout=600.0)
                    except Exception:  # noqa: BLE001
                        gen = ""
                    pred = parse_pred(gen)
                    gsm8k_correct += int(pred == gold and bool(gold))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    return UtilityResult(
        ok=True,
        kv_dtype=cfg.kv_dtype,
        passkey_cells=passkey_cells,
        gsm8k_correct=gsm8k_correct,
        gsm8k_n=gsm8k_n,
        gsm8k_error=gsm8k_error,
    )
