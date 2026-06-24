#!/usr/bin/env python3
"""Single-request prefill/decode throughput + peak VRAM across modes.

Drives the qw3 CLI binary directly (single-process path) for each
(seq_len x mode) cell. Parses the `native generate:` summary for prefill and
decode tok/s, and samples peak GPU memory via a background nvidia-smi poller.

Modes (single-request path):
  plain      : greedy decode, no MTP, no kvmem
  mtp        : --mtp-chain N --native-mtp-speculate
  kvmem      : --kvmem (plain decode over the assembled window)
  kvmem_mtp  : --kvmem + MTP (Phase C single-request path)

This is NOT the CB path (that's serve-based + concurrency). CB is benchmarked
separately via mtp_throughput_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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
    base_tokens = 285
    per_repeat = 270
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    base = _PASSAGE * n_repeats
    return f"Summarize the following passage in one paragraph.\n\n{base}\n\nSummary:"


def ctx_for(target_tokens: int, n_decode: int, min_ctx: int = 4096) -> int:
    need = target_tokens + n_decode + 1024
    ctx = min_ctx
    while ctx < need:
        ctx *= 2
    return ctx


_LINE = re.compile(
    r"\[qw3\] native generate: prompt_tokens=(\d+) "
    r"prefill=([0-9.]+)s \(([0-9.]+) tok/s\) "
    r"decoded=(\d+) decode=([0-9.]+)s \(([0-9.]+) tok/s\)"
)
_ACCEPT = re.compile(r"acceptance=([0-9.]+)")


@dataclass
class Cell:
    mode: str
    seq_len: int
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    prefill_tok_s: float = 0.0
    decoded: int = 0
    decode_s: float = 0.0
    decode_tok_s: float = 0.0
    acceptance: Optional[float] = None
    peak_vram_mib: int = 0
    base_vram_mib: int = 0
    delta_vram_mib: int = 0
    ok: bool = False
    error: str = ""


def gpu_mem_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    return int(out.splitlines()[0])


class VramPoller(threading.Thread):
    def __init__(self, interval_s: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval_s
        self.peak = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, gpu_mem_used_mib())
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop_polling(self):
        self._stop_evt.set()
        self.join(timeout=2)


def build_cmd(args, mode: str, prompt_file: Path, ctx: int) -> list[str]:
    cmd = [
        args.qw3, "--model", args.model,
        "--prompt-file", str(prompt_file),
        "-n", str(args.n_decode), "--ctx", str(ctx),
        "--prefill-chunk", str(args.prefill_chunk),
        "--kv-dtype", args.kv_dtype, "--temp", "0",
    ]
    if mode in ("mtp", "kvmem_mtp"):
        cmd += ["--mtp-chain", str(args.chain), "--native-mtp-speculate"]
    if mode in ("kvmem", "kvmem_mtp"):
        cmd += ["--kvmem",
                "--kvmem-block-tokens", str(args.kvmem_block_tokens),
                "--kvmem-budget", str(args.kvmem_budget),
                "--kvmem-interval", str(args.kvmem_interval)]
    return cmd


def run_cell(args, mode: str, seq_len: int, tmp_dir: Path) -> Cell:
    cell = Cell(mode=mode, seq_len=seq_len)
    ctx = ctx_for(seq_len, args.n_decode)
    prompt_file = tmp_dir / f"prompt_{seq_len}.txt"
    if not prompt_file.exists():
        prompt_file.write_text(make_prompt(seq_len), encoding="utf-8")
    cmd = build_cmd(args, mode, prompt_file, ctx)

    try:
        cell.base_vram_mib = gpu_mem_used_mib()
    except Exception:
        cell.base_vram_mib = 0

    poller = VramPoller()
    poller.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        poller.stop_polling()
        cell.error = "timeout"
        return cell
    finally:
        poller.stop_polling()

    cell.peak_vram_mib = poller.peak
    cell.delta_vram_mib = max(0, cell.peak_vram_mib - cell.base_vram_mib)

    m = _LINE.search(out)
    if not m:
        cell.error = "no summary line; tail:\n" + "\n".join(out.splitlines()[-15:])
        return cell
    cell.prompt_tokens = int(m.group(1))
    cell.prefill_s = float(m.group(2))
    cell.prefill_tok_s = float(m.group(3))
    cell.decoded = int(m.group(4))
    cell.decode_s = float(m.group(5))
    cell.decode_tok_s = float(m.group(6))
    a = _ACCEPT.search(out)
    if a:
        cell.acceptance = float(a.group(1))
    cell.ok = True
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", default="models/Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--modes", default="plain mtp kvmem kvmem_mtp")
    ap.add_argument("--seq-lens", default="4096 32768")
    ap.add_argument("--n-decode", type=int, default=128)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--kvmem-block-tokens", type=int, default=128)
    ap.add_argument("--kvmem-budget", type=int, default=131072)
    ap.add_argument("--kvmem-interval", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-json", default="/tmp/qw3_single_req_throughput.json")
    ap.add_argument("--tmp-dir", default="/tmp/qw3_bench_prompts")
    args = ap.parse_args()

    modes = args.modes.split()
    seq_lens = [int(x) for x in args.seq_lens.split()]
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cells: list[Cell] = []
    for seq_len in seq_lens:
        for mode in modes:
            print(f"[run] mode={mode} seq_len={seq_len} ...", flush=True)
            cell = run_cell(args, mode, seq_len, tmp_dir)
            cells.append(cell)
            if cell.ok:
                acc = f" acc={cell.acceptance}" if cell.acceptance is not None else ""
                print(f"  prompt={cell.prompt_tokens} "
                      f"prefill={cell.prefill_tok_s:.1f} tok/s "
                      f"decode={cell.decode_tok_s:.2f} tok/s "
                      f"peakVRAM={cell.peak_vram_mib} MiB "
                      f"(Δ{cell.delta_vram_mib}){acc}", flush=True)
            else:
                print(f"  FAILED: {cell.error}", flush=True)

    Path(args.out_json).write_text(
        json.dumps([asdict(c) for c in cells], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[done] wrote {args.out_json}")

    # Compact table.
    print("\n=== single-request throughput ===")
    print(f"{'seq':>7} {'mode':<10} {'prefill_t/s':>12} {'decode_t/s':>11} "
          f"{'peakVRAM':>10} {'accept':>7}")
    for c in cells:
        if not c.ok:
            print(f"{c.seq_len:>7} {c.mode:<10} {'FAIL':>12}")
            continue
        acc = f"{c.acceptance:.3f}" if c.acceptance is not None else "-"
        print(f"{c.seq_len:>7} {c.mode:<10} {c.prefill_tok_s:>12.1f} "
              f"{c.decode_tok_s:>11.2f} {c.peak_vram_mib:>8} MiB {acc:>7}")


if __name__ == "__main__":
    main()
