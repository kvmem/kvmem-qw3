"""Validate Q8 KV-cache vs FP16: peak VRAM, decode throughput, output parity.

Sweeps a set of prompt lengths and, for each, runs ./build/qw3 twice (fp16 and
q8 KV) on an identical greedy prompt. Reports:
  - peak VRAM (MiB)            -> memory savings of Q8
  - decode tok/s               -> native-Q8 vec vs FlashInfer-fp16 tradeoff
  - first divergence token idx -> output parity sanity (greedy)

Usage:
  python3 scripts/kv_q8_validate.py [--decode N] [--lens 4000,16000,64000]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from long_prompt_sweep import make_prompt  # type: ignore  # noqa: E402
from bench.vram import run_with_polling, wait_for_idle  # type: ignore  # noqa: E402

_QW3_LINE = re.compile(
    r"prompt_tokens=(\d+)\s+prefill=([\d.]+)s\s+\(([\d.]+)\s*tok/s\)\s+"
    r"decoded=(\d+)\s+decode=([\d.]+)s\s+\(([\d.]+)\s*tok/s\)"
)


def _write_prompt(text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="kvq8_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return Path(path)


def _cmd(model: str, ctx: int, n: int, pf: Path) -> list:
    return [
        str(ROOT / "build" / "qw3"),
        "--backend", "qwen-native", "--native-heavy",
        "--native-kernels", "cuda", "--native-linear-backend", "auto",
        "--model", model, "--raw",
        "-c", str(ctx), "-n", str(n),
        "--temp", "0", "--seed", "0",
        "--prompt-file", str(pf),
    ]


def _run(model: str, ctx: int, n: int, pf: Path, kv_dtype: str, timeout: float):
    env = os.environ.copy()
    if kv_dtype:
        env["QW3_KV_DTYPE"] = kv_dtype
    wait_for_idle()
    proc, peak = run_with_polling(_cmd(model, ctx, n, pf), env, timeout)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return {"ok": False, "err": out.strip()[-300:], "peak": peak}
    m = _QW3_LINE.search(out)
    if not m:
        return {"ok": False, "err": "no-parse: " + out.strip()[-300:], "peak": peak}
    # capture generated text (everything before the trailing stats line)
    gen = out.split("[qw3] native generate")[0]
    return {
        "ok": True,
        "prompt_tokens": int(m.group(1)),
        "prefill_s": float(m.group(2)),
        "prefill_tok_s": float(m.group(3)),
        "decoded": int(m.group(4)),
        "decode_s": float(m.group(5)),
        "decode_tok_s": float(m.group(6)),
        "peak": peak,
        "gen": gen,
    }


def _first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if a == b else n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "Qwen3.6-27B-Q8_0.gguf"))
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--lens", default="4000,16000,64000")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x]
    print(f"model={args.model}")
    print(f"decode={args.decode}  lens={lens}\n")
    hdr = (f"{'ctx':>7} {'dtype':>5} {'prompt':>7} {'peakMiB':>8} "
           f"{'pf_tok/s':>9} {'dec_tok/s':>9}")
    print(hdr)
    print("-" * len(hdr))

    for L in lens:
        prompt = make_prompt(L)
        pf = _write_prompt(prompt)
        ctx = int(L * 1.6) + args.decode + 256
        results = {}
        try:
            for dt in ("fp16", "q8"):
                r = _run(args.model, ctx, args.decode, pf, dt, args.timeout)
                results[dt] = r
                if not r["ok"]:
                    print(f"{ctx:>7} {dt:>5}  FAILED: {r['err']}")
                    continue
                print(f"{ctx:>7} {dt:>5} {r['prompt_tokens']:>7} {r['peak']:>8} "
                      f"{r['prefill_tok_s']:>9.1f} {r['decode_tok_s']:>9.2f}")
        finally:
            pf.unlink(missing_ok=True)

        if results.get("fp16", {}).get("ok") and results.get("q8", {}).get("ok"):
            f, q = results["fp16"], results["q8"]
            save = f["peak"] - q["peak"]
            spd = (q["decode_tok_s"] / f["decode_tok_s"]) if f["decode_tok_s"] else 0
            div = _first_divergence(f["gen"], q["gen"])
            divs = "identical" if div == -1 else f"@char {div}"
            print(f"  -> VRAM saved {save} MiB ({100.0*save/max(f['peak'],1):.1f}%), "
                  f"decode {spd:.2f}x fp16, output {divs}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
