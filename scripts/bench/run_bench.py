#!/usr/bin/env python3
"""Run the qw3 vs llama.cpp benchmark sweep, write JSON + HTML.

Usage (from repo root):
    python3 scripts/bench/run_bench.py --quick
    python3 scripts/bench/run_bench.py --comprehensive --out /tmp/bench.json
    python3 scripts/bench/run_bench.py --prompt-tokens "512 4096" --n-decode "64" \
        --mtp-chain "1 2" --trials 2

The MTP-enabled GGUF is required (see bench/config.py). Both engines run against
the same model.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench.config import BenchConfig
from bench.orchestrator import Orchestrator
from bench.report import render_report


def _ints(s: str):
    return [int(x) for x in re.split(r"[,\s]+", s) if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    preset = ap.add_mutually_exclusive_group()
    preset.add_argument("--comprehensive", action="store_true",
                        help="full 3D sweep (default)")
    preset.add_argument("--quick", action="store_true",
                        help="small smoke grid for CI/sanity")

    ap.add_argument("--model")
    ap.add_argument("--qw3")
    ap.add_argument("--llama-server")
    ap.add_argument("--prompt-tokens", type=str)
    ap.add_argument("--n-decode", type=str)
    ap.add_argument("--mtp-chain", type=str)
    ap.add_argument("--trials", type=int)
    ap.add_argument("--engines", type=str, help="comma list: qw3,llama")
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--llama-port", type=int)
    ap.add_argument("--out", type=str, default="/tmp/qw3_bench.json")
    ap.add_argument("--html", type=str, default=None,
                    help="HTML output path (default: <out>.html)")
    args = ap.parse_args(argv)

    cfg = BenchConfig.quick() if args.quick else BenchConfig.comprehensive()

    if args.model: cfg.model = args.model
    if args.qw3: cfg.qw3 = args.qw3
    if args.llama_server: cfg.llama_server = args.llama_server
    if args.prompt_tokens: cfg.prompt_tokens = _ints(args.prompt_tokens)
    if args.n_decode: cfg.n_decode = _ints(args.n_decode)
    if args.mtp_chain: cfg.mtp_chain = _ints(args.mtp_chain)
    if args.trials: cfg.trials = args.trials
    if args.engines: cfg.engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.no_plain: cfg.run_plain = False
    if args.no_mtp: cfg.run_mtp = False
    if args.llama_port: cfg.llama_port = args.llama_port

    out_json = Path(args.out)
    html_path = Path(args.html) if args.html else Path(str(out_json) + ".html")

    if not Path(cfg.model).exists():
        print(f"ERROR: model not found: {cfg.model}", file=sys.stderr)
        return 2

    orch = Orchestrator(cfg, out_json)
    store = orch.run()
    render_report(store, html_path)
    print(f"HTML report -> {html_path}")
    return 0 if not store.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
