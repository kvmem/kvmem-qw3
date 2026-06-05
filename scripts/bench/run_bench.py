#!/usr/bin/env python3
"""Run the qw3 vs llama.cpp benchmark sweep, write JSON + HTML.

DEPRECATED for qw3-only timing — use scripts/bench2/run.py (clean wall-clock
HTTP layer). Keep using THIS script when you need the llama.cpp side-by-side
comparison or the HTML report, which bench2 does not provide. See
scripts/bench/__init__.py.

Usage (from repo root):
    python3 scripts/bench/run_bench.py --full-1kout --resume
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
from bench.report_dynamic import render_dynamic_report


def _ints(s: str):
    return [int(x) for x in re.split(r"[,\s]+", s) if x]


def _floats(s: str):
    return [float(x) for x in re.split(r"[,\s]+", s) if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    preset = ap.add_mutually_exclusive_group()
    preset.add_argument("--comprehensive", action="store_true",
                        help="legacy multi-n_decode/multi-chain sweep")
    preset.add_argument("--quick", action="store_true",
                        help="small smoke grid for CI/sanity")
    preset.add_argument("--full-1kout", action="store_true",
                        help="requested full sweep: 1K..250K prompt targets, 1K output, plain + MTP")

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
    ap.add_argument("--kv-dtype", type=str,
                    help="qw3 KV-cache dtype: fp16 (default), fp32, q8, or fp8. "
                         "Run once per dtype into separate outputs to compare.")
    ap.add_argument("--no-utility", action="store_true",
                    help="skip the passkey + GSM8K utility (correctness) phase")
    ap.add_argument("--passkey-lens", type=str,
                    help="comma/space list of passkey context lengths")
    ap.add_argument("--passkey-depths", type=str,
                    help="comma/space list of needle depths (0..1)")
    ap.add_argument("--passkey-trials", type=int)
    ap.add_argument("--gsm8k-n", type=int)
    ap.add_argument("--util-port", type=int)
    ap.add_argument("--qw3-port", type=int)
    ap.add_argument("--llama-port", type=int)
    ap.add_argument("--out", type=str,
                    default="/tmp/qw3_llama_full_bench/qw3_llama_full_1kout.json")
    ap.add_argument("--html", type=str, default=None,
                    help="HTML output path (default: replace .json suffix with .html)")
    ap.add_argument("--static-html", action="store_true",
                    help="deprecated compatibility option; static HTML is now the default")
    ap.add_argument("--dynamic-html", action="store_true",
                    help="generate a report that fetches JSON at page load time")
    ap.add_argument("--resume", action="store_true",
                    help="load existing --out JSON and skip successful rows")
    ap.add_argument("--force", action="store_true",
                    help="rerun rows even if --resume finds successful results")
    ap.add_argument("--print-commands", action="store_true",
                    help="print recommended full-run/view commands and exit")
    args = ap.parse_args(argv)

    if args.quick:
        cfg = BenchConfig.quick()
    elif args.comprehensive:
        cfg = BenchConfig.comprehensive()
    else:
        cfg = BenchConfig.full_1kout()

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
    if args.kv_dtype: cfg.kv_dtype = args.kv_dtype
    if args.no_utility: cfg.run_utility = False
    if args.passkey_lens: cfg.passkey_lens = _ints(args.passkey_lens)
    if args.passkey_depths: cfg.passkey_depths = _floats(args.passkey_depths)
    if args.passkey_trials: cfg.passkey_trials = args.passkey_trials
    if args.gsm8k_n is not None: cfg.gsm8k_n = args.gsm8k_n
    if args.util_port: cfg.util_port = args.util_port
    if args.qw3_port: cfg.qw3_port = args.qw3_port
    if args.llama_port: cfg.llama_port = args.llama_port

    out_json = Path(args.out)
    html_path = Path(args.html) if args.html else out_json.with_suffix(".html")

    if args.print_commands:
        print_commands(out_json, html_path, cfg)
        return 0

    if not Path(cfg.model).exists():
        print(f"ERROR: model not found: {cfg.model}", file=sys.stderr)
        return 2

    html_path.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.dynamic_html:
        render_dynamic_report(out_json, html_path)
        print(f"Dynamic HTML report -> {html_path}")

    orch = Orchestrator(cfg, out_json, resume=args.resume, force=args.force)
    store = orch.run()
    if args.dynamic_html:
        render_dynamic_report(out_json, html_path)
    else:
        render_report(store, html_path)
    print(f"HTML report -> {html_path}")
    return 0 if not store.errors else 1


def print_commands(out_json: Path, html_path: Path, cfg: BenchConfig) -> None:
    report_dir = html_path.parent
    print("# Full qw3 vs llama.cpp benchmark: 1K/2K/4K/8K/16K/64K/128K/250K input targets, 1K output")
    print("mkdir -p " + str(report_dir))
    print(
        "python3 scripts/bench/run_bench.py --full-1kout --resume "
        f"--model {cfg.model} "
        f"--qw3 {cfg.qw3} "
        f"--llama-server {cfg.llama_server} "
        f"--mtp-chain \"{' '.join(str(c) for c in cfg.mtp_chain)}\" "
        f"--out {out_json} "
        f"--html {html_path}"
    )
    print()
    print("# Open the static report directly, or serve the directory from a local HTTP server.")
    print(f"cd {report_dir}")
    print(f"# Direct file: {html_path}")
    print("# Optional: python3 -m http.server 8017")
    print(f"# Optional browser URL: http://127.0.0.1:8017/{html_path.name}")
    print()
    print("# Useful partial/debug runs")
    print(
        "python3 scripts/bench/run_bench.py --quick --resume "
        f"--out {report_dir / 'quick.json'} --html {report_dir / 'quick.html'}"
    )
    print(
        "python3 scripts/bench/run_bench.py --full-1kout --resume --engines qw3 "
        f"--out {report_dir / 'qw3_only.json'} --html {report_dir / 'qw3_only.html'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
