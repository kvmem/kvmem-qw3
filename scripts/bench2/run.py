#!/usr/bin/env python3
"""Clean bench CLI: benchmark pre-launched OpenAI-compatible servers.

This driver NEVER launches a server. Start `qw3 serve` yourself, then point
the bench at the URL(s):

    # one server
    python3 scripts/bench2/run.py \
        --endpoint qw3=http://127.0.0.1:8080/v1 \
        --prompt-tokens "4096 16384 65536" --n-decode 128

    # fp16 vs fp8 side-by-side (two servers you started on 8080 and 8082)
    python3 scripts/bench2/run.py \
        --endpoint fp16=http://127.0.0.1:8080/v1 \
        --endpoint fp8=http://127.0.0.1:8082/v1 \
        --prompt-tokens "65536 131072 256000" --n-decode 128 \
        --out /tmp/fp8_vs_fp16.json

Timing is pure wall-clock over HTTP. With --method stream (default) a single
streamed chat request yields both prefill (TTFT) and decode tok/s. With
--method two_point, two non-streamed raw completions isolate decode.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench2 import driver, report  # noqa: E402


def _ints(s: str):
    return [int(x) for x in re.split(r"[,\s]+", s) if x]


def _parse_endpoint(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"endpoint must be label=url, got {spec!r}")
    label, url = spec.split("=", 1)
    return label.strip(), url.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", action="append", type=_parse_endpoint, required=True,
                    metavar="LABEL=URL",
                    help="a pre-launched server, e.g. fp16=http://127.0.0.1:8080/v1. Repeatable.")
    ap.add_argument("--prompt-tokens", type=str, default="4096 16384 65536")
    ap.add_argument("--n-decode", type=int, default=128)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--method", choices=["stream", "two_point"], default="stream")
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--out", type=str, default=None, help="write JSON results here")
    args = ap.parse_args(argv)

    endpoints = args.endpoint
    prompts = _ints(args.prompt_tokens)
    result = driver.run_grid(
        endpoints, prompts, args.n_decode, args.trials, args.method, args.timeout)

    d = driver.to_dict(result)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2))
        print(f"\nJSON -> {out}")

    print("\n" + report.render_text(d))
    n_err = sum(1 for c in d["cells"] if not c["ok"])
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
