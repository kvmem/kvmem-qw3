#!/usr/bin/env python3
"""Re-render the HTML report from an existing benchmark JSON store.

Usage:
    python3 scripts/bench/report_only.py /tmp/qw3_bench.json [--html out.html]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench.schema import BenchStore
from bench.report import render_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="path to a benchmark JSON store")
    ap.add_argument("--html", default=None, help="output HTML path (default: <json>.html)")
    args = ap.parse_args(argv)

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: not found: {json_path}", file=sys.stderr)
        return 2
    store = BenchStore.load(json_path)
    html_path = Path(args.html) if args.html else Path(str(json_path) + ".html")
    render_report(store, html_path)
    print(f"HTML report -> {html_path}  ({len(store.rows)} rows, "
          f"{len(store.errors)} errors, partial={store.partial})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
