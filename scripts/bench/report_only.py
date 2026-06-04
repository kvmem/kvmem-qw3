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
from bench.report_dynamic import render_dynamic_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="path to a benchmark JSON store")
    ap.add_argument("--html", default=None,
                    help="output HTML path (default: replace .json suffix with .html)")
    ap.add_argument("--static-html", action="store_true",
                    help="deprecated compatibility option; static HTML is now the default")
    ap.add_argument("--dynamic-html", action="store_true",
                    help="generate a report that fetches JSON at page load time")
    args = ap.parse_args(argv)

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: not found: {json_path}", file=sys.stderr)
        return 2
    html_path = Path(args.html) if args.html else json_path.with_suffix(".html")
    store = BenchStore.load(json_path)
    if args.dynamic_html:
        render_dynamic_report(json_path, html_path)
    else:
        store = BenchStore.load(json_path)
        render_report(store, html_path)
    row_count = len(store.rows)
    err_count = len(store.errors)
    partial = store.partial
    print(f"HTML report -> {html_path}  ({len(store.rows)} rows, "
          f"{err_count} errors, partial={partial})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
