"""Reuse the ordinary browser pilot with identical user fields in each pair."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import browser_pilot
import platform_pilots


def pair_identity(repetition):
    if type(repetition) is not int or repetition not in (1, 2):
        raise ValueError("Use the declared browser repetition")
    return "visual-context-browser-r" + str(repetition)


def admit_arguments(remaining, declaration, protocol, policy):
    def value(flag):
        if remaining.count(flag) != 1:
            raise ValueError("Declare each browser argument exactly once: " + flag)
        index = remaining.index(flag) + 1
        if index >= len(remaining): raise ValueError("Missing browser argument: " + flag)
        return remaining[index]
    output = Path(value("--output")).resolve()
    row = next((r for r in declaration["trials"] if r["id"] == output.name), None)
    if (row is None or row["kind"] != "normal-pilot" or row["policy"] != policy
            or value("--case") != "linux_browser" or value("--seconds") != "900"
            or value("--app") != "http://127.0.0.1:18217"
            or Path(value("--runtime")).resolve() != protocol.parent / "clients" / row["id"]
            or output != protocol.parent / "attempts" / row["id"]
            or remaining.count("--retire-task") != 1 or value("--linux-video-mode") != "vnc"):
        raise ValueError("Browser trial differs from the declared pair, budget or fresh runtime")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-protocol", type=Path, required=True)
    parser.add_argument("--image-history", choices=("full-retrievable", "recent-observations"), required=True)
    args, remaining = parser.parse_known_args()
    from visual_context_retest_queue import validate
    declaration = validate(args.comparison_protocol.resolve())
    row = admit_arguments(remaining, declaration, args.comparison_protocol.resolve(), args.image_history)
    expected = browser_pilot.expected
    sys.argv = [sys.argv[0], *remaining]
    # Guest fixture directories remain unique. Only the user-requested fields
    # are shared; old fixture data is never overwritten or read by the model.
    with patch.object(browser_pilot, "expected", lambda _: expected(pair_identity(row["repetition"]))):
        platform_pilots.main()


if __name__ == "__main__": main()
