"""Select one declared image policy through the unchanged OSWorld runner."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import osworld_candidate_run as original
from osworld_office_candidate_run import verify_auxiliary


def image_command(command, output_tokens, policy):
    if policy not in {"full-retrievable", "recent-observations"}:
        raise ValueError("Declare the matched image policy")
    result = original.candidate_command(command, output_tokens)
    if "-m" in result and result[result.index("-m") + 1] == "qw3_computer.app":
        if "--image-history" in result: raise ValueError("Image policy was already supplied")
        result += ["--image-history", policy]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-protocol", type=Path, required=True)
    parser.add_argument("--image-history", choices=("full-retrievable", "recent-observations"), required=True)
    args, remaining = parser.parse_known_args()
    from visual_context_retest_queue import validate
    declaration = validate(args.comparison_protocol.resolve())
    base_command, base_admit, base_save = original.candidate_command, original.admit, original.runner.save
    def command(values, tokens):
        # image_command must call the original adapter, not this patched hook.
        with patch.object(original, "candidate_command", base_command):
            return image_command(values, tokens, args.image_history)
    def admit(trial):
        validate(args.comparison_protocol.resolve())
        row = next((r for r in declaration["trials"] if r["id"] == trial.output.name), None)
        if (row is None or row["kind"] != "original-osworld" or row["task"] != trial.task
                or row["policy"] != args.image_history or trial.arm != "verified"
                or trial.framework.resolve() != Path(declaration["framework"])
                or trial.output.resolve() != args.comparison_protocol.parent / "attempts" / row["id"]):
            raise ValueError("Trial differs from the frozen image comparison")
        verify_auxiliary(trial)
        return base_admit(trial)
    def save(path, body):
        if Path(path).name in {"manifest.json", "result.json"}:
            body = dict(body, image_history=args.image_history,
                comparison_protocol_sha256=original.hashlib.sha256(args.comparison_protocol.read_bytes()).hexdigest(),
                comparison="Matched image presentation; original materials, initializer, grader and budgets retained")
        return base_save(path, body)
    sys.argv = [sys.argv[0], *remaining]
    with patch.object(original, "candidate_command", command), patch.object(original, "admit", admit), \
            patch.object(original.runner, "save", save):
        original.main()


if __name__ == "__main__": main()
