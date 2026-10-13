"""Validate public draft selection before opening a model or using a GPU."""
import os
import subprocess
import sys


def main():
    binary = sys.argv[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("QW3_")}
    cases = [
        (["--speculative-method", "dflash2"], "requires --dflash-model"),
        (["--speculative-method", "unknown"], "must be none|mtp|dflash2"),
        (["--speculative-method", "mtp", "--mtp-chain", "0"], "requires --mtp-chain > 0"),
        (["--speculative-method", "none", "--mtp-chain", "4"], "cannot use native MTP flags"),
        (["--speculative-method", "none", "--dflash-model", "draft"], "cannot use --dflash-model"),
        (["--speculative-method", "mtp", "--dflash-model", "draft"], "cannot use --dflash-model"),
        (["--dflash-model", "draft", "--mtp-chain", "4"], "are alternatives"),
        (["--mtp-chain", "4", "--dflash-model", "draft"], "are alternatives"),
        (["--speculative-method", "dflash2", "--dflash-model", "draft", "--native-mtp-trace"], "are alternatives"),
    ]
    for mode in ([], ["serve"]):
        for args, expected in cases:
            result = subprocess.run([binary, *mode, "--model", "/nonexistent/qw3-test.gguf",
                                     "-p", "hello", *args], env=env, capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and expected in result.stderr, (mode, args, result.stderr)
    help_text = subprocess.check_output([binary, "--help"], env=env, text=True)
    assert "--speculative-method" in help_text and "--dflash-model" in help_text
    print("public speculative CLI validation passed (18 invalid configurations)")


if __name__ == "__main__":
    main()
