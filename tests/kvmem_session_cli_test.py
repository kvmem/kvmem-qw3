"""Check opt-in session memory validation before model IO; no GPU required."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = sys.argv[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    missing_model = "/nonexistent/qw3-session-cli.gguf"
    if Path(missing_model).exists():
        raise AssertionError("test requires a nonexistent model path")
    base = ["--model", missing_model, "-p", "hello"]
    checked = 0

    def run(mode, flags, environment=env):
        return subprocess.run([binary, *mode, *base, *flags], env=environment,
                              capture_output=True, text=True, timeout=15)

    def rejected(mode, flags, expected, environment=env):
        nonlocal checked
        result = run(mode, flags, environment)
        if result.returncode == 0 or expected not in result.stderr:
            raise AssertionError((mode, flags, expected, result.returncode, result.stderr))
        if "failed to open GGUF" in result.stderr:
            raise AssertionError(("model IO preceded validation", flags, result.stderr))
        checked += 1

    parse_cases = [
        (["--kvmem-session-capacity", "-1"], "must be >= 0"),
        (["--kvmem-session-capacity", "2.5"], "invalid integer"),
        (["--kvmem-session-swap-gb", "-1"], "invalid GiB value"),
        (["--kvmem-session-swap-gb", "nan"], "invalid GiB value"),
        (["--kvmem-session-swap-gb", "inf"], "invalid GiB value"),
        (["--kvmem-session-swap-gb", "0.5GiB"], "invalid GiB value"),
        (["--kvmem-session-swap-gb", "1e300"], "GiB value too large"),
    ]
    for mode in ([], ["serve"], ["tokenize"]):
        for flags, expected in parse_cases:
            rejected(mode, flags, expected)

    enabled = ["--kvmem-session-capacity", "4", "--kvmem-session-swap-gb", "0.5"]
    for mode in ([], ["tokenize"]):
        rejected(mode, enabled, "session memory requires serve")
        rejected(mode, ["--kvmem-session-swap-gb", "0.5"], "session memory requires serve")

    controller_env = dict(env, QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS="1",
                          QW3_CONTINUOUS_KVMEM_CONTROLLER="1",
                          QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL="1",
                          QW3_CONTINUOUS_KVMEM_SESSION_REUSE="1")
    common = ["--kvmem", "--kvmem-query-conditioned", "--kvmem-prefix-cache",
              "--kvmem-repository-blocks", "256", "--ctx", "8192", "--kvmem-block-tokens", "32",
              "--kvmem-budget", "1024", "--kvmem-gen-budget", "2048", "--prefill-chunk", "128",
              "--kvmem-sink-blocks", "1", "--kvmem-recent-blocks", "4",
              "--continuous-batching", "--max-active", "2", "--kv-dtype", "fp8"]
    rejected(["serve"], ["--kvmem-session-swap-gb", "0.5"], "requires --kvmem-session-capacity")
    rejected(["serve"], enabled, "requires bounded continuous serving", controller_env)
    rejected(["serve"], [*common, *enabled, "--no-continuous-batching"],
             "requires bounded continuous serving", controller_env)
    rejected(["serve"], [*common, *enabled, "--kvmem-session-capacity", "1"],
             "must be >= --max-active", controller_env)
    rejected(["serve"], [*common, *enabled, "--kvmem-session-swap-gb", "0"],
             "requires a positive --kvmem-session-swap-gb", controller_env)
    rejected(["serve"], [*common, *enabled, "--kvmem-session-swap-gb", "1e-30"],
             "requires a positive --kvmem-session-swap-gb", controller_env)
    rejected(["serve"], [*common, *enabled, "--kv-dtype", "q8"],
             "does not support q8 KV scale planes", controller_env)
    for switch in ("QW3_CONTINUOUS_KVMEM_CONTROLLER", "QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL",
                   "QW3_CONTINUOUS_KVMEM_SESSION_REUSE"):
        rejected(["serve"], [*common, *enabled], "requires bounded continuous serving",
                 dict(controller_env, **{switch: "0"}))
    rejected(["serve"], [*common, *enabled], "continuous requests require QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS",
             dict(controller_env, QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS="0"))
    for flags, override in ((["--prefix-cache"], {}), ([], {"QW3_CONTINUOUS_MTP_LAYERED_VERIFY": "1"})):
        rejected(["serve"], [*common, *enabled, *flags], "ragged verification and request-owned prefix state",
                 dict(controller_env, **override))
    no_kvmem = [flag for flag in common if flag != "--kvmem"]
    rejected(["serve"], [*no_kvmem, *enabled], "requires --kvmem", controller_env)
    no_query = [flag for flag in common if flag != "--kvmem-query-conditioned"]
    rejected(["serve"], [*no_query, *enabled], "--kvmem-query-conditioned", controller_env)

    # Zero values must preserve the original route and its missing-model error.
    disabled = ["--kvmem-session-capacity", "0", "--kvmem-session-swap-gb", "0"]
    for mode in ([], ["serve"], ["tokenize"]):
        reference = run(mode, [])
        explicit_off = run(mode, disabled)
        if not (reference.returncode == explicit_off.returncode != 0 and
                reference.stderr == explicit_off.stderr and "failed to open GGUF" in reference.stderr):
            raise AssertionError(("default-off behavior changed", mode, reference.stderr, explicit_off.stderr))
    for flags in (common, [*common, *disabled], [*common, *enabled],
                  [*common, *enabled, "--kvmem-session-capacity", "2"]):
        result = run(["serve"], flags, controller_env)
        if result.returncode == 0 or "failed to open GGUF" not in result.stderr:
            raise AssertionError(("valid configuration rejected before model IO", flags, result.stderr))

    help_text = subprocess.check_output([binary, "--help"], env=env, text=True)
    if not all(flag in help_text for flag in ("--kvmem-session-capacity", "--kvmem-session-swap-gb")):
        raise AssertionError("session memory flags are absent from --help")
    # The manual GPU probe's plan must be inspectable even in a sandbox that
    # forbids sockets, with an empty placeholder model that cannot be loaded.
    with tempfile.TemporaryDirectory(prefix="qw3-session-probe-plan-") as tmp:
        root = Path(tmp)
        model = root / "placeholder.gguf"
        model.touch()
        output = root / "must-not-be-created"
        probe = Path(__file__).with_name("kvmem_multi_session_http_probe.py")
        result = subprocess.run([sys.executable, str(probe), "--binary", binary,
                                 "--model", str(model), "--output", str(output), "--dry-run"],
                                env=env, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or output.exists():
            raise AssertionError(("probe dry-run performed work or failed", result.stderr))
        plan = json.loads(result.stdout)
        reference, candidate = plan["reference"]["command"], plan["candidate"]["command"]
        if ("--kvmem-session-capacity" in reference or
                candidate[candidate.index("--kvmem-session-capacity") + 1] != "6" or
                candidate[candidate.index("--max-active") + 1] != "2"):
            raise AssertionError(("probe no longer compares default-off with excess retained histories", plan))
    print(f"PASS KVMem session CLI: {checked} invalid configurations, default-off compatibility, valid serving admission, probe dry-run")


if __name__ == "__main__":
    main()
