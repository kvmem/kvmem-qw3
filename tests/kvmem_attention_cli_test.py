"""Reject invalid attention policies before model loading; no GPU required."""
import os
import subprocess
import sys


binary = sys.argv[1]
env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
base = ["--model", "/nonexistent/attention-cli.gguf", "--kvmem", "-p", "hello"]
invalid = [
    (["--kvmem-middecode-recent-tokens", "-1"], "must be auto or >= 0"),
    (["--kvmem-middecode-recent-tokens", "2.5"], "invalid integer"),
    (["--kvmem-mandatory-parts", "tool"], "unknown mandatory part"),
    (["--kvmem-mandatory-parts", "system,system"], "duplicate mandatory part"),
    (["--kvmem-mandatory-parts", "system,"], "empty mandatory part"),
    (["--kvmem-mandatory-parts", ""], "empty mandatory parts list"),
    (["--kvmem-query-history", "yes"], "must be on|off"),
    (["--kvmem-query-history-decay", "-0.1"], "must be in [0,1]"),
    (["--kvmem-query-history-decay", "1.1"], "must be in [0,1]"),
    (["--kvmem-query-history-decay", "nan"], "must be in [0,1]"),
    (["--kvmem-query-history-decay", "inf"], "must be in [0,1]"),
    (["--kvmem-query-history", "on"], "requires --kvmem-query-conditioned"),
    (["--kvmem-action-combine", "raw"], "must be quota|score"),
    (["--kvmem-action-layers", "wat"], "must be last|spread3|all"),
    (["--kvmem-action-layers", "3,,63"], "must be last|spread3|all"),
    (["--kvmem-action-layers", "3,3"], "duplicate attention layer ID"),
    (["--kvmem-action-layers", ""], "empty attention layer policy"),
    (["--kvmem-prefill-attention-stride", "-1"], "must be in [0,1000000000]"),
    (["--kvmem-action-attention", "1"], "must be on|off"),
    (["--kvmem-action-pressure", "on"], "requires --kvmem-action-attention on"),
    (["--kvmem-prefill-attention", "on"], "requires --kvmem-action-attention on"),
    (["--kvmem-action-scope", "thinking"], "must be tool-call|non-thinking"),
    (["--kvmem-action-fraction", "nan"], "must be in [0,1]"),
    (["--kvmem-action-fraction", "1.1"], "must be in [0,1]"),
    (["--kvmem-action-decay", "-0.1"], "must be in [0,1]"),
    (["--kvmem-action-decay", "inf"], "must be in [0,1]"),
    (["--kvmem-action-interval", "0"], "must be in [1,4096]"),
    (["--kvmem-action-interval", "-1"], "must be in [1,4096]"),
    (["--kvmem-action-interval", "2.5"], "invalid integer"),
    (["--kvmem-prefill-attention-interval", "0"], "must be in [1,1000000000]"),
    (["--kvmem-prefill-attention-half-life", "nan"], "must be in [1,1000000000]"),
    (["--kvmem-action-attention", "on", "--kvmem-method", "h2o"], "requires --kvmem-method retrieval"),
    (["--kvmem-action-attention", "on", "--kvmem-semantic-expansion", "message"], "--kvmem-semantic-expansion none"),
]
history = ["--kvmem-query-history", "on", "--kvmem-query-conditioned"]
for flags, expected in [
    (["--kvmem-method", "h2o"], "--kvmem-method retrieval"),
    (["--kvmem-retrieval-method", "per-token"], "--kvmem-retrieval-method mean-k"),
    (["--kvmem-retrieval-method", "key-direction-adaptive"], "--kvmem-retrieval-method mean-k"),
    (["--kvmem-select-policy", "quota"], "--kvmem-select-policy topk"),
    (["--kvmem-semantic-expansion", "message"], "--kvmem-semantic-expansion none"),
    (["--continuous-batching"], "--no-continuous-batching"),
    (["--kvmem-archive", "/nonexistent/archive"], "online (non-archive) KVMem"),
    (["--kvmem-guided-reselect", "boundary"], "--kvmem-guided-reselect boundary|both"),
    (["--kvmem-guided-reselect", "both"], "--kvmem-guided-reselect boundary|both"),
    (["--kvmem-action-attention", "on", "--kvmem-action-combine", "score",
      "--kvmem-action-fraction", "1"], "100% attention score selection"),
]:
    invalid.append(([*history, *flags], expected))
for mode in ([], ["serve"]):
    for flags, expected in invalid:
        result = subprocess.run([binary, *mode, *base, *flags], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and expected in result.stderr, (flags, result.stderr)
    no_kvmem = subprocess.run([binary, *mode, "--model", "missing", "--kvmem-action-attention", "on"],
                             env=env, capture_output=True, text=True, timeout=10)
    assert "requires --kvmem" in no_kvmem.stderr, no_kvmem.stderr
    no_kvmem = subprocess.run([binary, *mode, "--model", "missing", *history],
                             env=env, capture_output=True, text=True, timeout=10)
    assert "requires --kvmem" in no_kvmem.stderr, no_kvmem.stderr

# Parse valid flags through the tokenize route, stopping at missing model IO.
# Prefill may feed semantic selection even with pressure attention disabled.
valid = ["--kvmem-action-attention", "on", "--kvmem-action-scope", "non-thinking",
         "--kvmem-action-combine", "score",
         "--kvmem-action-layers", "spread3", "--kvmem-prefill-attention-stride", "32",
         "--kvmem-action-fraction", "0.5", "--kvmem-action-decay", "0.5",
         "--kvmem-action-interval", "16", "--kvmem-action-pressure", "off",
         "--kvmem-prefill-attention", "on", "--kvmem-prefill-attention-interval", "8192",
         "--kvmem-prefill-attention-half-life", "8192"]
reference = subprocess.run([binary, "tokenize", *base], env=env, capture_output=True, text=True, timeout=10)
result = subprocess.run([binary, "tokenize", *base, *valid], env=env, capture_output=True, text=True, timeout=10)
assert result.returncode == reference.returncode != 0 and result.stderr == reference.stderr, result.stderr
result = subprocess.run([binary, "tokenize", *base, "--kvmem-action-combine", "quota"], env=env,
                        capture_output=True, text=True, timeout=10)
assert result.stderr == reference.stderr, result.stderr
# Query history remains usable with output attention disabled, and with either
# attention combination policy. Its off switch does not constrain old modes.
for flags in [
    ["--kvmem-middecode-recent-tokens", "auto"],
    ["--kvmem-middecode-recent-tokens", "0"],
    ["--kvmem-middecode-recent-tokens", "1024"],
    ["--kvmem-mandatory-parts", "default"],
    ["--kvmem-mandatory-parts", "all"],
    ["--kvmem-mandatory-parts", "none"],
    ["--kvmem-mandatory-parts", "system,query,tool-result"],
    ["--kvmem-mandatory-parts", "tool-call"],
    history,
    [*history, "--kvmem-query-history-decay", "0"],
    [*history, "--kvmem-guided-reselect", "off"],
    [*history, "--kvmem-guided-reselect", "middecode"],
    [*history, "--kvmem-query-history-decay", "1", *valid],
    [*history, "--kvmem-action-attention", "on", "--kvmem-action-combine", "quota",
     "--kvmem-action-fraction", "1"],
    [*history, "--kvmem-action-attention", "off", "--kvmem-action-combine", "score",
     "--kvmem-action-fraction", "1"],
    ["--kvmem-query-history", "off", "--kvmem-method", "h2o"],
    ["--kvmem-query-history", "off", "--kvmem-guided-reselect", "both"],
]:
    result = subprocess.run([binary, "tokenize", *base, *flags], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.stderr == reference.stderr, (flags, result.stderr)
for suffix in ("ACTION_ATTENTION", "ACTION_SCOPE", "ACTION_FRACTION", "ACTION_DECAY", "ACTION_INTERVAL",
               "ACTION_PRESSURE", "PREFILL_ATTENTION", "PREFILL_ATTENTION_INTERVAL", "PREFILL_ATTENTION_HALF_LIFE"):
    legacy = dict(env, **{"QW3_KVMEM_" + suffix: "1"})
    result = subprocess.run([binary, "tokenize", *base, *valid], env=legacy,
                            capture_output=True, text=True, timeout=10)
    assert "is no longer supported; unset it" in result.stderr, result.stderr
help_text = subprocess.check_output([binary, "--help"], env=env, text=True)
assert all(flag in help_text for flag in valid[::2])
assert "--kvmem-query-history" in help_text and "--kvmem-query-history-decay" in help_text
assert "--kvmem-mandatory-parts" in help_text
print("PASS attention/Query history CLI: validation, independent signals, supported combinations, legacy migration")
