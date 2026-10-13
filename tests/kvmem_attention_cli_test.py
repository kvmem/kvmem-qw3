"""Reject invalid attention policies before model loading; no GPU required."""
import os
import subprocess
import sys


binary = sys.argv[1]
env = {k: v for k, v in os.environ.items() if not k.startswith("QW3_")}
base = ["--model", "/nonexistent/attention-cli.gguf", "--kvmem", "-p", "hello"]
invalid = [
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
for mode in ([], ["serve"]):
    for flags, expected in invalid:
        result = subprocess.run([binary, *mode, *base, *flags], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and expected in result.stderr, (flags, result.stderr)
    no_kvmem = subprocess.run([binary, *mode, "--model", "missing", "--kvmem-action-attention", "on"],
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
for suffix in ("ACTION_ATTENTION", "ACTION_SCOPE", "ACTION_FRACTION", "ACTION_DECAY", "ACTION_INTERVAL",
               "ACTION_PRESSURE", "PREFILL_ATTENTION", "PREFILL_ATTENTION_INTERVAL", "PREFILL_ATTENTION_HALF_LIFE"):
    legacy = dict(env, **{"QW3_KVMEM_" + suffix: "1"})
    result = subprocess.run([binary, "tokenize", *base, *valid], env=legacy,
                            capture_output=True, text=True, timeout=10)
    assert "is no longer supported; unset it" in result.stderr, result.stderr
help_text = subprocess.check_output([binary, "--help"], env=env, text=True)
assert all(flag in help_text for flag in valid[::2])
print("PASS attention CLI: invalid policies, explicit options, independent semantic prefill, legacy migration")
