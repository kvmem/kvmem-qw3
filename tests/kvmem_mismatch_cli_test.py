"""CPU shadow probe must be explicit, validated, and independent of selection."""
import os
import subprocess
import sys

binary = sys.argv[1]
env = {k: v for k, v in os.environ.items() if not k.startswith('QW3_')}
base = ['--model', '/nonexistent/mismatch.gguf', '--kvmem', '--kvmem-query-conditioned',
        '--kvmem-index-placement', 'cpu', '--kvmem-mismatch-log', '/tmp/not-opened.jsonl']
for mode in ([], ['serve']):
    for flags, expected in [
        (['--kvmem-mismatch-interval', '0'], 'interval must be'),
        (['--kvmem-mismatch-samples', '5'], 'samples must be'),
        (['--kvmem-mismatch-samples', '-1'], 'samples must be'),
        (['--kvmem-mismatch-half-life', 'nan'], 'half-life must be'),
        (['--kvmem-mismatch-epsilon', 'inf'], 'epsilon must be'),
        (['--kvmem-mismatch-epsilon', '1'], 'epsilon must be'),
        (['--kvmem-mismatch-layers', '3,3'], 'duplicate attention layer'),
        (['--kvmem-mismatch-query-mode', 'invalid'], 'query-mode must be'),
        (['--kvmem-mismatch-query-mode', 'block-mean', '--no-kvmem-immutable-k'], 'requires --kvmem-immutable-k'),
        (['--kvmem-index-placement', 'gpu'], 'index-placement cpu'),
        (['--kvmem-method', 'h2o'], 'kvmem-method retrieval'),
        (['--kvmem-retrieval-method', 'key-direction-adaptive'], 'retrieval-method mean-k'),
        (['--kvmem-archive', '/nonexistent/archive'], 'without archive/repository/GC'),
    ]:
        r = subprocess.run([binary, *mode, *base, *flags], env=env, capture_output=True, text=True, timeout=10)
        assert r.returncode and expected in r.stderr, (flags, r.stderr)
reference = subprocess.run([binary, 'tokenize', '--model', '/nonexistent/mismatch.gguf'],
                           env=env, capture_output=True, text=True, timeout=10)
valid = subprocess.run([binary, 'tokenize', *base, '--kvmem-mismatch-layers', '3,31,63',
                        '--kvmem-mismatch-epsilon', '0.1', '--kvmem-mismatch-half-life', '8192'],
                       env=env, capture_output=True, text=True, timeout=10)
assert valid.returncode == reference.returncode and valid.stderr == reference.stderr, valid.stderr
print('PASS CPU mismatch CLI: early validation and independent opt-in')
