"""Real recorded histories, exact tokenizer IDs, with no repeated padding."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from kv_baselines.frontend import NativeBridge

sources = {
    'tuning': BASE / 'results/calibration_trace_termenv_dense_20260912/request_0061.json.gz',
    'evaluation': BASE / 'results/infllm_small_deepswe4_s1000_20260912/superjson-error-stack-serialization/small/attempt_01/server/inputs/e9ee4b6ad3f146469f15637b30781592.json',
}
bridge = NativeBridge(BASE.parents[1] / 'models/Qwen3.8-27B-BF16')
try:
    cases = {}
    for label, path in sources.items():
        raw = path.read_bytes()
        request = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        ids = bridge.call('render', request=request)['input_ids']
        cases[label] = {'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'input_ids': ids, 'available_tokens': len(ids)}
    # One identical short question after each prefix creates a decode workload;
    # this is a microbenchmark and does not produce an agent/task accuracy score.
    before = bridge.call('tokenize', text='\n<|im_end|>\n<|im_start|>user\n')['input_ids']
    query = bridge.call('tokenize', text='Review the technical details in the preceding development history and explain the next implementation and verification steps in detail.')['input_ids']
    after = bridge.call('tokenize', text='<|im_end|>\n<|im_start|>assistant\n<think>\n')['input_ids']
    cases['suffix'] = {'before': before, 'query': query, 'after': after}
    (HERE / 'inputs.json').write_text(json.dumps(cases)+'\n')
    print({k: v['available_tokens'] for k,v in cases.items() if k != 'suffix'})
finally:
    bridge.close()
