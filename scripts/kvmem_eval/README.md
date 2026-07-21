# KVMem LongMemEval runner

`run_eval.py` is the canonical, fixed evaluation harness: it loads the dataset,
renders prompts, calls qw3 through the OpenAI-compatible API, invokes the
DeepSeek judge, and writes JSONL/summary artifacts.

`run_kvmem_eval.py` is the single reusable process orchestrator. It starts qw3
with the requested KVMem parameters, waits for `/health`, invokes the unchanged
`run_eval.py`, records an exact manifest, and stops only the server process it
created.

Historical shell scripts under `/data/chaidi/kvmem_eval` are retained unchanged
for reproducing old results. New parameter sweeps should use this runner instead
of copying those scripts.

## Inspect a configuration

```bash
python3 scripts/kvmem_eval/run_kvmem_eval.py \
  --tag dryrun_bt32 \
  --block-tokens 32 \
  --sink-blocks 8 \
  --retrieval-method mean-k \
  --dry-run
```

Dry-run prints the server command, evaluation command, git commit, environment
flags, output paths, and expected judge without starting any process.

## Fast lifecycle smoke

```bash
python3 scripts/kvmem_eval/run_kvmem_eval.py \
  --tag smoke_bt32 \
  --block-tokens 32 \
  --sink-blocks 8 \
  --retrieval-method mean-k \
  --limit 1 --max-tokens 1 --no-judge
```

## Full 500-sample run

```bash
DEEPSEEK_API_KEY=... \
python3 scripts/kvmem_eval/run_kvmem_eval.py \
  --tag s500_32k_bt32_meank \
  --block-tokens 32 \
  --budget 32768 \
  --sink-blocks 8 \
  --retrieval-method mean-k
```

Defaults match the established `s500_32k_r05_current` envelope: Qwen3.6-27B
Q8_0, 262144 context, fp16 KV, 32K select/gen budgets, query-conditioned step
selection, GPU ratio 0.5, 64 GiB CPU tier, 256 GiB NVMe tier, thinking budget
4096, temperature 0.6, top-p 0.95, MTP chain 4, and max completion 32768.

The API key is read only from the environment and is never included in the
manifest or logs. By default, an existing tag is rejected; use a new tag or
explicit `--force`. Interrupted evaluations preserve the partial JSONL emitted
by `run_eval.py`.

## Incremental LongMemEval-M session ingest

`run_longmemeval_session_ingest.py` is a separate, causal harness for testing a
long-lived KVMem session. It does not change the canonical one-shot runner.
Historical assistant turns are supplied with `max_tokens=0`, so they are
teacher-forced prefill rather than regenerated. Once the prior rendered history
has filled the configured active capacity, the first user turn of each new
dataset session triggers semantic reselection and query replay; the remaining
turns keep that exact selected context.

The frozen ten-error launcher is:

```bash
DEEPSEEK_API_KEY=... \
scripts/kvmem_eval/run_longmemeval_m_session_ingest10.sh
```

It writes both a sample-level result JSONL and a request-level audit JSONL. The
audit verifies every ingest request returned `finish_reason=prefill_only`, and
records the session index, phase, fragment token count, selection mode, and
latency. Server-side `QW3_KVMEM_TRACE=1` logs the absolute query span, semantic
selection, aligned query replay, and retained session checkpoint.
