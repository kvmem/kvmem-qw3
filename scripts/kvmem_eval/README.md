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

## Storage optimization A/B levels

The reusable runner exposes the storage optimization profile without changing
the model, prompt, retrieval, or judge configuration:

```bash
python3 scripts/kvmem_eval/run_kvmem_eval.py \
  --tag storage_init \
  --optimization-level kvmem_init \
  --dry-run

python3 scripts/kvmem_eval/run_kvmem_eval.py \
  --tag storage_opt1 \
  --optimization-level opt_1 \
  --dry-run
```

`kvmem_init` is the frozen compatibility path. `opt_1` changes only CPU-tier
cache admission: blocks repeatedly selected by semantic retrieval are retained
ahead of one-pass streaming blocks. It does not change retrieval scores,
selected block IDs, KV values, or model outputs when the same blocks are
resident. `opt_2` through `opt_5` are reserved rollout levels and are rejected
until their implementations are complete.

The frozen ten-sample query-replay launcher accepts the same switch:

```bash
KVMEM_OPT_LEVEL=kvmem_init TAG=query_replay_init \
  scripts/kvmem_eval/run_longmemeval_m_query_replay10.sh

KVMEM_OPT_LEVEL=opt_1 TAG=query_replay_opt1 \
  scripts/kvmem_eval/run_longmemeval_m_query_replay10.sh
```

Use `QW3_KVMEM_PERF_TRACE=1` for timed reselection stages. In `opt_1`, the
server also emits `[kvmem-cache]` rows containing incoming CPU hit rate,
admissions, rejected admissions, and evictions.

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

For a stage-by-stage reselection profile, export
`QW3_KVMEM_PERF_TRACE=1`. The server log then contains
`[kvmem-reselect-perf]` rows for both prefill-pressure (`kind=explicit`) and
query-conditioned (`kind=semantic`) selections. This diagnostic synchronizes
timed GPU regions; do not use it for an uninstrumented throughput baseline.

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

## BEAM-10M

`run_beam_10m.py` evaluates the official 10M-token BEAM conversations. Each
history is sent once as a prefill-only request. Probing questions then restore
the same immutable history checkpoint through `--kvmem-prefix-cache`; generated
answers never enter the next question's context.

Download conversation 1 through the Hugging Face mirror. The downloader
explicitly unsets lowercase and uppercase HTTP(S)/ALL proxy variables before
opening a network connection:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
  uv run --no-project --with pyarrow --with requests \
  python scripts/kvmem_eval/download_beam_10m.py --conversations 1
```

Inspect the real sample without launching qw3:

```bash
python scripts/kvmem_eval/run_beam_10m.py \
  --data /data/chaidi/kvmem_eval/data/beam_10m \
  --conversation-ids 1 --dry-run

DRY_RUN=1 scripts/kvmem_eval/run_beam_10m_kvmem.sh
```

Run all 20 probing questions for conversation 1:

```bash
scripts/kvmem_eval/run_beam_10m_kvmem.sh
```

The runner writes an append-safe per-question JSONL, a request audit, a summary,
and a BEAM-compatible JSON containing `llm_response`. The launcher enables
prefix-cache tracing and fails unless every question restored a warm KVMem
checkpoint. Use BEAM's official type-specific evaluator on the compatible JSON;
`--no-judge` only disables the LongMemEval judge in the generic process
orchestrator.
