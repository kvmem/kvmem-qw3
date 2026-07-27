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

## Utility evaluation ledger

Every accuracy run, including smoke tests, partial failures, rejudges, and
derived retry/merge experiments, is indexed in
[`docs/kvmem_utility_evaluation.md`](../../docs/kvmem_utility_evaluation.md).
The accompanying
[`docs/kvmem_utility_evaluation_registry.json`](../../docs/kvmem_utility_evaluation_registry.json)
stores exact sample IDs, an ordered-ID fingerprint, parameters, result paths,
and artifact type.

An accuracy runner must preserve a unique tag, its exact `run_config.json`,
dataset/manifest or selected sample IDs, raw answers, per-sample evaluation,
summary, validation report, and server log. Rejudging never overwrites the
original evaluation; derived retry merges and text-replay controls must identify
their source artifacts.

After every accuracy test, refresh and validate the ledger from the repository
root:

```bash
python3 scripts/kvmem_eval/update_utility_evaluation.py
python3 scripts/kvmem_eval/update_utility_evaluation.py --check
```

The Markdown and JSON files are generated. Add only curated labels or historical
caveats to `scripts/kvmem_eval/utility_eval_overrides.json`; do not hand-edit
generated rows. The scanner copies only whitelisted configuration fields, so API
keys and arbitrary log contents cannot enter the registry.

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

`kvmem_init` is the frozen compatibility path. The three cumulative
optimization profiles are deliberately grouped by the bottleneck they target:

| Level | Added behavior | Primary objective |
|---|---|---|
| `opt_1` | heat-aware CPU admission and eviction | reduce the number of bytes loaded from SSD |
| `opt_2` | GPU-gather pages into contiguous slabs and batch D2H; with SSD, proactively write completed prefill blocks in the background, retain inclusive clean backing, and use bounded asynchronous positional writes | reduce stage-out latency |
| `opt_3` | CPU slots are gathered into two pinned slabs, copied by one H2D per slab, then GPU-scattered to cache pages; with SSD, also coalesce reads and pipeline SSD reads with H2D | reduce stage-in latency |

`opt_2` and `opt_3` work with CPU-only storage. In that mode CPU spill remains
exclusive so the tier only needs enough slots for non-resident blocks; retaining
every promoted CPU copy would otherwise require one CPU slot for every possible
context block and can deadlock a full working-set swap. When NVMe is configured,
inclusive SSD backing requires an arena large enough for one spill record per
possible context block. These profiles do not change retrieval scores, selected
block IDs, or KV values.

The portable default is a 64 MiB slab. CPU-only `opt_2` prewarms two pinned D2H
slabs (128 MiB), and `opt_3` adds two pinned gather/H2D slabs (128 MiB). A
reusable GPU staging slab adds at most 64 MiB and is shared by both directions.
With SSD, `opt_2` instead prewarms at most 512 MiB pageable write staging at the
default queue depth 8. It also prewarms four 64 MiB pinned write-through slabs:
for the current model a 2048-token immutable-K+MTP chunk occupies about 68 MiB,
so this is a two-chunk buffer. `opt_3` additionally uses the two pinned read
slabs per active read producer (two for CPU-only or SSD-only, four when CPU
hits and SSD misses may coexist). After
profiling a different host, override them with `QW3_KVMEM_IO_SLAB_MIB` and
`QW3_KVMEM_WRITE_QUEUE_DEPTH`; both values are range-checked and the queue
remains bounded. CPU gather defaults to four worker threads and can be tuned
from 1 through 16 with `QW3_KVMEM_CPU_GATHER_THREADS`. The SSD write-through
pool can be tuned from two through sixteen slabs with
`QW3_KVMEM_WRITEBACK_SLABS`, or disabled for a matched baseline with
`QW3_KVMEM_PREFILL_WRITEBACK=0`. The completed D2H slab also feeds the existing
heat-aware CPU cache by default; `QW3_KVMEM_PREFILL_CPU_ADMIT=0` is an
SSD-only diagnostic mode.

The frozen ten-sample query-replay launcher accepts the same switch:

```bash
KVMEM_OPT_LEVEL=kvmem_init TAG=query_replay_init \
  scripts/kvmem_eval/run_longmemeval_m_query_replay10.sh

KVMEM_OPT_LEVEL=opt_1 TAG=query_replay_opt1 \
  scripts/kvmem_eval/run_longmemeval_m_query_replay10.sh
```

Use `QW3_KVMEM_PERF_TRACE=1` for timed reselection stages. In `opt_1`, the
server also emits `[kvmem-cache]` rows containing incoming CPU hit rate,
admissions, rejected admissions, and evictions. In CPU-only `opt_2`, the trace
reports packed D2H batch count/bytes, exposed fence wait, and CPU scatter time;
its SSD mode additionally reports clean-backing reuse and completed write
bytes/syscalls. SSD mode also emits `[kvmem-writeback]` D2H-submit, SSD-submit,
and SSD-complete events, including the D2H tail that remained exposed after the
next prefill chunk. In `opt_3`, the trace reports CPU gather, packed H2D
batch/enqueue/wait times; SSD mode additionally reports bulk read batches,
positional-I/O syscalls, and read wait time.

The matched implementation benchmark and raw artifact tags are recorded in
[`docs/kvmem_storage_optimization_benchmark_20260723.md`](../../docs/kvmem_storage_optimization_benchmark_20260723.md).
The CPU-only packed D2H/H2D and GPU gather/scatter comparison is in
[`docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md`](../../docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md).
The SSD prefill write-through and mixed CPU+SSD pipeline benchmark is in
[`docs/kvmem_ssd_writeback_benchmark_20260724.md`](../../docs/kvmem_ssd_writeback_benchmark_20260724.md).

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

## Core performance ablation

The production path enables Proactive Stage-out, Hierarchical Reuse, and
Packed Rematerialization by default. A controlled run disables one group with
the repeatable server option:

```bash
--kvmem-optimize-off proactive-stage-out
--kvmem-optimize-off hierarchical-reuse
--kvmem-optimize-off packed-rematerialization
```

`--kvmem-optimize-off all` disables all three groups. It cannot be combined
with another value. The older `--kvmem-optimization-level` profiles are
deprecated reproducibility modes and cannot be combined with the new option.
Startup emits one `[kvmem-opt-status]` record per group; an unavailable
requested optimization produces an explicit error instead of silently
falling back.

Run the matched 512K single-sample matrix with:

```bash
scripts/kvmem_eval/run_agentlongbench_perf_ab.sh
```

The four sequential cells form a cumulative ablation:

```text
all-off
  -> proactive-stage-out
  -> proactive-stage-out + hierarchical-reuse
  -> all-on (+ packed-rematerialization)
```

This makes each adjacent difference attributable to the newly enabled group.
The cells use deterministic decoding by default and fail if an optimization
changes the generated answer. `summarize_kvmem_perf_ablation.py` aggregates
TTFT, stage-out, stage-in, assembly, total reselection time, GPU reuse, and
peak GPU memory into JSON and Markdown artifacts.

The canonical 512K results, exact metric definitions, hardware and runtime
parameters, complete implemented-optimization inventory, raw artifact paths,
memory control, and reproduction notes are recorded in
`docs/kvmem_performance_evaluation_20260726.md`.
