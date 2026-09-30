# DFlash 2 long-history native concurrency measurement

Started 2026-09-27 after the 9-stage short concurrency regression passed (264 responses).
This is a new performance workload, not a throughput interpretation of that short regression.

## Execution and evidence

- Worktree: `/home/chaidi/qw3-dflash2`, runtime commit `7ffa50c9c5911a13447cba6abb053dc1c4ed3561`.
- Qualified unchanged core archive: `benchmark/dflash2/lazy_attention_scores_build02/libqw3_core.a`.
- New timing observer: `tests/dflash_long_concurrency_probe.cpp`; standalone build and command provenance in `benchmark/dflash2/long_concurrency_build01/manifest.json`.
- Frozen CPU-rendered inputs: `benchmark/dflash2/long_concurrency_inputs01/manifest.json`.
- Background unit: `qw3-dflash-long-concurrency01.service`.
- Runner: `benchmark/dflash2/run_long_concurrency01.py`.
- Immutable plan: `benchmark/dflash2/long_concurrency01/plan.json`.
- Live progress: `benchmark/dflash2/long_concurrency01/manifest.json`.
- Each stage preserves config, native logs, exact callback token IDs/timestamps and raw 10 ms NVML samples. Completed stages also write `metrics.json` and `resources.json`.

Model/probe/input hashes are verified before inference. The runner holds the experiment lock,
waits for 60 seconds with no GPU compute processes before each stage, and checks progress every
15 seconds with a journal heartbeat every 5 minutes. A failed stage, foreign GPU process, or
resource-observer error stops this queue and only its own child. Stage timeout is 3 hours.

## Fixed workload

Target Qwen3.8-27B Q8_0, FP8 KV; existing official DFlash 2 draft. Every request retains
K=131072 and G=65536 tokens, block 32, prefill chunk 2048, page 16, context capacity 1048576.
The CPU tier is 64 GiB per slot, GPU memory ratio .95, and each of the shared main/MTP arenas
reserves `concurrency * (K + G + 8192)` token slots. These resource settings differ from the
previous single-request .44 GPU ratio and must not be hidden in historical comparisons.
There is no 48 GB GPU cap; memory is measured. A 110 GiB systemd host-memory limit confines
the owned test on the 125 GiB host. Capacity failures are recorded without reducing K or G.

Inputs are the unpadded original frozen DeepSWE prompt bodies, rendered by the same native
MiniSWE renderer as the previous replay. Selected indices from the 45-request manifest are
`9,10,21,22,24,25,36,37,39,40,42,43`: the first two frozen requests for each available task
at 131072, 196608 and 262144 thresholds, in source order. Tasks are Quill toolbar focus,
Effect SSE/HTTP API streaming, and DynamoDB Toolbox recursive schemas. Actual token counts
range from 131310 to 263467. Selection is frozen before inference, independent of outcomes.

The query is the token interval changed when the last tool message is temporarily emptied
in a second CPU rendering; inference receives the full, unchanged original prompt. Native
query-conditioned Force reselection, immutable source K, GPU Mean-K, and step updates remain
enabled. Initial guided generation and HTTP semantic pins are absent. Middecode refresh is
configured at 61440 tokens (steady 61440, maximum 2, private query maximum 4096), so these
at-most-4096-token continuations ordinarily do **not** exercise a middecode refresh. They
measure long-history selection and decoding; refresh-blocking performance needs a separate
explicit scenario. Joint prefill and joint draft remain off; the full per-slot controller
and ragged joint verification are enabled.

Sampling is temperature 1, top-k 20, top-p .95, repetition penalty 1, presence penalty 0,
top-k-first filtering, independent request RNG, natural EOS, maximum 4096 output tokens.
Each request starts cold with no prefix cache. Later prompts remain frozen and do not use
newly generated text. There is no tool execution or SWE task-success claim.

## Queue and metric definitions

The C2 pilot uses the first two real prompts and at most 64 output tokens; it is an admission
and genuine-joint-verification check, **not** a formal throughput result. On success:

1. Seed 1000: C1, C2, C4, each with all 12 inputs.
2. Seed 1001: C4, C2, C1, each with all 12 inputs.

Requests start in fixed groups of C; a group completes before the next group starts.
This is not sustained replenishing serving throughput. The runner requires a named joint
verification trace with the requested C before accepting a multi-request stage.

- Cold service throughput: sum of output tokens / sum of group submission-to-completion wall times.
  Model loading is excluded; queueing, cold prefill, reselection and generation are included.
- Aggregate callback decode throughput: sum of `output_tokens - 1` for every request / sum
  of each group's earliest-first-to-latest-last callback wall interval. Includes interference
  from other slots. Individual overlapping durations are **not** summed as an aggregate rate.
- Per-request TTFT, callback first-to-last decode rate, p95/max callback gaps, token-limit
  censoring, and actual joint-batch sizes are retained. Speculative callback bursts are
  observed user-output arrival times, not individual GPU kernel completion times.
- Native generation/speculation/refresh phase summaries remain available for diagnosis.
- NVML sampled whole-device/process peaks and host RSS are observations, not exact allocator maxima.

These measurements do not qualify the HTTP entrypoint: its current compatibility guard
routes realistic pinned/query/refresh requests outside the continuous route. No guard was
removed to obtain these numbers. Native cold C1/C2/C4 share the same new input/query/resource
policy; their results must be compared with each other. The previous 99.27/102.52 tok/s warm
HTTP replay is context only, not this queue's matched C1 control. No performance result is
asserted until the corresponding GPU stage has completed.
