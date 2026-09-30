# Long-history visual retrieval and KVMem comparison

Date: 2026-10-01. These are diagnostic histories with synthetic archive notes
and real recorded Windows screenshots. They do not measure live-task success.

## Growing history and visual invalidation

`long-history-probe-01` used a historical Edge screenshot at tokens
28,794–29,794, separated from the final question by synthetic archive notes.
The requested copy code, Name and Priority appeared only in that image, never
in the filler or an earlier assistant answer. The current screenshot showed
Notepad and did not contain the answers.

The short 2,133-token visual control passed. An ingestion request at 97,922
tokens returned only `READY`. The subsequent 158,911-token question returned
all three expected fields: `QW3-WINDOWS-2026-ALPHA`, `Alice`, `High`.

This request crossed the 131,072-token retrieval budget. The final trace used
the requested `sub-block-mean-k-max` scorer with no fallback, a complete query
and index, and selected 1,020 of 1,233 source blocks. All 1,000 historical
visual tokens were in the selected blocks. The retrieval policy allowed old
images to become retrievable; it did not pin every image. The final query
selected/replaced 209 blocks, taking approximately 1.82 seconds for selection
and 0.57 seconds for replay. DFlash accepted 34 of 48 drafted tokens and
reported no fallback.

The final request took 32.91 seconds after reusing 96,896 prefix tokens from
the ingestion request. This is a prefix-reuse result, not an isolated speedup
from sparse retrieval. A negative control replaced the historical image while
keeping every text part unchanged. The visual prefix diverged at token 28,794;
the engine invalidated the reusable checkpoint, recomputed, and returned null
for all three fields in 63.16 seconds. No stale visual answer was reused.

Evidence: `apps/computer_use/.runtime/long-history-probe-01/`, including
`retrieval-check.json` and `absent_image_control/result.json`, plus
`long-history-engine-01/{manifest.json,engine.log,retrieval.jsonl}`.
The initial probe and the later negative control ran separately; the current
`long_history_probe.py` combines both checks for subsequent reproduction.

## Matched cold KVMem on/off comparison

The exact 158,911-token positive and negative requests were frozen and hashed
in `replay-long-corpus-01`. `comparison-long-02` replayed both requests with
prefix reuse disabled on both sides. All four outputs passed their JSON field
checks. This is one repetition per case and policy, not a statistical estimate.

| KVMem | Historical image | First output (s) | Request (s) | Output tokens |
|---|---|---:|---:|---:|
| On | Present | 64.64 | 65.01 | 40 |
| Off | Present | 59.44 | 59.83 | 40 |
| On | Absent | 63.79 | 63.95 | 15 |
| Off | Absent | 58.99 | 59.16 | 23 |

The sampled whole-device GPU peak was **43.41 GiB on / 67.14 GiB off**, about
35% lower with KVMem. The cold positive request was about 9% slower with KVMem.
The configured logical context remains 1,048,576 on both sides; this allocation
difference is not the memory requirement of a tightly capped 159K dense server.
GPU usage was sampled once per second, so it is not a hard allocation bound.

Both profiles use the same Q8_0 target, BF16 visual tower, FP8 KV, DFlash 2,
seven maximum draft tokens, temperature zero and maximum output 256. KVMem uses
K131072/G65536, a 48GiB CPU tier and retrievable visual history. The first
request starts with a cold image cache; the second reuses its Notepad image.
Model and binary identities match `computer_use_model_assets_20261001.json`
and the short comparison report. DFlash and native MTP are never combined.

Evidence: `comparison-long-02/{summary.json,aggregate.json,requests.csv,output-checks.json}`
and per-profile raw chunks, engine logs and resource samples. Per-request data
is also checked in as `computer_use_long_comparison_20261001.csv`.
`comparison-long-01` retains two startup failures caused by a terminated
process remaining briefly visible in NVML after releasing its file lock.
The launcher now waits for both lock release and a clear GPU process list;
it does not stop another process to obtain the GPU.

The checked-in `check_long_replay.py` now reproduces the answer and final-trace
checks against this retained corpus: **4/4 passed**. The initial cold selection
uses a provisional window while the query/index are being built; the later
query-conditioned selection has the complete query/index and no scorer
fallback. The no-fallback result applies to that final selection, not every
provisional operation during cold ingestion.

### Memory accounting scope

The DFlash-on log reports the following components. Capacities, usage counters
and whole-device samples have different scopes and must not be added together.

| Component | Reported amount | Meaning |
|---|---:|---|
| Target weights | 27,260.6 MiB | Native uploaded tensors; rounded log value |
| DFlash weights | 3,671 MiB | Uploaded draft tensors; integer MiB log value |
| Vision tower | 878 MiB | CUDA vision weights |
| GPU KV pool | 6,710,886,400 bytes | Pool capacity; includes its used blocks |
| GPU KV blocks after positive request | 4,311,744,512 bytes | Used part of the pool, not another allocation |
| Mean-key retrieval index | 1,073,741,824 bytes | FP16 index allocation |
| Assembly RoPE table | 67,108,864 bytes | Position table |
| CPU KV tier after positive request | 5,301,600,256 bytes | Used counter within the 48 GiB configured tier |
| CPU raw K / spill | 2,617,245,696 / 2,602,565,632 bytes | Components of CPU tier usage |

The log also reports a 202.12 MiB pinned query capture, 48 MiB GPU query staging
and two raw transfer buffers. It does not give a complete disjoint allocation
breakdown for image embeddings, verifier workspaces, every transfer buffer or
temporary CUDA allocations. The 43.41 GiB sampled whole-device peak includes
more than the named weights and KV/index allocations. Process RSS and configured
CPU tier capacity are also distinct; these counters are not a full allocator
audit or a physical 48GB-card qualification.

The retained startup log also reports three pinned host I/O slab pools:
512 MiB for prefill writeback, 256 MiB for packed stage-out and 256 MiB for
read staging. These are capacity reports contributing to host memory pressure;
they must not be added to process RSS as if they were separate physical usage.
`summarize_memory.py` now reproduces component records and the sampled peak
from this retained run in `dflash2-kvmem-on/memory-evidence.json`. It explicitly
reports that tensor/workspace allocation traces are absent. The existing binary
supports those trace hooks, but a separate instrumented GPU run remains pending.

### Target-only and native MTP controls

`comparison-long-controls-01` adds target-only (`none`) and native MTP,
each with KVMem on/off, against exactly the same two frozen requests. Its
eight answer and final-retrieval checks passed. Together with the retained
DFlash pair, the matrix contains **12/12 passed checks across six profiles**.
Each case has one repetition. Prefix reuse is disabled throughout; the first
case starts with a cold vision cache, and the second shares its Notepad image.

| Decoder | KVMem | Positive request (s) | Negative request (s) | Profile GPU peak (GiB) |
|---|---|---:|---:|---:|
| Target only | On | 64.39 | 62.49 | 38.26 |
| Target only | Off | 60.12 | 59.31 | 61.89 |
| Native MTP | On | 64.62 | 62.63 | 39.63 |
| Native MTP | Off | 59.64 | 59.13 | 64.38 |
| DFlash 2 | On | 65.01 | 63.95 | 43.41 |
| DFlash 2 | Off | 59.83 | 59.16 | 67.14 |

For these single cold positive requests, KVMem reduced sampled GPU peak by
about 35–38%, while total request time increased by about 7–9%. The replies
contain only 40 output tokens: long prefill dominates, so faster speculative
decoding does not establish an end-to-end advantage here. Native MTP uses a
chain length of four; DFlash uses a maximum of seven draft tokens. They run
separately. These are observations from this setup, not statistical estimates
or recommendations for a smaller context allocation.

All six manifests have identical binary and launcher hashes and identical
frozen case hashes. Git commits differ because application helpers and reports
were updated between batches; no engine binary or launcher change occurred.
The final on-mode selections have complete query/index capture and the
requested scorer without fallback. Target-only selected 1,024/1,242 blocks;
MTP and DFlash selected 1,020/1,233 for the positive case and 1,021/1,233 for
the negative case. All negative outputs have null values for the three fields.

The merged `computer_use_long_matrix_20261001.csv` contains all 12 requests,
their originating run directories, timing, CPU/RSS and profile-level GPU
samples. The existing four-row DFlash CSV remains unchanged. Raw output and
retrieval checks are retained in each run's `visual-checks.json`.

## Real Codex multi-turn history

`codex-long-01` ran the pinned Codex app-server against QW3 and an observation
MCP fixture. It collected eight observations sequentially, each containing
600 synthetic archive notes and one recorded screenshot. Only observation 2
contained the historical browser image. The collection turn returned
`COLLECTION_READY`; a second user turn asked for the fields without another
tool call.

The final backend request retained all eight original image hashes and all
4,800 archive notes, with no answer in the text. Its prompt reached **168,845
tokens**. All eight MCP calls completed and the final three-field answer was
correct. The scorer captured all 88 query tokens and all 1,318 indexed blocks,
then selected 996 of 1,320 materialized blocks with no scorer fallback.
This demonstrates actual Codex history preservation and above-budget selection.

The full diagnostic took 75.62 seconds; its final answer request took 1.69
seconds, reusing 168,576 prefix tokens and prefilling 269. Server visible-output
latency for that request was 1.24 seconds. These warm timings are not comparable
to the cold table as an isolated retrieval speedup. The final request accepted
19 of 50 DFlash draft tokens, with no fallback. Sampled whole-device peak was
43.96 GiB and process RSS high-water was 35.66 GiB.

Evidence: `codex-long-01/{result.json,history-check.json,events.jsonl,resources.jsonl}`,
its exact inference requests, and `codex-long-engine-01/{manifest.json,engine.log,retrieval.jsonl}`.
The fixture never dispatches Windows input; repeated live desktop tasks remain
a separate acceptance item.

## Scope

The short on/off × target-only/native-MTP/DFlash matrix is reported separately
in `computer_use_kvmem_comparison_20261001.md`. Both short and above-budget
histories now cover all six profiles. No million-token, 48GB physical-card,
native Windows client, Mac or physical-desktop qualification follows from
these diagnostics.


## Separate CUDA allocation diagnostic

`memory-long-dflash-01` replays the identical frozen 158,911-token positive and
negative cases with DFlash, KVMem on/off and prefix reuse disabled. Both sides
use the same unchanged engine binary and new launcher hash, with existing tensor
and workspace trace hooks enabled. All four visual answer checks passed;
`summarize_memory.py --require-traces` passed on both completed engine logs
(701 on-mode events; 331 off-mode events, no inconsistent records).

| Counter | KVMem on | KVMem off |
| --- | ---: | ---: |
| Whole-device usage maximum at traced allocation/free events | 43.42 GiB | 67.14 GiB |
| One-second whole-device sampled peak | 42.76 GiB | 66.71 GiB |
| Traced owned tensors' simultaneous peak | 10.27 GiB | 34.24 GiB |
| Named workspace maximum capacity | 382.11 MiB | 255.67 MiB |
| Model process RSS high-water counter | 35.72 GiB | 32.03 GiB |

The event snapshots caught peaks missed by one-second sampling. Traced tensors
cover owned allocations of at least 1 MiB, not all weights, direct allocations,
views, CUDA-library internals or transient usage. Workspace capacities overlap
with other scopes and must not be added to device usage. Exit via SIGTERM means
remaining live trace counters are not a leak assessment. This is a separate
instrumented diagnostic, with one repetition per input, not another timing
benchmark or a physical 48GB qualification. Windows input probes ran concurrently
on CPU; no other GPU compute process shared the model GPU.

The compact exported counters and binary/log/launcher hashes are retained in
`docs/computer_use_memory_diagnostic_20261001.csv`; full logs, samples and scoped
reports are under `apps/computer_use/.runtime/memory-long-dflash-01/`.
