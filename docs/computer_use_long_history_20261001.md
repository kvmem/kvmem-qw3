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
in `computer_use_kvmem_comparison_20261001.md`. The above-budget comparison here
uses DFlash only. No million-token, 48GB physical-card, native Windows client,
Mac or physical-desktop qualification follows from these diagnostics.
