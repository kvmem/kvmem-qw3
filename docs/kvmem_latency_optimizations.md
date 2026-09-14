# KVMem latency optimizations

These switches preserve the full model-visible transcript. They do not summarize
history, change sampling, reduce the output limit, or change the KV dtype. The
OSWorld integration uses them only when explicitly enabled.

| Switch | Default | Candidate setting | Purpose |
|---|---:|---:|---|
| `QW3_OSWORLD_MESSAGE_PREFIX_CACHE` | 0 | 1 | Send only messages after the last acknowledged, identical prefix |
| `QW3_MESSAGE_PREFIX_CACHE` | 0 | 1 | Retain acknowledged immutable message prefixes |
| `QW3_PREPARED_PREFIX_CACHE` | 1, when message caching is enabled | 1 | Reuse parsed image identities, rendered prefix records and exact visual token maps; set 0 for legacy preparation |
| `QW3_PREPARED_PREFIX_CACHE_MIB` | 512 | 2048, validated on the complete 500-request replay | CPU metadata accounting limit for the prepared snapshots; independent of KV and vision feature tiers |
| `QW3_SERVER_TOKEN_CACHE` | 0 | 1 | Cache tokenization at atomic chat boundaries and decoded vocabulary bytes |
| `QW3_VISION_GPU_CACHE_MIB` | 512 | 4096 | Existing GPU cache of projected BF16 image features |
| `QW3_VISION_CPU_CACHE_MIB` | 0 | 8192 | Bounded CPU backing for evicted image features, uploaded only on demand |
| `QW3_KVMEM_DECODE_CHECKPOINT_TOKENS` | 0 | 1024 | Capture a replay checkpoint at an eligible block boundary after this decode interval |

The launcher forwards the client message-cache switch to the server unless an
explicit server setting overrides it. Run manifests record all these switches.

## Exact history transport and CPU preparation

The native Claude Code comparison still sends its complete original Messages
request on every call. Claude Code does not use the optional message-handle
transport extension described below. Its server-side prepared-prefix reuse
reduces repeated processing after receipt. With the native prepared cache and
direct request path enabled, `QW3_ANTHROPIC_WIRE_PREFIX=1` now compares exact
source-message bytes before constructing the JSON objects. Only the unmatched
suffix is parsed and converted; current request options are still validated.
Changed system messages fall back to the full request. Source records and
their Anthropic-to-internal message counts share the bounded prepared-cache
budget. This does not eliminate full-request transmission or the exact byte
comparison. Configuration and fallback behavior are documented in
`docs/claude_code.md`; validation artifacts are under
`benchmark/osworld_v2/claudecode_wire_prefix_20260909`.

The complete 454-request native replay reaches 982,173 input tokens. The
original path, wire-prefix path, and subsequent CPU allocation update produce
identical complete outputs, usage, stop reasons, KV reuse lengths, and six
capacity-recovery positions. Each has only the initial cold prefill. The last
50 regular requests average 2.360, 1.949, and 1.903 seconds of API TTFT,
respectively. Same-position windows following the first and last capacity
recoveries reduce API TTFT growth from 692 to 278 ms. Runs are sequential,
one per build; this unscored replay excludes Claude Code's history construction
and serialization. It does not establish a flat end-to-end curve or higher
task success.

The executor retains geometric capacity when copying visual override rows.
Copy assignment otherwise reallocates exactly the growing history length on
each append. All rows are still replaced and validated because feature-storage
ownership can change even when image contents match. This preserves text-only
handling and limits extra vector capacity to the chosen growth margin. The
final replay's last 50 row-copy phases average 7.7 ms, down from 47.2 ms in the
wire-prefix build. CUDA objects are unchanged. Final pure-text, real-query
replay, image rewrite, history shrink, and text-after-image checks pass; details
are in `claudecode_wire_prefix_20260909/row_capacity` under the same benchmark
directory.

The HTTP server uses a 64 KiB receive buffer. For a plain request with a known
Content-Length of at most 512 MiB, it reserves that many body bytes on the first
received data callback, after the existing payload admission check. This avoids
repeated string growth and copying. Larger, encoded, chunked and multipart
bodies keep their original handling; no request is shortened or rejected by
this reservation bound. The local httplib change is opt-in through
`CPPHTTPLIB_REQUEST_BODY_RESERVE_MAX`; the server defines the bound, which can
be overridden at build time (zero disables reservation).

The CPU-only HTTP checks under
`benchmark/osworld_v2/claudecode_http_ingress_20260909` and
`benchmark/osworld_v2/claudecode_http_reserve_20260909` preserve the complete
recorded request bytes and also verify chunked binary bodies. For the 290,783,388
byte request 500, median ingress was 741.6 ms with 4 KiB buffers, 558.2 ms with
64 KiB buffers, and 230.8 ms with 64 KiB buffers plus bounded reservation.
These are ingestion-stage measurements, not end-to-end API or task timings.

The complete replay in `benchmark/osworld_v2/claudecode_stream_cpu_20260909`
compares this CPU server update with the previously validated fast JSON path.
All 500 complete outputs, input hashes, token usage and 15 selection locations
match; the inference core and query generation/replay computations are unchanged.
The candidate has no zero-reuse request after initial startup. Mean API TTFT
changes from 2.333 to 2.085 seconds overall, from 2.174 to 1.927 seconds for
ordinary requests, and from 7.474 to 7.188 seconds for selection requests.
The last 50 ordinary requests improve from 3.459 to 2.531 seconds. At this
history size, the larger prepared cache keeps processing one new image per
request; the previous cache stopped advancing after request 426. These image
counts refer to preparation, not necessarily repeated GPU image encoding.

Streaming callbacks now share an immutable request state instead of copying
the full prompt and options on callback copies. Measured mean stream handoff
falls from 59.97 to 0.69 ms. This is work after preprocessing through provider
installation, not GPU queue time. That earlier candidate's regular TTFT grows from
1.138 seconds for the first 50 ordinary requests to 2.531 seconds for the last
50: full-request transmission, JSON parsing and final metadata materialization
remain history-dependent. This replay is unscored and does not establish task
success or equivalence of model capability.

The server keeps at most two immutable message snapshots, with a configurable
accounting limit that defaults to 512 MiB. The client computes an exact message-level common prefix with
its last acknowledged request, including image identity and any edited text.
Prepared-prefix caching keeps original messages separate from their rendered
forms. A validated prefix shares immutable records and image sources; only the
new suffix is parsed and hashed. The GPU vision frontend accepts these immutable
sources directly, avoiding repeated payload copies, hashes and byte comparisons
on identical source objects. It still resolves image features through its bounded
GPU/CPU cache; prepared records do not retain feature tensors.

Visual token mapping compares the actual newly rendered token prefix and exact
image identities/grids with the previous map. It reuses virtual IDs, row identities
and M-RoPE coordinates only before their common boundary, restarting at
`vision_start` if an edit intersects an image. Changed system/tools/template text
therefore invalidates affected positions even when message handles match. All
rows bind to the current request's storage. Cached maps contain CPU metadata only.
Final prompt/vector materialization still has linear work; this is not a claim
that the entire inference API now has constant-time history handling.

The next visual-index candidate replaces the executor's per-request hash table
for consecutive synthetic image IDs with an implicit row index. Every row still
passes the existing embedding validation. A nonconsecutive ID materializes the
original mapping, including duplicate detection; reordered or sparse callers
therefore retain their semantics. Lookup changes neither embedding contents,
scatter order, M-RoPE coordinates nor query generation/selection/replay.
CPU reference-map parity and executor compile checks are recorded under
`benchmark/osworld_v2/claudecode_visual_index_20260909`. A combined candidate in
`benchmark/osworld_v2/claudecode_visual_copies_20260909` also moves temporary
image/conversation JSON values instead of repeatedly copying their base64
payloads. It borrows immutable generation options during synchronous stateless
calls when no adjustment is required. Explicit sessions, missing fingerprints,
and output capping retain a private copy. Backend visual metadata is retained
only for explicit sessions and named-cache operations; the executor still owns
the current request's complete map independently.

All 500 original requests pass 1,500 complete CPU conversion comparisons.
Mean conversion time is 59.2 ms before and 39.4 ms after; for the last 50
requests it is 69.1 and 36.9 ms. These measurements exclude JSON parsing,
transport, GPU work, and other preparation. They are not API TTFT results.
The combined native build replaces exactly three CPU archive members, preserving
all CUDA objects. GPU parity passes 14 KVMem and seven ordinary-service checks,
including explicit visual sessions and named-cache restores. A real compaction
input reuses 227,472 tokens and prefills 3,487 tokens in 2.264 seconds; its
64-token greedy continuation equals a fresh-cache run. This is a correctness
check, not a measurement of full native summary generation. The complete
500-request replay now passes identical outputs, 15 selection locations, cache
reuse and prefilled-token counts. Mean ordinary API TTFT falls from 1.927 to
1.832 seconds (4.9%); the last 50 ordinary requests fall from 2.531 to 2.341
seconds (7.5%). Selection-request TTFT is 7.188 versus 7.091 seconds, with the
full original query generation and replay retained. Sampled peak server RSS is
100.386 versus 100.048 GiB, with no swap in either replay. This is a fixed-input
comparison between two KVMem versions, not a new desktop success trial or a
comparison against ordinary qw3. Ordinary latency still grows with history;
the next parser optimization therefore remains warranted.
The expanded `claudecode_matched15_toolvalidated_20260909` cohort
was paused to prioritize this optimization; its three completed pairs use the
previously frozen STREAM binary and contain none of this candidate's results.

`QW3_VISUAL_CPU_TRACE=1` enables diagnostic timing of CPU visual metadata work.
The backend's `executor_setup_ms` encloses the executor's `total_ms`; these
timers must not be added together. The switch defaults off. It does not change
vision placement, feature resolution, model computation or replay semantics.

A subsequent JSON scanner candidate is prepared under
`benchmark/osworld_v2/claudecode_json_simd_20260909`. The local yyjson change
uses SSE2 to scan runs of ordinary ASCII string bytes, stopping before quotes,
backslashes, controls and non-ASCII input so the existing decoder retains its
validation. Vector reads are bounded by the supplied input length. The build
option `QW3_JSON_ASCII_SIMD` enables it where SSE2 is available; disabling the
option or building on an unsupported target retains the scalar scanner.
The candidate, frozen scalar baseline and ASan/UBSan build each pass 8,731
valid/malformed documents with exact values, types and errors. All 500 original
requests also produce identical parsed fields, types and floating-point bits.
The disabled SIMD object's bytes match the frozen scalar object; CMake ON/OFF
configuration checks verify the option wiring. In the separate, alternating
order CPU comparison, mean parsing time falls from 85.076 to 56.813 ms (33.2%);
the last 50 requests fall from 176.449 to 121.677 ms (31.0%). These timers
exclude protocol conversion, transport, inference and desktop actions, and
must not be interpreted as an API TTFT improvement.

The new native binary replaces only `yyjson.c.o` in the parser archive. The
entire validated visual-copy inference core, including CUDA objects, is
unchanged. GPU correctness passes 14 KVMem and seven ordinary-service checks.
On the real compaction input, warm KV state reuses 227,472 tokens and prefills
3,487 tokens in 2.259 seconds, versus 101.120 seconds for a fresh 230,959-token
prefill; the 64-token continuations match exactly. The complete 500-request
inference replay now passes output/usage/stop equality, identical source-reuse
counts and all 15 selection positions. Every noninitial request has positive,
consistent cache reuse. Mean nonselection API TTFT falls from 1.8324 to 1.8044 s
(1.53%); the final 50 nonselection requests fall from 2.3411 to 2.3036 s. Mean
selection TTFT falls from 7.0910 to 7.0584 s, while guided generation, semantic
selection and suffix replay are essentially unchanged. This is one complete
fixed-input run per variant, not a new scored desktop trial. Both runs have
zero swap and similar minimum host-available memory; their different RSS
high-water marks alone do not establish algorithmic memory savings.

The next fresh 052 pilot is queued under
`benchmark/osworld_v2/claudecode_no_query_replay_20260909`. After default behavior
and ablation state checks pass, it uses one frozen binary for full KVMem,
ordinary qw3 and the separately authorized no-replay ablation. The full method
still retains guided generation, selection and replay. The originally prepared
two-arm-only pilot was superseded before launch. Fixed-15 expansion remains
pending evaluation of the live pilot.

Shared records are conservatively charged to every snapshot; in-flight requests
can retain evicted snapshots until they finish. Oversized snapshots are not
acknowledged. Setting `QW3_PREPARED_PREFIX_CACHE=0` restores the original raw
snapshot and full preparation path. Streaming and clients without message-cache
negotiation retain their existing paths. Tools and generation options are sent
with every request. An expired handle returns a dedicated HTTP 409
before inference; the client then resends the complete messages. A server that
does not acknowledge the extension continues to receive full requests.

The tokenizer cache splits only at a special token that the loaded tokenizer
recognizes as an atomic `<|im_start|>` boundary. It falls back when the tokenizer's
BOS behavior is incompatible. A vocabulary-sized immutable decoded-byte table
accelerates semantic span mapping without changing the byte offsets. Screenshot
URLs and trace metadata also reuse the original immutable image strings.

## Vision placement and capacity

The visual neural network, its projection, and the language model remain on the
GPU. Image decoding/resizing/patch preparation, message handling, and the CPU KV
and index tiers still use CPU resources.

The server's `pre_validate_ms` timer is broader than CPU validation: it spans
`server_json_end` through `server_render_start`, including incremental vision
preparation and the vision encoder. It therefore includes GPU work when a new
screenshot is encoded. Likewise, wrapper/preprocessing totals contain nested
stages and must not be added to those stages. Use the explicit visual CPU timers
for metadata-copy costs; the complete preprocessing total is not an estimate of
CPU work that can be removed.

On GPU feature-cache eviction, the optional CPU tier copies the original BF16
bytes without changing their precision. A CPU hit initially creates only row
metadata and a reference to those bytes. `DeviceInputEmbeddingStorage::resolve`
uploads that image once per request only when actual prefill/replay needs a row.
Duplicate images share the same deferred upload. Warm historical KV hits do not
require uploading all historical image features.

The CPU cache is bounded and uses LRU eviction, including accounting for the
stored image key. In-flight requests may temporarily retain entries after LRU
eviction. If a feature is absent from both tiers, the original image is encoded
again. Thus the setting is a cache budget, not a guarantee that every possible
2M-token image history fits. No NVMe is used for this cache.

## Decode checkpoints

A three-entry ring keeps recent aligned decode states. Capture is opportunistic:
it waits for a committed block boundary and a complete MTP prefix after the
configured interval. The interval is not a hard upper bound on replay length.
Entries are published only after the turn's source index is finalized. Reuse
requires an exact token/visual prefix, `C <= LCP`, compatible query coordinates,
current selection generation, complete source index and MTP state, and alignment.
The newest eligible ring state can become P; the existing P replay path restores
its state and metadata together. A rewind invalidates entries beyond its origin.
Unrelated requests clear the ring. A deeper rewrite can still require cold
prefill when no retained state is safe.

This is separate from the CPU background GC. Snapshot capture copies small
model states on the GPU; it has measured, nonzero overhead. It does not copy the
entire historical KV cache. Short ordinary responses below the interval do not
create periodic decode checkpoints. The MTP route is the route covered by the
new periodic capture.

## Validation and measurement boundaries

Prepared-prefix artifacts are under `benchmark/osworld_v2/prepared_prefix_20260908`.
`qw3-vision-request-preparation MODEL.gguf` checks 180 actual-tokenizer histories
with append, tail/image/grid/system rewrites, exact raw/rendered/token/visual-row/
M-RoPE parity, eviction, byte limits and in-flight snapshot lifetime.
The paired GPU service test checks 9 scenarios, including tool changes and
CPU-feature-cache hits, against the same binary with the switch disabled.
`qw3-vision-cpu-cache` additionally exercises immutable prepared image sources
and checks exact BF16 data under reordered/duplicate hits and capacity fallback.

The CPU microbenchmark uses the real optimized 052 step 499->500 messages and
500 source images. It stubs feature storage/grids and excludes visual encoding,
GPU prefill, decode and the complete server harness. Its common ChatML wrapper
produces 1,068,179 tokens (the full server uses extra wrappers). Across three
paired samples per mode, median CPU preparation including rendering/tokenization
and mapping fell from 553.923 ms to 30.341 ms, with identical mapping digests.
The incremental request reused 998 messages, 499 images and 1,066,011 mapped
tokens. These numbers are CPU-stage measurements, not observed TTFT improvements
or a new scored 052 trajectory.

Artifacts are under `benchmark/osworld_v2/kvmem_latency_optimization_20260908`.
The original scored OSWorld run remains paused; the following are unscored
regressions, not new task-success results.

- `test_message_transport.py` and `test_trace_instrumentation.py`: exact prefix,
  tail/image edits, expiry retry, and unchanged screenshot metadata.
- `qw3-message-prefix-cache`: immutability, eviction, capacity and restoration.
- `qw3-chat-token-cache MODEL.gguf`: randomized rewrites, partial Unicode/special
  tokens, million-token visual history, vocabulary bytes and semantic span maps.
- `qw3-vision-cpu-cache MODEL_DIR PYTHON WORKER IMAGE1 IMAGE2 IMAGE3 IMAGE4`:
  real GPU eviction, reordered/duplicate CPU hits, exact BF16 restoration, and
  bounded-cache fallback. Use distinct approximately 2040-row screenshots.
- `scripts/osworld_v2/test_latency_optimizations.py`: controlled GPU A/B for a
  capped long response, tail rewrite, short append, deeper branch, unrelated
  prompt, and expired message handle. Supply the binary, launch settings, and
  saved server command via the documented command-line arguments.
- `scripts/osworld_v2/kvmem_full_history_preflight.py`: multimodal history growth
  beyond 256K and semantic replay, including a forced small GPU feature cache.

For the 132K-history, 4096-output-token regression, the same six model responses
and token counts were observed before and after optimization. Tail rewrite TTFT
fell from 2.531 s to 0.608 s; a deeper branch fell from 48.499 s to 1.137 s. These
are server TTFT measurements. The tested rewrite shrank replay from 4238 to 878
tokens. Three periodic checkpoint captures cost approximately 14 ms in total.

For the actual first 323 recorded images of OSWorld task 008, with the original
4 GiB GPU cache and identical GPU vision kernels, visual preparation fell from
23.098 s (115 re-encoded images) to 0.348 s (one new image). A second identical
request took 0.136 s with no re-encoding. These numbers measure vision preparation
alone, not complete API TTFT or action latency.

The candidate was built consistently with current CPU core headers; unchanged
CUDA objects from the frozen experiment were reused. The A/B source and binary
hashes are recorded in the artifact provenance. CPU compilation does not alter
the original GPU kernel objects. End-to-end OSWorld success and per-step latency
still require a new scored paired run; these regressions do not imply every
KVMem step must be faster than an ordinary qw3 step with a different transcript.
