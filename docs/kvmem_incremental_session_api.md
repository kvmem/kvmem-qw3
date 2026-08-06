# KVMem incremental session API

This API keeps LongMemEval-specific parsing outside qw3. The server exposes only
generic inference operations; callers decide what a session is, which user text
is a retrieval query, and when reselection occurs.

## Request fields

Chat Completions accepts these opt-in fields when `--kvmem` and
`--kvmem-query-conditioned` are enabled with mean-k retrieval:

- `kvmem_session_id`: non-empty ID of the single active persistent session.
- `kvmem_session_op`: `start`, `append`, or `finish`. `start` resets model and
  KVMem state; later operations must use the same ID.
- `kvmem_reselect`: `auto`, `force`, or `off`. `force` requires an explicit
  `kvmem_query_span` or `kvmem_query_message_range`; `off` forbids one.
- `kvmem_prefill_window`: `pressure` permits the normal capacity-pressure path
  to rebuild a deterministic sink+recent window when the physical page pool is
  close to full (and before a later append when a sparse window is already
  active); it does not shrink a persistent session merely because one request
  fragment ended. `keep_selected` retains the last semantic selection and fails
  explicitly if its reserved append headroom is exhausted. `semantic_chunk` is
  a one-shot long-prefill mode described below.
- `kvmem_prefill_semantic_start_tokens`: optional block-aligned threshold for
  `semantic_chunk`. Zero inherits `--kvmem-prefill-budget`.
- `kvmem_prefill_semantic_query_tokens`: optional suffix cap for the retrieval
  query derived from each physical prefill chunk. Zero uses the whole chunk.

`start` and `append` require `max_tokens=0`. `finish` may decode normally.

## Semantic chunk prefill

`kvmem_prefill_window="semantic_chunk"` is an opt-in one-shot construction
policy for testing whether better context selection during long prefill improves
the retained KV representation. After the configured threshold, every physical
prefill chunk follows this sequence:

1. capture an ephemeral block-aligned target/DeltaNet/MTP boundary;
2. run a provisional target-model pass only to capture retrieval Q;
3. restore the boundary, score only sealed historical blocks, and install one
   semantic selection;
4. replay the complete chunk once to commit target KV, DeltaNet state, the
   Adaptive/Mean-K index, and MTP prefix KV under that selected window.

The provisional pass never registers blocks, captures raw K/index rows, builds
MTP draft-prefix KV, or starts tier writeback. Thus DeltaNet and MTP state are
committed exactly once. Query replay preserves the already assembled selection
and does not invoke a second `set_selection`. When all history still fits at the
first threshold, scoring cannot alter the identity result; that event skips the
provisional pass and commits the chunk once.

After a durable replay, proactive CPU/NVMe writeback starts immediately so its
D2H can overlap the next provisional target pass. Ephemeral checkpoints do not
wait for unrelated raw-K persistence, and a rollback with no durable suffix
does not execute the general tier-draining truncate path.

The mode currently requires a fresh one-shot request, block-aligned history,
query-conditioned Mean-K/SubBlockMeanK (including Adaptive prototypes), query
recomputation, and local-position MTP. It is deliberately incompatible with
persistent sessions, prefix-cache reuse, transcript replay, inline refresh, and
logit dumping until those lifecycle combinations have dedicated invariants.

## Prefill-only contract

`max_tokens=0` now means a real prefill-only transaction. The server renders
only the supplied messages and does not append a synthetic assistant generation
header. It performs tokenization, prefill, KVMem registration/offload, requested
reselection, query replay, mean-K index maintenance, checkpoint capture, and MTP
prefix priming, but never samples or commits a generated token. The response has
empty assistant content, zero completion tokens, and
`finish_reason="prefill_only"`.

## State and retrieval invariants

The session continuously builds the position-invariant per-layer mean-K source
index even before a query exists. Therefore a later query can score every prior
block, including blocks already offloaded from GPU.

After semantic selection, query recomputation is enabled by default. qw3 keeps a
model-state checkpoint at the last block-aligned position and at most
`block_tokens-1` following token IDs. If a new query begins in that partial
block, qw3 restores the checkpoint and replays the saved tail plus the new query
against the selected context. No historical selected window is re-prefilled.
`--no-kvmem-recompute-query` disables this behavior; the legacy
`QW3_KVMEM_RECOMPUTE_QUERY=0|1` environment override remains available for old
experiments.

The old 512-token query truncation has been removed. The full explicitly marked
user-query span participates in scoring. Very long queries consequently allocate
and compute proportionally more query rows; failures are explicit allocation or
capacity errors rather than silent truncation.

## LongMemEval-M policy

The reference harness is
`scripts/kvmem_eval/run_longmemeval_session_ingest.py`. Before 256K rendered
tokens it ingests sessions using pressure selection. Afterwards, for each
session containing a user turn, it:

1. teacher-forces any causally preceding assistant-only prefix under pressure;
2. marks the first user turn as the query and forces semantic reselection;
3. replays that query against the selected approximately 224K context;
4. teacher-forces the rest of the session with `keep_selected`;
5. repeats at the next session, then forces one final selection for the benchmark
   question and decodes with a 32K generation reserve.

Sessions with no user turn cannot causally supply a retrieval query and remain
on pressure selection. Both this count and assistant-prefix deviations are
recorded in the result JSONL.
