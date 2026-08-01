# KVMem in-process milestone checkpoint benchmark

## Implementation status

The repository currently implements only the opt-in `kvmem-session` benchmark
described in this document. Its milestone objects are private to one benchmark
process and cannot be named, queried, or loaded by an HTTP request.

The request-level local cache API proposed below is a design only. In
particular, the code does **not** currently contain:

- a `freeze_to_local` or `kvmem_cache` request field;
- a cache-ID registry or cache status endpoint;
- hash-based cache lookup or validation;
- a way to send only a new query and name a saved checkpoint;
- TTL/LRU eviction, versioning, tenant isolation, or concurrent writers;
- durable checkpoints that survive server restart.

## Purpose

Long-context performance experiments previously issued one semantic query at
each context length. A single observation mixes cold-start noise, block
residency, retrieval variance, and query replay variance. The optional
`kvmem-session` milestone mode prefills a context once, captures the exact
logical state, and issues multiple queries without rebuilding the history.

This is an in-process benchmark checkpoint. It is not yet a persistent
cross-process checkpoint format.

## CLI

```bash
./build/qw3 kvmem-session \
  --model /path/to/model.gguf \
  --session-ladder 1M,5M,10M \
  --session-query-tokens 32 \
  --session-decode-tokens 1 \
  --session-repeat-queries 20 \
  --session-repeat-mode frozen \
  [ordinary KVMem options]
```

New controls:

- `--session-repeat-queries N`: issue `N` queries at every ladder target. The
  default is `0`, which preserves the original profiling path.
- `--session-repeat-mode frozen`: restore the same milestone before every
  query. This measures independent query latency on one fixed history.
- `--session-repeat-mode sequential`: restore once, then let later probes see
  earlier probe turns. This models a warm multi-query agent session.

The Python driver exposes the same controls as `--repeat-queries` and
`--repeat-mode`. It writes parsed rows to `repeated_queries` in the output JSON.

## Exact lifecycle

When repeated-query mode is enabled, each ladder point runs as follows:

1. Incrementally ingest only the context delta with `max_tokens=0`.
2. Do ordinary pressure selection while the bounded GPU pool fills, but do not
   perform a synthetic final semantic selection during history ingest.
3. Capture the milestone at the exact requested logical position.
4. Save all state needed by an in-process branch:
   - executor position and logical main/MTP page counts;
   - hidden state, MTP prefix hidden state, DeltaNet recurrent and convolution
     states;
   - KVMem registration and compact-window metadata;
   - the API session's block-aligned checkpoint and unaligned tail tokens;
   - the selected working-set block IDs.
5. Append a synthetic query, force semantic reselection, replay the query, and
   decode the requested probe tokens.
6. In `frozen` mode, restore step 4 before every query. In `sequential` mode,
   continue from the preceding probe.
7. Restore the milestone after all probes so the next ladder point appends only
   its true context delta.

Restoration truncates branch-only KVMem blocks and re-materializes the captured
working set. Checkpoint capture and restore time are reported separately and
are excluded from the query's `total_ms`.

## Output

Each probe emits one machine-parseable line:

```text
[kvmem-session-query] turn=... query=... mode=frozen base=... \
  span=[...,...) final=... capture_ms=... restore_ms=... total_ms=... \
  semantic_ms=... replay_ms=... decode_ms=... score_ms=... \
  stage_in_ms=... stage_out_ms=... assemble_ms=... decoded=...
```

For each context length, report at least median, mean, p95, and standard
deviation for total query time, semantic selection, query replay, retrieval
score, stage-in, stage-out, and assembly. At least 20 queries are recommended.

## Compatibility and limitations

- All new behavior is opt-in. With `repeat_queries=0`, the legacy profiler does
  not receive the new CLI flags and keeps its previous prefill/decode lifecycle.
- The checkpoint stores small model state on the device but does not duplicate
  the complete historical KV. Historical blocks remain authoritative in the
  existing GPU/CPU/NVMe tiers.
- Frozen restoration fixes logical history and selected GPU working set. It does
  not currently flush the OS page cache or erase lower-tier heat, so a separate
  `canonical-cold` policy is still needed for controlled cold-I/O measurements.
- The snapshot cannot survive process exit. Cross-process recovery requires a
  durable manifest, persistent NVMe backing rather than the current ephemeral
  unlinked file, model/config hashes, logical block-to-backing mappings, and GPU
  re-materialization on attach.
- Restored runs intentionally cannot replace full-prefill throughput or pressure
  stage-out experiments, because those costs are skipped by restoration.

## Proposed request-level local freeze cache (not implemented)

### Objective

Allow an ordinary inference request to freeze its completed KVMem state under a
local cache ID. Later requests can name that cache and send only a new query,
without resending or re-prefilling the long history.

The explicit cache ID is the primary lookup mechanism. A content fingerprint is
secondary metadata for integrity checking and deduplication; it is not the
normal access key.

### Proposed save request

For a prefill-only context ingest:

```json
{
  "messages": [
    {"role": "user", "content": "...long context..."}
  ],
  "max_tokens": 0,
  "kvmem_cache": {
    "save": {
      "id": "experiment-sample-001",
      "scope": "local",
      "when": "after_request",
      "ttl_seconds": 86400
    }
  }
}
```

With `max_tokens=0`, `after_request` means the state immediately after prompt
prefill. On a generating request, it means the state after successful decode
and final KV/KVMem registration. The response should return authoritative cache
metadata:

```json
{
  "kvmem_cache": {
    "id": "experiment-sample-001",
    "version": 1,
    "status": "ready",
    "position": 1048576,
    "fingerprint": "sha256:...",
    "scope": "local"
  }
}
```

The server must not report `ready` until pending raw-K persistence, mean-K index
capture, KVMem registration, and tier writes required by the checkpoint have
reached a recoverable boundary.

### Proposed query-only load request

```json
{
  "messages": [
    {"role": "user", "content": "What is the final answer?"}
  ],
  "max_tokens": 32768,
  "kvmem_cache": {
    "load": {
      "id": "experiment-sample-001",
      "mode": "frozen",
      "required": true
    }
  }
}
```

The server restores the named checkpoint, renders the supplied messages as a
continuation suffix (without a second BOS/system prefix), performs semantic
reselection and query replay, and decodes normally. The client does not resend
the frozen history.

`required=true` makes a missing, evicted, incompatible, or corrupt cache an
explicit error. The server must not silently fall back to a full prefill because
that would invalidate latency measurements and can change model behavior.

### Load modes

`frozen` is a read-only branch:

```text
checkpoint -> query A -> answer A -> discard branch
checkpoint -> query B -> answer B -> discard branch
```

Each request starts from the same logical checkpoint. Query-specific retrieval
and query replay still occur. The original checkpoint remains unchanged.

`append` is an optimistic, sequential update:

```text
checkpoint v3 -> query -> answer -> atomically publish checkpoint v4
```

An append request should carry `expected_version`. Publishing succeeds only if
the named cache is still at that version, preventing two concurrent requests
from silently overwriting each other. The first implementation may atomically
replace the old version rather than retain a version tree.

### Identity, lookup, and fingerprints

The recommended identity model is:

```text
explicit cache ID  -> lookup and lifecycle control
version            -> ordered updates / compare-and-swap
fingerprint        -> compatibility validation and deduplication
```

The fingerprint should cover at least:

- model and tokenizer identity;
- model/KVMem configuration that affects state interpretation;
- canonical prefix token IDs and logical checkpoint position;
- checkpoint format version.

A random opaque server-generated ID is safest by default. A client-provided
alias is useful for controlled experiments, but must be scoped to the
authenticated tenant. Hash-only lookup is insufficient because it requires the
caller to possess or recompute the full prefix, does not naturally represent
sequential versions, and makes lifecycle control less explicit.

### Proposed status endpoint

```text
GET /v1/kvmem/caches/{cache_id}
```

It should report `creating`, `ready`, `evicted`, `incompatible`, or `failed`,
plus version, logical position, TTL, last-access time, and GPU/CPU/NVMe block
residency. Cache IDs must be unguessable or authorization-scoped so one tenant
cannot restore another tenant's history.

### Local registry contents

The registry entry needs to retain or reference:

- executor and DeltaNet recurrent/conv state;
- main/MTP logical page counts and MTP prefix state;
- API block-aligned checkpoint and partial-block tail tokens;
- KVMem registered position, active-window metadata, and selected block IDs;
- references to historical GPU/CPU/NVMe block authority and retrieval indices;
- configuration hashes, cache version, TTL, and refcounts.

It should not duplicate the complete historical KV for every checkpoint. Frozen
branches share immutable historical backing and allocate only branch suffix
state. Append should use copy-on-write or atomic replacement.

### Correctness constraints before implementation

The current benchmark can truncate one private branch because it owns a single
active milestone. A general cache registry cannot do that blindly: truncating
one restored branch must not release blocks referenced by another saved cache.
Named caches therefore require block/reference ownership or copy-on-write before
multiple versions are exposed.

The request layer must also define continuation tokenization explicitly. Sending
only a new chat message must append the correct role/template tokens to the
saved token stream; it must not add a new conversation BOS or duplicate system
prompt. Cache restore must validate model, tokenizer, KV dtype, block geometry,
retrieval method, and checkpoint format before touching executor state.

### Suggested implementation phases

1. Single-process registry, server-generated IDs, `save` plus read-only
   `frozen` loads, explicit cache-miss errors, no version tree.
2. Atomic `append` with `expected_version`, one current version per cache ID,
   refcounted shared history, and TTL/LRU eviction.
3. Status/delete endpoints, quotas, tenant isolation, metrics, and controlled
   concurrent frozen readers.
4. Persistent manifests and attach after process restart, as described below.

## Next phase: persistent attach

The persistent format should use one append-only canonical backing store shared
by 1M/5M/10M manifests. A manifest records a high-watermark plus logical block
metadata, retrieval-index offsets, recurrent/MTP state, configuration hashes,
and selected block IDs. It must never serialize CUDA pointers, GPU physical page
IDs, pinned-host pointers, or asynchronous I/O handles. Restore attaches the
backing file and materializes only the active KVMem window.
