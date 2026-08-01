# KVMem local checkpoint and milestone benchmark

## Status

Two related mechanisms are implemented:

1. `qw3 kvmem-session` has private in-process milestones for repeated latency
   probes at fixed context lengths.
2. `/v1/chat/completions` exposes a named, process-local KVMem checkpoint API.
   A client can prefill a history once, save it under an explicit ID, send only
   a later query, and either discard that query branch (`frozen`) or publish a
   prefill-only continuation (`append`).

The request API is deliberately process-local. It retains executor state and
references the existing GPU/CPU/NVMe KVMem pools; it does not serialize CUDA
pointers or all model state into a portable file. It therefore does **not**
survive a server restart. Persistent attach is a separate future phase.

## Request-level local checkpoints

### Server requirements

The server must run with ordinary KVMem and query-conditioned mean-k retrieval.
Cache operations use the serialized plain request route, not continuous
batching. No extra CLI switch is required for the API itself.

For the Q8 GGUF used by the current smoke test, the complete command was:

```bash
QW3_Q8_BF16_MAIN=0 QW3_KVMEM_TRACE=1 ./build/qw3 serve \
  --model ./models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 18080 \
  --ctx 131072 --prefill-chunk 2048 --mtp-chain 4 \
  --kvmem --kvmem-block-tokens 512 \
  --kvmem-budget 32768 --kvmem-gen-budget 32768 \
  --kvmem-method retrieval --kvmem-retrieval-method mean-k \
  --kvmem-query-conditioned --kvmem-index-placement gpu \
  --kvmem-gpu-memory-ratio 0.5 --kvmem-cpu-gb 8 \
  --kvmem-opt-stage-out off --kvmem-opt-stage-in on \
  --kvmem-opt-pack on --temp 0
```

`QW3_Q8_BF16_MAIN=0` is a workaround for the existing Q8/BF16 batch RMSNorm
limitation and is not part of the checkpoint design.

### Save after prefill

```json
{
  "messages": [
    {"role": "user", "content": "...long history..."}
  ],
  "max_tokens": 0,
  "kvmem_reselect": "off",
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

The first implementation intentionally requires `max_tokens=0`. This gives an
exact teacher-forced endpoint: the request prefills the messages without adding
an assistant generation header, registers KVMem state, and publishes cache
version 1 only after the request succeeds. `ttl_seconds` is optional, where 0
means no TTL; the current maximum is one year.

The explicit cache ID must contain 1–128 characters from
`[A-Za-z0-9_.:-]`.

### Frozen query

```json
{
  "messages": [
    {"role": "user", "content": "What is the final answer?"}
  ],
  "max_tokens": 32768,
  "kvmem_reselect": "force",
  "kvmem_query_message_range": {
    "message_begin": 0,
    "message_end": 1
  },
  "kvmem_cache": {
    "load": {
      "id": "experiment-sample-001",
      "mode": "frozen",
      "required": true,
      "expected_version": 1
    }
  }
}
```

Only the new query is transmitted. Its rendered tokens are appended at the
saved logical position, semantic reselection and query replay run normally,
and decoding produces an answer. When the request ends, the query/answer branch
is discarded and the exact saved executor, recurrent, API-boundary, tail, pool,
and selected-window state is restored. Repeated frozen queries therefore start
from the same history version.

`expected_version` is optional for frozen reads but recommended for controlled
experiments. `required` is effectively always true: a missing, expired,
evicted, or failed cache never silently falls back to full prefill.

### Incremental append

```json
{
  "messages": [
    {"role": "user", "content": "...next teacher-forced history fragment..."}
  ],
  "max_tokens": 0,
  "kvmem_cache": {
    "load": {
      "id": "experiment-sample-001",
      "mode": "append",
      "required": true,
      "expected_version": 1
    }
  }
}
```

Append restores version 1, prefills only the supplied suffix, and atomically
replaces the registry entry with version 2. `expected_version` is mandatory;
a stale writer receives HTTP 409. Append currently also requires
`max_tokens=0`: generated assistant output cannot yet be used as the exact
published continuation boundary. A caller should submit recorded user,
assistant, and tool messages as teacher-forced history fragments.

### Response metadata

Both save and load responses include:

```json
{
  "kvmem_cache": {
    "id": "experiment-sample-001",
    "version": 2,
    "status": "ready",
    "position": 52037,
    "fingerprint": "fnv1a64:...",
    "scope": "local",
    "created_at": 1785,
    "last_access_at": 1786,
    "expires_at": null,
    "selected_blocks": 64,
    "total_blocks": 102,
    "residency": {
      "gpu_bytes": 1800404992,
      "cpu_bytes": 1853882368,
      "nvme_bytes": 0
    }
  }
}
```

The FNV-1a value covers relevant model/KVMem configuration, logical position,
and canonical history token IDs. It is diagnostic integrity metadata, not an
authentication primitive and not the lookup key. Lookup uses explicit
`id + version`.

### Status and delete

```text
GET    /v1/kvmem/caches/{id}
DELETE /v1/kvmem/caches/{id}
```

GET returns current metadata, including `ready`, `expired`, `evicted`, or
`failed`. DELETE marks the cache evicted and releases its saved state. A later
load is explicit failure, not a cold-prefill fallback.

Load error mapping is:

- 400: invalid operation or parameters;
- 404: unknown ID;
- 409: `expected_version` conflict;
- 410: known but expired or evicted cache.

### Registry contents and pool ownership

A ready entry saves:

- executor position, logical main/MTP pages, hidden state, MTP prefix state,
  and DeltaNet recurrent/convolution state;
- API block-aligned checkpoint, partial-block tail, and canonical session token
  IDs;
- selected KVMem block IDs and tier-usage metadata;
- ID, version, TTL timestamps, position, and configuration fingerprint.

Historical K/V, raw-K, mean-K indices, and tier manifests remain authoritative
in the existing executor-owned GPU/CPU/NVMe pools. The cache entry references
that lineage instead of duplicating all historical KV, so saving a 1M-token
history does not create a second 1M-token KV allocation.

The current executor and block store support one live lineage. Consequently,
creating a different named cache, issuing an unrelated cold request, or using
the legacy mutable session API evicts any previous ready named cache. Tombstone
metadata remains queryable. Supporting multiple independent ready IDs requires
block refcounts/copy-on-write and independent tier manifests; claiming that
without those mechanisms would permit stale GPU/CPU/NVMe references.

The HTTP server also serializes these operations with its generation mutex.
There are no concurrent frozen readers or concurrent writers in this phase.

### Reusable smoke test

With a server running at port 18080:

```bash
python3 scripts/kvmem_local_cache_smoke.py \
  --base-url http://127.0.0.1:18080/v1 \
  --cache-id local-smoke-001 \
  --history-repeats 5200
```

The script verifies save, two frozen queries, append to version 2, a frozen
query against version 2, stale append rejection, status, and delete.

The initial real-model validation used 52,023 prompt tokens:

| Operation | Result | Wall time |
|---|---:|---:|
| Save | ready, v1, position 52,023 | 14.966 s |
| Frozen query 1 | `CERULEAN-7319`, still v1/52,023 | 1.475 s |
| Frozen query 2 | `CERULEAN-7319`, still v1/52,023 | 0.767 s |
| Append | ready, v2, position 52,037 | completed |
| Frozen query v2 | `amber` | completed |
| Stale append expecting v1 | HTTP 409 | expected failure |

Trace logs confirmed that suffix-relative query spans were shifted to the saved
logical position, query replay ran, and mean-k retrieval reported no fallback.

## Private `kvmem-session` milestone benchmark

The request API above is for real experiment clients. The existing profiler is
still useful for low-variance repeated timing at synthetic context ladders:

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

- `frozen` restores the same private milestone before each probe.
- `sequential` restores once and lets later probes see earlier probe turns.
- Capture/restore includes the executor, block-aligned API checkpoint, tail and
  complete host-side session token lineage, plus the selected working set.
- Capture and restore time are reported separately from query `total_ms`.

Each probe emits:

```text
[kvmem-session-query] turn=... query=... mode=frozen base=... \
  span=[...,...) final=... capture_ms=... restore_ms=... total_ms=... \
  semantic_ms=... replay_ms=... decode_ms=... score_ms=... \
  stage_in_ms=... stage_out_ms=... assemble_ms=... decoded=...
```

For each context length, report median, mean, p95, and standard deviation over
at least 20 queries. Restoration deliberately skips full history prefill and
pressure stage-out; it must not replace cold-prefill throughput experiments.

## Future persistent attach

Cross-restart recovery needs an append-only canonical backing store and a
portable manifest recording high-watermark, logical block metadata, retrieval
index offsets, recurrent/MTP tensors, model/config hashes, and selected block
IDs. It must never serialize CUDA pointers, GPU physical-page IDs,
pinned-host pointers, or live asynchronous-I/O handles.

That phase also needs:

- durable checkpoint file format and atomic manifest publication;
- block reference counts/copy-on-write for multiple named histories;
- attach-time compatibility validation and lazy active-window materialization;
- quota/LRU policy, authentication/tenant isolation, and corruption handling.

Until those are implemented, `scope=local` means the currently running qw3
server process, not merely the same machine.
