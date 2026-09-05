# Native multimodal CPU frontend

This adds an opt-in native Qwen3.8 image path to `qw3 serve`. The language model
and all existing text/KVMem behavior remain unchanged unless the server is
started with `--vision-cpu-model` and a request contains an image.

## Start

```bash
./build/qw3 serve \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --vision-cpu-model models/Qwen3.8-27B-BF16 \
  ...existing options...
```

The directory must be the original Hugging Face checkpoint containing the
tokenizer, processor, config, safetensor index, and `model.visual.*` weights.
The persistent Python worker loads only the vision tower on CPU in BF16. The
projected FP32 image rows are transferred to the existing GPU language-model
prefill; the language-model weights never move to CPU.

The worker defaults to `.venv/bin/python` when it exists and otherwise uses
`python3`. Packaging overrides are available through
`QW3_VISION_CPU_PYTHON` and `QW3_VISION_CPU_WORKER`; they are intentionally not
additional public CLI flags.

`QW3_VISION_CPU_THREADS` selects the CPU vision worker's intra-op thread count.
When unset, the server uses the online CPU count. Exact repeated image payloads
are cached in process memory; `QW3_VISION_CPU_CACHE_MIB` controls the default
512 MiB cache and `0` disables it.

## Request formats

OpenAI Chat Completions accepts `image_url` content blocks whose URL is a
base64 `data:` URI. The Anthropic Messages adapter accepts native
`{"type":"image","source":{"type":"base64",...}}` blocks and preserves their
ordering relative to text.

V1 does not fetch remote HTTP image URLs. This keeps request processing
deterministic and avoids adding an implicit network client to the server.

## KVMem behavior

Every complete `<|vision_start|> ... <|vision_end|>` span is registered as an
atomic mandatory KVMem span. Visual embeddings are replayed from request-local
CPU output whenever their synthetic image-pad token IDs are re-prefilled. A
request is rejected when its mandatory blocks do not fit the active KVMem
budget; it is never silently truncated.

Dense prefill, selected-window construction, immutable raw-K rebuild, query
de-RoPE, query replay, decode, and MTP verification all use Qwen's exact
interleaved T/H/W M-RoPE layout. KVMem keeps both source and compact-window
three-axis positions so a semantic remap never falls back to scalar RoPE.

Process-local warm-prefix checkpoints retain lightweight per-row visual
identities in addition to the request-level fingerprint. Prefix admission
compares the source identity and M-RoPE coordinate of every visual row before
the candidate checkpoint. This permits an accumulated `[A,B,C] -> [A,B,C,D]`
request to reuse the language-model KV for `A,B,C`, without retaining the old
projected tensor solely for cache validation; changing or reordering a prior
image shortens the reusable prefix at its first visual token.

Named `kvmem_cache` checkpoints include the projected embeddings, complete
M-RoPE positions, mandatory spans, and request fingerprint. Frozen loads
therefore send only the new text query. Append loads reuse the last aligned
checkpoint and replay at most one old partial block before adding a suffix,
which keeps packed Adaptive/Fixed4 indexes exact. Persistent KVMem session
appends retain prior images and can add new images; their virtual token IDs and
M-RoPE coordinates are rebased into the combined source sequence.

## Validation

The CPU frontend smoke test validates processor output shape and grid metadata:

```bash
.venv/bin/python tests/vision_cpu_worker_smoke.py \
  --model models/Qwen3.8-27B-BF16
```

With a multimodal KVMem server already listening, the end-to-end canary sends
the image once, runs named-cache save/frozen/append/frozen-v2, then independently
runs persistent-session start/append/finish. Later requests intentionally omit
the image:

```bash
.venv/bin/python scripts/multimodal_kvmem_cache_smoke.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model Qwen3.8-27B-Q8_0.gguf \
  --image /absolute/path/to/image.png \
  --history-repeats 200
```

`scripts/multimodal_prefix_cache_canary.py` separately exercises accumulated
full-transcript warm reuse: turn one sends image A, and turn two resends A while
appending image B. With `QW3_KVMEM_PREFIX_CACHE_TRACE=1`, the second request
must report a prefix-cache HIT. Its optional `--replacement-image` case changes
historical A while preserving the token layout and must report a safe MISS.

`qw3-kvmem-immutable-k` additionally checks exact scalar/three-axis immutable-K
rebuild and query de-RoPE parity on CUDA.

## Deliberate restrictions

- no continuous batching for image requests;
- no `kvmem_round_padding` combined with images;
- no remote image URLs or video/audio inputs.

`kvmem_round_padding` fails explicitly because inserting padding after vision
finalization would invalidate its exact embedding/M-RoPE token map. Text-only
requests retain their previous routing, prefix-cache, MTP, and KVMem behavior.
