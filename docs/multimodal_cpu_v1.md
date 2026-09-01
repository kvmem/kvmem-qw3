# CPU vision frontend V1

V1 adds an opt-in native Qwen3.8 image path to `qw3 serve`. The language model
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
budget; it is never silently truncated. Multimodal requests do not read or
write the process-local warm prefix cache because V1 checkpoints do not yet
serialize image embeddings.

Initial dense prefill and ordinary decode use exact Qwen T/H/W M-RoPE. The
existing KVMem compact-window re-RoPE operation is still scalar-position based;
adding a persistent per-source-token 3D position table is the remaining V2
correctness item for image spans that are physically remapped by semantic
selection. V1 therefore provides safe storage/replay and no-crash integration,
but KVMem multimodal quality above the selection budget must not yet be reported
as native-parity accuracy.

## Deliberate V1 restrictions

- no continuous batching for image requests;
- no append/session continuation or frozen local-cache capture for images;
- no `kvmem_round_padding` combined with images;
- no remote image URLs or video/audio inputs.

All restrictions fail explicitly. Text-only requests retain their previous
routing, prefix-cache, MTP, and KVMem behavior.
