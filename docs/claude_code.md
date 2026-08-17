# Claude Code compatibility

`qw3 serve` exposes the Anthropic Messages subset used by Claude Code:

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- text, thinking, `tool_use`, and `tool_result` content blocks
- non-streaming Messages responses and named Anthropic SSE events
- opaque thinking signatures for multi-turn round trips

Start the server normally, then point Claude Code at it:

```bash
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --ctx 262144 \
  --enable-thinking

ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=local-qw3 \
claude --model Qwen3.6-27B
```

The API key is a client-side compatibility value; qw3 does not authenticate
local requests. `anthropic-version`, `anthropic-beta`, prompt-cache metadata,
and request metadata are accepted without changing Qwen inference semantics.

## KVMem behavior

When the server is launched with `--kvmem`, Anthropic requests automatically
derive mandatory semantic spans for:

1. the system prompt and tool definitions;
2. the current user task;
3. the live assistant tool-call and tool-result trajectory;
4. active `CLAUDE.md`/`AGENTS.md` policy carried in recognized
   `<system-reminder>` frames.

The spans consume the ordinary fixed KVMem selection budget. They are kept
during pressure prefill and later reselections. The resolved sink and semantic
spans form the hard, block-deduplicated set; recent tokens use only the budget
that remains. If the hard set exceeds `--kvmem-budget`, the request returns HTTP
413 with the required budget instead of silently dropping harness control
text. Requests carrying semantic spans use the serialized generation route
until continuous batching has per-row semantic-selection metadata.

For a typical Claude Code prefix, use a KVMem budget comfortably above the
system/tool prefix size. The exact size depends on the Claude Code version,
enabled tools, skills, project instructions, and tokenizer.

## Current limitations

Image, document, redacted-thinking, and Anthropic server-tool blocks are not
yet supported. They return an explicit `invalid_request_error`; unsupported
content is never silently removed. OpenAI-compatible endpoints keep their
existing request and response formats.

See [harness_compatibility.md](harness_compatibility.md) for reminder lifetime,
budget priority, OpenCode, and DeepSeek Harness details.
