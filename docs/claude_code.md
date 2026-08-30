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
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --ctx 1048576 \
  --enable-thinking --native-mtp-speculate --mtp-chain 4 \
  --kvmem --kvmem-budget 65536 --kvmem-gen-budget 32768 \
  --kvmem-query-conditioned --kvmem-update-mode step \
  --kvmem-guided-reselect both \
  --kvmem-guided-thinking-tokens 0 \
  --kvmem-guided-query-tokens 256 \
  --kvmem-middecode-trigger-tokens 28672 \
  --kvmem-middecode-max-refreshes 2

ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=local-qw3 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
DISABLE_COMPACT=1 \
claude --model Qwen3.8-27B
```

The API key is a client-side compatibility value; qw3 does not authenticate
local requests. `anthropic-version`, `anthropic-beta`, prompt-cache metadata,
and request metadata are accepted without changing Qwen inference semantics.
For controlled long-context evaluation, the two Claude Code environment
variables keep the client-advertised context equal to the server context and
disable Claude Code's own automatic compaction. They are supported by the
locally tested Claude Code 2.1.148; record both values in experiment manifests
because they are client-version-sensitive controls rather than API fields.

For a dense 256K arm that intentionally keeps Claude Code auto-compaction on,
the custom model name needs an explicit capability marker and auto window:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=local-qw3 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144 \
CLAUDE_CODE_AUTO_COMPACT_WINDOW=262144 \
claude --model 'Qwen3.8-27B[1m]'
```

Claude Code 2.1.148 otherwise treats an unknown custom model as 200K and
reports an effective 180K compaction window; it only honors
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` directly when compaction is disabled. The
`[1m]` suffix is a Claude Code capability marker and is stripped before the
request reaches qw3. With the explicit 262,144 window, Claude reports an
effective 242,144-token window and begins compaction at about 229,144 tokens,
which closely matches a 229,376-token (224K) KVMem selection budget.

## KVMem behavior

When the server is launched with `--kvmem`, Anthropic requests automatically
derive exact spans for the complete system/tool protocol, the first durable
root task, the newest real user instruction, and the latest applicable
`CLAUDE.md`/`AGENTS.md` policy frames. Root and current are deduplicated for a
single-turn request; retaining both on a continuation prevents a terse
"continue the original task" message from displacing its acceptance criteria. The
newest unfinished tool transaction is a separate raw live suffix. Earlier
assistant/tool rounds return to the historical retrieval pool.

The original user task is a retrieval scoring query, not a positional pin to
the end of the prompt. KVMem selects old evidence, then replays the fitted live
tail. Short live suffixes remain fully active. When a single tool result or
parallel result set is larger than the active budget, `KvmemRequestPlan`
retains bounded protocol/head/tail anchors, leaves the complete body in durable
KVMem retrieval storage, and replays only the fitted contiguous tail. Raw and
fitted blocks are logged separately; oversized tool output is not an HTTP 413
condition by itself.

The finalized request plan also verifies that a reusable M/P checkpoint is
both resident and a valid prefix of this request. If it is missing or stale,
the suffix is safely pressure-prefilled and indexed. The server does not
silently run a semantic selection with an old task query. If the suffix since
a valid checkpoint already exceeds the generation reserve, it follows the
same pressure-ingest path instead of being appended past A+B.

`--kvmem-guided-reselect both` enables semantic refresh at request boundaries
and safe mid-generation boundaries. If the initial prompt C is below the KVMem
budget A, crossing A alone does not refresh or discard history: the first epoch
continues to A+T. The default T is 28,672 tokens for a 32K generation reserve,
leaving 4K physical safety headroom. A genuinely new user query above A
refreshes immediately using that real query. Stable tool continuations and a
long single response use a private compact query only after T additional
tokens. Private tokens are rolled back and are not included in Anthropic
response content or usage. There are at most two refreshes per request. Because coding agents
normally stop each HTTP response at a tool call, the same threshold also
accumulates prompt growth across tool-result continuations of one stable real
user task. Crossing T upgrades that one `reselect=off` continuation to a
private guided selection; it does not require the harness to synthesize a user
message. The native Anthropic adapter normally presents these continuations as
`auto`; qw3 suppresses same-task `auto` below the threshold and promotes only
the threshold-crossing turn, so Claude Code does not reselect after every tool
result. Requests carrying semantic spans use the serialized generation route
until continuous batching has per-row semantic-selection metadata.
Use `--kvmem-update-mode step` with guided reselection: the guided boundary and
mid-decode epochs are the selection cadence. The legacy interval mode would
also reselect every N decoded tokens using the last query, adding redundant
selection work between semantic refreshes.

The default zero-token private-thinking cap still conditions the retrieval
query on the full live context; it asks for the compact query directly. A
positive cap is retained for ablations, but the extra private decode is charged
to refresh latency and is never exposed to the client.

For a typical Claude Code prefix, use a KVMem budget comfortably above the
system/tool prefix size. The exact size depends on the Claude Code version,
enabled tools, skills, project instructions, and tokenizer.

### Single-trajectory evaluation mode

The current Claude Code KVMem runner deliberately disables the `Agent` tool
with `--disallowedTools Agent`. QW3 still has one warm KVMem state per server,
so interleaving parent and subagent trajectories would overwrite that state
and invalidate prefix-cache measurements. The frozen analyzer also treats an
observed `Agent` tool call as a protocol error, which catches client versions
that ignore the CLI restriction. Multi-trajectory KVMem state remains out of
scope for this runner.

After guided reselection, MTP performs one target-model bridge step to rebuild
the draft prefix under the selected window and then resumes speculative
decoding. M/P checkpoint admission requires the MTP prefix length and every
required MTP V page to be resident or durably backed. A defensive spill with
missing MTP V is stored as main-model-only data, marks later MTP checkpoints
non-resumable, and never terminates the request; target-model verification
remains authoritative.

The DeepSWE runner writes model patches, summaries, status, and exit artifacts
group-writable so a host-side verifier can consume container-owned files. Its
service watchdog terminates an owned QW3 process group if the inner batch
runner disappears. For a resumable queue, launch the batch through
`benchmark/claude_deepswe_ab/durable_batch_supervisor.py` under tmux; completed
arms are reused, unexpected runner exits are restarted, and explicit
`paused_for_investigation` states are not retried.

Streaming HTTP status alone is not a success signal: once SSE headers have
been committed, a later inference failure still has status 200. Every request
now emits `stream_completed`, `terminal_status`, and `terminal_error` in
`[qw3-server-accounting]`; batch reports treat a terminal stream error as a
failed arm.

## Current limitations

Image, document, redacted-thinking, and Anthropic server-tool blocks are not yet
supported. They return an explicit `invalid_request_error`; unsupported
content is never silently removed.
OpenAI-compatible endpoints keep their existing request and response formats.

See [harness_compatibility.md](harness_compatibility.md) for reminder lifetime,
budget priority, OpenCode, and DeepSeek Harness details.
