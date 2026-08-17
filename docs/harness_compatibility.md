# Harness compatibility

`qw3 serve` derives request-scoped KVMem mandatory spans for Claude Code,
OpenCode, and DeepSeek Harness without requiring private fields in the JSON
body.

## Routes and detection

| Harness | Route | Detection |
|---|---|---|
| Claude Code | `POST /v1/messages` | Internal Anthropic route context |
| OpenCode | `POST /v1/chat/completions` | Product identity signal when present; otherwise the compatible-tool structural fallback below |
| DeepSeek Harness | `POST /v1/chat/completions` | `User-Agent: deepseek-harness/...` or an `x-deepseek-harness-*` header |
| Compatible tool client | `POST /v1/chat/completions` | Nonempty top-level `tools` plus a leading system/developer message |

OpenCode provider stacks do not have to expose a stable product identity
header, so correctness does not depend on one. `x-session-affinity` and
`X-Session-Id` are not sufficient by themselves to identify OpenCode because
they are common routing headers. The compatible-tool fallback protects the
structural tool/system prefix when product identity is absent or stripped by a
proxy.

## Protected semantic regions

The prompt renderer records the exact byte provenance of every region before
tokenization. It does not assume a fixed prefix length.

- `SystemControl`: chat-template control text, complete tool schemas and tool
  protocol instructions, and leading system/developer messages.
- `CurrentQuery`: the latest real user task. Pure harness reminder messages and
  user-role `<tool_response>` compatibility messages are skipped when finding
  it.
- `LiveToolTrajectory`: every rendered assistant/tool/message segment after the
  current query.
- `ProjectPolicy`: the latest recognized workspace-instruction policy carried
  inside `<system-reminder>` frames, including `AGENTS.md` and `CLAUDE.md`.

A message containing both a real task and an appended reminder remains the
current query. A standalone reminder after the task is control context, not a
replacement user query.

Workspace policies use latest-baseline/latest-path semantics. A newer complete
DeepSeek Harness workspace baseline supersedes earlier baselines; a newer
`Instructions from: <path>` frame supersedes an older copy for the same path.
Removal notices remain visible as the latest state.

The server recognizes complete, balanced harness frames. It does not make every
arbitrary occurrence of the words `system-reminder`, `AGENTS.md`, or
`CLAUDE.md` permanent. For an unidentified generic client, only frames with a
known persistent workspace-policy signature receive control semantics.

## KVMem budget priority

Physical blocks are charged once even when sink and semantic spans overlap:

```text
hard: sink union SystemControl/CurrentQuery/LiveToolTrajectory/ProjectPolicy
soft: recent blocks while capacity remains
then: retrieval/profile candidates
```

The hard set never expands the configured selection window. If it exceeds the
active request budget, the server returns HTTP 413 and reports the harness,
mandatory token requirement, active budget, and span counts. Increasing sink
tokens cannot replace semantic spans: sink is positional, while workspace
policy and the live query may occur anywhere in the prompt.

Requests with mandatory spans currently fall back from continuous batching to
the serialized generation path. This preserves correctness until the batched
executor has per-row semantic selection metadata.

## Out of scope

Codex is not included in this compatibility layer because its custom provider
path uses the OpenAI Responses wire API. `qw3 serve` does not currently expose
`/v1/responses`.
