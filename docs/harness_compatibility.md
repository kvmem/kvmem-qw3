# Harness compatibility

`qw3 serve` derives request-scoped KVMem semantics for Claude Code, OpenCode,
and DeepSeek Harness without requiring private fields in the JSON body. Stable
control, the stable root task, the current instruction, the latest durable workspace policy, the live tool
suffix, and the retrieval scoring query have separate lifetimes.

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

## Semantic regions

The prompt renderer records byte-accurate provenance before tokenization. It
does not assume a fixed token count or pin everything after the first query.

- `SystemControl` is the chat-template control text, complete tool schemas and
  protocol instructions, and leading system/developer text.
- `RootTask` is exactly the first durable real-user task. It remains fixed for
  later continuation/finalization turns and is omitted as a separate pin when
  it is the same message as `CurrentQuery`.
- `CurrentQuery` is exactly the newest real user instruction. User-role tool results
  and reminder-only messages do not replace it.
- `ProjectPolicy` contains only the latest applicable `AGENTS.md`, `CLAUDE.md`,
  or complete workspace-policy frames. Later copies supersede older copies for
  the same path, so policy cost does not grow once per tool round.
- `live_suffix` is separate from the exact mandatory spans. It identifies the
  newest unfinished assistant tool call plus all parallel results, or the
  current user turn when no tool transaction is open. Tool calls and results
  are ordinary retrieval candidates, including the latest transaction. They
  do not gain mandatory protection from belonging to `live_suffix` by default.
  Only the
  final KV block is hard and replayed to seed generation (32 tokens at the
  production block size, possibly sharing a boundary with the result tail).
  A real user instruction retains its separate mandatory lifetime.

Completed and current assistant/tool rounds are ordinary historical retrieval
candidates. The current task can score all tool evidence without making the
interval from that task to the prompt end non-evictable. Explicit policy, visual
and client pins retain their own independent lifetimes; optional experimental
file-evidence pins still require their explicit configuration.

## Configure mandatory parts

`mandatory_tokens` in the trace is the measured hard-retention total. Configure
which automatic harness lifetimes contribute to that total with
`--kvmem-mandatory-parts`:

| Part | Region | Default |
|---|---|---|
| `system` | System/developer control and tool schemas/protocol | On |
| `root` | First durable real-user task | On |
| `query` | Current real-user instruction | On |
| `policy` | Latest applicable workspace policy frames | On |
| `tool-call` | Current assistant tool call, including its arguments | Off |
| `tool-result` | Current tool results, including parallel results | Off |

```sh
# Current defaults: ordinary retrieval for tool calls/results.
--kvmem-mandatory-parts system,root,query,policy

# Also protect results, while leaving call arguments retrievable.
--kvmem-mandatory-parts system,root,query,policy,tool-result

# All six lifetimes, or no automatic semantic lifetimes.
--kvmem-mandatory-parts all
--kvmem-mandatory-parts none
```

`default` selects `system,root,query,policy`. A comma-separated list replaces
the selection, so omitted parts are off. Unknown/empty/duplicate parts fail
before model loading. `tool-call` and `tool-result` refer to the current
transaction; completed tool rounds remain ordinary retrieval candidates.

An identified harness or compatible tool client can override a request on
either `/v1/chat/completions` or `/v1/messages`:

```json
{"kvmem_mandatory_parts": {"tool-call": false, "tool-result": true}}
```

The object changes only the named switches from the server's configuration.
A string or array replaces the selection, e.g. `"system,query"` or
`["system", "query"]`; `[]`/`"none"` disables all six, and `"default"` inherits
the configured server selection. Each request is independent.

Turning off a lifetime makes its original source retrievable; it does not
delete text, alter tool arguments, change the scoring query, or synthesize a
replacement. Overlap with another enabled lifetime still protects a block.
For a single user turn, `root` and `query` identify the same message: either
enabled switch protects it. Fixed sink/recent allocation, visual/client pins,
optional experimental file-evidence pins, and the final decode-seed block have
their own independent rules. To remove the fixed bands as well, explicitly set
`--kvmem-sink-tokens 0 --kvmem-recent-tokens 0`. `none` therefore need not produce
zero `mandatory_tokens`.

Mid-decode guided refreshes have a separate generated-tail policy. Configure it
with `--kvmem-middecode-recent-tokens auto|N`. The default `auto` preserves the
existing bounded recent tail and protects the complete open epoch at an emergency
refresh. An explicit token count replaces that policy at both safe and emergency
refreshes; `0` disables these additional pins, including open-fragment protection.
For a retention ablation with only a 1K fixed sink, use:

```sh
--kvmem-mandatory-parts none --kvmem-sink-tokens 1024 \
--kvmem-recent-tokens 0 --kvmem-middecode-recent-tokens 0
```

The final decode-seed block and any explicit client/visual/evidence pins remain
independent. Disabling hard retention does not remove their source text from the
input or historical retrieval index.

Selected regions are charged once at KV block granularity within the existing
budget. Capacity fitting can reduce oversized hard regions; enabling a part
does not enlarge the GPU window. Trace fields expose `mandatory_parts`,
`spans_tool_call`, `spans_tool_result` and the resulting `mandatory_tokens`.

## Retrieval query and live suffix

`score_query_span`, raw `live_suffix_span`, and fitted `live_replay_span` are
independent. The first produces retrieval Q, the second identifies the complete
unfinished transaction in durable KVMem storage, and the third is the bounded
contiguous tail that must be present immediately before decode. For a tool
continuation, that replay tail is only the final KV block; selection can recall
the rest of the transaction by relevance. The executor
never uses the scoring query's start block as a `pin_from_block` boundary. It
selects the historical window first and then replays only the fitted tail.

With `--kvmem-guided-reselect boundary` or `both`, a genuinely new user query
above A selects immediately. A trajectory first observed below A instead keeps
its dense A+B epoch and performs its first private guided selection at A+T.
With `middecode` or `both`, a long response may do the same at a safe
tool/code/text boundary after the configured number of generated tokens.
Private planning and query tokens are rolled back and never emitted through the
API. A malformed private query falls back to the original task query.
If one tool callback jumps past the remaining A+B headroom, the same private
refresh occurs at that request boundary, but the source is pressure-ingested
and indexed first. It is never appended keep-selected beyond the physical
generation reserve.

## KVMem budget priority

`KvmemRequestPlan` first counts the raw protocol lifetimes, then produces a
bounded active-window fit. Physical blocks are charged once when categories
overlap:

```text
durable/indexed: complete prompt, including every byte of large tool results
hard active: sink + protocol/task/policy anchors + final decode-seed block
soft active: query-selected blocks from all tool calls/results and other history
replay: final block for a tool continuation; fitted tail for a user instruction
```

By default the current tool round is excluded from the hard union except for
its final decode-seed block, whether its body is short, long or larger than the budget.
If the remaining required spans exceed the active budget, deterministic
capacity fitting protects the decode seed, sink and fair semantic anchors. A
real user suffix retains its fitted head/tail behavior. One eighth of the
active block budget is reserved for query-selected evidence in this fitted
case. The omitted middle
blocks are not dropped or summarized: pressure prefill writes them to the
CPU/NVMe-backed KVMem store and builds their content index. Logs expose raw,
fitted, soft-retrievable, and retrieval-reserve block counts. A tool result larger than the active
budget therefore does not return HTTP 413. HTTP 413 is reserved for the
declared logical context/storage envelope, not for an A-token working-set fit.

The backend finalizes this same plan against the concrete M/P prefix
checkpoint before touching device state. Missing, stale, or out-of-range
checkpoints use a bounded pressure-prefill rebuild; they never promote a stale
task query into an unrequested semantic reselection. Independently of the
normal A+T trigger, an incremental suffix larger than B is pressure-ingested
instead of being appended to the old A-token epoch; this is a last-resort
safety gate, not an additional steady-state reselection cadence.

For a 64K selection budget and 32K generation reserve, a production starting
point is:

```text
--kvmem-guided-reselect both
--kvmem-update-mode step
--kvmem-guided-thinking-tokens 0
--kvmem-guided-query-tokens 256
--kvmem-middecode-trigger-tokens 28672
--kvmem-middecode-max-refreshes 2
```

The production default generates the context-aware private retrieval query
directly (`--kvmem-guided-thinking-tokens 0`). A positive value enables an
additional private chain-of-thought ablation, but its decoded tokens add
latency without becoming part of the retrieval query. The guided query cap
accepts at most 512 tokens. Each mid-decode refresh starts
a new generation epoch; the 32K reserve remains separate from the 64K selected
context. A generated query that reaches its cap without EOS, a line boundary,
or terminal sentence punctuation is discarded and safely falls back to the
last complete query. `step` mode is required here so the explicit semantic
epochs are not interleaved with legacy fixed-interval reselections.

For A=64K, B=32K, T=28K, the first dense epoch refreshes at 92K and retains a
4K physical safety margin. Tool calls split a logical agent run into many short model requests. For a
stable real-user task, qw3 therefore also accumulates assistant/tool prompt
growth across those requests. When the same trigger is reached, one tool
continuation is internally promoted from `kvmem_reselect=off` to a guided
semantic epoch. Direct harness adapters that omit the field and therefore
arrive as `auto` use the same gate: same-task continuations below the threshold
are treated as `off`, rather than generating a private query after every tool
result. A shorter replacement trajectory resets this state, and the bounded
per-task tracker never changes ordinary Qwen or non-harness requests.

Requests with semantic spans currently fall back from continuous batching to
the serialized generation path. This preserves correctness until the batched
executor has per-row semantic selection metadata.

The controlled Claude Code runner is stricter: it also disables the `Agent`
tool and rejects any observed subagent call during analysis. This keeps one
logical trajectory on the server's single KVMem warm slot. It is a harness
restriction, not continuous batching or multi-trajectory KVMem support.

When both harness semantic handling and `--kvmem-prefix-cache` are active, the
renderer keeps historical assistant reasoning framing stable across a later
real-user turn. This prevents the chat template from rewriting the prefix at
the beginning of a long tool trajectory. Ordinary serving without that exact
combination retains the existing compact-history rendering.

## Out of scope

Codex is not included in this compatibility layer because its custom provider
path uses the OpenAI Responses wire API. `qw3 serve` does not currently expose
`/v1/responses`.
