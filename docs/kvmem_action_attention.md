# Sampled action attention for KVMem

This opt-in experiment combines Query retrieval with attention used while
producing generated output. The scope is configurable: tool-call content only
(the compatible default), or all non-thinking output. It does not change K/V or recompute Q*K.
It is not yet evidence that tool error rates improve.

## Configuration

Use the CUDA + FlashInfer build with online KVMem, ordinary block selection,
`--kvmem-method retrieval`, and `--no-continuous-batching`. Plain target decoding
and the default batched DFlash verifier are supported. Archive attach and
continuous batching are rejected when this experiment is enabled. The diagnostic
sequential MTP verifier is also rejected; accepted-prefix sequential replay is
supported.

```sh
qw3 serve --model /path/to/model.gguf --kvmem --no-continuous-batching \
  --kvmem-budget 36864 --kvmem-gen-budget 16384 \
  --kvmem-action-attention on \
  --kvmem-action-scope non-thinking \
  --kvmem-action-layers spread3 \
  --kvmem-action-interval 16 \
  --kvmem-action-fraction 0.5 \
  --kvmem-action-decay 0.5 \
  --kvmem-action-pressure on \
  --kvmem-prefill-attention on \
  --kvmem-prefill-attention-interval 1024 \
  --kvmem-prefill-attention-stride 32 \
  --kvmem-prefill-attention-half-life 8192
```

Keep the existing model, DFlash, KV precision, sink/recent, and other launch
options. All switches use `on|off`; scope values use hyphens. Attention capture,
pressure scoring and prefill scoring default to `off`; fraction defaults to
`0.25` (the example explicitly selects 50/50). The scope defaults to `tool-call`.
When enabled, the sampling defaults are `spread3`, prefill interval 1024 and
Query stride 32. The old sparse policy can be selected explicitly with `last`,
prefill interval 8192 and stride 0. All three attention switches still default
to off, so ordinary runs do not enable this policy implicitly.
`non-thinking` includes ordinary answers, prose, generated code and tool calls
outside thinking. Values are validated before CLI model loading and when the
executor is configured. Use a new service/run to change an experimental policy.
The running frozen DeepSWE cohort is not changed by rebuilding this source.

Configuration is passed from `EngineOptions::kvmem_attention` through
`KvMemStoreConfig::attention`, without modifying or reading environment settings.
The CLI rejects the old `QW3_KVMEM_ACTION_*` / `QW3_KVMEM_PREFILL_ATTENTION*`
configuration variables with a migration message: unset them and pass these
options explicitly. This avoids silently running an old script with capture off.
Historical launch scripts and their frozen executables are left intact.

Set all three switches to `off` to disable the experiment.
`--kvmem-action-fraction 0` keeps semantic Query selection while recording
statistics, useful for measuring collection overhead. `--kvmem-action-pressure off`
keeps capacity recovery on its previous policy; prefill collection can still
supply the semantic selector independently.

The interval, layers and quota are engineering starting points, not tuned accuracy
optima. `--kvmem-action-layers` accepts `last`, `spread3`, `all`, or an explicit
comma-separated list of zero-based standard attention layer IDs. `spread3` selects
the first, lower-middle and last standard layers, deduplicated for small models;
Qwen3.8-27B selects layers 3,31,63. GDN and invalid/duplicate IDs are rejected.
All query heads participate in each selected layer. Normalize each head's
distribution, average across heads, then take an equal mean across selected
layers. This keeps the score scale independent of the number of layers.
This is true normalized attention for sampled rows, not full-layer/full-token H2O.

## Collection and acceptance

A parser follows consumed generated token IDs. In `tool-call` scope, only tokens
between `<tool_call>` and `</tool_call>` and outside `<think>`/`</think>` qualify.
In `non-thinking` scope, all generated content outside those thinking boundaries
qualifies, including responses with no tool call. Initial thinking state also
honors the opening marker already supplied by the prompt template. This is a
token-boundary rule, not a semantic classifier of prose that sounds like reasoning.

The four boundary markers themselves, prompt prefill (including user input and
tool results), private Query probes, Query replay and KV recomputation are
excluded in both scopes. Those historical tokens can still receive importance
as attended-to keys; they do not supply the sampled query rows. The first eligible
token and every sixteenth subsequent eligible token are sampled; the counter
advances only through accepted target rows. All eligible content in the same
assistant response shares one counter. This expands the eligible output region,
not the sampling density. Logs include `scope`; the legacy `action_tokens` field
counts accepted eligible content tokens for that scope.

The FlashInfer attention variant taps the logits already produced by the normal
Q*K operation. Unsampled calls use the original kernel instantiation. Sampled
calls store only selected query rows. Selected layers reuse the same logit scratch
sequentially; a small CUDA reduction adds weighted block mass to one shared output
tensor, with one host transfer after accepted rows are known. It never
reads K again or launches a second Q*K computation. Attention output arithmetic
is unchanged.

DFlash may produce and verify rejected rows. Capturing such a row is provisional:
its statistics and parser state are committed only for the accepted prefix.
Checkpoint rollback excludes the rejected suffix; target replay replaces its
provisional statistics. Accepted-prefix sequential replay commits each accepted
row before reusing scratch. Private branches cannot commit statistics.

Both thinking boundary state and the sampling counter follow accepted prefixes:
a rejected think-close cannot enable collection, and a rejected think-open cannot
suppress subsequent accepted prose. Broader scope reuses the existing capture
kernel and adds no Q*K computation. It can sample more rows on prose-heavy tasks
than tool-only scope; the old tool-only timing is not a measured overhead bound
for this mode. This is an engineering extension, not a claim of exact ActKV
reproduction.

Prefill has two independent sampling controls. The `interval` sets how often an
input computation chunk is observed; `stride` sets Query spacing inside an
observed chunk. A stride of 0 preserves last-row-only sampling. For a 1024-token
chunk, stride 32 captures rows 31,63,...,1023, including the final row. Rows are
anchored at the chunk end; partial chunks preserve that final observation. The
example observes every 1024-token chunk, rather than only one chunk every 8192
input tokens. Near-capacity forced observation still overrides a larger interval.
At most 64 Query rows fit in one captured transaction; a denser configuration
fails with an explicit request to increase stride or reduce prefill chunk size.
It never silently truncates the requested samples.

Each Query attends only to causally visible keys in the current GPU window.
Statistics do not newly score evicted keys. Prefill mass and visible exposure are
averaged over sampled Query rows, then the historical mass/exposure are decayed
once per accepted chunk. A key not yet visible to an early Query does not receive
an exposure count for that Query. Additional layers do not count as additional
temporal exposures. Private probes, replay, recompute and rejected rows remain
excluded. The collected rows are a sparse estimate, not a representation of every
token's attention.

The multilayer/Query-stride validation and isolated 27B efficiency measurements
are recorded in `runtime/multilayer_attention_20261011/`. Accuracy must be checked
in a new frozen cohort; the old last-layer/last-row answers remain separate.
Five measured alternating trials after warmup, with other GPU services stopped,
showed median additional target-model time of 0.981% for every-chunk prefill
with `spread3`/stride32, 0.066% for single-token output and 0.500% for an 8-token
target verifier. These measure fixed tokens and windows; prefill has no eviction,
and the probe does not load the DFlash draft. They do not establish a bound on
all serving workloads or an accuracy improvement. The 32-token default is the
densest measured full-input candidate under the preset 2% median overhead limit.

Collection works both before and after the first window selection. Selected
blocks retain their original historical identity. Newly appended context (for
example, the latest Read result) is divided at original block boundaries and also
receives credit before the next selection. Warm prefix reuse invalidates score
indices but retains the sparse window's block IDs, packed bases and lengths
when action tracking is enabled. A short KeepSelected tool turn can therefore
credit the original selected keys without reassembly; a history rewind trims
only the discarded suffix. The real-model regression checks restore + truncate
+ action capture and resident pressure candidates without a new selection.
A final emitted token that is never fed back through the target model has no
measured attention.

## History and selection

At the end of a response containing accepted samples:

`H_new(block) = decay * H_old(block) + mean_sampled_action_mass(block)`.

A response without samples leaves history unchanged. Within a response, mass is
collected against original historical block positions, so reselection/packed slot
changes cannot change its identity. Bounded repositories additionally validate
slot generation. Recycled slots initialize with zero history. Statistics live in
the in-memory block store; new conversations/reset clear their block history.

Sink/recent/mandatory blocks are selected first. Of the remaining budget, 75% is
selected by the existing Query score; 25% is selected from remaining blocks with
positive historical action score. Any unused quota falls back to Query ranking.
There is no mixing of differently scaled raw scores, no additional KV capacity,
and no unconditional pinning of previously used evidence. History from the
current response is used beginning with the next response's selection.

### Action attention during capacity recovery

`--kvmem-action-pressure on` additionally changes the capacity-recovery path.
It requires action capture to be enabled and defaults to `off` for compatibility.
Sink and request-scoped mandatory blocks retain precedence within the same fixed
prefill budget. Every remaining slot first competes by action score, without
consulting Query scores. Only blocks in the currently installed window and its
committed appended suffix are attention candidates; cold history is retrieved
by the ordinary Query/guided-query selector instead.

During a response with accepted samples, the read-only pressure score is
`decay * H_old + accepted_round_mass / accepted_round_samples`. Otherwise it is
`H_old`. The overlay includes only committed accepted target rows and never
commits history or applies decay a second time. Private generation, Query replay,
thinking and rejected speculative rows cannot add evidence. Unfilled slots use
the newest resident blocks, including blocks with no action score. Existing
mandatory ranges protect required new input; this feature does not promise to
pin every complete Read message or add a hidden recent-token quota.

The new work happens only at a pressure selection: host metadata collection and
sorting one resident window. It does not rerun Q*K, copy a full attention matrix,
or add a GPU synchronization. Window installation still has its existing remap
and transfer costs. `[action-pressure]` records activation, candidate count,
attention-selected blocks and remaining fallback slots. Timing and behavioral
results are saved separately from the original capture-only validation.

The 2026-10-10 follow-up uses a 50/50 Query/action quota on normal semantic
selection, guided `both`, an 8,192-token growth trigger and at most eight guided
refreshes per model request. These are experiment settings rather than changed
defaults. With `tool-call` scope, single-turn QA without tool calls has no action
mass, so its action quota falls back to Query; its pressure policy falls back to
resident recency. With `non-thinking`, its generated answer can supply mass. A
cold request has no generated-output history at initial prompt selection. With
prefill collection enabled, input attention can now fill its attention quota,
as described below. Output attention itself cannot affect an earlier selection.

## Optional prefill attention for cold input and semantic selection

Enable `--kvmem-prefill-attention on` with `--kvmem-action-attention on`.
The prefill interval, Query stride and half-life are independently configurable.
Pressure attention is independently optional; enable it to use these scores
for intermediate capacity recovery too.

It samples the first ordinary committed prefill chunk and then at the configured
input-token interval (rounded to chunk ends). Each sampled chunk supplies
Query rows according to `--kvmem-prefill-attention-stride`, at all configured
standard attention layers. Sampling
is additionally forced when GPU headroom is within roughly one more chunk, so
capacity recovery can use newly ingested evidence. `--kvmem-prefill-attention-interval 1`
samples every chunk, at greater cost. Capture uses the existing Q*K logits, supports row indices
beyond 63, and does not shrink the prefill compute batch. The reduced block mass
is committed before the next chunk's capacity check.

Input statistics are separate from generated-output/action statistics. Prompt
input (including user text, tool observations, and teacher-forced transcript text)
supplies these input query rows. This mode does not parse roles or thinking markers
inside a rendered input transcript; the non-thinking parser applies to generated
output only. Private Query branches, Query replay, provisional prefill, KV
recomputation and speculative verification do not supply input statistics. A
rewind into scored input invalidates the prefill score cache; reset clears it.

Each currently visible historical block maintains `(mass, exposure)`. At sample
end `t`, with the previous sample at `t_prev`:

```
d = 2 ** (-(t - t_prev) / half_life_tokens)
mass = d * mass + sum(visible_query_mass) / sampled_query_count
exposure = d * exposure + visible_query_count / sampled_query_count
score = mass / exposure
```

Newly visible blocks start with zero mass/exposure. This prevents identical
attention from scoring higher simply because a block was observed in more chunks;
token-distance decay limits early-context bias but cannot guarantee an unbiased
ranking. Only the latest observed window is retained in this temporary score map,
keyed by original history positions, so it stays bounded and cannot credit a
recycled physical slot. Partial pieces of one block share one exposure.

Capacity recovery still protects sink/mandatory blocks first. If its resident
candidate set has no positive output-attention scores, remaining slots use
prefill scores; zero/unavailable scores fall back to resident recency. Otherwise
the original output-attention policy wins.

At normal semantic Query/guided selection, sink/recent/mandatory blocks retain
priority. The existing `--kvmem-action-fraction` quota chooses its history source
before allocating Query slots:

- If any live historical block has positive committed output attention, use
  output attention. Query overlap does not cause a switch to input scores.
- Otherwise, use positive prefill means for that quota. The first long input
  can therefore use, for example, 50% Query and 50% input-attention slots.
- Missing, zero, or already-selected input candidates leave slots to Query.
  The Query ranking still spans all retrievable history; prefill statistics
  only cover the latest sampled visible window, not all evicted history.

This is a quota union, not an arithmetic blend of raw scores. Selection remains
chronological and within the same budget. Prefill scores never overwrite
`action_score`. A response with accepted output samples in flight suppresses
input fallback; the existing normal selector consumes output history once that
response is committed. Capacity recovery can use the accepted-round overlay
immediately. Input and output scores are never added together. The semantic
fallback does not force another sample, rerun Q*K, or issue a GPU transfer; it
uses the host statistics already collected. Last-row sparse samples need not
include the final prompt token. No guided query is run between prefill chunks.

Only the requested attention rows are materialized, reusing storage across layers,
followed by GPU block reduction and one host transfer/update per accepted chunk.
These operations have nonzero cost; measure
prefill time separately from policy-induced window changes. This setting is an
engineering baseline, not an accuracy-tuned policy.

## Validation

- `qw3-kvmem-attention-cli`: option parsing, invalid policy rejection before model
  loading, independent semantic prefill mode, explicit legacy migration.

- `qw3-kvmem-action-attention`: both scopes, parser acceptance/rollback, thinking exclusion,
  idempotence, Query fallback, quota/sink handling, history decay.
- `qw3-action-attention-capture`: FP16/FP8; decode and causal prefill; multirow,
  reversed physical pages, split KV; attention output bit parity and probability
  mass against an independent Q*K oracle; last-row sampling in 1,024-token chunks.
- The same capture test checks evenly spaced 2/4/32/64 causal Query rows and
  weighted multi-layer accumulation, without changing normal attention outputs.
- `qw3-kvmem-action-model-probe MODEL`: both scopes, real target-model verification/replay,
  exclusion, rejected thinking boundaries, logits parity, selection remapping, and reset;
  optional-prefill isolation, cold semantic quota and output priority,
  cold/sparse pressure, rewind, repeated capacity recovery.
- `qw3-kvmem-action-model-probe MODEL --bench`: alternating fixed-token target
  benchmarks with a 36,864-token window and 16,384-token generation reserve;
  serial and eight-row verifier shapes. Same selected window and tokens in both
  arms; compilation, model loading and prompt prefill are outside the timing.
  This probe uses tool-call scope; the correctness suite covers both scopes.
- `qw3-kvmem-action-model-probe MODEL --bench-prefill`: alternating capture off/on,
  identical 36,864-token inputs and 1,024-token chunks, with a 36,864 selection
  budget and 16,384 generation reserve; 8,192-token sampling interval, no eviction.
  Discard trial zero (warmup).

Measured results are recorded separately under `runtime/action_attention_20261010`.
