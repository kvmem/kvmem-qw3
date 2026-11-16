# Query relevance history

Query history is an opt-in host-side history for the existing Mean-K retrieval
signal. It gives repeatedly relevant blocks a decaying preference across input
requests. It is separate from sampled output attention, prefill mass/exposure,
and the window-local profile. It does not implement GQA max or change attention
QK kernels.

## Enable

Add these options to an existing online KVMem invocation:

```sh
--kvmem --kvmem-query-conditioned \
--kvmem-method retrieval --kvmem-retrieval-method mean-k \
--kvmem-select-policy topk --kvmem-semantic-expansion none \
--no-continuous-batching \
--kvmem-query-history on --kvmem-query-history-decay 0.5
```

History defaults to `off`; decay defaults to `0.5` and accepts finite values in
`[0,1]`. Output attention can remain off, or retain its existing `quota` or
`score` combination. A 100% attention `score` policy rejects Query history
because that policy has no Query contribution. The general retrieval/profile
`--kvmem-select-policy quota` is a different policy and is not supported here.
Archive attachment, continuous batching, semantic-chunk input, inline refresh,
transcript replay, initial/boundary guided Query, and API-session thinking
compaction are not supported. These combinations fail explicitly. Decode-time
guided refresh remains supported after the input observation is committed.

## Scoring and selection

For the valid historical blocks in a complete Mean-K score frame, normalize the
nonnegative raw scores to sum to one:

```text
q_t(b) = raw_query_score(b) / sum(raw_query_score)
candidate_t(b) = decay * parent_history(b) + q_t(b)
```

Normalization happens before accumulating history. Mean-K already averages
heads/layers but sums query tokens, so storing unnormalized scores would let a
long question contribute more total historical weight than a short question.
The valid domain excludes unscored/recycled repository slots and blocks beyond
the accepted input. Partially overlapping blocks that extend past that input
boundary do not contribute. Sink/recent masking follows the existing scorer.
There is no top-fraction gate. Missing current mass contributes zero; old
history decays when there is a valid positive observation. An input with no
Query, or a complete all-zero observation, keeps its parent's scores unchanged.
With a complete all-zero score frame, that retained history still ranks eligible
blocks; an incomplete frame never publishes a history-based selection.

The original `retrieval_score` remains available for diagnostics and other
existing consumers. A separate effective Query vector feeds ordinary TopK or
the Query side of attention quota/score fusion. Existing sink, recent and
mandatory-block precedence and the selection budget are unchanged. Output
attention and prefill-source precedence are unchanged. Capacity recovery keeps
its existing resident-only policy.

Below the selection budget, Query capture and scoring still run when history
is enabled, so a subsequent over-budget request can inherit evidence. This does
not force window selection while all blocks fit. It does add a lightweight
Mean-K scoring pass where ordinary serving would not need one. No new full
attention QK pass is introduced; score normalization, history maps and input
identity bookkeeping also have nonzero host time/memory costs.

## Input transactions and branches

One outer logical input request owns one history transaction:

1. Verify the incoming token/media identity against the prior accepted input.
   Choose the latest recorded boundary inside the verified prefix and strictly
   before the new input end.
2. Each complete scoring pass computes a new preview from that immutable parent.
   Repeated scoring, masking diagnostics and query replay replace the preview;
   they never accumulate on top of the previous preview.
3. Commit once after final input processing/replay and before decode, including
   a request with `max_tokens=0`. Failed input transactions abort their preview.

Incomplete/provisional/private/partial-layer scores cannot teach history.
Generating a fresh response for identical input therefore replaces that
input's observation from its parent. Editing or rewinding selects the matching
earlier parent and discards the abandoned future from the current branch.
Named cache snapshots retain independent immutable branches. Physical KV
rebuild positions do not determine logical parentage. CPU/SSD offload does not
erase Query history; physical repository generations are checked when mapping
scores back to a current selection.

The backend currently uses the complete input identity conservatively: it does
not guess an earlier message boundary from the Query span or strip an assistant
header without proof. A template rewrite that no longer matches that boundary
can therefore cause an earlier parent or no parent to be used.

After input commit, decode-time guided refresh retains its existing current
Query ranking. It does not add another Query-history observation or decay the
input history again. Accepted output attention continues its own lifecycle.
This is an input-history feature, not per-output-token Query history.

History maps and boundary nodes are retained in process memory with their
conversation/cache lineage. Snapshots share immutable nodes; no full history
is copied into per-token speculative checkpoints. Metadata grows with observed
blocks/rounds; this feature does not provide restart persistence.

## Validation

`QW3_KVMEM_TRACE=1` emits `[query-history]` with the committed logical round,
input boundary, whether a positive Query was observed, and historical block
count. Repeated preview does not increase that committed round count.

Host coverage includes normalization, repeated preview/commit, regeneration,
editing, empty observations, snapshot isolation, identity checks, selector
fusion/quota compatibility, CLI rejection and executor lifecycle tests with a
recording device backend. Run:

```sh
ctest --test-dir build-query-history-host --output-on-failure \
  -R 'qw3-kvmem-(query-history|action-attention|attention-cli|store|page-table|repository|prefix-history|prefix-reuse-policy)'
```

These host tests do not establish GPU numerical parity, overhead or answer
quality. Quality evaluation needs matched continuous multi-turn conversations,
including short follow-up requests and topic changes. Independent cold requests
start without a parent history and cannot establish the benefit of this feature.

The 2026-10-15 BFCL long-context pilot ran eight paired closed-loop trading
conversations with 44 distinct user queries per arm, a 32,768-token budget,
50% action-attention quota, and Query-history decay 0.5. Both arms passed 4/8
full conversations and 26/44 diagnostic turn checks. History committed across
all 101 enabled-arm input requests; all 21 matched-input selection snapshots
changed their selected blocks. Explicit tool errors decreased from 2 to 0 and
one retry request was avoided, without improving conversation pass rate.
Two natural-language responses per arm hit the shared output limit; official
BFCL grading checks tool state/results rather than answer completeness. All 16
scores were independently reproduced from saved calls. This is a small,
single-domain observation, not evidence of a general quality improvement.
See the [report and recorded protocol](../artifacts/bfcl_query_history_ab_32k_20261015/report.md).
