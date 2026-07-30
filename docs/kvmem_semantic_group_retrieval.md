# KVMem semantic-group retrieval

This design adds two independent, opt-in controls on top of
`sub-block-mean-k`:

1. score fine-grained slices and materialize a complete semantic group
   (`round` or `message`);
2. replace group MaxSim with length-normalized attention mass.

They can be enabled separately or together. The default configuration remains
ordinary block retrieval.

## CLI

Complete-message expansion with 32-token scoring:

```text
--kvmem-retrieval-method sub-block-mean-k
--kvmem-block-tokens 512
--kvmem-subblocks 16
--kvmem-semantic-expansion message
```

`message` mode requires:

```text
kvmem-block-tokens / kvmem-subblocks = 32
```

Length-normalized mass can be added to the same run:

```text
--kvmem-group-score-reduce length-normalized-mass
--kvmem-group-length-alpha 0.5
```

Round expansion uses the same scorer and selector:

```text
--kvmem-semantic-expansion round
--kvmem-group-score-reduce length-normalized-mass
--kvmem-group-length-alpha 0.5
```

`--kvmem-round-retrieval` remains a compatibility alias for
`--kvmem-semantic-expansion round`.

The two reducers are:

- `max`: the previous semantic-group MaxSim behavior;
- `length-normalized-mass`: sum globally normalized attention probabilities
  in a group, then normalize for group length.

When semantic expansion is active, this group reducer supersedes
`--kvmem-subblock-reduce`; the latter continues to control ordinary
non-grouped block retrieval.

`alpha` must be in `[0,1]`:

- `0`: raw group mass, which favors longer groups;
- `0.5`: square-root length normalization and the recommended initial value;
- `1`: mean mass density, which most strongly penalizes long groups.

## Scoring

Let `s` be a fine-grained scoring slice, normally 32 tokens. For every query
layer, query token, and attention head, KVMem first computes the same global
softmax over all retrievable slices as ordinary sub-block mean-K:

```text
p(s | q,l,h) = softmax_s(q[l,h] · meanK[s])
```

For semantic group `G`, raw mass is:

```text
Mass(G) = sum_(q,l,h) sum_(s in G) p(s | q,l,h)
```

The final length-normalized score is:

```text
Score(G) = Mass(G) / |G|^alpha
```

Here `|G|` is the number of scoring slices overlapped by the group, not its
UTF-8 byte length or number of physical transfer blocks. This keeps the score
definition stable across message contents and tokenization.

The CUDA implementation supports both the original fused path and the scalable
tiled path used when the total number of scoring slices exceeds the fused
shared-memory limit. Both paths compute the same global softmax and group
reduction.

## Semantic boundaries

In `message` mode:

- ordinary Chat Completions requests derive historical message boundaries from
  the server's fully rendered chat template;
- the final user query is excluded because it is replayed/pinned separately;
- user, assistant, and tool-result messages are independent groups;
- the initial system/developer prefix is not scored as a candidate group and
  remains governed by the configured sink retention;
- flattened AgentLongBench prompts supply original-message byte spans through
  `kvmem_retrieval_group_spans`.

The AgentLongBench runner enables this with:

```text
--kvmem-message-expansion
```

In `round` mode, the caller supplies round spans. The existing runner flag is:

```text
--kvmem-round-only
```

Dataset/session parsing remains outside KVMem. The store only receives ordered
half-open token spans.

## Budget behavior

Sink, recent, replay-pinned, and oracle-mandatory blocks are charged first.
Candidate semantic groups are then sorted by semantic score.

For each candidate:

1. compute the union of physical blocks overlapped by the complete group;
2. deduct blocks that are already selected;
3. admit the group only if all remaining blocks fit;
4. otherwise skip it without partially materializing it.

Adjacent groups that share a physical boundary block are charged for that block
only once. The final selected physical blocks are restored in chronological
order.

Consequently, selection never exceeds `kvmem-select-budget`. A single message
larger than the remaining budget is skipped rather than truncated; this is
intentional because partial admission would violate semantic completeness.

## Trace records

With `QW3_KVMEM_TRACE=1`, the relevant records are:

```text
[bs-semantic-groups] configured groups=...
[bs-mean-softmax] ... groups=... group_reduce=... alpha=...
[bs-semantic-select] mode=... reduce=... alpha=...
```

The final record includes selected group/block counts, groups skipped for
budget, and unused block slots.
