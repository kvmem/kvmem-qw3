# Experimental DeltaNet State-Edit Retrieval for KVMem

## Status

**Status: archived experimental method; not recommended for use.**

As of 2026-07-12, the implementation remains in the source tree for
reproducibility and possible future research, but its CLI entry points are
disabled. The supported KVMem retrieval methods should be preferred.

This decision is based on both accuracy and systems results:

- DeltaNet retrieval with cumulative decay: **41/102 = 40.20%**.
- DeltaNet retrieval without cumulative decay: **44/102 = 43.14%**.
- Matched 32K mean-k result: **79/102 = 77.45%**.
- The default 17-layer DeltaNet summary occupied approximately **31.88 GiB** and
  increased mean TTFT from roughly 35.8 seconds for mean-k to 46.5 seconds.

The method is model-specific: it requires recurrent DeltaNet/Gated-DeltaNet
layers and cannot be used by models containing only normal attention layers.
That restriction conflicts with KVMem's goal of providing a broadly applicable
KV-memory mechanism.

## Motivation and Method

For each historical token block `j`, selected DeltaNet layers and heads capture
the recurrent state at the end of the block, `S_j`, together with the block's
cumulative scalar decay `a_j`. The block is represented by its net state edit:

\[
E_j = S_j - a_j S_{j-1}.
\]

For a current L2-normalized DeltaNet query `q_t`, the experimental scorer reads
the edit and uses the value-vector norm:

\[
r_{j,t} = d_j \left\|E_j^T q_t\right\|_2,
\qquad
d_j = \exp(G_M-G_j).
\]

Scores are aggregated with TopKMean over query tokens and value heads, followed
by per-layer RMS normalization and equal-weight layer averaging. An optional
mode disables the post-block decay by setting `d_j=1`.

The source implementation includes:

- recurrent-state boundary snapshots and per-block decay accumulation;
- capture of the actual convolution-processed, L2-normalized DeltaNet query;
- a CUDA block scorer implementing `||E_j^T q||`;
- query/head/layer aggregation;
- memory-budget-capped layer selection, including even and late-layer policies;
- score-dump and answer-session block-recall diagnostics.

The original design discussion remains in
[`kvmem_deltanet_retrieval_design.md`](kvmem_deltanet_retrieval_design.md);
this document
records the implementation outcome and current product decision.

## Experimental Findings

### Decay is too aggressive over long histories

On prompts of approximately 109K--115K tokens, full cumulative scalar decay
strongly biased selection toward recent blocks. Accuracy by answer position was:

| Answer position | decay on | decay off | mean-k |
| --- | ---: | ---: | ---: |
| Oldest 25% | 1/17 | 7/17 | 13/17 |
| 25%--50% | 5/28 | 9/28 | 20/28 |
| 50%--75% | 17/34 | 16/34 | 28/34 |
| Most recent 25% | 18/23 | 12/23 | 18/23 |

Disabling decay recovered some early-history questions but harmed recent-history
ranking, producing only a 2.94 percentage-point net improvement. A useful decay
would need temperature, clipping, or learned calibration rather than a binary
on/off choice.

### The retrieval target is misaligned

KVMem selects token-wise KV blocks that are later consumed by normal attention.
Mean-k scores those blocks in the same attention query/key space. DeltaNet
retrieval instead measures the magnitude of a recurrent-state edit read by a
DeltaNet query. A large recurrent contribution does not imply that restoring the
corresponding normal-attention KV block will help the current generation.

The L2 norm also discards direction. New writes, corrections, erasures, and
unrelated high-energy edits can all receive large positive scores. The scorer
does not apply the recurrent layer's downstream normalization, gate, or output
projection when estimating block utility.

### Layer and head aggregation is noisy

The original configuration selected 17 of 48 recurrent layers and gave every
layer equal weight after RMS normalization. This can amplify weak layers by
normalizing them to the same scale as useful layers. TopKMean over 48 value heads
similarly favors per-block extreme values and head energy.

Small diagnostics on 12 stratified samples compared the last two recurrent
layers (61 and 62) with only the last layer (62):

| Configuration | Summary memory | Mean TTFT | Gold-session MRR |
| --- | ---: | ---: | ---: |
| late layers 61+62 | ~3.75 GiB | 34.26 s | 0.1246 |
| late layer 62 | ~1.88 GiB | 33.49 s | 0.1437 |

The selected-block sets had a mean Jaccard similarity of only 0.708, with 30--56
of 128 blocks changing after adding one layer. This indicates substantial
layer-level ranking instability.

### Session recall does not imply answer-block precision

In the 12-sample diagnostic, every configuration selected at least one block
from every gold answer session. However, answer sessions often covered 7--65
blocks, and only a small fraction of their blocks survived. The method could
identify the general session region without reliably selecting the local block
containing the answer. This is consistent with the low end-to-end accuracy.

### Memory changes the prefill execution path

Allocating full fp32 state matrices consumed enough GPU memory to activate the
bounded KVMem pool during prefill. Even the single-layer 1.88 GiB summary still
triggered three mid-prefill reselections in the measured configuration, whereas
the matched mean-k path performed one main selection. The DeltaNet query and
later state edits were therefore produced after normal-attention context had
already been compressed, making the comparison more than a scorer-only change.

## Known Implementation Limitations

- Applicable only to hybrid models containing DeltaNet-style recurrent layers.
- High memory cost: one fp32 `d_v x d_k` matrix per block, head, and layer.
- The full-query CUDA scorer uses dynamic shared memory proportional to query
  length; long query spans require redesign rather than the current launcher.
- Prefix-cache/session continuation does not preserve complete historical
  DeltaNet boundary summaries.
- Clean-query capture preserves normal-attention queries but not the DeltaNet
  query buffer.
- Query-containing tail blocks participate in scoring and layer normalization.
- Configured `topk_q` values above eight are internally capped by the CUDA
  scorer.

These issues are another reason the implementation should not be re-enabled as
a supported CLI method without further work.

## Possible Future Research

If this direction is revisited, the most promising changes are:

1. Measure exact answer-local block Recall@K before end-to-end generation.
2. Use one or a few validated late recurrent layers instead of equal weighting
   across many evenly spaced layers.
3. Normalize per head across blocks and control edit-energy bias.
4. Replace pure contribution norm with a direction-aware contribution aligned
   to the full recurrent output, ideally after the actual gate/norm/output path.
5. Use softened/clipped decay rather than full cumulative decay.
6. Compress summaries with fp16/int8, low-rank state edits, or random projections
   so enabling the scorer does not change the prefill memory path.
7. Implement correct summary preservation for prefix-cache continuation.

Until those questions are resolved, DeltaNet state-edit retrieval should be
treated as an archived research prototype rather than a KVMem feature.
