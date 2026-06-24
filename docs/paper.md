# Attention Distribution Divergence Measurement

For each generated token during decoding, we compute a block-level attention
distribution over the visible KV cache. The historical KV cache is partitioned
into fixed-size KV blocks, where each block contains 128 consecutive tokens.
For a given generated token, attention is first computed at the token level
using the actual query vector of the current token and the cached key vectors in
the KV cache. Specifically, for each attention head and each transformer layer,
we recompute the attention scores between the current query and all visible
cached keys, apply softmax normalization over the visible sequence, and obtain
the token-level attention weights. We then aggregate these token-level attention
weights into KV blocks: the attention mass of a block is defined as the sum of
the attention weights assigned to all tokens inside that block. Thus, for a KV
block \(b\), its attention mass is

```text
A_{t,l,h}(b) = sum_{j in b} softmax(q_{t,l,h} k_{j,l,h}^T / sqrt(d)),
```

where \(t\) is the generated token index, \(l\) is the layer index, \(h\) is
the attention head index, \(q_{t,l,h}\) is the query vector of the current
generated token, \(k_{j,l,h}\) is the cached key vector of historical token
\(j\), and \(d\) is the head dimension. This aggregation produces, for each
generated token, a distribution over KV blocks rather than over individual
tokens. For the KL-divergence stability analysis, we exclude the first sink
block from the distribution and renormalize the remaining block masses so that
they sum to one. For the block-selection coverage analysis described later, the
sink block is instead treated as an always-retained block, matching the actual
KV-memory policy.

After obtaining the block-level attention mass for each generated token, we
average the distribution across attention heads and transformer layers to obtain
a single block-level attention distribution for that token. Let this averaged
distribution be denoted as \(A_t(b)\), where \(b\) indexes KV blocks:

```text
A_t(b) = Normalize((1 / (L H)) * sum_{l=1}^{L} sum_{h=1}^{H} A_{t,l,h}(b)),
```

where \(L\) is the number of layers and \(H\) is the number of attention heads.
This distribution describes how the model, on average across layers and heads,
allocates its attention mass over historical KV blocks when generating token
\(t\). Importantly, this block-level distribution is not estimated using a
block representation such as an averaged key. It is computed by recomputing the
token-level softmax attention weights from the actual query and cached keys, and
then summing the resulting token-level weights within each KV block.

We then compute a sliding-window attention distribution along the decoding
trajectory. Each window contains 128 consecutive generated tokens, and adjacent
windows are shifted by a stride of 32 generated tokens. For a sliding window
\(W_i\), we average the token-level block distributions \(A_t(b)\) over all
generated tokens inside the window:

```text
P_i(b) = Normalize((1 / |W_i|) * sum_{t in W_i} A_t(b)).
```

The resulting \(P_i(b)\) is the average block-level attention distribution of
the model during the corresponding decoding window. Intuitively, it indicates
which regions of the historical KV cache the model attends to while generating
the tokens in that window.

To measure how the model's attention focus changes over time, we compute the KL
divergence between the block-level attention distributions of adjacent sliding
windows:

```text
D_i = KL(P_i || P_{i+1})
    = sum_b P_i(b) log_2(P_i(b) / P_{i+1}(b)).
```

The KL divergence is computed over the union of KV blocks appearing in the two
adjacent distributions. In implementation, a small smoothing constant
\(\epsilon = 10^{-9}\) is added to every block probability before normalization
to avoid undefined values when a block receives zero mass in one of the two
windows. Because we use \(\log_2\), the divergence is measured in bits.

Each point in the plotted curve corresponds to one value \(D_i\), i.e., the
divergence between two adjacent decoding windows. Low values indicate that the
model attends to similar KV blocks across the two windows, while high values
indicate that the model's attention has shifted to different regions of the KV
cache. The red vertical dashed lines in the figure mark agent step boundaries.
These boundaries are used only as annotations and are not used in the
computation of \(D_i\). Therefore, the KL curve itself is computed purely from
the sequence of adjacent decoding windows. If the curve remains low within a
step but spikes near a step boundary, this indicates that the model's attended
context is stable during a step and changes significantly when the agent
transitions to a new step.

One possible confounding factor is that the number of visible KV blocks can grow
during decoding, since newly generated tokens are appended to the KV cache. To
control for this effect, we also compute a fixed-context variant in which only
the prefill context blocks available at the beginning of each step are included,
while KV blocks corresponding to newly generated tokens are excluded. Under this
fixed-context setting, the same sliding-window procedure and KL divergence
computation are applied. The observation that step-boundary spikes remain
significant under this fixed-context variant suggests that the divergence peaks
are not primarily caused by the growth of visible KV blocks during decoding, but
instead reflect a genuine shift in the model's attention distribution over the
historical context.

# Attention Coverage Measurement

In addition to measuring the stability of the attention distribution, we also
measure how concentrated the distribution is. This analysis is intended to
evaluate whether a step-level KV selection policy can keep only a small number
of historical KV blocks while still covering most of the model's attention mass.
The resulting plot is similar in spirit to a cumulative coverage or ROC-style
curve: the x-axis represents the amount of selected KV blocks, and the y-axis
represents the fraction of total attention mass covered by those selected
blocks.

We use the same window-level block attention distribution \(P_i(b)\) defined
above. For each decoding window \(W_i\), \(P_i(b)\) represents the average
attention mass assigned to KV block \(b\) during that window. To compute the
coverage curve, we sort all candidate KV blocks by their attention mass in
descending order:

```text
P_i(b_1) >= P_i(b_2) >= ... >= P_i(b_N),
```

where \(b_1, b_2, ..., b_N\) are the KV blocks sorted by attention mass. The
cumulative coverage of the top \(k\) selected blocks is then defined as

```text
C_i(k) = sum_{r=1}^{k} P_i(b_r).
```

This value measures how much of the model's attention mass can be covered if we
keep only the highest-weight KV blocks in that decoding window. For example, if
\(C_i(8) = 0.65\), then the top 8 selected KV blocks cover 65% of the total
attention mass for window \(W_i\).

Because the actual KV-memory policy always retains the attention sink, our
primary coverage analysis treats the first KV block as an always-kept sink
block. Therefore, the reported top-\(k\) coverage is computed as follows: first
include the sink block, then select the highest-weight non-sink blocks until the
total number of selected blocks reaches \(k\). Formally, let \(s\) denote the
sink block and let \(R_i\) be the non-sink KV blocks sorted by descending
attention mass. The sink-kept top-\(k\) coverage is

```text
C_i^{sink}(k) = P_i(s) + sum_{r=1}^{k-1} P_i(R_i[r]).
```

Thus, `top 1` corresponds to keeping only the sink block, `top 2` corresponds
to keeping the sink plus the highest-weight non-sink block, and `top 8`
corresponds to keeping the sink plus the top 7 non-sink blocks. We also report
fractional coverage values such as top 10% and top 20%, which select the
corresponding fraction of KV blocks after including the always-kept sink block.

As in the fixed-context KL analysis, the primary coverage experiment uses only
the prefill context blocks available at the beginning of each step and excludes
KV blocks corresponding to newly generated tokens. This avoids conflating
coverage with the growth of the KV cache during decoding and directly measures
how concentrated the model's attention is over the historical context available
for step-level selection.

Empirically, after keeping the sink block, the attention distribution remains
substantially concentrated. Across the evaluated SWE-bench agent traces, the
sink block alone covers roughly 6% to 12% of the measured attention mass. When
we keep the sink plus a small number of additional high-weight historical
blocks, the cumulative coverage increases quickly: selecting a total of 8
blocks typically covers about 60% to 75% of the attention mass, selecting a
total of 16 blocks covers about 71% to 82%, and selecting the top 20% of blocks
usually covers about 77% to 85%. These results suggest that a step-level KV
selection policy can retain the sink block and a relatively small set of
additional high-attention historical blocks while preserving most of the
attention mass used by the model.
