# KVMem：基于 DeltaNet 的历史块召回方案

## 1. 核心思路

假设历史已经被划分成 \(M\) 个 KV 块。每个历史块额外保存一个 DeltaNet 状态摘要。

新步骤到来后，使用当前步骤在 DeltaNet 层产生的 query 去读取每个历史块的状态增量：

\[
\boxed{
\text{历史块状态增量}
\xrightarrow{\text{当前 Query 读取}}
\text{块贡献向量}
\xrightarrow{\text{聚合}}
\text{块召回分数}
}
\]

最终根据分数选择对应的历史 KV 块。

DeltaNet 使用固定大小的递归状态存储历史信息，并通过 delta rule 对状态进行写入和修正，因此这里检索的不是原始 token key，而是每个历史块对递归状态产生的净编辑。([arxiv.org](https://arxiv.org/abs/2412.06464?utm_source=openai))

---

## 2. 每个历史块保存的内容

对于历史块 \(j\)，在每个选定的 DeltaNet 层 \(\ell\) 和 head \(h\) 上保存：

\[
E_j^{(\ell,h)}
=
S_j^{(\ell,h)}
-
a_j^{(\ell,h)}S_{j-1}^{(\ell,h)}
\]

其中：

- \(S_{j-1}\)：进入历史块前的状态；
- \(S_j\)：历史块结束后的状态；
- \(a_j\)：历史块内部的累计衰减；
- \(E_j\)：历史块对状态产生的净编辑。

同时保存历史块结束位置的累计 log-decay：

\[
G_j^{(\ell,h)}
=
\sum_{u=1}^{j}\log a_u^{(\ell,h)}
\]

因此，每个历史 KV 块对应的 DeltaNet 检索摘要为：

\[
\boxed{
\mathcal D_j=
\left\{
E_j^{(\ell,h)},
G_j^{(\ell,h)}
\right\}_{\ell,h}
}
\]

这里不建议直接使用：

\[
S_j-S_{j-1}
\]

因为其中混入了旧状态的自然衰减。

---

## 3. 获取当前 Query

新步骤到来后，获取当前输入在选定 DeltaNet 层中的 query：

\[
Q^{(\ell,h)}
=
\left[
q_1^{(\ell,h)},
q_2^{(\ell,h)},
\ldots,
q_T^{(\ell,h)}
\right]
\]

其中 \(T\) 是用于召回的 query token 数量。

必须使用 DeltaNet 实际读取 recurrent state 时使用的 query，包括相同的：

- Query projection；
- Head 切分；
- L2 normalization；
- 其他模型内部预处理。

Qwen3.6 使用 Gated DeltaNet 和完整注意力交错的混合结构，因此这里只提取其中 DeltaNet 层的 query。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-27B?utm_source=openai))

---

## 4. 修正历史块的后续衰减

第 \(j\) 个历史块结束后，还经过了后续历史块。它的状态编辑可能已经发生衰减。

设最后一个历史块为 \(M\)，则近似后续衰减系数为：

\[
d_j^{(\ell,h)}
=
\exp
\left(
G_M^{(\ell,h)}
-
G_j^{(\ell,h)}
\right)
\]

得到该块在当前时刻的近似剩余状态增量：

\[
\boxed{
\widetilde E_j^{(\ell,h)}
=
d_j^{(\ell,h)}
E_j^{(\ell,h)}
}
\]

如果暂时不想处理后续衰减，第一版也可以直接令：

\[
d_j^{(\ell,h)}=1
\]

但是建议同时实验“有衰减”和“无衰减”两个版本，因为 scalar decay 只考虑了全局遗忘，没有精确描述后续 token 对该块进行的方向性擦除。

---

## 5. 使用 Query 读取历史块

对于当前 query token \(q_t^{(\ell,h)}\)，计算它从历史块状态增量中读取出的 value 向量：

\[
\boxed{
c_{j,t}^{(\ell,h)}
=
\left(
\widetilde E_j^{(\ell,h)}
\right)^\top
q_t^{(\ell,h)}
}
\]

如果推理框架中的状态布局是 \(d_v\times d_k\)，则写成：

\[
c_{j,t}^{(\ell,h)}
=
\widetilde E_j^{(\ell,h)}
q_t^{(\ell,h)}
\]

二者本质相同，只需要与实际 DeltaNet 状态的矩阵方向保持一致。

\(c_{j,t}^{(\ell,h)}\) 表示：

> 历史块 \(j\) 对状态产生的编辑中，有多少内容能够被当前 query token 读取出来。

---

## 6. 单个 Query 的块分数

对贡献向量取 L2 范数：

\[
\boxed{
r_{j,t}^{(\ell,h)}
=
\left\|
c_{j,t}^{(\ell,h)}
\right\|_2
}
\]

展开后为：

\[
\boxed{
r_{j,t}^{(\ell,h)}
=
d_j^{(\ell,h)}
\left\|
\left(E_j^{(\ell,h)}\right)^\top
q_t^{(\ell,h)}
\right\|_2
}
\]

使用范数而不是带符号的标量，是因为历史块可能产生：

- 正向写入；
- 对旧信息的修正；
- 负向更新；
- 擦除操作。

这些变化都可能对当前推理重要，不应该只保留“正向贡献”。

---

## 7. 聚合多个 Query Token

对于一个历史块，当前步骤的不同 query token 会得到不同分数：

\[
r_{j,1}^{(\ell,h)},
\ldots,
r_{j,T}^{(\ell,h)}
\]

建议使用 top-\(k_q\) 平均：

\[
\boxed{
r_j^{(\ell,h)}
=
\operatorname{TopKMean}_{t}
\left(
r_{j,t}^{(\ell,h)}
\right)
}
\]

例如只平均分数最高的 4 个 query token。

这样能够避免：

- 对所有 query 求平均造成相关信号被稀释；
- 只取最大值容易受到单个异常 token 干扰。

如果当前输入非常短，可以直接使用：

\[
r_j^{(\ell,h)}
=
\max_t r_{j,t}^{(\ell,h)}
\]

---

## 8. 聚合多个 Head

每个 DeltaNet head 得到一个块分数：

\[
r_j^{(\ell,1)},
\ldots,
r_j^{(\ell,H)}
\]

建议同样使用 top-\(k_h\) 平均：

\[
\boxed{
r_j^{(\ell)}
=
\operatorname{TopKMean}_{h}
\left(
r_j^{(\ell,h)}
\right)
}
\]

这样可以让少量真正相关的 head 决定召回结果，而不是被大量无关 head 稀释。

第一版可以取：

- 最高 4 个 head；
- 或最高 \(10\%\sim25\%\) 的 head。

---

## 9. 聚合多个 DeltaNet 层

不同层的分数尺度可能差异较大。因此先在每个层内部，针对所有历史块进行归一化。

例如使用 RMS 归一化：

\[
\widehat r_j^{(\ell)}
=
\frac{
r_j^{(\ell)}
}{
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
\left(r_i^{(\ell)}\right)^2
}
+\epsilon
}
\]

然后跨层聚合：

\[
\boxed{
s_j
=
\sum_{\ell\in\mathcal L}
w_\ell
\widehat r_j^{(\ell)}
}
\]

第一版可以使用相同层权重：

\[
w_\ell=\frac{1}{|\mathcal L|}
\]

最终得到每个历史 KV 块的 DeltaNet 召回分数：

\[
s_1,s_2,\ldots,s_M
\]

按照分数排序并选择对应的历史 KV 块即可。

---

## 10. 完整召回公式

将上述过程组合起来，核心公式为：

\[
\boxed{
s_j
=
\sum_{\ell\in\mathcal L}
w_\ell
\operatorname{Normalize}_{j}
\left[
\operatorname{TopKMean}_{h}
\left(
\operatorname{TopKMean}_{t}
\left[
\left\|
\exp(G_M^{(\ell,h)}-G_j^{(\ell,h)})
\left(E_j^{(\ell,h)}\right)^\top
q_t^{(\ell,h)}
\right\|_2
\right]
\right)
\right]
}
\]

其中：

- \(E_j\)：历史块产生的状态净编辑；
- \(G_M-G_j\)：该编辑经过后续历史后的近似衰减；
- \(q_t\)：当前步骤的 DeltaNet query；
- 向量范数：该历史块可被当前 query 读取的贡献大小；
- Query、head、layer 聚合：得到最终块分数。

---
