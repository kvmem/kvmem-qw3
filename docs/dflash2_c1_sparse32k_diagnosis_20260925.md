# 单请求 KVMem：32K 性能回退与事实完整性诊断

本阶段没有改变模型运行时，沿用 Q8_0＋E4M3 FP8 target KV 和官方草稿。
单请求、多并发后置，目标仍 active。当前证据既没有满足任务质量，也没有满足
性能目标；不集成或部署。完整表格见 `dflash2_c1_sparse32k_greedy_20260925.md`。

## 输出一致，但性能回退

冻结的 32K 历史、256-token 工作窗口、CPU mean-K 索引场景，MTP chain=4 与
DFlash 七草稿各 17/17 响应逐 token 等于无投机基线。三路线都仅答对 1/16，
都有 3 次长度截断，所有失败保留。生命周期检查确认实际稀疏检索与 warm replay。

DFlash/MTP 热请求吞吐比为 0.940222，观察到约 5.98% 回退。
2159 个相同输出 token，DFlash 656 轮、MTP 688 轮：每轮有效输出约 3.29/3.14，
而 DFlash 累计 verify 为 17.919 秒，MTP 为 16.539 秒。低接受收益和较贵的验证
解释了这次计时表现，但仍需重复/干预实验才能确定最佳优化策略。

## 实际选中块与事实跨度

新增只读 block-ranking/selection dump，保持原始策略，没有 oracle。再次执行
全部 17 个 none 请求，输出 IDs 和响应字节与原基线完全一致。每个 selected
block 的 token 数组逐一与冻结 input IDs 的原坐标切片核对。

16 个正式请求需要的 20 条事实中，完整选中 **0** 条，部分选中 **12** 条，
完全未选中 **8** 条。以 `retrieval-h32768-r01` 为例，事实跨两个块：
`[21299,21312)` 未选中，其原文是 `Test 01, service gamma, revision 1: the`；
`[21312,21336)` 已选中，其原文是 `current marker is NOTE-81BD5BFC. This revision supersedes older records for this service.`。
回答确实识别了该标记，却称无法关联 service gamma。这是可验证的事实片段丢失，
不是推测中的 tokenizer 错误。它支持对这类错误的解释，不能独自证明全部错误
的因果关系；DeltaNet 状态还可能保留未选中的历史信息。

本轮只诊断，没有扩大窗口、改变检索规则、注入正确事实位置或修改评分。

## 资源观测与局限

新增 benchmark-only NVML/proc 采样，并使用 CLOCK_MONOTONIC 与请求窗口对齐。
采集 30437 个样本，请求间隔 10ms；实际最大间隔
46.994ms，共 34 个间隔超过 20ms。
与正式 query 窗口相交的最大间隔 21.093ms，历史摄入
窗口相交最大 30.377ms。所有原始时间保留，不假称每次
观测都达到 20ms 以内。

观察到整卡 used-memory 峰值 32039043072 字节，
worker RSS 峰值 31767584768 字节。前者含 driver，
后者含映射权重；都不是算法独占内存。尚没有 allocator high-water 和完整
index/pinned/snapshot/draft-cache 细分，故正式资源/性能资格仍未完成。
新增观察器输出不变不等于性能开销为零；所有这些时间只是诊断结果。

## 后续优化实验

已建立 3/5 草稿的顺序工程实验（`run_c1_sparse_widths.py`），沿用全部冻结样本，
不按单个样本挑宽度，每个宽度要求与 none 逐 ID 一致。默认七草稿保持不变。
该实验在本报告封存时仍在运行，其结果不计入本阶段完成证据。采样、128K 有界
C1、全量公开任务与独立重复仍需继续；公开参考资格也未通过。

证据索引：`benchmark/dflash2/q8_c1_optimization_stage10.json`。
