# 32K 稀疏单请求：三路线完整 greedy 对照

配置为冻结 `sparse_32k_w256_c1_warm`：Qwen3.8-27B Q8_0、E4M3 FP8 target KV，
32768-token 历史，256-token 工作窗口，1024-token 生成预留，32-token 块，
sink=1/recent=2，CPU mean-K 索引，2 GiB CPU tier。MTP chain=4，DFlash 七草稿。
每请求独立摄入完整冻结历史，不复用其他回答；query 从该历史 checkpoint 恢复。
当前运行时与 stage09 相同，continuous batching 关闭，没有部署或集成。

## 正确性与任务质量

1 次预热加 16 次正式请求，MTP、DFlash 各 **17/17 输出 token ID** 与无投机完全一致。
每条路线的 17 次 warm 生命周期日志均通过：摄入无 prefix HIT，query 恰好复用
32768-token 历史，捕获完整查询，使用 mean-K 且无 fallback，检索索引大于工作集，
执行 warm query replay。典型为 1025 个索引块选择 8 个块。

三条路线的严格 FINAL 标记评分均为 **1/16**，均有 **3 次 256-token 长度截断**。
输出一致证明这组测试中投机没有改变目标行为，不证明 KVMem 检索质量可用。
部分回答拿到标记却无法确认对应服务名；实际 block 选择仍需单独导出核对，
不能仅凭回答文本确定根因，更不能通过扩大预算或删失败样本补过冻结测试。

## 诊断性能

| 路线 | 解码中位数 tok/s | 热请求吞吐 tok/s | 含历史摄入 tok/s | TTFT 中位数 秒 |
| --- | ---: | ---: | ---: | ---: |
| none | 44.73 | 41.81 | 8.04 | 0.224 |
| mtp | 96.06 | 87.60 | 8.86 | 0.230 |
| dflash | 87.95 | 82.36 | 8.82 | 0.230 |

DFlash/MTP 热请求吞吐比为 **0.940222**，本次观察到约 **5.98% 回退**。
这是各一个进程、固定顺序、启用 KVMem PERF_TRACE 的 greedy 诊断，尚没有独立
交替重复和完整资源计量；不能用它替代正式性能资格或 T=1 公共推理任务。
包含历史摄入的速度与仅解码、仅热请求速度分别列出，不能混用。

两条投机路线都输出 2159 个正式 token。MTP 用 688 个验证批次，DFlash 用 656 个，
实际每轮输出分别约 3.14 和 3.29。DFlash 草稿接受数/提议数为 1514/4494，
MTP 为 1469/2723。DFlash 减少验证轮数的收益有限，而每轮验证更贵。
累计诊断 verify 计时分别为 MTP 16.539 秒、DFlash 17.919 秒；draft 为 3.812
和 4.020 秒。这些是原生累计/舍入计数，不是独立 GPU exclusive 时间。

下一步用 3、5 草稿做明确标记的工程实验，保留默认七草稿和全部基线，要求
完整固定请求仍与 none 逐 ID 一致；不按单个样本挑最优宽度，不据此直接修改默认。
同时进行原策略的 selected-block 诊断和资源观测；它们不改变答案或检索策略。

完整原始记录：`benchmark/dflash2/c1_frozen_sparse32k_greedy02/`。
哈希索引：`benchmark/dflash2/q8_c1_sparse32k_greedy_audit01.json`。
此报告封存本次完整 greedy 结果；此前的 progress 文档保留为当时快照。
目标未完成，多并发继续后置。
