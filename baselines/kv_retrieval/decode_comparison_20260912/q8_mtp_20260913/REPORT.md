# Qwen3.8-27B Q8_0 / FP8 KV / MTP 吞吐复测

独占同一 GPU，绑定 GPU 所在 NUMA 的 CPU；同一真实 DeepSWE 历史前缀，固定 128 个输出 token，贪心生成、忽略 EOS。每项三次重复。吞吐统计首个模型 token 后的 127 个输出，初始 prefill、guided query 和 query replay 单独保留。

KVMem：128K 选择＋128K 生成预算，block=32，CPU 历史/index，NVMe=0，GC 关闭；MTP=4 主组保留 guided-reselect=both 开关及完整 replay。请求提供显式 query span，240K 实际观测到 query replay，没有私有 guided-query 生成日志。32K MTP=0 对照低于选择预算，用于观察配置切换后的吞吐。

| 方法 | 输入 | Decode 中位数 [最小, 最大] token/s | MTP 接受率 |
|---|---:|---:|---:|
| kvmem_q8_mtp0 | 32K | 43.18 [43.10, 43.95] | — |
| kvmem_q8_mtp4 | 8K | 111.66 [111.54, 112.30] | 66.9% |
| kvmem_q8_mtp4 | 32K | 93.18 [93.09, 93.20] | 56.8% |
| kvmem_q8_mtp4 | 120K | 89.77 [89.65, 90.86] | 73.1% |
| kvmem_q8_mtp4 | 240K | 72.59 [67.83, 82.48] | 56.9% |
| qw3_q8_mtp4 | 8K | 115.04 [114.85, 115.08] | 63.2% |
| qw3_q8_mtp4 | 32K | 106.74 [106.23, 106.76] | 65.2% |
| qw3_q8_mtp4 | 120K | 94.00 [93.74, 94.13] | 72.5% |
| qw3_q8_mtp4 | 240K | 73.32 [73.21, 73.33] | 70.7% |

这些是部署配置的 decode 测量，不是完整 agent 成功率或端到端 latency。每个请求的 prefill、首 token、完整请求时间均保留在 export/runs.csv。重复相同 prompt 会撤回先前生成的尾部，checkpoint 回退也不得从原始记录中删除。

输出一致性边界：32K 的 MTP=0/4 输出文本逐次相同为 False。两组输入 IDs 相同，但前缀缓存的先前请求路径和解码执行不同；本次未做状态对齐的数值等价测试。240K KVMem 三次输出也不同。吞吐及接受率为真实观测，不能宣称逐 token 等价加速或推断质量。

InfLLM / InfiniGen 的现有 Qwen 适配未实现 MTP；它们不在这张 Q8/MTP 对照表中。此前 BF16/MTP=0 结果保留为不同配置的补充数据。

量化权重身份见 weight_identity.json；启动参数、MTP 实际草稿/接受计数及服务器原始日志见 runs/。50 token/s 是待检验的预期，不是筛选或停止测试的条件。
