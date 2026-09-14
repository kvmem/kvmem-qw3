# Qwen3.8-27B decode 吞吐对比

原作者代码的 OPT/Mistral 测试已先完成；下列主结果统一到本地 Qwen3.8-27B 原始 BF16 权重、FP8 KV、batch=1、关闭 MTP、同一 GPU 独占运行。外部方法使用 HF/FLA/PyTorch，在独立目录适配 Qwen，没有为此修改生产 qw3。

测试输入为同一条真实 DeepSWE 历史的不同长度前缀，所有方法的输入 token 完全一致。每项生成 128 tokens，统计首 token 后的 127 次 decode；每项重复三次，表中为中位数，单位 token/s。

| 方法 | 8K | 32K | 120K | 240K |
|---|---:|---:|---:|---:|
| 普通 qw3，全 GPU KV | 24.43 | 24.13 | 23.00 | 21.66 |
| KVMem，128K 选择 + 128K 生成 | 23.98 | 23.66 | 22.49 | 22.45 |
| HF，全 GPU KV | 21.32 | 19.98 | 16.13 | 12.76 |
| HF，全 CPU KV | — | 16.69 | 7.69 | — |
| InfLLM，小窗口约 6.3K | 16.14 | 15.65 | 14.95 | 14.69 |
| InfiniGen，选择上限 2K | 11.74 | 9.85 | 8.44 | 8.24 |

小窗口结果保留原方法的小工作集配置，并不意味着实际注意力窗口或准确率与 KVMem 相同。

128K 预算补充：输入固定 240K，生成 32 tokens，统计后 31 次 decode，每项仅一次，不能与上表当作相同长度的重复测量。InfiniGen 的 128K 是选择上限，实际本次平均约 8,329 tokens/query head。

| 方法 | Decode token/s |
|---|---:|
| KVMem，128K 预算 | 22.18 |
| InfLLM，block=128，原块整理 | 0.63 |
| InfLLM，block=128，等价 CPU 查找优化 | 3.48 |
| InfLLM，block=2048，改变选块粒度 | 4.04 |
| InfiniGen，上限 128K | 2.48 |

InfLLM 的等价 CPU 优化保持选块、槽位及拷贝顺序，真实模型的全部 32 个输出 token 也一致；本次单测加速 5.49 倍。改变 block=2048 的另一个配置会改变选择行为，不能视作等价优化。

这些结果支持当前同模型系统实现的 decode 吞吐比较。HF 与 native 后端的算子及 residual 精度仍有差异，不能把全部倍数归因于 KV 算法，也不能由此推断 agent 准确率、compaction 停顿或端到端 latency。加载、校准、prefill、初始选择/query replay 均单独记录，未计入 decode。

发现并保留了一个限制：对 KVMem 重复相同 240K completion prompt、撤回上一轮生成后，两次触发 no_reusable_checkpoint，cold prefill 为 109.575 / 108.352 秒。这不是持续追加的 Claude Code 流量测试，不能推断其在线发生率；本轮未修改冻结的引擎来修复。

完整实现与精度边界、三次测量范围、InfiniGen GQA 部分索引的额外显存开销及所有原始数据，见 REPORT.md、METHOD_AUDIT.md、export/、matched_budget_export/。原模型原生复现结果见 ../native_upstream_export/REPORT.md。
