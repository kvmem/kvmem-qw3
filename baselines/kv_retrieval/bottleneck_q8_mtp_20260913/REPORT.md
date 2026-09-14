# InfiniGen / InfLLM：吞吐瓶颈归因

原先的低吞吐不能全部归到 InfiniGen 方法本身。本轮通过不改变检索选择规则和输出的实现优化，直接测到了可消除的开销。剩余差距仍包含缓存、Qwen 适配和 MTP 调度成本，不能把“优化后剩余百分比”直接命名为“算法固有成本”。

## 设置与计时口径

- 单张 RTX PRO 6000 Blackwell Server Edition，GPU 独占；CPU affinity 0–15、32–47，实际预取工作线程的 Torch 线程数也同步调整。PCIe 实测 Gen 5 ×16。
- 同一 Qwen3.8-27B Q8_0 GGUF，FP8 E4M3FN 历史 KV，外部 HF/FLA BF16 激活，batch=1，greedy。主模型和 MTP 预测头都使用真实 Q8 权重。
- 来自同一条已记录 DeepSWE 交互历史的两个固定输入：32,768 和 245,760 tokens。每个输入只重复计时，不代表两个独立任务，更不是完整 agent 准确率评估。
- InfLLM：128 initial + 4096 local + 16×128 retrieved tokens，GPU 历史块缓存容量 32 blocks。InfiniGen：每 query head 最多 2048 个历史 token，alpha=4，partial ratio=0.2。两者保留原先的小窗口设置，**工作集并不完全相同**；本轮前后优化的各自选择规则、窗口和精度均不变。
- 每次生成 128 tokens，以后 127 个有用输出除以完整 decode wall time。MTP 的 draft、每个候选的检索、搬运、验证、拒绝和状态提交均计入。模型加载、prefill、校准、CUDA graph 建立、恢复到完全相同的初始状态不计入 decode。
- InfLLM 复用作者 ContextManager、representatives、检索、LRU 和注意力；InfiniGen 复用作者派生的 threshold/mean-count/top-k 选择与 offloading 思路。Qwen、GQA、Q8、FP8、MTP 是本地适配。这里不是作者原生 OPT/Llama 配置的官方成绩。

## 交错复核的结果

单位为有用输出 token/s，每项三次中位数。线程数和方法顺序交错；所有慢重复也保留。表中 InfiniGen 来自最后一轮同条件线程复核；InfLLM 来自同轮 GPU 独占正式测量。首次八模式矩阵、全部复核以及合并六次结果另见 MEASUREMENTS.md 和 CSV，未覆盖原始结果。

| 历史 tokens | MTP | 原适配：2 线程 | 原适配：8 线程 | 等价优化：8 线程 | InfLLM：2 线程 |
|---|---:|---:|---:|---:|---:|
| 32,768 | 0 | 11.35 | 12.97 | 18.93 | 20.82 |
| 32,768 | 4 | 8.41 | 9.53 | 10.12 | 20.83 |
| 245,760 | 0 | 8.05 | 9.44 | 13.63 | 19.18 |
| 245,760 | 4 | 6.75 | 7.67 | 9.42 | 18.72 |

“等价优化”包含 GPU 整数索引排序/去重、将逆映射留在 GPU，以及预测 Q 时不再计算不会被使用的 gate 投影。仍然逐 token 检索，仍使用 CPU 历史 KV，仍做原来的 attention 和 MTP 验证。每次输出都与此前独立完整 prefill 的参考输出逐 token 相同。Q-only 投影在此原型中复制约 0.5 GiB 的 Q8 权重，需计入其实际部署成本。

## 有多少差距可归为实现开销

先把吞吐转换为 **ms/有用输出 token**，再计算：

`(原适配时延 − 等价优化时延) / (原适配时延 − InfLLM 时延)`。

这表示在当前固定输入上，通过已验证实现改动消除的“观测差距”，不是跨任务统计置信下界，也不是算法成本的精确拆分。不能直接相减 token/s 来计算耗时占比。

| 历史 | MTP | 原适配 ms/token | 优化后 ms/token | InfLLM ms/token | 已消除的观测差距 | 优化后 token/s 范围 |
|---|---:|---:|---:|---:|---:|---|
| 32,768 | 0 | 88.09 | 52.84 | 48.02 | 88.0% | 18.83–19.49 |
| 32,768 | 4 | 118.97 | 98.79 | 48.02 | 28.4% | 9.92–19.27 |
| 245,760 | 0 | 124.16 | 73.39 | 52.15 | 70.5% | 13.28–14.29 |
| 245,760 | 4 | 148.09 | 106.17 | 53.43 | 44.3% | 9.14–9.43 |

MTP 尤其需要保留波动边界。首次矩阵在 32K 的 optimized_both 中出现过 10.20–19.47 token/s；后续交错复核并未得到稳定的约 19 token/s。降低 Python GIL 切换间隔也未消除所有慢重复。因此不应挑选最快重复作为论文基线，也不应把剩余 MTP 差距全部归为方法必需的计算。

## 哪些瓶颈有直接证据

1. **CPU 索引整理确实很贵，而且可以优化。** 正式分项计时中，原适配在 32K / 240K、无 MTP 时，CPU 排序和去重累计约 22.9 / 29.8 ms 每有用输出。GPU 整数处理保持完全相同的每头选择和时间顺序；CPU 排序、去重调用耗时降至接近零。H2D 字节数只减少约 1%，说明提升主要不是减少了被选择的历史。
2. **CPU gather 与 PCIe 搬运是成本，但并不是整个低吞吐的解释。** 同一组无 MTP 分项记录中，CPU gather 约 6.9 / 14.5 ms 每输出，H2D stream span 约 5.0 / 8.8 ms；主线程等待预取 future 约 29.9 / 53.2 ms。等待包含预测、索引同步、CPU 处理和搬运流水线的影响，不能全部叫作 PCIe 时间。所有这些 host/stream span 相互重叠，不能加成 100% wall time。
3. **预测路径的额外 gate 投影是 Qwen 适配成本，单独去掉收益有限。** 它不会用于 InfiniGen 的预测分数，移除后所有输出相同。但 Qwen 的 head RMSNorm 与 RoPE 仍然完整计算，不能沿用原 OPT 的线性折叠假设而跳过它们。只删除不用的 gate，并未稳定消除主要瓶颈。
4. **全历史检索索引确实随历史增长。** 当前适配为 24 个 query head 分别保存 51 维 FP16 partial key。240K 时主模型 16 个 FA 层索引约 8.965 GiB；相同前缀的完整 FP8 K+V 约 7.5 GiB。这与 Qwen 的 24:4 GQA、当前索引表达及 FP16 索引精度有关，不能推广为原 OPT 实现的固定内存比例。
5. **MTP 增加的候选检索有真实工作量。** InfiniGen 32K / 240K 的 draft 接受率分别约 67.2% / 72.3%；127 个有用输出对应 172 / 163 个目标验证位置，即多 35.4% / 28.3%。本轮仍为每个候选独立执行因果正确的检索，并保留拒绝后的恢复。额外工作会放大昂贵的检索路径；其实际延迟还取决于批处理和跨层重叠，不能仅由接受率推算。
6. **MTP 适配的预取启动位置更晚，削弱了跨层重叠。** 单 token 路径通过前一层 input_layernorm hook 启动下一层 forecast（`infinigen/InfiniGen/qwen/backend.py`）。当前 MTP `verify.py::full_attention` 则先算目标层 Q/K/V，再调用 `infinigen_pipeline.enqueue`，之后逐候选等待 future。后续候选需要前面候选的 K/V，存在真实因果依赖，但当前实现也失去了普通路径用于掩盖 CPU 处理的部分计算区间。因此这不是作者原生单 token 流水线的直接 MTP 性能，也还不能作为方法的最优重叠实现。下一步可研究将不可变前缀的预测提前，并与候选尾部合并，必须继续保留完整逐候选选择和验证。

## InfLLM 为什么更快；还有哪些适配成本不能算成方法下限

InfLLM 即使每 token 检索，也经常不必从 CPU 再取数据：

| 历史 | MTP | InfLLM 检索块已在 GPU | 已在注意力缓冲区 |
|---|---:|---:|---:|
| 32,768 | 0 | 92.7% | 76.7% |
| 32,768 | 4 | 93.3% | 76.8% |
| 245,760 | 0 | 74.6% | 56.1% |
| 245,760 | 4 | 76.0% | 57.1% |

InfiniGen 的独立检索记录中，当前 token 选择的不可变历史 KV 行，有如下比例在前一个 token 刚被选择过：32,768 tokens：56.9%；245,760 tokens：40.4%。统计按去重后的 (token, KV-head) 行加权，排除每层首个冷步和新增/可能重写的尾部。

当前适配没有把这些行作为上一 token 的驻留缓存复用。这给出了进一步减少重复 CPU gather/H2D 的机会，**没有实现该缓存，也没有把复用比例当成加速比**。因此，已测得的 CPU 搬运也并非全部不可避免。

## CUDA 时间线

另用 Nsight Systems 记录了三个 32-output 诊断。时间范围包括很短的初始状态恢复和种子采样；以下为插桩条件下的 GPU 活动区间并集，不是常规运行中的 SM occupancy。kernel、copy 和 memset 跨 stream 重叠已经去重。不能用 trace 中的吞吐替代上面未插桩的正式数据。

| 32K trace | 计时范围 s | GPU 有 kernel 的时间占比 | 没有已记录 GPU 工作的时间占比 | Python GC s |
|---|---:|---:|---:|---:|
| MTP=0, full | 3.623 | 26.6% | 69.4% | 0.0001 |
| MTP=4, full | 5.514 | 11.9% | 84.7% | 0.0000 |
| MTP=4, optimized_bookkeeping | 5.183 | 13.9% | 82.5% | 0.0001 |

在这两个原适配的短 trace 内，MTP 使 GPU kernel 活动并集从约 0.963 秒降到 0.656 秒，但总范围从约 3.623 秒升到 5.514 秒；H2D 实际复制时间只从约 0.139 秒升到 0.175 秒。这支持“模型批处理节省的 GPU 计算没有转成 wall-time 收益，host 调度、预取等待和大量小算子发射成为瓶颈”的判断。由于插桩开销和短输出段，这些百分比不用于算法成本的定量归因；未插桩计时才用于前面的收益表。

逐 kernel、CUDA runtime 和 H2D/D2H 实际活动见 nsight_top_kernels.csv、nsight_runtime_calls.csv、nsight_memcpy.csv。GC 回调只在这三个 trace 中启用，常规吞吐没有这些探针。

## 非部署用的诊断与可比较边界

首次正式矩阵还测试了：提前给出完全相同的 partial Q、提前给出完全相同的选中 ID、用完整 GPU FP8 历史替代 CPU gather/H2D，以及组合条件。这些干预保留输出，用于查找成本；提前知道 Q/ID 和额外保存完整 GPU 历史都不是可部署的低显存 baseline。

为使诊断只改变数据来源，InfiniGen harness 在所有模式都分配了完整 GPU shadow，但只有 gpu_source 模式才读取/维护它。普通 full 和等价优化仍从 CPU 历史收集 KV。该额外分配意味着本报告的峰值 GPU allocated 值不能当成生产 InfiniGen 内存占用。

oracle 的各项收益存在重叠、流水线交互和测量波动，不能相加；组合 oracle 也还保留了 CPU 去重等处理，不是理论性能上界。

KVMem 之前约 72.59 token/s 的 240K 数据来自原生 qw3 引擎、128K 工作集和不同的残差/算子实现。本轮没有重跑该参照。它可以作为部署吞吐参考，**不能把它与外部 HF/FLA 适配的完整倍数差称为 KV 检索算法本身的加速比**。若论文需要纯方法归因，需要同引擎、同数值路径和明确的工作集/质量约束。

## 数据与复现入口

- `formal_infinigen.json` / `formal_infllm.json`：108 条未插桩正式计时，12 条单独 profile，额外 tuning/capture 分开存放。
- `thread_confirmation.json`：48 条交错线程/等价优化复核、18 条 GIL 诊断和历史行复用统计。
- `throughput_repetitions.csv`、`thread_confirmation_summary.csv`、`all_controlled_repetitions_summary.csv`：不筛掉慢重复。
- `MEASUREMENTS.md`、`analysis.json`、`confirmed_attribution.json`：完整表格与可复算的时延差。
- `probe.py`：隔离的诊断与等价优化；`run_experiment.py`、`check_threads.py`：GPU 独占、同输入、逐 token 校验和计时。
- `formal_source.tar.gz` 保留首次矩阵源文件；每份正式 JSON 带 source hashes。trace-only GC 探针为之后追加，不改变首次矩阵代码。`implementation_manifest.json` 记录导入的适配实现。
- OSWorld 的已完成结果有 SHA256 保护；中断的 038 KVMem 记录完整保留在 `interrupted_038_kvmem/`，不当作失败结果。恢复验证另见 `resume_status.json`。
