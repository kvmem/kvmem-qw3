# 与具名现有工作的最小直接对比

2026-09-12。根据用户“核心还是和现有工作的对比”的补充，替代上一方案的实验优先级。当前仅完成方案和代码入口审计，未接通新算法，也未启动对比实验。

## 推荐基线

首个基线选 InfLLM。其分块历史、模型 Q 检索、CPU 卸载、GPU 局部窗口与历史缓存，直接对应“现有 KV retrieval + offloading 是否已经足够”的问题。作者代码公开，方法范围相对集中，适合作为最先接入的候选；实际接入成本仍须由原型确认。

第二个基线选 FlexiCache：原实现公开，已利用 attention head 的时间稳定性降低重排和搬运频率，能避免仅与较早方法比较。原实现基于 vLLM，涉及 head 策略、缓存池与内核，接入成本预计更高。默认 host KV 池为 180 GB，不能未经调整就放到本机 125 GiB 内存上运行；配置调整应保留方法语义并报告实际容量。

LouisKV 机制相关性很高，但本轮从论文页面与公开搜索尚未确认可取得的官方实现；不能把搜索不到当作确定未开源。若取得作者实现，可作为低频检索基线的优先候选。Practical Online KV Cache Compaction for LLM Agents 同样高度相关，尤其是 query 选择和永久压缩，但本轮也未确认官方可运行代码。ReKV 对多模态相关，但先做纯文本可减少适配范围。

来源：[InfLLM 作者仓库](https://github.com/thunlp/InfLLM)、[InfLLM ContextManager](https://github.com/thunlp/InfLLM/blob/main/inf_llm/attention/context_manager.py)、[FlexiCache 作者仓库](https://github.com/NazmulTakbir/FlexiCache)、[LouisKV](https://arxiv.org/abs/2510.11292)、[在线 agent compaction](https://arxiv.org/abs/2608.00902)、[ReKV 作者仓库](https://github.com/Becomebright/ReKV)。

## 最小主实验

三组：普通 qw3 + Claude Code 原生 compaction、InfLLM、完整 guided KVMem。首先在四个完整 DeepSWE 文本任务上运行一个 seed，使用原始任务、工具与判分器，不预置额外历史，不人为强制压缩。任务清单应先冻结，并说明小样本是可行性/诊断对比，不能代表总体能力。第二阶段再加入 FlexiCache 和更多任务。

两种 KV 方法都关闭客户端摘要。它们都可能消除 summary stall，因此二者之间的核心比较是任务质量、常规推理成本、维护峰值和整条轨迹耗时。不能预设 KVMem 的 TTFT 一定更低。

先用现有文本请求归档验证增量处理、首次检索、后续召回和尾部重渲染，再启动完整任务；归档重放本身不产生 agent 成功率。缩小任务数量和模态范围来节省时间，不以不完整的原算法代替具名基线。

## 接入与命名边界

InfLLM 当前官方模型 patch 列出 Llama、Mistral、Qwen2；本地 qw3 的 Qwen3.8 混合模型还不能直接使用。不能声称只更换一个参数即可开始。优先审查将作者的完整检索注意力路径适配到当前模型标准 attention 层的方案，保留 recurrent 层的原计算；这是需要验证的模型适配，不是现成支持。

如果采用 qw3 后端移植，应保留并逐项核对 InfLLM 的代表 token 选择、检索粒度、局部/历史 attention 与位置规则、CPU 存储、GPU 缓存及实际采用的 LRU/异步优化。用作者实现对照同一组 Q/K/V 输入的选块、attention 输出与 cache-hit 行为，作为移植验证的一部分；这不能代替完整模型与会话生命周期测试。不得用 KVMem 的 Mean-K、全局工作集或低频调度替换原算法后仍标作 InfLLM。

表格应将这种结果明确标为 “InfLLM (adapted to Qwen3.8; qw3 backend)”；若算法语义发生实质改变，则重新命名并列出差异，不将其当作原论文忠实复现。适配实现的性能结论也只对该实现成立。

优先保持相同的模型权重、量化、KV dtype、采样、客户端和设备。如果原实现不能支持当前 FP8/MTP 等配置，需要寻找共同可验证的配置并同步重跑 KVMem；不能直接拿不同模型或不同加速配置的旧数字相除。本地配置解析器接受部分 legacy qwen3 GGUF 元数据，但发布说明将其他模型列为 experimental；不能因此假定更换为 8B 模型没有适配成本。

来源：[InfLLM 模型 patch](https://github.com/thunlp/InfLLM/blob/main/inf_llm/utils/patch.py)、[本地支持范围](/home/chaidi/qw3/docs/release_baseline.md:55)。

共享客户端、工具和安全的增量接口，核查各组前缀复用。若基线因接入错误每轮完整 prefill，不能把它解释为原方法固有成本。保留各算法所需的状态管理，不将 KVMem 的 guided query/replay 强加给其他方法。

## 最小结果表

按任务列出官方得分/成功、动作数、输出 tokens、agent 总时间、API 累计时间、step TTFT、下一工具动作等待、decode TPOT/吞吐，以及实际 GPU/CPU 内存。固定请求重放另外记录检索与搬运分项，区分冷启动和增量阶段。

预算先提供一个相同物理内存上限的配置，再在重放中至少提供两个检索预算点。KVMem 的 128K selection + 128K generation headroom 可能令实际活动窗口接近 256K；不能将它与始终只有 128K 的基线描述为全程等量 attention。算法的 block/head 布局不同，不强行把所有 block 大小设置成 32；报告实际 bytes 和平均/峰值活动 KV。

复用作者默认/论文推荐配置作为起点，在独立校准轨迹上选择预算内的配置，保留缓存命中和异步重叠优化。能力明显下降的方法不能仅凭更快 latency 被判定获胜。

## 时间成本判断

目前没有足够证据保证具名基线能在数小时内接通。主要成本是模型与缓存生命周期适配及正确性验证，减少 benchmark 样本不能免去这一步。先完成 InfLLM 单请求、增量请求和第一次历史检索的原型验证，才能给出可信运行排期。

若短期只完成原模块的固定 Q/K/V 性能比较，可以单列为组件级实验；不得据此宣称完成了 agent 级直接比较。消融仍可辅助定位差异，但不替代上述具名基线。
