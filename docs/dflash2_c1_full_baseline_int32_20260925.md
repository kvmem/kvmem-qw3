# 单请求：完整无投机基线与对齐权重读取优化

本轮补齐 GSM8K 128 题的无投机基线，并增加一个默认关闭的 Q8 验证内核候选。
主线仍为 Qwen3.8-27B Q8_0 权重、E4M3 FP8 目标 KV、KVMem 单请求。
工作区是 `/home/chaidi/qw3-dflash2`；未集成主目录、替换服务或开展多并发开发。
公开性能、质量和单请求完整验收仍未完成。

## 完整 GSM128 诊断对照

冻结 xhigh prompts，T=1、top-k=20、top-p=.95、top-k-first，最多 4096 输出。
每条路线一个独立加载进程，128 个测量请求；所有截断都保留并计入质量评分。
任务计时包含请求间客户端开销，排除模型加载和预热。

| 路线 | 输出 token | 任务耗时 | 任务吞吐 | 最终答案正确 | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无投机 | 73,202 | 1629.442 s | 44.925 tok/s | 117/128 | 6 |
| 原生 MTP-7 | 66,825 | 783.903 s | 85.246 tok/s | 119/128 | 4 |
| DFlash 2 | 65,127 | 490.328 s | 132.823 tok/s | 119/128 | 4 |

无投机结果与用户此前约 40 tok/s 的观察相符，早先 21/11.5 tok/s 不能沿用为当前基线。
此处 DFlash/none 的诊断比值为 2.9566，DFlash/MTP-7 为 1.5581。

**版本边界：**无投机使用 build12（CLI `22c6d3c7…`），缓存候选关闭；
MTP/DFlash 是前一阶段 CLI `b7500937…` 的完整运行，并未在本轮重跑。
它们不是同一构建的正式三路对照，也未完成三次独立交替重复或统计质量非劣验收。
这些完整数字不代表本轮新候选的完整任务速度或得分。

KVMem W8192/gen4096 可容纳所选 prompt 与全部输出，属于 all-fit 路径，
不代表长历史稀疏检索、压缩或换页后的吞吐。无投机计时期间没有编译、评分或其他 GPU 任务。
评分在模型运行结束后执行，沿用已有固定评分器与答案；只认 `</think>` 之后的最终答案。

证据：`q8_c1_public_gsm_none_full01.json`、`q8_c1_quality_gsm_none_full01.json`、
`q8_c1_public_gsm_all_routes_{summary,quality}01.json`，均在 `benchmark/dflash2/`。

## 接受长度与瓶颈

[公开模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 的接受长度是
每个请求先计算输出 token/验证轮数，再对请求取算术平均。
本机完整 GSM128 的 DFlash 按此口径为 **5.347**，公开值为 **5.46**。
此前报告的 **4.685** 是全任务 token 总数/总轮数，长回答的权重更大，不能直接与 5.46 比较。
本机 MTP-7 的请求均值为 4.094，聚合值为 3.388。

既有 DFlash 日志中，128 个请求的验证阶段累计 398.973 s，约占完整任务耗时的 81.4%；
草稿累计 71.593 s，约占 14.6%。这些是取整后的运行计数器，不是独立 kernel 的排他计时。
证据：`q8_c1_public_gsm_acceptance_diagnostic01.json`。

公开 GSM8K C1 为 H200/FA3 上的 236.1 tok/s。本机 RTX PRO 6000 Blackwell、运行引擎
以及模型卡未明确的目标/KV 精度都不同；当前绝对吞吐仍低于公开值。
接受长度接近不等于性能或质量验收通过，也不能按显存带宽直接换算为已达标。

## 新增的对齐读取候选

`src/mmvq_q8.cu` 新增 `QW3_Q8_ADD_STREAM_INT32=1`，默认关闭，且需同时开启
已有的 `QW3_Q8_ADD_STREAMING=1`。仅作用于 rows=5120、cols=17408、验证宽度=8
的 Q8 FFN down residual-add。该形状的量化平面起点和行跨度满足对齐要求：
用一次 32 位 streaming 读取替代两次 16 位读取及拼接，保持八元素整数点积、
FP32 累加、四 warp 归约与残差写回顺序。其他形状沿用原路径。

没有增加 BF16 目标权重或 BF16 目标 KV 支持。官方 BF16 草稿，以及已有内部算术保持原样；
原生草稿 attention/cache 仍为 FP32。

独立算子实验轮换 284,098,560 字节权重，超过两倍 134,217,728 字节的 GPU L2。
10 组交替 int16/int32 试验全部更快，中位调用耗时由 **0.086929 → 0.084396 ms**，
配对速度比几何均值为 **1.03091**。两版寄存器数分别为 88、80，无寄存器溢出。
这些是同进程内算子试验，不是十组独立完整模型结果。

随后用同一 build13 做三组独立完整模型进程对，顺序为 off/on、on/off、off/on。
用例为 4K 工程日志、KVMem W1024、T1/k20/p.95、256 输出，SwiGLU reuse 与旧 streaming 均开启。
各进程取三个 GPU warm 样本中位数：

| 组 | int32 关闭 | int32 开启 | 速度比 |
| --- | ---: | ---: | ---: |
| 1 | 119.985 tok/s | 121.195 tok/s | 1.01009 |
| 2 | 120.046 tok/s | 121.404 tok/s | 1.01132 |
| 3 | 119.929 tok/s | 121.187 tok/s | 1.01049 |

配对几何均值 **1.01063（+1.06%）**。这是已有 DFlash 配置中的小幅内核优化，
不能替代公开五任务测试或 DFlash 与 MTP 的对照。

候选 CLI 完整 SHA-256：
`ec8af19176e4c693fdec3963a74b6188577fb1d7dd3d0beba33f6832fc862949`。
增量补丁、算子源码、编译记录、资源报告在 `q8_c1_stream_int32_experiment01/`。

## 候选验证范围

- 独立候选通过 91 个 linear 算例和 CUDA memcheck，零错误，包含真实 FFN down
  形状及输出保护区；不是把旧 90 算例计数当成新分支覆盖。
- 实际模型构建通过 4K 历史下残差与完整词表 logits 对照，max-abs=0、argmax 差异为零。
  数值检查不区分正负零，不能扩大为所有浮点位模式一致。
- 三组工程计时中的同路线 CPU/GPU 输出检查通过。
- 五项公开任务各 8 个测量请求及 1 个预热，共 45 个响应，与上一 streaming 配置的
  完整文本、choices、usage、seed 和结束原因均一致。已有对应小样本评分可以沿用，
  不等于本轮候选已经取得完整 GSM128 或五任务质量资格。

新候选五任务小样本的完整任务吞吐为 GSM8K **159.419**、MATH-500 **140.985**、
HumanEval **121.320**、MBPP **122.957**、MT-Bench **112.832 tok/s**。
它们不能替换前面的 GSM128 数字，也没有独立重复，不能作为正式公开性能结论。
MBPP strict 接口评分问题和 MT-Bench judge 尚未解决，不因响应一致就自动获得资格。

证据：`q8_c1_build13.json`、`q8_c1_stream_int32_{parity,pairs,summary}01.json`、
`q8_c1_public_stream_int32_pilot01.json`、`q8_c1_stream_int32_public_{identity,summary}01.json`。

## 同机公开实现

本轮实际启动 vLLM 0.30.0 + GGUF 插件，使用真实 Q8 GGUF、目标 FP8 KV 和官方 safetensors 草稿。
未经修改的 V2 路径继承目标 GGUF loader，导致草稿加载失败；V1 则明确拒绝 DFlash2 草稿。

上游提交 [c723a831](https://github.com/vllm-project/vllm/commit/c723a831a81cb4ff89ea6d61b2d15109306ed1bc)
已有显式草稿 load-config 选择。本轮只将这一选择逻辑适配到独立 import overlay，
安装包源码不变；没有移植该提交的 daemon/IPC 功能，也没有修改推理内核或采样。
它是**带加载适配的参考端**，不能冒充未经修改的公开版本。

该适配已使 Q8 目标和草稿权重成功加载，日志记录模型内存 31.33 GiB。
随后发现 nvcc、ninja 的 PATH 配置问题；原始失败记录保留，runner 已补齐虚拟环境和 CUDA
路径，并在启动前检查 nvcc/ninja/c++。参考启动的最终状态记录在 `vllm_q8_startup_audit01.json`。
最终 `dflash_loader_backport03` 通过：greedy 和 T=1 各生成 16 token，两个请求的
137 个输入 token ID 均与冻结数据完全一致，输出 token ID 与 usage 数量一致。
首次 JIT 与初始化约 504 s，此数字只记录启动开销，不是生成性能。
所有参考进程已关闭，GPU 无残留进程。

这只关闭混合加载、tokenizer 和短生成 smoke 检查，不提供参考吞吐或质量资格。
参考端尚未验证 none/DFlash 的 greedy 对照、完整质量、与原生匹配的采样过滤顺序、
容量策略和优化后的编译/CUDA graph 配置。它的草稿 KV 为 BF16，原生草稿 attention/cache
为 FP32，不能宣称全部内部精度相同。正式比较前仍需协议修订和上述资格验证。

后续继续统一最终构建和配置，补完整任务及独立重复、同机参考资格、当前配置的剩余
guided/compaction 和冻结 KVMem 矩阵检查。CUDA graph 的批量路径目前还受 host 页表拷贝限制，
是否适合优化需进一步 profiling 和状态验证。单请求验收通过后再开发多并发。
