# 单请求：完整 GSM8K 对照与残差投影缓存候选

本阶段继续以 Qwen3.8-27B Q8_0 权重、E4M3 FP8 目标 KV、KVMem 单请求为主线。
新增一个默认关闭的 CUDA 缓存候选；单请求公开性能与质量验收仍未完成，多并发继续后置。
工作目录为 `/home/chaidi/qw3-dflash2`，未集成到主目录或运行服务。

## 完整 GSM8K：128 个样本

以下是**加入本轮缓存候选之前**的选定配置，CLI SHA-256 为
`b75009375182e734f14ea90ae6c6270c48ef4bef90dfd20a76e461fe57419d7f`。
两条路线均启用 GPU target sampling、SwiGLU reuse=1，连续批处理关闭。
使用冻结的 xhigh 原始 prompt、T=1、top-k=20、top-p=.95、top-k-first、最多 4096 输出。
MTP 和 DFlash 均配置七个草稿 token。

| 路线 | 测量样本 | 输出 token | 完整任务耗时 | 任务吞吐 | 最终答案正确 | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 MTP-7 | 128 | 66,825 | 783.903 s | 85.246 tok/s | 119/128 | 4 |
| DFlash 2 | 128 | 65,127 | 490.328 s | 132.823 tok/s | 119/128 | 4 |

这次运行的吞吐比为 **1.5581**。每条路线只有一个独立加载进程，因此不能据此宣布
独立重复、统计质量非劣或五任务性能验收通过。两种采样算法的输出长度不同。

任务计时从第一个测量请求准备开始，到最后一个响应解析结束，包含请求间开销，
排除模型加载和预热。所有截断响应保留；未结束 thinking、没有可用最终答案的响应
计为错误。评分使用已固定的 GSM8K 答案和 Math-Verify，最终答案分数为 92.97%。
整个文本中出现正确数值的诊断分数不替代最终答案分数。

此组 KVMem W8192/gen4096 可容纳完整 prompt 和输出，属于 all-fit 路径；
不能用它代表稀疏检索或长历史压缩后的吞吐。128 样本的无投机对照仍待补齐。

前 8 个 GSM8K 样本约 155 tok/s 的结果不能代表完整任务。本次事后分组显示：
99 个不超过 512 token 的响应平均每轮输出 5.45 token；6 个超过 2048 token 的响应
平均每轮输出 3.99 token，合计占 21,776 个输出 token。该分组用于解释整体耗时，
没有从汇总中排除长响应，也不是控制长度的性能实验。

证据：`q8_c1_public_gsm_{full,mtp_full}01.json`、
`q8_c1_public_gsm_full_pair_summary01.json`、
`q8_c1_quality_gsm_{full,mtp_full}01.json`、
`q8_c1_public_gsm_length_diagnostic01.json`，均在 `benchmark/dflash2/`。

## 本轮代码改动与实际收益

`src/mmvq_q8.cu` 新增 `QW3_Q8_ADD_STREAMING=1`，默认关闭。
仅当 Q8 residual-add 的输出行数为 5120、输入列数为 17408、验证宽度为 8 时，
使用 `__ldcs` 流式权重缓存读取提示。整数点积、FP32 累加和四 warp 归约顺序保持原样。
其他矩阵形状继续使用原路径。没有新增 BF16 目标权重或 BF16 目标 KV 支持。

独立算子实验轮换超过两倍 GPU L2 容量的权重，五轮交替测试中，FFN down 的调用
中位数由 0.105150 ms 降到 0.089930 ms；该计时还包含输入量化和后端调用开销。
较小的 attention 投影使用同一提示反而变慢，因此没有启用。另一种 cg 缓存策略、
两行/四行复用候选均更慢，保留实验记录但没有加入运行路径。

**完整模型收益明显较小。** 在 4K 工程日志、KVMem W1024、256 输出的固定用例中，
三个独立进程对依次采用 off/on、on/off、off/on：

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| 进程内中位数再取中位数 | 119.326 tok/s | 120.014 tok/s |
| 三个进程中位数范围 | 119.264–119.652 | 119.852–120.052 |

三个配对速度比为 1.00577、1.00334、1.00493，几何均值 **1.00468（+0.47%）**。
这比较的是 DFlash 内核开关，不能替代公开任务或 DFlash/MTP 对照。
真实生成 profiling 确认新内核执行了 2176 次，中位耗时 90.753 μs；此前相同公开
prompt 的 FFN down 诊断中位数为 93.856 μs。两个 profiler 运行仅用于诊断。

候选 CLI SHA-256 为
`22c6d3c767b3d1e997a85dadc9616e2e814908f3f97818f095f3241d9407e636`。
本轮增量补丁保存在 `benchmark/dflash2/q8_c1_add_cache_experiment01/runtime_change.patch`。

## 验证范围

- 当前运行代码通过 91 个 linear 算例和 CUDA memcheck，零错误。其中新增真实
  `5120×17408`、宽度 8、带输出保护区的算例，确保实际覆盖受形状限制的新分支。
- 4K 历史下，全模型残差和完整词表 logits 与串行路径逐元素数值一致。
  算子及探针的零误差检查不区分正负零，不能扩大为所有浮点位模式完全一致的证明。
- 三组工程计时中的 CPU/GPU 同路线输出检查通过。
- 五项公开任务各 8 个测量样本，加上预热，共 45 个响应：文本、choices、usage、
  seed 和结束原因均与之前配置一致。保留截断响应；这属于输出回归，不等于质量验收。

候选五任务小样本的完整任务吞吐分别为 GSM8K 157.665、MATH-500 139.143、
HumanEval 119.862、MBPP 121.326、MT-Bench 111.460 tok/s。它们不是 128 样本结果，
也没有三次独立交替重复，不能替换前面的完整 GSM8K 数字或作为正式提速结论。

证据：`q8_c1_build12.json`、`q8_c1_add_streaming_{memcheck,parity}01.json`、
`q8_c1_add_streaming_{pairs,summary}01.json`、`q8_c1_add_streaming_profile_audit01.json`、
`q8_c1_public_add_streaming_pilot01.json`、
`q8_c1_add_streaming_public_{identity,summary}01.json`。

## 同机公开实现对照

已在独立环境安装 vLLM 0.30.0 与 GGUF 插件，依赖检查、CUDA 扩展导入、CPU 配置解析通过。
vLLM 源码对应 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，插件对应
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`。实际安装的 DFlash2 源文件哈希与该版本一致。
GGUF 已迁移到独立插件，配置方式见
[vLLM 官方说明](https://docs.vllm.ai/en/latest/features/quantization/gguf/)。

配置探针已确认目标为 GGUF/Qwen3_5、目标 KV 为 fp8_e4m3，草稿仍为
DFlash2DraftModel、safetensors 加载、无草稿权重量化。混合加载需要明确 HF 配置解析
和独立 draft_load_config，避免将目标的 GGUF 解析/加载设置错误继承给草稿。

**尚未加载模型或执行参考 GPU 推理，没有参考吞吐结果。** Native 草稿保留 FP32
attention/cache，准备中的 vLLM 草稿 KV 为 BF16；该内部精度差异已单列，不能宣称
所有内部精度一致。实际权重加载、分布/质量、tokenizer、KV 和性能资格仍待验证。
记录在 `benchmark/dflash2/vllm_q8_reference_audit01/`。

[公开模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 的 GSM8K 为
236.1 tok/s，使用 H200/FA3；本机为 RTX PRO 6000 Blackwell，且模型卡未明确目标与
KV 精度。当前结果仍未达到该绝对数字，不能把硬件差异当成已经达标的证明。

后续先补完整无投机对照、新候选的完整任务与独立重复、同机公开实现资格，以及当前
配置下尚未关闭的 KVMem guided/compaction 和冻结矩阵检查；单请求验收后再做多并发。
