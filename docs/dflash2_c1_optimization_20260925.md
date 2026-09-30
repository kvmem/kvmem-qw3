# DFlash 2 单请求优化记录（2026-09-25）

本阶段遵循用户最新顺序：先优化并验证 KVMem 单请求，再继续多并发。
主模型为 Qwen3.8-27B Q8_0，KV 为 E4M3 FP8；官方草稿及既有内部 BF16
算术不变。本阶段没有新增 BF16 主模型加载或 BF16 KV 支持。
工作仍在隔离目录 `/home/chaidi/qw3-dflash2`，未集成到运行服务。

**状态：已获得局部、可重复的提速；尚未完成公开五任务吞吐/质量验收。**

## 当前保留的内核优化

`src/mmvq_q8.cu` 的 fused Q8 SwiGLU 在一次权重读取后计算多个验证 token。
`QW3_Q8_SWIGLU_REUSE_ROWS=1` 显式复用 gate/up 权重的量化值和 scale，保留
串行解码的 Q8_1 量化、四 warp 归约及 FP32 累加顺序。单 token 分支保持原算术。
开关默认关闭，尚未据此改变生产配置。

两行/四行 CTA 和向量载入尚无可靠收益，未选为候选配置。随后试验的按 token
分配 warp、两个物理 warp 模拟四个累加器都通过算子精确检查，但没有额外收益，
已撤出运行源码。源码快照和结果分别保留在
`benchmark/dflash2/q8_c1_columns_experiment01/`、`q8_c1_virtual_experiment01/`。

## 已关闭的测量和正确性证据

固定工程用例：4K 历史、KVMem W1024、256 输出、T=1/k=20/p=.95。
三个独立加载进程对，顺序为 off/on、on/off、off/on；每进程三个热样本。
按进程中位数聚合，端到端吞吐 **113.104 → 119.462 tok/s**，三个配对速度比的
几何均值为 **1.055691（+5.57%）**。目标验证约 **1.720 → 1.602 s**，草稿约
0.344 s。该数据比较 DFlash 内核开关，不是 DFlash 与 MTP 或公开任务的对比。
证据：`benchmark/dflash2/q8_c1_swiglu_summary01.json`。

当前候选通过 90 个 linear 算例及 CUDA memcheck（零错误），对应
`q8_c1_selected_{operator,memcheck}02.json`；可选 vector 路径也在
`q8_c1_vector_memcheck02.json` 实际执行了包括 32/64/96 列的 90 个算例。
这些检查没有启用独立的 wide fanout 测试组，不扩大覆盖范围。
当前最终 binary 默认七草稿还通过普通 QW3 的 44 个响应及 bounded KVMem 的
32 个响应回归：`q8_c1_selected_dense01.json`、`q8_c1_selected_bounded01.json`。
4K KVMem 历史后四个位置、验证宽度 1–8，全部 64 层残差和完整词表 logits
与单 token 解码精确一致。证据为 `q8_c1_reuse_final_parity01.json`。
各记录限定其 source/binary SHA，不用旧记录代替后续版本的新验证。

审计发现旧 `reuse_final/vector ...01` 算子日志实际只有 66 个 linear 算例，
虽然其 manifest 中测试源码已列出新增维度，当时测试 binary 仍运行旧列表。
已保留原始记录并在 `q8_c1_operator_coverage_erratum01.json` 勘误；上述新运行
重新构建并从实际日志确认 90 项，不能仅凭源码 hash 推断执行了哪些测试。

公开格式初筛：五任务各一个测量样本及一个 warmup，固定 xhigh prompt，
最大 4096 输出，KVMem W8192 全量容纳上下文。以下为同一 binary 的 off/on
HTTP 实测；十组输出（含 warmup）的文本、token 数、usage 和停止原因完全一致。

| 任务 | 开关关闭 tok/s | reuse=1 tok/s | 比率 |
| --- | ---: | ---: | ---: |
| GSM8K | 146.179 | 149.112 | 1.0201 |
| MATH-500 | 146.446 | 147.334 | 1.0061 |
| HumanEval | 167.822 | 168.584 | 1.0045 |
| MBPP | 137.436 | 138.976 | 1.0112 |
| MT-Bench | 107.525 | 108.178 | 1.0061 |

证据：`benchmark/dflash2/q8_c1_public_reuse_pair01.json`。每任务只有一个样本，
不是任务集平均值；MATH-500、MT-Bench 样本在 thinking 内达到 4096 上限，
保留为 length 结果，不能当成质量通过。raw completion 缺少查询 span，实际使用
window-profile fallback 且全部上下文可容纳，不能称为稀疏检索性能。

标准七草稿、reuse=1 配置随后扩展为每任务 8 个样本，合计 40 个测量响应
和 5 个 warmup；**不能与上面单样本表直接相减当成性能变化**。
按每任务总输出 token / 总 HTTP 耗时聚合：

| 任务 | 8 样本聚合 tok/s | 每请求平均有效输出/验证轮 | 到达长度上限 | thinking 未关闭 |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 155.329 | 5.62 | 0 | 0 |
| MATH-500 | 137.864 | 5.33 | 1 | 1 |
| HumanEval | 118.343 | 4.78 | 3 | 3 |
| MBPP | 119.910 | 4.44 | 3 | 2 |
| MT-Bench | 110.266 | 4.27 | 2 | 2 |

这是一个加载进程中的初筛，不是独立重复或公开 128 样本复现；任务答案正确率
尚未评分，结构检查也不能替代评分。证据为 `q8_c1_public_expanded7_01.json`、
`q8_c1_public_expanded7_summary01.json`。仍未达到公开吞吐。

先前初筛的 none 约 44–45 tok/s、MTP-7 约 65–108 tok/s，可用于初步定位。
其中 none 运行的一部分与两线程编译重叠，不能用于正式性能对照；已记录在
`q8_c1_build02_context.json`、`q8_c1_public_baseline01_summary.json`。

随后重新完成了 **GSM8K 同一组 8 个输入**的干净对照：同一 binary、Q8_0/FP8、
相同 KVMem/采样参数，计时期间没有编译或其他 GPU 测试重叠。

| 算法 | 聚合 tok/s | 输出 token 总数 |
| --- | ---: | ---: |
| 无投机 | 44.421 | 2234 |
| 原生 MTP-7 | 106.893 | 2090 |
| DFlash 2，七草稿/reuse=1 | 155.329 | 2083 |

DFlash 在该小样本上的吞吐比为 none 的 3.497×、MTP-7 的 1.453×。
不同算法的采样文本/长度不同；这不是相同输出的加速比，也不是完整任务质量
或独立重复验收。三路均没有命中长度上限，thinking 均关闭；答案正确率未评分。
none 的 44.4 tok/s 与用户先前约 40 tok/s 的基线相符，不能沿用早先不匹配
精度/计时条件的 21/11.5 tok/s。证据：`q8_c1_public_gsm_controls01.json`、
`q8_c1_public_gsm_controls_summary01.json`，DFlash 使用上面的 expanded7 记录。

## Profiling 的解释边界

公开 GSM8K 单样本 CLI profiling 对齐了服务端强制的 MMQ/HGEMM 设置。
8-token SwiGLU 累计 **0.34858 → 0.33512 s（约 -3.86%）**；全部 CUDA kernel
累计 **1.22673 → 1.20580 s**。内核收益小，端到端 0.5%–2% 的初筛结果与之
一致。不能把工程用例的 5.57% 推广为公开任务收益。

上述时间包含 prefill，属于插桩运行而非正式吞吐测量。原始 profile manifest
沿用了工程用例的局限性文字，已在 `q8_c1_public_profile_pair01.json` 中明确勘误；
命令、冻结 prompt 和日志不变。后续 runner 已修正元数据。

## 公开目标与尚未完成项

[官方模型卡](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 使用单 H200、
SGLang/FA3、七个草稿 token、T=1/k=20/p=.95、xhigh、最多 4096 新 token。
五任务 C1 DFlash 为 184.0–236.1 tok/s。模型卡未明确目标/KV 精度。
本机为 RTX PRO 6000 Blackwell Server Edition，不能宣称硬件/精度相同。
[NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) 标称显存带宽 4.8 TB/s，
[本机 GPU 型号](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
为 1597 GB/s。带宽是需要分析的条件差异，不能直接换算成性能达标证明。

private controller 新增实验参数 `QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS`，
默认仍为 checkpoint 的七个；范围为 1 到 block_size−1，使用现有短块草稿/
验证路径。它不改变 checkpoint 的 block_size，也不替代独立合批控制器的配置。

六个草稿的候选通过当前 binary 的 32 个 bounded KVMem 响应检查，含跨 none/
DFlash 的 greedy 比较、同一路径的采样重复、稀疏查询/复用/尾部改写、取消与
继续生成（`q8_c1_width6_backend01.json`）。CPU/GPU sampler 的固定 seed 输出
一致，工程用例热 GPU 中位数却下降到 **108.428 tok/s**（一个独立进程）。
五任务各一个公开样本为 149.420/143.389/119.097/145.187/108.231 tok/s，
收益不一致，因此**不采用六个草稿作为默认优化**。不同草稿长度的采样文本
不同，这些数字只是筛选结果，不构成相同输出的配对加速比或质量通过。
证据：`q8_c1_width6_timing_summary01.json`、`q8_c1_public_width6_summary01.json`。

完整公开任务与独立进程重复、其余四任务的干净 none/MTP-7 控制、长历史/归档/恢复的当前
版本验证仍未完成。多并发验收继续后置，整个目标保持 active。

## 复现入口

使用新 ID，串行运行 GPU 任务：

```sh
python3 benchmark/dflash2/run_q8_c1.py --run-id NEW_PARITY --case parity --env QW3_Q8_SWIGLU_REUSE_ROWS=1
python3 benchmark/dflash2/run_q8_c1.py --run-id NEW_TIMING --case timing --env QW3_Q8_SWIGLU_REUSE_ROWS=1
python3 benchmark/dflash2/run_q8_c1_public.py --run-id NEW_PUBLIC --routes dflash mtp7 none --samples 8 --env QW3_Q8_SWIGLU_REUSE_ROWS=1
```

所有脚本保存命令、环境覆盖、代码/binary SHA、日志和闭合后的文件 SHA；
公共任务脚本另外保存完整响应。默认 `performance_qualified=false`。
