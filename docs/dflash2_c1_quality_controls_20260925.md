# DFlash 2 单请求：完整对照与质量诊断（2026-09-25，第二阶段）

本记录补充已封存的 `dflash2_c1_optimization_20260925.md`，不修改旧证据。
目标仍为 KVMem、Qwen3.8-27B Q8_0 主模型、E4M3 FP8 KV、单请求。
多并发开发继续后置；没有新增 BF16 主模型或 BF16 KV 支持，没有部署服务。

## 本轮优化筛选

尝试在 Q8 residual-add 多 token 验证中显式复用量化权重和 scale，保持旧的
FP32 累加顺序。算子输出精确检查通过。微基准轮换超过 GPU L2 两倍的权重
数据，避免只测试驻留缓存的小矩阵；交替执行三个 reference/candidate 轮次。

主验证宽度 8、输出维度 5120 时，输入维度 5120/6144/17408 的速度比依次为
0.9957/0.9995/1.0063；宽度 7 的 FF down 反而慢约 2.7%。没有继续把它推广
为模型级优化。候选已从运行源码撤出，源码及测量保存在
`benchmark/dflash2/q8_c1_add_experiment01/`。

当前保留上一阶段 `QW3_Q8_SWIGLU_REUSE_ROWS=1` 候选和默认七个草稿 token。
恢复后的 `src/mmvq_q8.cu` SHA 为
`c6a6524ccd739ad763d08a9fc44f3714ff4f6704479096e23d3a95eb620869a9`。
默认开关仍关闭；这轮 residual-add 试验没有带来新的已确认端到端提速。

当前 binary 的静态 CUDA 资源审计显示，宽度 8 的 SwiGLU 原分支每线程
96 个寄存器、reuse=1 分支 108 个，两者报告的 stack/local 均为零。
两行/四行变体为 147/185 个寄存器。记录为
`benchmark/dflash2/q8_c1_build09_resources01.{json,txt}`。这可用于筛选下一轮
寄存器/占用率试验，但不是动态 occupancy 测量，也不能据此保证降低寄存器
就会提速。

## 可复现的质量诊断

新增 `scripts/dflash_freeze_quality.py` 从本地缓存固定官方数据集答案。它检查
缓存路径中的 revision、缓存内容 SHA，以及全部五任务各 160 个输入与已冻结
官方 formatter 的逐条一致性。记录位于 `benchmark/dflash2/public_quality_v1/`。

`scripts/dflash_score_public.py` 检查已关闭测量的 manifest/transcript SHA，保留
全部样本，包括长度截断。数学题使用独立环境中的 Math-Verify 0.9.0；代码题
使用官方测试。生成代码在 bwrap 中执行，不暴露工作区、模型、网络或 GPU，
并限制资源。评分器先验证自己的正负例及提前退出处理，并验证 canonical
代码能够通过同一组官方测试。环境版本锁位于
`benchmark/dflash2/public_quality_requirements01.txt`。

评分规则预先固定：只取 `</think>` 后首个 Python/无语言代码块；没有 final
答案的响应仍计为失败。数学同时报告全文和 final 答案，不把 thinking 中的
候选答案代替最终答案。MT-Bench 尚未配置独立 judge，明确记为未评分。

官方 MBPP 吞吐 prompt 只包含自然语言题目，没有官方测试要求的函数名。
因此同时保留两种结果：原始代码直接执行官方测试的严格结果；以及唯一顶层
函数映射到唯一官方测试函数名的诊断结果。后者仅增加别名，不修改算法，
不是标准 pass@1，也不代替严格结果。无法唯一映射时记为 unresolved。

首轮 DFlash 每任务 8 条的诊断：GSM8K 7/8，MATH-500 7/8，HumanEval 5/8，
MBPP 严格 0/8、函数名映射诊断 5/8。GSM8K 的无投机及 MTP-7 都是 7/8。
MBPP 严格失败中有六条函数名不匹配，两条 thinking 未结束。映射后仍保留
一个列表/元组处理失败，没有根据测试修复生成算法。
证据为 `benchmark/dflash2/q8_c1_quality_initial03.json`。这不是质量非劣验收。

随后补齐其余任务的无投机/MTP-7 对照，质量诊断如下。各单元均为相同的
八道题、各一次采样；不同算法使用相同 seed 不保证采样文本相同。

| 任务 | 无投机 | MTP-7 | DFlash 首轮 |
| --- | ---: | ---: | ---: |
| GSM8K final 正确 | 7/8 | 7/8 | 7/8 |
| MATH-500 final 正确 | 8/8 | 7/8 | 7/8 |
| HumanEval 官方测试通过 | 5/8 | 4/8 | 5/8 |
| MBPP 严格测试通过 | 0/8 | 0/8 | 0/8 |
| MBPP 唯一函数名映射诊断 | 6/8 | 5/8 | 5/8 |
| MT-Bench | 未评分 | 未评分 | 未评分 |

新增对照评分记录为 `benchmark/dflash2/q8_c1_quality_remaining01.json`。
MATH、MBPP 有一个样本的差距，既不能据此宣布退化，也不能忽略或宣称非劣。
其中 DFlash 首轮和 GSM 对照使用上一构建，剩余四任务对照使用恢复源码后的
新构建；记录中的运行源码 SHA 相同、binary SHA 不同，不能把它写成全表同一
binary。第二轮 DFlash 专门用于补齐新 binary 的性能与输出回归对照。

本表使用冻结的公开格式输入：每任务前八个测量样本、xhigh、T=1/k=20/p=.95、
最多 4096 新 token、KVMem W8192/gen4096/ctx16384、prefix cache 关闭、C1。
这里全部上下文均可容纳；raw prompt 没有查询 span，走 window-profile fallback。
因此它衡量启用 KVMem 的 all-fit 路径，不是长历史稀疏检索性能。

第二轮现已关闭：45 个响应（40 个测量、5 个 warmup）的文本、choices、usage、
停止原因、seed 和样本身份与第一轮完全一致，因此上述 DFlash 答案评分同样
适用于这些响应。证明文件为 `q8_c1_public_repeat02_identity.json`，不是在新
输出上直接套用旧分数。两轮记录的运行源码相同、binary SHA 不同。

GSM8K 的两组对照也已在新 binary 上重跑，分别九个响应（含 warmup）与旧组
完全一致。对应 `q8_c1_public_gsm_{none,mtp}02_identity.json`，质量评分仍为
各 7/8。至此下面五任务的三路性能对照全部使用相同 binary、记录源码和环境。
binary SHA 为：
`b75009375182e734f14ea90ae6c6270c48ef4bef90dfd20a76e461fe57419d7f`。

| 任务 | 无投机 tok/s | MTP-7 tok/s | DFlash 第二轮 tok/s | DFlash/MTP |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 44.433 | 106.393 | 154.743 | 1.454 |
| MATH-500 | 45.197 | 84.192 | 137.677 | 1.635 |
| HumanEval | 45.282 | 72.848 | 118.566 | 1.628 |
| MBPP | 45.306 | 76.149 | 119.931 | 1.575 |
| MT-Bench | 45.301 | 69.813 | 110.118 | 1.577 |

五任务相对 DFlash 第一轮波动均小于 0.4%，
是复测稳定性证据，不是新的优化收益。新摘要为
`benchmark/dflash2/q8_c1_public_expanded7_summary02.json`。每轮每任务仍仅
8 个样本，不能替代 128 样本及至少三次独立交替的正式测量。
完整同一 binary 汇总为 `benchmark/dflash2/q8_c1_public_same_binary_summary02.json`。
无投机 44.4–45.3 tok/s 与用户原先约 40 tok/s 的观察相符，旧的 21/11.5
tok/s 不再作为当前 Q8/FP8 对照。DFlash 相对 MTP 为 1.45–1.64×；算法间输出
文本及长度不同，不能把它解释成相同输出序列的执行加速比。

剩余四任务的干净无投机/MTP-7 吞吐分别为：MATH 45.197/84.192，HumanEval
45.282/72.848，MBPP 45.306/76.149，MT-Bench 45.301/69.813 tok/s。
它们按输出总 token / HTTP 请求总耗时聚合，包含请求内全部工作，计时期间没有
编译或其他 GPU 任务。少量 CPU 评分/配置审计同时进行，因此仍作为筛选测量，
不是严格隔离且至少三次独立交替的正式验收。证据为
`benchmark/dflash2/q8_c1_public_remaining_controls_summary01.json`。

## 同机公开框架对照的限制

固定的 SGLang 源码 revision 为 `9f6fc553322e897a22f30320dae750f4a40a0b84`。
使用绝对 GGUF 路径、离线模式且隐藏 CUDA，调用其原始 `get_config`，在
Transformers 读取配置时得到 `GGUF model with architecture qwen35 is not supported yet.`。
证据为 `benchmark/dflash2/sglang_q8_config_probe02.{json,log}`。

这只证明该固定环境中的直接 GGUF 配置加载路径不可用；未启动 GPU 推理，
未测量 SGLang Q8/FP8 性能，不扩大为所有版本或所有转换路径都不支持。
更早的 probe01 是相对路径解析错误，不能用来证明架构不支持。
源码预审另发现 generic GGUF loader 缺少 qwen3_5 → qwen35 名称映射，记录
在 `sglang_q8_loader_audit01.json`，没有为制造对照去修改公开框架源码。

## 验收状态

完整 128 样本/任务、至少三次独立交替重复、质量非劣验证、MT-Bench judge、
当前候选的长历史与归档恢复验证仍需推进。公开 H200 绝对吞吐仍仅作外部参照，
不能据本机小样本结果宣布达标。整个目标保持 active。
