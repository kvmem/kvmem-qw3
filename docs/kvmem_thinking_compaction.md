# Thinking 完成后的事务式压缩（实验性）

## 2026-09-29：当前 thinking 的独立摘要

当前实现改为独立短上下文摘要。摘要输入只有摘要指令和本轮新生成的
thinking token（移除末尾 `</think>`），不传入主任务 prompt、旧 thinking
或工具历史。目标模型权重共享；新的 executor 从位置 0 开始，重新 prefill，
使用独立的页表、KV 页面、recurrent/conv 状态和 logits，不复制父请求状态。
摘要采用目标模型 greedy 解码，不启用 MTP/DFlash 草稿。

父请求暂停期间，摘要从该请求的空闲 GPU 页配额中取得独占租约，包含已有
private reserve。整段输入必须装得下，否则跳过摘要；不会截断原 thinking。
摘要不创建 CPU/NVMe 历史仓库。临时 executor 和页面在恢复主请求前释放；
另外仍需为摘要回放保留仓库空闲槽位。独立计算状态和每次最多 256 tokens
的 prefill scratch 仍会增加显存开销，不等同于零额外显存。

无效摘要保持父请求原状态，无需摘要分支回滚；有效摘要仍按原事务流程恢复
thinking 前的状态、截断本轮 thinking、回放摘要后继续。只处理当前新生成的一轮。

串行服务继续使用 `--thinking-summary`。bounded KVMem、step 更新和原生
MTP/DFlash 的要求保留。continuous batching 现通过完整 bounded controller
接入，需要既有开关：

```sh
export QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS=1
export QW3_CONTINUOUS_KVMEM_CONTROLLER=1
export QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL=1
# 使用 DFlash 时还需 export QW3_CONTINUOUS_DFLASH=1
```

这些环境变量配合 `--continuous-batching --thinking-summary` 及正常 bounded
KVMem 参数使用。原有采样、会话复用等功能的开关继续独立配置。
摘要复用父请求的调度槽位，在输入 prefill 块、每个摘要 token 和替换回放块
完成后让出，其他请求可以继续执行。摘要 token 目前独立计算，尚未联合合批。
恢复/截断本身仍作为完整事务执行，可能造成短暂等待。

以下 2026-09-22/23 内容为旧版继承父上下文的摘要分支及其历史测试记录；其中
“仅串行”和“摘要失败恢复结束快照”不描述新版。历史质量与性能数字不能用作
独立摘要或并发版本的验证结果。新版本验证记录见
[独立摘要验证](thinking_summary_isolation_20260929.md)。

2026-09-22。实现默认关闭。当前策略为 thinking ≥ 1,024 tokens 才摘要，摘要自然结束。下文区分新版验证与旧版 DeepSWE 结果；任务质量与性能必须独立对照。

用户要求：正常生成 thinking，在 `</think>` 处运行一个额外摘要分支，回到 thinking 前的模型状态，输入短摘要后继续生成答案/工具调用。

## 执行过程

1. 在 prompt prefill、guided retrieval/replay 全部完成后，以 prompt 末尾的 `<think>\n` 为边界保存 DeltaNet recurrent/conv、hidden、主/MTP KV 和位置状态。
2. 原 thinking 走普通 MTP 生成与仓库发布路径，服务端暂缓向客户端输出。保留边界快照以备压缩成功后回滚。
3. 在 `</think>` 处检查长度；不足 1,024 tokens 则发送原 thinking 并继续同一解码循环，摘要、回滚、重放耗时均为零。仍有一次边界快照和缓冲开销。
4. 长 thinking 若跨过工作集切换、旧快照失效则跳过；否则暂停，补齐尚未 forward 的 closing token，并保存 thinking 结束状态及下一 token。
5. 在原 thinking 状态后开启私有摘要分支，使用同一模型 greedy 解码，空 `<think>` 前缀避免再生成长推理，不使用 MTP。提示词要求只保留继续任务所需的最关键、最短描述，没有固定词数或目标 token 数。
   输出修复版进一步明确主任务暂停、待办不能写成已完成，并预填 `Summary: `（不计入保留摘要）。私有分支约束工具、thinking、角色标记及代码围栏的生成，按解码字节检查跨 token 拼接；只在 GPU argmax 被禁止时复制 logits 并选择最高分合法 token。主分支采样不变。自然终止使用 EOS/`im_end`，不把再次生成的 `</think>` 当作摘要结束。
6. 摘要自然停止后检查格式及是否变短。资源保险上限随原 thinking 长度和可用私有空间变化；达到上限而未自然结束则丢弃摘要，绝不把截断文本当作有效摘要。可选旧参数仍允许显式指定上限。
7. 成功时丢弃私有分支，恢复 thinking 前的完整状态并截断原 thinking 的仓库历史，再输入短摘要及 `</think>`、正常重建主/MTP KV 和索引。
8. 无效摘要只恢复 thinking 结束快照，原 thinking 已在仓库中，无需重放原文。未闭合的 thinking 也直接保留；取消请求尽可能恢复有效边界，并使 warm cache 失效。
9. 客户端收到的 reasoning 与 warm prefix history 使用同一保留序列，然后继续答案/工具生成。

这条路径复用 guided query 的私有缓存隔离和 `capture_transient_state/restore_state`。新增的 private generation 入口不采集、不覆盖检索 query Q。

状态回滚包含 DeltaNet 的 recurrent 和 conv 状态，仅删除 attention KV 不等价。摘要也不是原 thinking KV 的数值压缩：它会重新经过模型，因此后续输出允许变化，需用任务成功率验证。

## 边界与失败处理

- 本版仅支持 bounded repository（`--kvmem-repository-blocks`）、普通串行 KVMem、step 更新、native MTP、隐式完整历史请求。named sessions、archive、continuous batching、显式 transcript replay、inline refresh 和 logits dump 不支持此组合。
- 私有摘要按当前可用空间准入；原 thinking 可以正常触发容量恢复，但跨工作集切换后跳过本轮压缩，不强行恢复旧快照。
- 太短、未完整闭合的 thinking，以及空摘要、未自然结束的摘要、包含结构标签/工具标记/代码围栏的摘要、没有变短的摘要，都保留原始 thinking。短 thinking 无回滚/重放；无效摘要仅回滚摘要分支，无原文重放。
- CUDA/状态恢复错误不当作有效摘要；请求报错。流取消会退出私有分支并恢复边界。
- 流式生成在原 thinking 和摘要计算期间通过空进度回调维持心跳并检测断连；空进度不算 completion token。客户端可见 reasoning 的首 token 会变晚，TTFT 必须如实计入。
- 原始 thinking 已消耗的主模型生成预算仍计入 `max_tokens`。成功压缩不会赠送额外答案 token；HTTP `finish_reason=length` 使用可见 token 加 discarded token 判定。摘要分支的额外生成开销单独统计。
- 当前只压缩新生成的一轮 thinking。不会自动重写或回收启用前已经存在的旧 thinking。
- 输出修复版的 `[thinking-summary-output]` JSON 日志保存原始摘要、停止原因和屏蔽次数，供诊断而不发送给客户端；指标解析器排除此载荷内可能引用的日志文本。格式约束不保证语义准确，任务质量仍依赖 DeepSWE verifier。详见 [输出修复验证与新版批次](thinking_summary_output_fix_20260922.md)。

## 配置

在原 KVMem+MTP 服务命令上增加：

```bash
--thinking-summary --thinking-summary-min-tokens 1024
```

Chat 请求用 `thinking_summary: true` 开启自然摘要，`thinking_summary_min_tokens` 默认 1024。`thinking_summary: false` 关闭。兼容旧参数 `thinking_summary_tokens`：正值显式限长并开启；仅传此参数且为 0 时关闭；与 `thinking_summary: true` 同传 0 表示不设固定上限。`thinking_budget` 仍是原 thinking 的强制结束预算。阈值使用当前生成 token 序列（包含 closing token 和可能的强制结束引导），并非字符数。

## 验证

新版还增加“已发布原 thinking → 私有摘要 → 回滚 → 替换”的逐字节对照，以及无效摘要恢复结束快照的比较；HTTP 检查使用默认 1K 阈值并检查短 thinking / 无效摘要的 replay_ms 为零。新版构建和主机测试保存在 `benchmark/thinking_compaction_natural_1k_20260922`；冻结 GPU 验证与新 DeepSWE 批次在 `benchmark/thinking_compaction_natural_1k_deepswe_20260922`。新版 GPU 状态 oracle 已通过 63、511、1535 三个边界，两个分支的 replacement 与直接计算逐字节一致。

新版 baseline/summary HTTP 均已通过：默认 1K 长段样例 1624→84 tokens；短段 248/88/303/42 tokens 的 summary/rollback/replay 时间均为零；强制无效摘要测试原文 93 tokens 保留、replay_ms=0；取消恢复及下一轮 prefix HIT 通过。5 项主机测试通过，新版 DeepSWE 已启动，性能与任务质量结果尚待完成。

以下具体通过数和 DeepSWE 结果属于旧 128/256 策略：

模型状态 oracle（不等同于任务质量）：

```bash
build-query-attention-probe-cuda/qw3-thinking-compaction-probe \
  models/Qwen3.5-4B-Q8_0-with-MTP.gguf
```

它在多个不对齐边界执行“长私有分支→回滚→短序列”，与新 executor 直接计算“同一前缀→同一短序列”比较 recurrent、conv、hidden、主/MTP raw K、实际 K/V 和 logits，并检查私有分支没有发布历史或切换工作集。63、511、1535 三个边界比较均为逐字节相同，包含内部 MTP 拒绝分支的嵌套回滚。

真实 HTTP 检查覆盖 streaming/non-streaming、下一轮 prefix HIT、summary fallback、forced thinking close、Anthropic Read 工具调用，以及 thinking 期间断开流后继续请求。样例 thinking 分别从 346→59、248→56 tokens；这些仅证明压缩路径确实执行，不是 DeepSWE 节省率。

首次 HTTP 验证在关闭摘要的 baseline 中也复现 CUDA 越界：MTP 拒绝或回滚使 bounded repository 的物理槽位不再连续，而首次 selection 前的 attention 仍把历史位置当槽位。现已使 bounded repository 从初始空窗口开始维护历史→物理页映射；reset 同样重建空窗口。两组共享此修复。

DeepSWE 对照入口：

```bash
python3 scripts/kvmem_eval/run_thinking_compaction_deepswe.py \
  --experiment benchmark/thinking_compaction_natural_1k_deepswe_20260922
```

两组使用同一冻结 binary、模型、seed、任务容器、sampling、28,672 thinking cap、128K selection、64K generation reserve、512K 固定仓库、CPU 索引和现有规则 GC。baseline 关闭摘要，新的 summary 默认自然结束、1K 阈值。旧 v2 冻结实验保持 128/256。每个任务/arm 使用独立服务和容器，任务间交替先后顺序。默认采用已有四任务 pilot，不用它推断全数据集成功率。

本次修复后实际运行目录为 `benchmark/thinking_compaction_deepswe_20260922_v2`；原目录保留首次失败及修复验证证据。v2 的 `actual-start.json` 记录了首个 agent 容器与成功的真实 Anthropic 请求，`state.json` 跟踪进度，任务完成后写入 `results.json` 和 `comparison.json`。请勿向正在计时的服务发送额外探测请求。

首个 `psd-tools-blend-range-api` 完整 pair 已结束：两组 verifier 均为 F2P 45/45、P2P 979/979，服务错误为 0。baseline→summary 的 agent 耗时为 2431.099→2643.288 秒（+8.73%）；末次请求 live 为 8252→6452（−21.81%），ready 为 8130→9930，总容量均为 16384 槽位。日志 CPU bytes 下降 26.87%，GPU KV bytes 增加 3.54%。成功压缩 29 次，直接少保留 32424 个 thinking tokens；另有短 thinking 原文重放 75 次、无效摘要回退 17 次、准入条件不满足跳过 13 次。首个样本体现为仓库占用减少，但没有加速或降低 GPU KV bytes。完整原始证据和指标解释见 [首对报告](../benchmark/thinking_compaction_deepswe_20260922_v2/first-pair-report.md)。旧批次已按用户要求立即停止：保留 3 个完整 trial，termenv baseline 中途停止，不纳入完成耗时或配对比较。

工作树后续加入 `rejection_reason` / `summary_stop_token` 诊断，区分 token-limit、空输出及 chat/thinking/tool/fence 标记；不放宽原有接受条件。原 v2 冻结二进制不变，新增诊断供后续复现使用。`postlaunch-diagnostics-check.json` 记录该诊断版本的主机检查和构建结果。`pause_zero_compaction_after_pair.py` 原用于首对结束仍零压缩时暂停诊断；首对已发生 29 次压缩，守卫记录 `continue`。后续策略切换已停止旧 supervisor 和 agent 容器，详见 `policy-stop.json`。

脚本先等待本机 GPU 空闲，不停止其他服务；状态 oracle 和 HTTP 集成检查成功后才启动 agent。`state.json` 区分 `prepared`、`waiting_for_gpu`、`state_oracle`、`http_oracle`、`running_deepswe`、`finished` 和 `failed`。排队状态不代表 DeepSWE 已实际开始。

## 指标

- 正确率：verifier reward=1 且 F2P/P2P 完整通过。容器退出 0、agent 自报完成均不足以判成功。
- 时间：agent elapsed、服务 request wall、thinking 原始 tokens、summary decode tokens、summary/rollback/replay 时间、工具次数与 prefix HIT/MISS。保留全部失败/超时；不能仅筛选变快或成功的样本。
- 两组会生成不同的后续推理和工具调用；任务级耗时差包含执行路径变化，不能单独归因于 KV 更少或解码内核更快。仅在完整 verifier 通过后比较同任务资源开销，失败组仍须报告质量与耗时。
- KV：每轮记录 `slots/ready/live/writing/retired/releasing`、GPU/CPU/index bytes；保留最后一次请求快照、峰值 live 和最低 ready。
- `slots` 是固定预分配容量。压缩主要降低 live 占用、增加可复用 ready；不能把 reserved capacity 或 `nvidia-smi` 不下降解释为无收益，也不能把累计 reclaimed 当作末尾实际释放量。
- 结束指标取服务 teardown 前；清空会话之后的零占用不能比较。异步 retired/releasing 尚未变成 ready 的部分必须保留原状态，不冒充可复用空间。
- 仓库还存在 Cancelled 状态，当前冻结日志未单独输出此列，因此已打印状态计数不保证加起来等于 slots。首个 baseline 的差额为 2；不能将这个差额算入 ready。各槽位状态逐个采样，也不是全仓库原子快照。
- 在相同容量下两组可能都接近 GC 水位。进一步研究应扫描物理仓库容量，比较相同任务质量所需的最小容量，并增加随时间的占用曲线。

## 已有轨迹的只读调查

`benchmark/kvmem_thinking_gc_design_20260922/payload_census.json` 来自已有 82 条 Claude Code / Qwen3.8-27B DeepSWE 轨迹，约 2.05 GB 日志。去掉 streaming 重复增量后，11,750 个 thinking block 约占所统计文本载荷字符数的 43.12%。对低、中、高比例各一条轨迹，aggregate thinking 字符数与 streaming delta 字符数独立核对相同。

该统计排除 system、模板、signature 等，单位是 Unicode 字符；它不是 token 比例、最终存活 KV 比例，也不是可安全回收的比例。样本含失败和复用的旧运行，不代表新的对照实验。

Qwen3.8 官方默认保留历史 thinking，强调 agent 决策连续性与减少重复推理，因此这项压缩应当作为需要验证的质量/资源权衡：[模型说明](https://huggingface.co/Qwen/Qwen3.8-27B#disable-preserved-thinking)。

## 1K 阈值的依据

用当前 GGUF 的原生 tokenizer 对历史 aggregate thinking 文本重新分词，去除 streaming 重复增量：82 条旧轨迹、11,750 段、6,995,712 tokens。中位数 103，P90 1,159，P95 2,424；长尾承担大部分 token。

| 阈值 | 触发段数 | 触发比例 | 覆盖 thinking tokens |
| --- | ---: | ---: | ---: |
| 256 | 3,496 | 29.75% | 90.57% |
| 512 | 2,306 | 19.63% | 84.36% |
| **1,024** | **1,317** | **11.21%** | **74.05%** |
| 2,048 | 691 | 5.88% | 61.25% |

独立查看刚完成的未压缩 psd-tools baseline：137 段 / 74,880 tokens；1K 触发 17 段（12.41%），覆盖 66.86% 的 thinking tokens。用户据此选定 1K。该比例是候选原文覆盖量，不是压缩后节省率；质量及耗时收益仍由新对照检验。

原始统计和可复现脚本见 [distribution.json](../benchmark/kvmem_thinking_gc_design_20260922/token_length_distribution/distribution.json) 及同目录 `analyze.py`。这是文本重新分词，排除了模板/closing 等，不完全等同于实际生成 token 边界；旧轨迹包含失败，非随机样本。
