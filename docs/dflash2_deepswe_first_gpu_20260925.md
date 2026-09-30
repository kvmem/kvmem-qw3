# DeepSWE 首轮 GPU 结果与后续测试

2026-09-25 后续审计：原 live02 及等待中的 replay/K128 进程已退出，旧 manifest
没有完成收尾，不能继续视为运行中。Quill 已完成结果仍有效；Effect 已记录 251 个
完成请求、加权 decode 84.24 tok/s，但 agent 以 BadGatewayError 退出，任务未通过。
本地性能归因、公开目标权重的纠正及中断边界见
`dflash2_deepswe_cause_diagnosis_20260925.md`。下文“正在运行/等待”保留为历史检查点。

用户追加要求按旧 K128K/G64K 补测：已新增独立配置和执行器，对齐 K、prefill、G、
刷新阈值，复用全部 45 个输入及三项真实任务，保留 MTP4/DFlash 对照和 48GB 上限。
准备与运行入口见 `dflash2_k128_comparison_20260925.md`；下述既有结果仍属于 K64K/G32K，
不追认为新窗口的性能。

本轮是 build17、Q8_0 目标权重、FP8 KV、K64K/G32K、GPU Mean-K、七草稿。
没有进入并发优化。以下均为部分结果，不能算长任务验收通过。

最新：Quill 已完成官方任务，13/13 项 F2P、22/22 项 P2P 通过，reward=1，
无异常；117 个请求的 native/HTTP/trajectory token 数一致。最长历史 181,185 token，
输出 110,399 token，decode 1,416.030735 秒，加权 **77.9637 tok/s**。
64–128K 为 **76.8705 tok/s**，128–256K 为 **75.4612 tok/s**。
整卡采样峰值 **45,276,266,496 字节（45.28GB）**，216,819 个样本，无外部 GPU
计算进程。最大采样间隙 228.5ms，2 次超过 20ms；仍不是 allocator 瞬时峰值保证。
实际 agent 执行约 34.14 分钟，官方 verifier 约 49.87 秒。
完整证据：`deepswe_long_live02_quill_summary01.json` 与
`deepswe_long_live02_quill_resources01.json`。Effect 任务正在运行，Dynamo 和固定
输入 MTP/DFlash 重放随后串行执行。单任务通过不构成整个长任务验收通过。

按用户要求核对旧记录：同一 Quill 任务的原生 MTP4 整体 decode 为 58.5550 tok/s，
本轮观察值提高 33.15%；64–128K 从 61.2595 到 76.8705（+25.48%），128–256K
从 52.8373 到 75.4612（+42.82%）。旧配置是 K128K/G64K、刷新阈值 61,440、
CPU tier 110GiB、不同构建及生成轨迹；这些比例不能隔离 DFlash 的净收益。
另外两项旧整体 decode 为 Effect 56.8336、Dynamo 55.7591 tok/s。计算及原始日志
哈希见 `deepswe_historical_decode_comparison01.json`。

边界优化已有独立实验构建 `deferred_refresh_build01`，主工作区及 build17 未替换。
原型已接入该实验执行器并编译，但尚未执行 GPU 测试。回归程序
`deferred_refresh_regression_build02` 检查 greedy 开关一致性和开启优化后的 CPU/GPU
采样一致性；`deferred_refresh_regression01` 等待现有 replay02 进程及实验锁。
首个回归构建因 JSON 头路径错误失败，修正后的 build02 另行保留；没有运行模型重试。

## 首轮固定轨迹重放

`benchmark/dflash2/deepswe_long_replay01` 完成 7 个请求，第 8 个在 68,058-token
历史处终止于接口错误。模型生成达到 4,096-token 上限时，canonical 工具调用尚未
闭合，流式解析器报 `incomplete canonical tool call`。原始 SSE、native 日志、
完整内存采样及失败 manifest 均保留，没有把第 8 个请求当作成功响应。

| 已完成历史范围 | 输出 token | decode 时间 | token 加权 decode |
| --- | ---: | ---: | ---: |
| 32,935 / 33,459 / 37,140 | 5,826 | 75.799784 s | 76.8604 tok/s |
| 65,688（单请求） | 1,141 | 11.156395 s | 102.2732 tok/s |

32K 组三个请求中包含一次 length 截断；没有删掉这类响应。该组每次验证批平均输出
3.005 token，验证 32.141 ms、草稿 6.010 ms。作为历史诊断参照，GSM128 的 build13
对应每批 4.685 token、验证 27.950 ms、草稿 5.115 ms。两者构建和负载不同，不能
作为同条件性能对照，但说明草稿接受率下降和每批执行变慢都影响了本轮速度。
全局 token/batch 与此前按请求平均的接受长度口径也不同，不能混用。

首轮整卡采样峰值为 **43,560,796,160 字节（40.5692 GiB）**，自有进程峰值为
42,867,884,032 字节；16,617 个样本，未发现其他 GPU 计算进程。
请求采样间隔为 10ms，实际最大间隙约 209.7ms，超过 20ms 的间隙有 2 次。
它只支持“本次不完整运行中观测峰值低于 48GB”，不证明更长历史或瞬时峰值合格。
还没有同配置 MTP 进程的显存对照，不能据此断言 DFlash 总增量。

## 真实任务与重放修正

实际 mini-swe-agent 使用非流式 chat 请求。第二轮重放保持原冻结 messages、tools、
采样及 4,096-token 上限，改用这一实际接口；旧流式失败仍作为接口行为记录保留。
统计同时记录是否用尽输出预算，以免工具调用 finish reason 掩盖 length 截断。

首次真实任务尝试 `deepswe_long_live01` 在模型启动前失败：主机非 root 进程尝试
bind 中继的 443 端口，错误地把权限问题当作端口检查。已改为只读连接检查，实际绑定
仍由 Docker 中继完成。这次没有 agent 输出或模型运行，不构成按模型结果重试。

当前执行顺序：

1. `deepswe_long_live02`：原先冻结的三个实际任务，计划见
   `deepswe_long_live_plan02/plan.json`，每个任务一次官方 agent/verifier。
2. `deepswe_long_replay02`：同一 45 请求的非流式重放，两轮 DFlash、一轮 MTP4。
   等待同一个实验锁，不能与真实任务抢占 GPU。

真实 quill 任务的前 18 轮已完成 6,095 个输出 token，decode 总计 66.9431 秒，
约 91.05 tok/s，历史到 26,249 token；17 轮复用了前缀，无 HTTP 错误。
这是运行中快照，不是任务成功率或长历史结果。需完成后读取官方 F2P/P2P、所有
请求、刷新与恢复状态、完整 NVML 时间线，再确定下一步优化。

证据：`deepswe_long_replay01_partial_diagnosis.json`、
`deepswe_long_replay01_resource_analysis.json`、
`deepswe_gsm_acceptance_diagnosis01.json`，均在 `benchmark/dflash2/`。

## 真实任务跨窗口后的诊断

quill 第 40 轮从 91,674-token 历史生成 5,114 token，decode 108.746887 秒，
即 **47.0266 tok/s**。本轮剩余刷新阈值是 2,534 token，但工具调用尚未闭合，
直到第 5,114 token 才到达安全边界。当前 `kvmem_middecode_draft_limit` 在阈值
之后返回零，`drafts.empty()` 分支逐 token 执行，因此等待边界的 2,580 token
无法受益于投机。

本轮实际生成了 241-token 私有检索查询（5.850 秒），从 3,025 个历史块选择
2,048 个块，重选总计 0.427 秒。DFlash 日志各阶段合计仅 39.011 秒，另有
69.736 秒 decode 时间未落入这些计数器；该差额包含串行分支和其他控制开销，
不能全部当作已直接测量的串行 kernel 时间。原先的 `plain_s=0` 只计最终回退循环，
不会覆盖 `drafts.empty()`，也不能据此声称整个请求均在投机。

后续优先检查这段边界等待路径。优化必须保持首次安全边界、紧急容量上限及回滚；
若用草稿检测工具闭合，应先验证闭合候选，再决定提交/暂停，不能依据草稿内容在
p/q 验证前丢掉闭合候选而改变采样分布。当前没有修改运行中的 build17。

分析器增加了每轮实际选择事件和总耗时，区分普通选择、历史块超过窗口的选择、
guided mid-decode refresh；这些时间已包含于 native 阶段中，不重复相加。
证据保存在 `benchmark/dflash2/deepswe_live_deferred_refresh_diagnosis01/`。

运行中第 88 轮检查点（`deepswe_long_live02_progress01/summary.json`）：历史最长
151,111 token，>=64K 请求加权 decode **76.9544 tok/s**，整卡观测峰值
**45,272,072,192 字节（45.27GB）**，未发现其他 GPU 计算进程或 HTTP 错误。
官方 verifier 尚未结束，不能据此认定任务正确性通过。去除发生 mid-decode refresh
的请求仅作为瓶颈诊断时，其他长请求约 83.64 tok/s；正式指标仍保留全部请求。

边界处理的 CPU 原型保存在 `benchmark/dflash2/deferred_refresh_policy01/`，
已检查阈值、紧急预留、分片工具/围栏标记和三个精确单 token 目标概率。
它尚未接入 native 执行器，不能称为已实现或已测得提速；完整 GPU 基线仍用 build17。
