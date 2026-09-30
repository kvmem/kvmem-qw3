# K128K/G64K 同配置补测

2026-09-25 后续检查：下述最初启动的队列 PID 839168 及其等待的原重放进程已不在，
manifest 停留于等待状态。本轮尚无 K128K/G64K GPU 结果，不能称队列仍在运行。
冻结计划及 CPU 检查保留；详见 `dflash2_deepswe_cause_diagnosis_20260925.md`。

用户要求按旧 Quill 记录的 K128K/G64K 再测试。本轮新增独立运行，保留既有
K64K/G32K 实验、构建及结果，不修改正在运行或等待中的脚本。

## 配置与对照

使用 build17、Qwen3.8-27B Q8_0 目标权重、FP8 目标 KV、单请求。
仅将当前冻结命令中的四个参数调整为旧记录值：

| 参数 | 现有 K64K/G32K | 本次 K128K/G64K |
| --- | ---: | ---: |
| KVMem budget | 65,536 | 131,072 |
| Prefill budget | 65,536 | 131,072 |
| Generation budget | 32,768 | 65,536 |
| Middecode refresh trigger | 28,672 | 61,440 |

旧记录的 thinking budget 同样为 28,672，本次保持不变。GPU Mean-K、guided both、
private query 4,096、immutable K、prefill chunk 2,048、block 32、逻辑 ctx 1,048,576、
CPU tier 64GiB、GPU ratio .44、T=1/k20/p.95 和原七草稿配置均保留。
GPU ratio 不是硬上限，也不能代替实际窗口和显存检查；分析时核对日志中的有效
选取窗口和生成预算，如运行时缩小窗口则不能称为 K128K/G64K 验证通过。

本轮对齐旧 K/G 策略，不等于重现旧服务全部设置：旧版还有不同的二进制、GPU ratio=1、
CPU tier 110GiB、3M ctx。历史 58.6 tok/s 与新结果的差值仍包含构建差异；
本轮 MTP4/DFlash 使用相同 build17、窗口、采样、输入和资源设置，可作同配置比较。

## 固定工作负载及顺序

1. 等待已经运行的 K64K/G32K 真实任务和其后排队的固定重放进程退出；等待发生在
   GPU 实验锁之外，防止阻塞已有队列。再取得 `long_running_gpu.lock`，每个服务
   启动前要求 GPU 连续空闲 60 秒，不终止其他任务。
2. 原封复用 `deepswe_long_requests01` 的 45 个冻结请求，三个 route 顺序为
   DFlash seed1000、MTP4 seed1000、DFlash seed1001，每个 route 都运行全部请求。
   使用实际 MiniSWE 的 nonstream 接口，保留自然 EOS、max_tokens=4096、所有短输出、
   截断与失败，任务内保持原输入顺序。固定重放不执行工具，不证明任务完成。
3. 用相同 K128K/G64K 的 DFlash seed1000 各执行一次原三项真实任务：Quill、Effect、
   Dynamo。使用官方 agent 循环、原 90 分钟 deadline 及 verifier；不按模型结果重试。
4. 如发生资源、HTTP 或基础设施错误，保留失败运行，停止本轮后续阶段以供诊断。

## 指标

按历史长度 <32K、32–64K、64–128K、128–256K、>=256K 分桶，主指标为总输出
token / 总 native decode 秒数，同时报告 TTFT、完整请求速度、输出长度、刷新、
草稿/验证时间、终止原因和工具状态。分别报告 MTP4、两次 DFlash，以及已完成的
K64K/G32K 结果，不用简单平均每请求 tok/s，也不删掉慢请求。

显存继续以整卡 **48,000,000,000 字节** 为上限，独立 NVML observer 请求 10ms
采样，记录实际间隙、设备 UUID、计算进程、进程显存、整卡峰值和 CPU RSS。
越界时只停止本轮拥有的 GPU 进程并保留失败，不提高上限或静默缩小 K/G。
NVML 采样峰值不等于 allocator 瞬时峰值。

正确性、长历史输出量、两次 >=130 tok/s 及转入并发条件仍按
`dflash2_long_running_acceptance_20260925.md`；本轮配置准备或排队不构成 GPU 通过。

执行器：`benchmark/dflash2/run_deepswe_k128_comparison.py`。CPU 准备冻结原输入哈希、
build17 二进制及 core、模型哈希、依赖源码、完整命令和官方任务环境；启动前重新核对。
输出 `replay/` 与 `live/` 使用原汇总器支持的 manifest 格式，原始记录保留独立目录。

本轮冻结计划：`benchmark/dflash2/deepswe_k128_plan01/plan.json`，SHA256
`127f15c34645d5b0c1baa89ee423e3a53653a4ccff308f1265d98ef716e45943`。
`cpu_checks.json` 已逐项核对三条命令仅改变上述四个参数，且全部与旧 attempt.json
中的数值一致，原 K64K/G32K 命令未改变。CPU 准备核对全部 45 个请求哈希通过。

执行队列已启动：`benchmark/dflash2/deepswe_k128_comparison01/manifest.json`。
等待已有重放 PID 684646（start ticks 289593156）退出，再竞争共享实验锁；当前
已有真实任务及后续重放保留。准备完成时尚未执行 K128K/G64K GPU 推理。
