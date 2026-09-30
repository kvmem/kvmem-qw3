# DFlash2 在 DeepSWE 上的吞吐归因

本轮为既有日志及公开冻结证据的 CPU 分析，没有启动新的 GPU 性能测试。
结论：已确认本地 KVMem 边界处理存在投机暂停开销；长历史的验证执行耗时也值得
优先分析。任务内容影响草稿接受长度，但现有证据不能判定 DFlash2 不适合 DeepSWE，
也不能把公开成绩与本地的全部差额归给 qw3。

## 公开记录究竟是什么任务

此前引用的 124 请求 / 102.57 tok/s、117 个无重叠请求 / 125.36 tok/s，来自作者
2026-08-24 约两小时二十二分钟的日常 agent 使用。发布了脱敏的服务遥测及耗时，
没有任务说明、私有提示词和对话内容，不能识别具体任务或认定为 DeepSWE/SWE-bench。

更正此前“官方 FP8 权重”的说法：冻结发布
`68e05c793c656816553c0bcdc9093e945b22bde1` 的目标实际是
`orcarouter/Qwen3.8-27B-Uncensored-FP8`，revision
`9228df5c6c9c509e1019f83b4e085cf643118bac`。这是修改过的 FP8 衍生模型，
不只是把本地同一个 Q8_0 文件换了精度。

其 DFlash2 草稿来自 `incoai/Qwen3.8-27B-DFlash2`，公开 revision 与本地记录不同，
但实际 `model.safetensors` SHA256 **完全一致**：
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`。
config SHA256 也一致：
`873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`。
因此不能将差异解释为本地用了不同版本的草稿权重；运行时实现仍需独立验证。

来源：[作者对真实使用记录的说明](https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/)、
[冻结来源记录](https://github.com/jpezzulli/sglang-rtxpro6000/blob/68e05c793c656816553c0bcdc9093e945b22bde1/PROVENANCE.md#checkpoints)、
[模型哈希](https://github.com/jpezzulli/sglang-rtxpro6000/blob/68e05c793c656816553c0bcdc9093e945b22bde1/evidence/models/checkpoints.json)、
[指标定义与结果](https://github.com/jpezzulli/sglang-rtxpro6000/blob/68e05c793c656816553c0bcdc9093e945b22bde1/RESULTS.md)。

## 本地最新记录

当前可核对的 DFlash 结果均为 build17、Q8_0 目标 + FP8 KV、K64K/G32K，
不是尚未产出结果的 K128K/G64K 补测。所有已完成响应都保留，吞吐按
`总 completion token / 总 native decode 秒数` 计算；没有把工具执行时间算入 decode。

| 任务 | 已完成请求 | 输出 token | 加权 decode | 最长历史 | 正确性/完成状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| quill-shared-toolbar-focus | 117 | 110,399 | 77.96 tok/s | 181,185 | 官方 F2P 13/13、P2P 22/22，通过 |
| effect-sse-httpapi-streaming | 251 | 129,077 | 84.24 tok/s | 262,047 | agent 以 BadGatewayError 退出，不能认定任务通过 |

两者 native/HTTP/trajectory 的输出计数均一致。Quill 有 10 个、Effect 有 6 个请求
落在 50–70 tok/s；刷新请求还会更低。旧原生 MTP4 记录中 Quill/Effect/Dynamo 整体
分别为 58.56/56.83/55.76 tok/s，需与新 DFlash 记录区分。旧配置和生成轨迹不同，
不能把这组差值全部解释为算法提升。

Effect 的 Pier job 已结束，记录 NonZeroAgentExitCodeError；官方 verifier 在该次
中断输出上为 F2P 0/47、P2P 70/70、reward=0。它必须作为失败运行保留，不能删掉，
也不能据此把基础设施中断判成 DFlash 数值错误。该任务的完整资源收尾清单未写出，
本分析不追加显存验收通过声明。

## 已经分离出的两类损失

### 1. KVMem 等待安全边界时停止投机

`src/kvmem_refresh_policy.hpp` 的 `kvmem_middecode_draft_limit` 到达刷新阈值后
返回零；若工具调用或代码围栏尚未闭合，后端 `drafts.empty()` 分支逐 token 执行，
直到达到安全边界。这是本地执行策略引入的损失，不是草稿模型对某项任务的接受率。

Quill 第 40 个请求：历史 91,674、输出 5,114，刷新阈值剩余 2,534 token，后续
等待闭合的 2,580 token 走串行路径。整体 47.03 tok/s。另有私有检索查询约 5.85 秒，
真正重选约 0.427 秒。不能简单说主要耗时都来自 KV 搬运。旧 `plain_s=0` 没有覆盖
该零草稿分支；剩余未计入 DFlash 阶段的时间也不能全部当作已测量的串行 kernel 时间。

保持安全闭合边界的实验优化已有独立 CPU 构建，但 GPU 正确性和性能尚未验证，
不能把预期收益写成实测。细节见 `dflash2_deepswe_first_gpu_20260925.md`。

### 2. 避开刷新后，验证成本仍限制吞吐

以下仅为 **100,000–150,000 输入 token、无 middecode refresh** 的诊断子集。
没有用删除慢请求后的数字替换上面的正式整体结果。

| 指标 | Quill | Effect |
| --- | ---: | ---: |
| 请求数 | 40 | 41 |
| 每次验证批对应的输出 token | 3.611 | 4.125 |
| 草稿接受比例 | 37.40% | 44.78% |
| 验证日志计时 / 批 | 37.85 ms | 37.95 ms |
| 草稿日志计时 / 批 | 6.09 ms | 6.08 ms |
| 总 native decode 时间 / 批 | 44.79 ms | 44.86 ms |
| 加权 decode | 80.62 tok/s | 91.95 tok/s |

这些值满足 `吞吐 ≈ 每批输出 / 每批耗时`。以当前每批产出不变，130 tok/s 要求
每批总耗时约 27.78 / 31.73 ms，当前约 44.8 ms；这是优化所需幅度的算术说明，
不是已经证明可达到的收益。验证阶段占该子集 decode 日志计时约 84.5%，是优先
profiling 的方向。计时未设置逐阶段 GPU 同步且秒数经过舍入，不能据此确认某个
CUDA kernel、内存带宽或 CPU 调度分别占多少。

GSM128 历史 build13 的每批输出约 4.685、验证 27.95 ms、草稿 5.11 ms；早期
DeepSWE 32K 三个重放请求约 3.005、32.14 ms、6.01 ms，说明内容相关的草稿产出
下降和每批成本上升都存在。但两个构建、任务、历史长度不同，不是因果 A/B。

公开 SGLang 的 100–150K 单请求遥测平均接受长度约 3.565，也不是每轮都接受
七个草稿。本地长历史约 3.6–4.1 的产出没有显示“几乎不接受草稿”。不过公开数值是
遥测窗口的平均值，本地是整个子集的 batch 加权值，任务及终止处理也不同；不能
直接做百分比相除，宣称本地接受率已经与对方一致。

## 如何进一步区分实现与草稿模型

1. **同一 Q8_0 的基础执行与验证成本。** 固定输入、K/G、FP8 KV、采样，测无投机、
   MTP4、DFlash2；同时用固定 token 序列测 target 的单步与八行验证耗时，避免不同
   自由生成轨迹把内容难度带入 kernel 比较。参考框架必须先验证能读取同一 Q8_0。
   仅把目标改成 FP8 后测 SGLang，不能隔离框架差异。
2. **同一文本前缀的草稿/目标一致性。** 对冻结 DeepSWE continuation 回放，检查
   草稿分布、接受长度、目标 hidden features、位置及状态。分别覆盖未刷新、刷新
   前后和 KVMem 重建；若只有本地/刷新后变差，优先修适配。固定 seed 的自由生成
   不保证算法之间生成相同 token，不能替代这个对照。
3. **内容适配问题。** 在参考实现和本地相同权重/采样都正常时，再比较相同长度的
   推理文本、代码编辑、工具 JSON、错误日志等冻结前缀。如果两边在特定内容上都
   有一致的低接受长度，才有证据将这一部分归为草稿模型的任务分布问题。
4. **边界优化 A/B。** 在完整长任务之外用同一边界回归，先保证输出分布、回滚、
   安全闭合和显存上限，再验证减少零草稿串行段的收益。主线继续 Q8_0 + FP8 KV，
   不通过改 BF16 权重、温度或缩短历史制造更高分数。

## 运行状态与复算

本次检查时，原 live02 runner、server、replay02、K128 comparison01 和回归队列
对应旧进程已不在；相关 manifest 仍停在 running/waiting。首次检查 GPU 无计算进程；
本轮结束前复查已有其他 RULER 实验占用 GPU（PID 912973），未作干预。
这些状态不能被描述为“仍在后台排队”；K128 补测尚无新 GPU 结果。尚未确定进程
退出原因，不从 BadGatewayError 反推根因，也未在本轮重新启动完整 GPU 测试。
原始失败/未收尾记录均保留。

复算脚本：`benchmark/dflash2/analyze_deepswe_dflash_causes.py`。
输出：`benchmark/dflash2/deepswe_cause_diagnosis01/summary.json`，包含所有聚合分子
分母、子集边界、刷新请求、源文件哈希、公开 checkpoint 与本地实际文件哈希。
来源快照位于 `benchmark/dflash2/pro6000_public_research20260925_01/frozen_27b_agentic/`。
