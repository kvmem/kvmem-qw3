# KVMem 联合预填充：实验及正确性失败记录

状态：**实验路径，默认关闭，未通过输出一致性检查，不计入性能验收。**
实现仅在 `/home/chaidi/qw3-dflash2`；没有集成回 `/home/chaidi/qw3`。
目标模型为 Qwen3.8-27B Q8_0，KV 为 E4M3 FP8。

## 主目录与开发工作区的能力边界

主目录的 `docs/continuous_batching_sampling_kvmem_20260924.md` 记录了采样、
窗口感知 body batch 和 MTP ragged 验证入口，以及该阶段的 CPU 检查。
不能把那些入口描述为完整 bounded KVMem 多会话并发服务。

DFlash 工作区后来增加了非 bounded 请求的完整事务控制器调度、联合目标验证、
MTP/DFlash 联合草稿及 GPU 采样。已结束的短场景 GPU 证据及限制见
`dflash2_matched_controller_20260925.md` 等阶段报告；历史通过记录保留原构建身份，
不代表本次新增预填充路径通过。

当前入口仍明确禁止 bounded repository、GC 或 thinking compaction 与
continuous batching 组合。完整会话/归档/缓存复用并发生命周期、每请求存储配额、
CPU/NVMe 仲裁和独立摘要上下文仍未完成。单请求吞吐不能证明这些能力。

## 本次实验改动

- `QW3_CONTINUOUS_KVMEM_BATCHED_PREFILL=1` 显式启用控制器的预填充合批。
  请求在预填充边界交还调度器，以各自页表、窗口位置和 token 段共同执行。
- executor hooks 补充每请求 query、raw K、DFlash 特征及预填充完成状态捕获。
  私有 query probe 等不兼容路径保留原 executor 执行。
- 调度器分别记录预填充准备和计算时间，避免相加包含等待的逐请求耗时。
- HTTP 日志区分 `dispatch` 与 `speculation_configured`，避免把串行调度标签
  `plain` 误读为无投机。配置字段本身不能证明某次请求实际执行了 DFlash，
  仍需核对原生执行摘要；新日志的 HTTP 模型验证尚未运行。

本次 host/CUDA 构建完成，36 项 host CTest 通过。
日志为 `benchmark/dflash2/q8_joint_prefill_{host_build01,host_ctest01,cuda_build01}.log`。

## 已结束的 GPU 对照

`greedy_off01` 与 `greedy_on01` 使用相同源码哈希和同一个 CUDA 测试二进制，
分别关闭/开启预填充合批。非 bounded KVMem，C4，约 800 token 历史，工作集
256，固定四条输入，greedy，包含反向请求顺序及取消。每路线每配置 16 个响应，
总计 96 个响应。每配置内部重复检查通过，但跨配置检查失败：

| 路线 | off/on 完整文本及回调一致 | 不一致行（零基） |
| --- | ---: | --- |
| none | 12 / 16 | 1、5、9、13 |
| MTP | 12 / 16 | 1、5、9、13 |
| DFlash | 12 / 16 | 1、5、9、13 |

关闭合批时，none 与 MTP、DFlash 各 16/16 一致。开启后分别为 14/16 一致，
不同项为行 1、5。失败集中于同一输入的重复和取消变体，不能把 12 个差异
解读成 12 条独立输入，也不能把单配置重复通过解释为优化正确。

已保留原始日志、转录与哈希；运行清单中的 `passed_scoped` 仅表示其探针内部
检查通过。配对汇总 `benchmark/dflash2/q8_joint_prefill_summary01.json` 明确为
`failed_correctness`，`performance_qualified=false`，失败路线不生成加速比。
汇总程序返回 1，失败输出保存在同名 `.log`，这是正确性拒绝而非运行基础设施故障。

尚未确定差异根因。下一步比较逐请求与合批的矩阵算术、recurrent/attention
状态及窗口捕获，再重跑相同对照；不能在根因修复前开启默认值或报告吞吐收益。
长历史与 bounded 正式矩阵仍未运行，本阶段未完成目标。

## 已有单请求吞吐应单独解读

`dflash2_gpu_sampling_20260925.md` 的 C1、4K 历史、W1024、256 输出、T=1
热请求诊断，端到端中位数为 none 43.744、MTP 113.892、DFlash 112.903 tok/s。
这是每路线一个进程内三次测量；不同算法随机输出不同，未经过跨路线任务质量验收，
不能据此宣称正式 2.58 倍收益或超过原生 MTP，更不能称为并发吞吐。

另一个 833-token 历史、W256 的 greedy 短诊断输出一致，none 44.933、DFlash
85.617 tok/s，约 1.905 倍，同样不代替正式独立重复与公开测试。
