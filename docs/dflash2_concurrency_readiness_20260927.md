# DFlash 2.0 / KVMem 多请求优化启动评估

2026-09-27。用户确认单请求优化到此结束，转而评估多请求优化是否可以开展。

**结论：可以进入多请求性能开发。** 开发工作树已有非 bounded KVMem 的请求控制器、
共享页池及 DFlash 联合验证，具备从 C2 开始测量和改进的基础。当前证据不代表
长历史多请求性能已经合格，也不代表完整 bounded repository 多会话服务已经实现。
本次只核对源码和封存记录、保存启动评估，没有启动新的 GPU 基准或更改执行代码。

## 范围与基线

- 工作树为 `/home/chaidi/qw3-dflash2`，分支 `codex/dflash2-qwen38`；
  `/home/chaidi/qw3` 主目录仍是较早的采样/body batch 适配阶段，不能混用能力描述。
- 继续使用 Qwen3.8-27B Q8_0 目标权重、FP8 目标 KV、K128K/G64K、
  temperature=1、top-k=20、top-p=0.95；普通 QW3 只做回归。
- 单请求最终结果为两种子各 45 个冻结请求 99.2733 / 102.5191 tok/s，
  无中途刷新。独立刷新事件长请求为 46.7630 / 52.8140 tok/s。
  这些是单请求指标；不再以稳定达到 130 tok/s 为结束条件。
- 保留单请求已验收源码、二进制和证据。当前开发源码与单请求完成清单的六个
  相关源码哈希一致，旧并发测试的源码身份不同，必须重新执行适用的回归。
- 优先测量吞吐并记录显存，不通过缩小每请求 K/G 或改变精度制造并发增益。
  48GB 容量优化按此前用户的吞吐优先顺序处理；单请求采样峰值不能外推并发容量。

## 已有能力与证据边界

| 部分 | 已有内容 | 仍缺内容 |
| --- | --- | --- |
| 请求状态 | 每请求 executor、RNG、草稿缓存、特征、checkpoint 和接受/回滚；控制器在安全点交还调度器 | 当前单请求改动叠加后的并发复验、长期持续运行 |
| GPU KV 页池 | 共享底层 arena；每请求驻留配额和私有查询借页；取消/销毁释放；高物理页号与配额分离 | 长历史容量准入、CPU/NVMe 总配额与公平调度 |
| 联合目标验证 | DFlash ragged verifier、每请求特征和 logits 行偏移、GPU 采样、槽位补充 | 当前长历史算术路径、性能及真实服务负载验证 |
| 联合草稿 | 已合批卷积、attention、选择和输入转换；请求间缓存独立 | 草稿投影 GEMM 仍逐请求执行，没有证实整体提速 |
| KVMem 控制器 | 短场景查询、回放、刷新、外部取消和排队补位已有通过记录 | 前缀/持久会话/归档/压缩完整并发生命周期 |
| 联合 prefill | 已有默认关闭的实验入口 | off/on 输出一致性失败尚未解决，不能开启作为性能候选 |
| bounded repository | 串行入口已有实现 | 当前明确拒绝 continuous batching；GC、thinking compaction 也有串行限制 |

封存证据分别为：

- `q8_dflash_cb_summary01.json`：448 响应，其中 416 为 KVMem；
  覆盖 C4/C16、查询、刷新、取消、槽位补充。工作集仅 256 token，历史约 800 token。
- `q8_joint_draft_summary01.json`：352 响应，其中 320 为 KVMem，
  另有 140 个固定输入 GEMM 检查；`end_to_end_speedup_qualified=false`。
- `q8_matched_controller_summary01.json`：544 响应，96 个串行/批量 greedy
  响应匹配；串行控制仍保留全局 CB 数值配置，不是默认单请求数值路径。
- `q8_joint_prefill_summary01.json`：`failed_correctness`。
  none/MTP/DFlash 各 16 个 off/on 响应只有 12 个一致，保留失败，不能算提速。

这些是不同历史阶段的证据，不能将数量相加后称为当前构建的验证覆盖。

## 优先检查的执行问题

1. **长历史联合验证不能直接继承单请求优化结论。**
   `QwenExecutor` 构造函数在全局 CB 开启时禁用原有内部 BF16 激活路径；
   `supports_continuous_verify_batch()` 要求 FP32 hidden。
   目标权重仍是 Q8_0，激活算术的差别不等于新增 BF16 目标权重支持。
   当前联合 verifier 调用 `attention_prefill_batch_paged_ragged_gated_device()`；
   单请求固定八行/1024-token KV 分块的 stable prefill 在另一套规划入口，
   不能仅打开原开关就声称 ragged 路径享有相同调度和数值性质。
   先比较同一 CB 数值配置下的独立/联合执行，再单列默认 C1 对照。

2. **请求间仍有整体等待。**
   `CooperativeRequest` 每次只恢复一个控制器，直到它 yield 或结束。
   联合 prefill 关闭时，原 prefill 不在块边界交还该调度器；私有查询和事务回放
   也没有完整的独立可调度事件。调度器还要求所有 resident 请求就绪后才验证。
   因此一个请求的长 prefill/刷新可能阻塞已就绪请求；这是源码判断，尚无当前
   C2/C4 实测等待占比。应增加请求阶段时间线，并检查其他请求 token 间隔。

3. **联合草稿中的线性层没有真正合成共享 GEMM。**
   `DFlashBatchExecutor::linear()` 调用 `dflash_linear_blocks()`，保持固定的
   每请求 GEMM 形状以避免历史的批形状数值漂移。旧合并 GEMM 曾在 140 个算子
   用例中有 71 个不满足逐位一致，并导致固定种子模型重复不一致。
   可以优化，但必须区分允许的数值差异和状态/RNG 隔离错误，保持可靠算术对照。
   既有宽 Q8 verifier 算子实验可复用为候选，历史算子加速不能算整模型加速。

4. **真实 agent 的复用与冷启动需要分开测。**
   现有控制器并发探针是冷请求；显式持久会话不进入此路由。
   冻结请求回放可用于 decode 基准，但不能代表 DeepSWE 每轮工具返回后能复用
   历史 KV 的服务吞吐。需补会话所属关系、prefix lease、刷新提交/回滚、取消后
   清理及缓存复用，再验证实际 agent 任务。bounded repository 接入是独立里程碑。

## 建议实施顺序

1. **建立当前构建的 C1/C2/C4 基线与正确性入口。**
   先覆盖不同长度、反向提交、取消、槽位补充、一个请求刷新而另一个继续生成；
   再使用冻结 DeepSWE 输入，保持每请求 K128K/G64K。
   逐级增加历史至 128K/256K，保留全部失败、短输出和截断。
   由同一构建提供 none/MTP/DFlash 同策略对照；默认 C1 与 CB C1 分列。
   联合 prefill 保持关闭；草稿合批、宽 Q8 候选分别 A/B，不能一次全部开启归因。

2. **优先测量并优化联合目标验证。**
   单请求 verify 约占 decode 的 81%，所以值得首先排查；此比例不是并发实测。
   检查 ragged 长历史 FP8 attention 的分块与规划、Q8 线性层权重重用，以及
   不同请求长度/接受宽度变化时的数值与恢复。根据 C2/C4 profiling 决定实施顺序。

3. **拆解 prefill/查询/刷新造成的队首等待。**
   先让可靠的独立 prefill 在安全块边界 yield，再研究联合 prefill。
   私有查询需要独立的请求内状态与调度事件；这不等于实现 thinking 摘要 sub-agent。
   刷新提交、回滚和取消必须保持事务语义，不能把共享 CUDA 后端同时交给多线程。

4. **按实测占比继续草稿 GEMM、会话复用和资源管理。**
   会话/缓存复用是进入真实长任务服务测试前的必要工作。
   CPU/NVMe 配额、GPU 准入及 bounded repository 多会话独立验证；不能通过删除
   现有拒绝条件代替实现。C2/C4 有稳定收益后再拓展 C8/C16。

## 计时规则

- **服务总吞吐**：所有请求主回答输出 token / 从首个提交到最后一个完成的墙钟时间，
  排除模型加载，包含请求期间的 prefill、刷新和调度等待。
- **decode 诊断吞吐**：先独立定义统一 decode 测量窗口；报告该窗口内实际主回答
  token、墙钟时间和其他阶段重叠，不能把重叠请求的 decode 秒数相加当总吞吐分母。
- 单独报告每请求速度、TTFT、TPOT/token 间隔 P50/P95、最长停顿、草稿接受率、
  实际批宽、验证/草稿/查询/重选/等待耗时、GPU 整卡峰值和 CPU/pinned 内存。
- 现有 `continuous_batching_benchmark.py` 的 `tokens_per_s` 使用组墙钟时间，
  但 `decode_tokens_per_s` 是输出除以逐请求 decode 秒之和；后者不能当并发总吞吐。
- 以相同输入集合的组墙钟吞吐比较 C1/C2/C4，不相加各请求 tok/s，
  不把主回答之外的私有查询 token 算入服务产出。真实 agent 完成时间另计工具耗时。

源码定位：`src/qwen_native_backend.cpp` 的 `run_continuous_controlled_requests()`、
`src/qwen_executor.cpp` 的构造函数及 `supports_continuous_verify_batch()`、
`src/kernels_cuda.cu` 的 ragged attention、`src/flashinfer_prefill_adapter.cu` 的
独立/ragged 规划、`src/dflash_batch_executor.cpp` 的 `linear()`、
`src/shared_kv_page_pool.hpp`。

历史报告：
[控制器](dflash2_continuous_controller_20260925.md)、
[联合草稿](dflash2_joint_draft_20260925.md)、
[宽 Q8 verifier](dflash2_wide_q8_verifier_20260925.md)、
[联合 prefill 失败记录](dflash2_joint_prefill_20260925.md)、
[单请求最终结果](dflash2_single_request_results_20260927.md)。
