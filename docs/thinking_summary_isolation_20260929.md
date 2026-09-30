# 当前 thinking 独立摘要与 continuous batching

2026-09-29，工作目录 `/home/chaidi/qw3`。这是一次执行路径改造；吞吐和任务质量尚未做基准对照。

## 行为

主请求生成完整的当前 thinking 后暂停。摘要输入由固定摘要指令、本轮原始 thinking token（去掉末尾 `</think>`）和摘要输出前缀组成。没有主任务 prompt、旧 thinking、工具历史，也不继承父请求的 attention KV、DeltaNet recurrent/conv 或 logits。输入末尾再次要求压缩原文，避免把原文中的任务指令当成新任务继续执行。

临时 `QwenExecutor` 共享目标模型权重，从位置 0 重新 prefill，只用目标模型 greedy 解码。父请求可使用原生 MTP 或 DFlash；摘要自身不加载第二份权重、不运行草稿模型。它是内部摘要任务，不提供通用 agent 工具调用接口。

只有自然结束、格式合法且更短的摘要才会替换原文。成功后恢复 thinking 开始处的父请求快照，截断当前 thinking，再回放摘要和 closing token 以重建状态。无效摘要直接保留父请求结束时的状态，`rollback_ms=0`、`replay_ms=0`。未闭合、太短或工作集已切换的 thinking 沿用原有跳过逻辑。

## GPU 与 CPU 资源

- 摘要在父请求空闲 GPU 页配额和已有 private reserve 内取得独占租约，使用新页表；不能覆盖父请求或其他请求的页面。
- 准入时一次预留摘要全部页面，至少容纳完整输入及一个输出 token，并留出父请求尾页。输入放不下就跳过，不截断 thinking。输出上限还受模型上下文容量、原文长度和替换回放可用仓库槽位约束。
- 摘要结束、取消或异常退出时释放租约，再进行父请求的替换回放。独立 recurrent/conv、logits 和最多 256 token 的 prefill scratch 会增加临时显存；共享权重不代表零额外显存。
- 摘要不创建 CPU/NVMe KV 仓库，host 侧只增加输入/输出 token、文本和必要的日志/格式检查缓冲。父请求自己的历史仓库继续占用原配额；其恢复边界在摘要期间保持保护。
- `QW3_KVMEM_PROBE_RESERVE_TOKENS` 仍控制既有私有页预留（默认 8192）。更长的当前 thinking 可能需要增加它，并相应增加显存预算；实际能否准入还取决于父请求剩余页和其他私有租约。

## 开启

新二进制为 `/home/chaidi/qw3/build-dflash-integration/qw3`。在既有 bounded KVMem、step 更新、MTP/DFlash 命令上增加：

```sh
--thinking-summary --thinking-summary-min-tokens 1024
```

continuous batching 还需要现有完整 controller：

```sh
export QW3_EXPERIMENTAL_BOUNDED_CONTINUOUS=1
export QW3_CONTINUOUS_KVMEM_CONTROLLER=1
export QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL=1
# 父请求使用 DFlash 时：
export QW3_CONTINUOUS_DFLASH=1
```

配合 `--continuous-batching --max-active 2` 及上述摘要参数。采样、session reuse 等开关仍按原配置单独开启。bounded repository、step 更新、隐式完整历史和原生投机执行的前置条件保留；named sessions、archive、显式 transcript replay、inline refresh 等原有限制未放开。

摘要沿用父请求的调度槽位，无需等待额外空槽。每个输入块、摘要 token 和替换回放块完成后通过 controller yield，让其他请求继续执行。两个父请求都进入摘要时也可以轮流推进。当前摘要 token 独立计算，尚未与其他请求联合组成 GPU batch；恢复和仓库截断仍作为完整事务执行。

日志用 `summary_context=isolated-current-thinking` 标识新路径，`private_prompt_tokens` 是整个独立摘要输入的 token 数，`summary_yields` 是该请求摘要及替换回放让出的次数。`summary_ms` 在并发模式下包含等待其他请求的时间，不能直接解释为 GPU 计算时间。

## 验证方法

在仓库目录下运行：

```sh
ctest --test-dir build-dflash-integration-host --output-on-failure
build-dflash-integration/qw3-thinking-summary-isolation-probe \
  models/Qwen3.5-4B-Q8_0-with-MTP.gguf
build-dflash-integration/qw3-thinking-compaction-probe \
  models/Qwen3.5-4B-Q8_0-with-MTP.gguf
python3 tests/thinking_summary_http_probe.py \
  --binary build-dflash-integration/qw3 \
  --model models/Qwen3.5-4B-Q8_0-with-MTP.gguf \
  --continuous --output /tmp/qw3-summary-isolation/http-continuous-4b-prompt
python3 scripts/kvmem_eval/test_thinking_compaction_http.py \
  --binary build-dflash-integration/qw3 \
  --output /tmp/qw3-summary-isolation/http-default-1024-4b-prompt
```

GPU 隔离探针比较两种不同 token、不同长度的父历史：相同当前 thinking 的摘要 token 和全部末次 logits 必须逐字节一致。还检查父请求 KV、recurrent/conv、hidden、logits、位置和仓库历史不变，拒绝超容量输入，以及正常/提前放弃时页面全部归还。

并发 HTTP 探针使用测试专用 64 token 阈值，覆盖摘要期间另一个请求确实继续 verification、两个占满父槽位的摘要、无效摘要、短 thinking、缓存续接及摘要中断连后继续请求。默认 1024 阈值由独立 HTTP 回归覆盖，同时检查 streaming/non-streaming、工具调用、取消和 prefix-cache HIT。

## 本次结果

| 检查 | 结果 |
| --- | --- |
| CUDA / host 构建 | 通过 |
| Host CTest | 44/44 通过 |
| GPU 独立状态与页面归还 | 两种父历史的摘要 token/logits 完全一致；父状态逐字节不变；页面全部归还 |
| 原有事务回滚探针 | 63、511、1535、4095 token 边界全部逐字节一致；四种 query 恢复检查通过 |
| 默认 1024 阈值 HTTP | 通过；长样例 1625 → 126 tokens；8 个记录请求及一次取消探测 |
| Continuous HTTP | 通过；10 个完成请求及一次摘要中途断连；5 次成功压缩，样例 281 → 65 tokens |
| 摘要失败 | 显式 1 token 上限触发回退；父请求无回滚和原文重放 |
| 缓存续接 | prefix-cache HIT；并发 `summary-cache` 会话命中 |

结果位于 `/tmp/qw3-summary-isolation/`：`host-prompt-ctest.log`、`gpu-isolation-4b-prompt.log`、`gpu-rollback-4b.log`、`http-default-1024-4b-prompt/result.json` 和 `http-continuous-4b-prompt/validation.json`。`changes.diff` 对照本次开始前的工作目录快照，避免把已有 DFlash/CacheBlend 改动归入本次修改；`source-sha256.json` 记录最终源码和服务二进制校验值。

第一次默认阈值补测没有成功压缩：仅在原文前放指令时，4B 长样例继续推理直至资源上限，正确回退为原文。增加原文末尾的摘要指令后，上述完整默认阈值和并发回归均通过。初次失败日志保留在 `http-default-1024-4b/`，没有放宽摘要接受条件。这是有限样例验证，不保证任意长 thinking 的摘要质量。

首轮验证时，本机 GPU 被已有 vLLM 服务占用约 81 GiB，因此以上结果使用 Qwen3.5-4B Q8 + FP8 KV + 原生 MTP。随后用户释放出约一半显存，已完成 27B/DFlash 的状态隔离、回滚、双并发和默认阈值 HTTP 补测，详见 [27B/DFlash 验证](thinking_summary_27b_dflash_validation_20260929.md)。两轮检查均不证明完整任务成功率或吞吐提升。
