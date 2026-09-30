# 27B / DFlash 当前 thinking 独立摘要验证

2026-09-29，补测 [独立摘要执行方式](thinking_summary_isolation_20260929.md)。

## 配置与范围

- Target：`models/Qwen3.8-27B-Q8_0.gguf`。
- Draft：`models/Qwen3.8-27B-DFlash2-BF16`，BF16、5 层、block size 8。
- KV：FP8，page size 16；bounded KVMem，step 更新，CPU 索引。
- 测试服务使用独立 localhost 端口，测试顺序执行，完成后关闭。已有 vLLM 进程保持运行。
- 开始时 RTX PRO 6000 96 GiB 显卡约空闲 47.6 GiB。每 3 秒采样整卡和进程显存，采样峰值不等于高精度瞬时峰值。
- 服务二进制冻结自 `build-dflash-integration/qw3`。本次只扩展测试入口，没有修改运行时实现。

证据目录：`benchmark/thinking_summary_isolation_27b_dflash_20260929/`。其中保留二进制、测试脚本快照、源码 SHA-256、模型路径/文件大小/时间戳、每项命令、退出码、HTTP 响应、服务日志及显存采样。模型文件未额外计算完整权重哈希。

并发功能检查使用小规模资源配置：ctx 8192、K/prefill K 1024、G 2048、repository 256 × 32 tokens、CPU tier 总额 3 GiB、private reserve 4096、max active 2；摘要阈值 64、thinking budget 256 是测试专用。默认阈值串行检查使用阈值 1024、thinking budget 1600、G 4096、CPU tier 1 GiB。它们不代表 128K/256K 上下文或高并发负载已完成验证。

## 已完成的检查

| 检查 | 结果 |
| --- | --- |
| GPU 回滚 | 63、511、1535、4095 token 边界的 recurrent、conv、主 KV、DFlash 缓存和 logits 均与干净参考逐字节一致 |
| 检索 query 恢复 | 四种 query 情况全部通过；warm stash 可用；无效恢复被拒绝 |
| 摘要隔离 | 两种不同 token 和长度的父历史得到相同摘要 token 与 logits；父 KV/recurrent/conv/hidden/logits 和 DFlash 缓存保持不变 |
| 页租约 | 拒绝超容量输入，正常完成及提前放弃均归还全部页面 |
| 双并发 HTTP | 10 个完成请求及一次摘要中途断连通过，5 次成功压缩，样例 280 → 90 tokens |
| 默认 1024 阈值串行 HTTP | 8 个完成请求及一次取消探测通过；长样例 1624 → 154 tokens；streaming/non-streaming、Anthropic Read 工具调用、缓存续接、未闭合 thinking、无效摘要和取消恢复均通过 |
| 并发推进 | 第一轮摘要有 92 次让出，期间同行请求执行 91 次 DFlash verification；两个占满父槽位的请求均完成摘要 |
| 失败与续接 | 1 token 上限导致无效摘要，回滚/重放耗时均为零；短 thinking 跳过；prefix-cache 和 `summary-cache` 会话命中；摘要断连后新请求成功 |

并发服务日志包含实际 `route=dflash`、`enabled=true` 和 `fallback=false`，并出现 136 次双请求 ragged verification。首轮可见摘要保留了 `src/cache.py`、公共 API 不变、测试尚未运行和待检查源代码的事实；后续请求仍回答未运行测试。这是有限样例检查，不是完整任务成功率评测。

两组 HTTP 日志中 DFlash 投机执行均开启且没有 fallback，没有 HTTP 500 或 `terminal_status=error`。默认阈值工具用例本身只有 43 个 thinking tokens，因此工具调用检查属于短 thinking 跳过路径，不能解释为“长 thinking 压缩后的工具调用质量”已完成验证。

四项 GPU 测试均退出 0，完整机器可读汇总在证据目录的 `validation.json`。进程显存采样如下：

| 测试 | 测试进程采样峰值（MiB） | 整卡采样最低空闲（MiB） |
| --- | ---: | ---: |
| DFlash 回滚 | 33288 | 15419 |
| 独立摘要隔离 | 32920 | 16055 |
| 双并发 HTTP | 40038 | 8665 |
| 默认阈值串行 HTTP | 35250 | 13457 |

最高约 39.1 GiB，期间整卡最低仍空闲约 8.5 GiB。测试结束后仅余原 vLLM 进程占用 48514 MiB，整卡空闲恢复为 48715 MiB。此处显存是运行该配置的实测占用，不是摘要功能相对于关闭摘要的增量。

本次为隔离探针增加了可选 DFlash 权重参数及父 DFlash 缓存快照比较；为默认阈值 HTTP 脚本增加 `--dflash` 参数和环境隔离。最终服务二进制与前一轮 4B 验证一致。

## 复现

在仓库目录运行，逐项等待退出，避免多个 27B 测试争用剩余显存：

```sh
build-dflash-integration/qw3-thinking-summary-isolation-probe \
  models/Qwen3.8-27B-Q8_0.gguf models/Qwen3.8-27B-DFlash2-BF16
build-dflash-integration/qw3-thinking-compaction-probe \
  models/Qwen3.8-27B-Q8_0.gguf models/Qwen3.8-27B-DFlash2-BF16
python3 tests/thinking_summary_http_probe.py \
  --binary build-dflash-integration/qw3 \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --dflash models/Qwen3.8-27B-DFlash2-BF16 \
  --continuous --output /tmp/summary-27b-dflash-continuous
python3 scripts/kvmem_eval/test_thinking_compaction_http.py \
  --binary build-dflash-integration/qw3 \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --dflash models/Qwen3.8-27B-DFlash2-BF16 \
  --output /tmp/summary-27b-dflash-default-1024
```

并发探针不加 `--continuous` 即为串行功能检查。HTTP 脚本会清除继承的 `QW3_*` 环境变量，显式配置本次测试开关。

这轮验证用于确认状态隔离、DFlash 回滚和并发生命周期。没有进行吞吐对照，也没有运行完整 DeepSWE/任务成功率评测。
