# KV Memory 选择性重计算

此实现借鉴 [CacheBlend](https://arxiv.org/html/2405.16444v3) 的“保留大部分 KV、只重算少量历史 token”思路，面向当前 Qwen 混合注意力执行器。它是可开关的实验路径，默认关闭，不声称等价于原论文，也不保证优于原有 KV Memory。

## 执行方式

在 `kvmem_begin_query_replay()` 安装选中窗口之后设置待重算标记。第一次真实 forward 前：

1. 排除未改变的连续历史前缀和本来就会完整 replay 的 query suffix。
2. 复用 `KvMemBlock::retrieval_score`，按分数选完整检索块。当前常用配置为每块 32 tokens。比例和绝对 token 上限共同限制预算；不额外计算一轮全窗口 token 分数。
3. 将选中块按窗口位置排序，合并历史和窗口位置都相邻的区间。
4. 复用 `forward_n_tokens()` 和现有分页因果注意力内核，在每个区间的实际窗口位置覆盖 KV。计算只涉及选中的 token；attention 仍读取其前方的完整选中窗口。
5. 恢复原 query 边界的隐藏状态、DeltaNet/conv 状态及逻辑位置，然后执行原有完整 query replay 和生成流程。

plain、MTP、incremental session 和 semantic-chunk 只要经过该 query replay 入口，就共用这一实现。纯容量收缩和不使用 query replay 的路径不会额外触发重计算。未启用时不保留历史 token，也不分配备份。

## 为什么保留为实验选项

CacheBlend 原论文通过层间 KV 偏差挑选 token。本实现复用 query 检索分数，以检索块为计算粒度，因此属于 query-guided block recomputation。这样无需新打分内核，也无需实现不连续 query 行的稀疏 attention 内核。

Qwen 的 DeltaNet 层有顺序依赖，无法像纯 Transformer 那样任意跳过历史而维持精确循环状态。本实现临时将 recurrent/conv 清零，让被选择的文本按历史顺序通过循环层；完整 attention 仍读原窗口的因果前缀。临时循环状态最后丢弃，query 使用原来的边界状态。这是显式近似，不能解释为完整 compact prefill 的等价实现。

## 原始缓存与资源

- 重算只覆盖 GPU 工作窗口；不写入 immutable raw K、全历史检索索引、CPU/NVMe backing 或 MTP 历史 KV。
- 每层使用两个按原始 dtype 分配的连续备份张量，只备份被修改的物理行。重排、stage-out、历史截断和可能涉及旧块的 proactive writeback 前恢复原始行。
- 普通 checkpoint 和 transient MTP snapshots 只保存计算状态，不复制历史 KV，因此保留当前覆盖，让 query replay 和后续 decode 使用重算后的窗口。
- 文本 token 按历史块保留在 host，用于 warm 请求和会话重算。bounded repository 模式在选窗处按 live 历史块清理，物理 slot 复用不作为历史身份。
- archive 加载后没有在本进程计算过的历史 token 不可重算，以 `missing_source_tokens` 记录。当前不扩展 archive 格式。
- 当前明确拒绝 M-RoPE 多模态输入和 Q8_KV 块量化布局；FP16、FP8 使用现有分页计算路径。BF16/F32 分支未作为本次模型验收配置。

## 使用

```bash
QW3_KVMEM_BLEND_RATIO=0.10 \
build/qw3 serve ...
```

`RATIO` 范围为 `[0,1]`，默认 `0`。本次按用户指定使用 `0.10`，即当前 Query 召回窗口约 10% 的 token，按已有检索分数选择完整块。32K 窗口、32-token 块时为 3264 tokens（102 块，9.96%）。不足一块不重算。可选 `QW3_KVMEM_BLEND_MAX_TOKENS` 仅用于额外限制，默认 `UINT32_MAX`，不压低比例；设置时实际预算为 `min(floor(窗口 token 数 × ratio), max_tokens)`。

日志 `[kvmem-blend]` 包含实际 tokens、window_tokens、blocks、runs、missing_source_tokens 和 ms。ms 包括该调用内备份、状态操作和重算，不包含未来恢复临时覆盖的耗时，也不是完整请求 TTFT。

## 验证与复现

实现位于 `src/kvmem_blend.cpp`；选择策略为 `src/kvmem_blend_policy.hpp`；executor 复用原有 forward、query replay、checkpoint 和 tier 生命周期。

```bash
cmake -S . -B build
cmake --build build -j 8 --target qw3 qw3-kvmem-blend-policy qw3-kvmem-blend-model-probe
ctest --test-dir build -R '^qw3-kvmem-(blend-policy|store|prefix-reuse-policy|refresh-policy|request-plan|archive)$' --output-on-failure
QW3_KV_DTYPE=fp8 build/qw3-kvmem-blend-model-probe models/Qwen3.5-4B-Q8_0.gguf
QW3_KV_DTYPE=fp16 build/qw3-kvmem-blend-model-probe models/Qwen3.5-4B-Q8_0.gguf
QW3_TEST_BOUNDED=1 QW3_KV_DTYPE=fp8 build/qw3-kvmem-blend-model-probe models/Qwen3.5-4B-Q8_0.gguf
```

模型 probe 检查目标 KV 确实变化、其他物理行不变、恢复 source KV 精确一致、immutable raw K 不变、未来 KV 扰动不能影响历史重算，以及 recurrent/conv、hidden、page table、历史/窗口位置一致。bounded FP8 配置还强制触发容量压力，验证入库后的源 token 保留及 checkpoint 不提前撤销重算覆盖。

四类大海捞针复用已有冻结 RULER-family 数据和 scorer：`niah_single_2`、`niah_multikey_2`、`niah_multivalue`、`niah_multiquery`。模型 Qwen3.8-27B-Q8_0，FP8 KV，MTP4，32K 检索/prefill 预算，先盲 ingest、再给 query，完整 query replay；64K 输入，深度 10/50/90，seed=1000，关闭/开启各 12 次。两组统一 thinking budget=4096、max output=8192，其他配置继承冻结参考实验。

```bash
.venv/bin/python -u scripts/kvmem_eval/run_cacheblend_niah.py \
  --reference /data/chaidi/qw3-experiments/ruler_niah_rerope_ablation_20260925 \
  --out /data/chaidi/qw3-experiments/cacheblend_20260925/niah64k_r10 \
  --binary /data/chaidi/qw3-experiments/cacheblend_20260925/bin/qw3
```

结果位于 `/data/chaidi/qw3-experiments/cacheblend_20260925/`，包含 binary、配置、逐样本回答、模型日志、GPU 并行进程记录和回归测试日志。本次为小样本诊断，不是完整官方 RULER 分数。另一组实验同时占用 GPU，wall time 不作为独占 GPU 的性能结论。

## 验收结果

2026-09-26 完成全部 24 次有效运行，进程正常退出，本任务启动的服务已停止。每格包含深度 10%、50%、90% 的三个样例：

| 测试 | 关闭重算：全答对 | 开启 10%：全答对 | 两组答案覆盖率 |
| --- | ---: | ---: | ---: |
| 单针 `niah_single_2` | 3/3 | 3/3 | 100% |
| 多键 `niah_multikey_2` | 3/3 | 3/3 | 100% |
| 多值 `niah_multivalue` | 3/3 | 3/3 | 100% |
| 多查询 `niah_multiquery` | 3/3 | 3/3 | 100% |

24 个输出的数字集合也全部与答案精确一致，均正常 `stop`，没有达到输出长度上限。12 次开启运行均实际重算 102 块、3264 tokens；query 边界前的窗口为 32704 或 32736 tokens，实际比例 9.97%–9.98%，`missing_source_tokens=0`。没有把正常 query replay 的 token 计入重算预算。

6 项 CPU 回归测试全部通过。真实模型 FP16、FP8 和 bounded FP8 的三组 probe 均通过；最后一组包含压力入库、源 token 保留和 checkpoint 生命周期回归。

在本次共享 GPU 条件下，重算阶段平均 5741.71 ms，中位数 5740.40 ms，范围 4422.64–6723.69 ms。每次 102 块合并为 49–65 次连续 forward。回答请求的平均 wall time 为关闭 5.22 s、开启 11.55 s；它包含选窗、query replay 和生成，不含之前的盲 ingest，也不是 TTFT。并行任务和输出长度会影响这些数值，不能由此得出独占 GPU 的固定减速倍数。

这轮样例在关闭时已达到满分，因此只能证明这些样例上的功能和缓存一致性通过，不能证明重算提高了质量。实现通过复用检索分数、合并连续块、每层连续 KV 备份和既有 forward/attention 路径控制额外工作，没有新增 CUDA 内核；但多次 forward 和混合循环层仍有明显成本，不能声称已经实现理论最优效率。保留默认关闭，使用上述环境变量显式开启 10%。

### 结果来源与可审计性

完整结果与核验文件：

- [逐样本结果](/data/chaidi/qw3-experiments/cacheblend_20260925/niah64k_r10/results.json)
- [分组汇总](/data/chaidi/qw3-experiments/cacheblend_20260925/niah64k_r10/summary.json)
- [验收审计与耗时统计](/data/chaidi/qw3-experiments/cacheblend_20260925/niah64k_r10/acceptance-audit.json)
- [运行二进制身份](/data/chaidi/qw3-experiments/cacheblend_20260925/niah64k_r10/binary-identity.json)
- [仅本任务的代码补丁](/data/chaidi/qw3-experiments/cacheblend_20260925/implementation.patch)

关闭组运行时，开启路径继续修正了 checkpoint 覆盖保留、reset 恢复和源 token 记录时机；这些修正不改变关闭路径的数值运算，但两组二进制不同，不能宣称是同一个二进制的 A/B。关闭组 SHA256 前缀为 `b7bbbc8677c9`，最终开启组为 `48b6415c4adf`，完整 SHA256、配置及进程身份已保存。首次开启试运行因源 token 缺失触发实现断言，修复后重跑；失败记录保留在 `on/failed-attempts`，不计入最终 24 次验收结果。该失败样例的答案本身正确，排除原因是实现不满足重算覆盖要求。

补丁以开始任务时的文件快照为基准，仅包含本次实现，保留工作区原有修改。源码 SHA256 清单位于同目录的 `source-sha256.json`。

后续已完成 [256K 完整四类测试](kvmem_cacheblend_256k_20260926.md)：开启组 60 个样例，复用历史基线后覆盖率从 60.42% 提高至 90.00%；同配置的 29 个配对子集也观察到改善。上述 64K 结果保持原样。
