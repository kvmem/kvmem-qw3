# KVMem 效率评测：方法、结果与复现说明

本文记录截至 2026-07-26 已完成的 KVMem 效率实验，重点回答三个问题：

1. KVMem 额外开销主要发生在哪些阶段；
2. Proactive Stage-out、Hierarchical Reuse 和 Packed Rematerialization
   分别消除了多少开销；
3. 当前结果应当如何复现，以及哪些数字还不能作为最终论文结论。

对应实现提交为 `062b403`（`feat(kvmem): optimize tiered window
rematerialization`）。本文件记录的是原始测量结果，不用理论带宽替代实测时间。

## 1. 评测范围

### 1.1 核心消融实验

核心结果来自同一个 AgentLongBench 512K 样本：

```text
stable_sample_id:
adb0765b3c59611b3b9923d5b06e6dfddf70021a16c0ea948c219c40085642e5

request prompt tokens: 515,029
task: Count Frequency (Tool)
reference/final answer: 0
```

四个配置严格顺序执行，每次都重新启动服务，使用相同模型、样本、采样参数和
KVMem budget。四次请求的最终解析答案均为 `0`，评分均为正确。该实验只使用 CPU
tier，不读取 SSD，因此表中的 stage-in/stage-out 数字不包含 NVMe 延迟。

### 1.2 硬件

| 组件 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 97,887 MiB |
| Driver | 580.159.03 |
| CPU | 2 × AMD EPYC 9124，32 physical cores / 64 logical CPUs |
| NUMA | 2 nodes |
| Host memory | 125 GiB |
| Swap | disabled |

### 1.3 共同推理参数

| 参数 | 值 |
|---|---|
| Model | `Qwen3.6-27B-Q8_0.gguf` |
| Backend | qwen-native CUDA |
| `--ctx` | 655,360 |
| KV dtype | FP16 |
| Prefill chunk | 2,048 tokens |
| KVMem context budget | 204,800 tokens（200K） |
| Generation reserve / request max output | 32,768 tokens（32K） |
| Block size | 32 tokens |
| Retrieval | query-conditioned mean-k |
| Sink / recent blocks | 8 / 0 |
| CPU tier | 64 GiB |
| NVMe tier | disabled |
| GPU memory ratio | 0.51 |
| Query replay | enabled |
| Immutable raw-K | enabled |
| MTP | enabled，chain 4 |
| Thinking budget | 4,096 |
| Temperature / top-p | 0 / 0.95 |
| Seed | 20260722 |

核心消融使用 canonical raw-K 构建：驻留但 compact position 改变的 K 也从
position-free raw-K 重新构建，避免低精度 re-RoPE 误差成为性能开关之间的隐藏变量。
在当前提交上复现这一口径时，应设置：

```bash
QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1
```

该设置让任意非零位置变化都触发 raw-K refresh。若不设置，则测试的是当前生产默认的
“最多 8 次原位 re-RoPE 后刷新”策略，结果不应与本文 canonical 表格混合。

## 2. 四组消融如何定义

“all-off”不是历史旧提交，也不是关闭 KVMem。它保留 immutable raw-K、bounded GPU
page pool、正确性检查和公共异步传输基础设施，只关闭论文中定义的三个优化组。

| Cell | `--kvmem-optimize-off` | Proactive stage-out | Hierarchical reuse | Packed rematerialization |
|---|---|---:|---:|---:|
| all-off | `all` | off | off | off |
| proactive-only | `hierarchical-reuse,packed-rematerialization` | on | off | off |
| proactive-plus-reuse | `packed-rematerialization` | on | on | off |
| all-on | 不传 | on | on | on |

相邻两行只增加一个优化，因此相邻差值用于解释该优化的贡献：

```text
all-off
  -> + proactive-stage-out
  -> + hierarchical-reuse
  -> + packed-rematerialization
```

三个优化组的含义如下：

- **Proactive Stage-out**：每个已完成 prefill chunk 提前建立 CPU clean
  backing，让压力点 stage-out 主要变成 page ownership 更新。
- **Hierarchical Reuse**：保留连续两次选择重叠的 GPU blocks，仅 stage-in
  新进入的 blocks；CPU tier 使用 retrieval-aware/heat-aware 策略。
- **Packed Rematerialization**：把零散 host blocks gather 到连续 pinned slab，
  批量 H2D，再由 GPU scatter/re-RoPE；CPU stage-in 与 raw-K assembly 可以重叠。

## 3. 执行方法

### 3.1 一次执行全部四组

GPU 空闲时，从仓库根目录运行：

```bash
cd /home/chaidi/qw3

QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
TAG_PREFIX=agentlongbench_512k_kvmem_core_ablation_$(date +%Y%m%d_%H%M%S) \
scripts/kvmem_eval/run_agentlongbench_perf_ab.sh
```

脚本会：

1. 按上述顺序启动四个独立 qw3 服务；
2. 对同一 sample 运行统一 AgentLongBench evaluator；
3. 以 1 秒间隔记录 `nvidia-smi`；
4. 从 `[kvmem-reselect-perf]`、`[kvmem-assembly-perf]` 和
   `[kvmem-reuse]` 日志提取分阶段耗时；
5. 生成 JSON 和 Markdown 汇总。

本机四组完整执行约需 25 分钟。使用当前 production 8-remap 默认进行新测试时，去掉
`QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1`。

### 3.2 只执行部分 cell

```bash
CELLS=all-off,all-on \
QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS=1 \
TAG_PREFIX=kvmem_perf_quick_$(date +%Y%m%d_%H%M%S) \
scripts/kvmem_eval/run_agentlongbench_perf_ab.sh
```

可用 cell：

```text
all-off
proactive-only
proactive-plus-reuse
all-on
```

### 3.3 输出位置

默认输出：

```text
/data/chaidi/kvmem_eval/results/${TAG_PREFIX}_${CELL}/
/data/chaidi/kvmem_eval/logs/${TAG_PREFIX}_${CELL}_server.log
/data/chaidi/kvmem_eval/logs/${TAG_PREFIX}_${CELL}_runner.log
/data/chaidi/kvmem_eval/logs/${TAG_PREFIX}_${CELL}_gpu.csv
/data/chaidi/kvmem_eval/results/${TAG_PREFIX}_performance_summary.json
/data/chaidi/kvmem_eval/results/${TAG_PREFIX}_performance_summary.md
```

汇总器也可以单独运行：

```bash
scripts/kvmem_eval/summarize_kvmem_perf_ablation.py \
  --log-root /data/chaidi/kvmem_eval/logs \
  --results-root /data/chaidi/kvmem_eval/results \
  --cell all-off=TAG_ALL_OFF \
  --cell proactive-only=TAG_PROACTIVE \
  --cell proactive-plus-reuse=TAG_REUSE \
  --cell all-on=TAG_ALL_ON \
  --output-json /tmp/kvmem_perf.json \
  --output-markdown /tmp/kvmem_perf.md
```

## 4. 指标口径

- **TTFT total**：从提交请求到第一个 reasoning token。它已经包含完整 prefill 和
  所有 reselection，不能再把 reselection 加到 TTFT 上。
- **Reselection total**：本次请求中所有 `[kvmem-reselect-perf] total_ms`
  的和，是 TTFT 的子集。
- **Stage-out / stage-in / assembly**：reselection 内部诊断计时。stage-in 和
  assembly 在 all-on 中可以重叠，所以子项不保证严格相加等于 reselection total。
- **TTFT minus reselection**：`TTFT - sum(reselection total)`，只是用于检查控制
  变量的残差，不是单独插桩得到的 kernel timer。
- **Post-TTFT generation**：首个 reasoning token 之后的 reasoning + final answer。
- **Peak GPU**：1 秒采样一次的进程级 `nvidia-smi used_memory` 最大值；极短暂的峰值
  可能被漏采。

每个 cell 都发生 12 次 reselection：

```text
11 × explicit pressure reselection
1  × final semantic/query-conditioned reselection
```

触发位置在四组中完全相同：

```text
235520, 266240, 296960, 327680, 358400, 389120,
419840, 450560, 481280, 512000, 515029,
以及最终 515029 semantic reselection
```

## 5. 核心结果

### 5.1 Request-level latency

| Cell | TTFT (s) | Reselection within TTFT (s) | TTFT minus reselection (s) | Post-TTFT generation (s) | Full request (s) | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| all-off | 341.8 | 64.0 | 277.9 | 37.7 | 379.5 | 48,062 |
| proactive-only | 337.8 | 58.8 | 279.0 | 37.3 | 375.1 | 48,004 |
| proactive-plus-reuse | 309.9 | 30.7 | 279.2 | 37.8 | 347.7 | 48,004 |
| all-on | 287.2 | 8.5 | 278.7 | 37.8 | 325.0 | 48,320 |

非重选残差始终在 277.9--279.2 秒，说明四组的模型 prefill 主体基本稳定。

从 all-off 到 all-on：

- reselection：`63.957 s -> 8.488 s`，降低 **86.73%**，即 **7.53×**
  working-set-management speedup；
- TTFT：`341.844 s -> 287.176 s`，降低 **15.99%**；
- full request：`379.530 s -> 324.973 s`，降低 **14.37%**；
- peak GPU 只增加 258 MiB（0.54%）。

### 5.2 Reselection breakdown

以下均为一次请求内 12 次事件的累计时间：

| Cell | Selection (s) | Stage-out (s) | Stage-in wall (s) | Assembly (s) | Reselection total (s) | GPU reuse / natural overlap |
|---|---:|---:|---:|---:|---:|---:|
| all-off | 0.033 | 3.645 | 29.039 | 31.239 | 63.957 | 0% |
| proactive-only | 0.039 | 0.268 | 28.995 | 28.992 | 58.843 | 0% |
| proactive-plus-reuse | 0.053 | 0.042 | 1.613 | 28.478 | 30.730 | 100% |
| all-on | 0.051 | 0.050 | 0.310 | 7.901 | 8.488 | 100% |

### 5.3 每个优化解决了什么

**Proactive Stage-out**

```text
stage-out: 3.645 s -> 0.268 s
reduction: 92.65% (13.60×)
```

它基本移除了压力点同步 CPU copy，但不减少下一窗口需要加载的数据，所以 stage-in
仍约 29 秒，TTFT 只下降约 4 秒。

**Hierarchical Reuse**

```text
stage-in:          28.995 s -> 1.613 s
reselection total: 58.843 s -> 30.730 s
stage-in reduction: 94.44%
```

连续选择中自然重叠的 58,417 个 block events 全部在 GPU 复用；需要 stage-in 的
block events 从 76,798 降至 18,381。该优化主要减少传输量，而不是提高单次 PCIe
传输速度。

**Packed Rematerialization**

```text
assembly:          28.478 s -> 7.901 s
stage-in wall:      1.613 s -> 0.310 s
reselection total: 30.730 s -> 8.488 s
assembly reduction: 72.26% (3.60×)
```

该优化把不可避免的 raw-K refresh 组织成跨 block 的 CPU gather、批量 H2D 和 GPU
scatter/re-RoPE pipeline。all-on 中 assembly 仍占 reselection 的约 93.1%，是下一阶段
最值得继续优化的剩余瓶颈。

## 6. GPU memory 对照

使用 `scripts/kvmem_eval/run_dense_memory_baseline.sh` 做了不启用 KVMem 的 FP16/MTP-4
近似活动容量对照：

| 配置 | Logical input/history | Active/prefilled tokens | Peak GPU (MiB) |
|---|---:|---:|---:|
| Dense, no KVMem | 235,520-token prompt | 235,520 | 47,854 |
| KVMem all-on | 515,029-token history | 200K context + 32K reserve | 48,320 |

KVMem 在保存约 515K 逻辑历史的同时，峰值仅比该 dense 对照高 466 MiB（0.97%）。
这个比较用于说明 GPU bounded working set，不是吞吐量比较：两条请求的 logical prompt
长度不同，而且 dense 对照比 KVMem 的 232K active capacity 少一个 2K chunk。

复现 dense memory control：

```bash
cd /home/chaidi/qw3
scripts/kvmem_eval/run_dense_memory_baseline.sh
```

原始结果：

```text
/data/chaidi/kvmem_eval/results/
  dense_fp16_mtp4_232k_memory_baseline_20260726/memory_summary.json
```

## 7. Bounded re-RoPE 的性能/准确率权衡

核心四组表使用 canonical raw-K rebuild。之后又测试了允许驻留 block 原位
re-RoPE 的策略：

| 配置 | 9-sample accuracy | Mean TTFT (s) | Mean assembly/request (s) |
|---|---:|---:|---:|
| FP16 canonical raw-K | 5/9 | 282.700 | 7.020 |
| FP16, refresh after 32 remaps | 4/9 | 277.562 | 1.183 |
| FP8, refresh after 8 remaps | 4/9 | 294.217 | 0.804 |

这不是严格的效率消融：三组在 `temperature=0.6` 下生成长度不同，FP8 还改变了
KV dtype。它只能说明原位 re-RoPE 能显著降低 assembly，不能据此接受一个阈值。

确定性单样本控制进一步发现：

- FP16/32-remap：assembly `8.033 s -> 1.233 s`，但 canonical 正确答案 `1`
  被改成错误答案 `0`；
- FP8/8-remap：assembly `4.614 s -> 0.795 s`，bounded 与 canonical 都回答
  `true`，最终文本一致。

因此 FP16=32 已被否决。当前提交将 FP16 和 FP8 默认都设为最多 8 次；**截至本文，
最终 FP16/8-remap profile 尚未完成同样的端到端性能消融**，不能把 FP16/32 的
1.183 秒 assembly 当作当前默认结果。

数值 sweep：

```text
/data/chaidi/kvmem_eval/results/rope_remap_drift_20260726/metrics.txt
```

## 8. 当前实现的完整效率优化清单

本节给出截至提交 `062b403` 的完整实现盘点。这里需要明确区分三种概念：

1. `--kvmem-optimize-off` 只控制论文消融中的三个高层优化组；
2. 正确性、可扩展性和有界内存所需的公共基础设施在四个 cell 中始终保留；
3. FP8、prefill chunk、FlashInfer workspace 等属于运行 profile，不由三个
   `optimize-off` 开关控制。

因此，`--kvmem-optimize-off all` 的准确含义是“关闭三个论文级性能组”，不是
“回到最早历史提交，也不是关闭所有 KVMem 优化”。这样设计可以保证消融只改变
数据移动和重构策略，不改变检索结果、KV dtype、内存容量或正确性路径。

### 8.1 三个论文级优化组

| 优化组 | 默认 | 主要解决的问题 | 当前包含的实现 |
|---|---:|---|---|
| Proactive Stage-out | on | 压力点同步 D2H/CPU/SSD 写入阻塞 prefill | chunk 完成后预写回；GPU gather + packed D2H；有界 pinned slabs；异步 CPU admission；配置 SSD 时后台持久化；clean-backing ownership 与压力点 metadata-only release |
| Hierarchical Reuse | on | 相邻 reselection 重复加载相同 blocks | GPU set-difference reuse；只加载新进入 blocks；inclusive CPU/SSD clean backing；retrieval-aware heat/frequency CPU admission；CPU 优先、SSD 只处理 cold misses |
| Packed Rematerialization | on | 零散小传输和逐 block re-RoPE/assembly | block-major raw-K；跨 block CPU gather；连续 pinned H2D；GPU batched scatter+RoPE；双 buffer；persistent worker pool；V stage-in 与 raw-K assembly 重叠；SSD span coalescing；RoPE sin/cos table |

三个组的代码控制入口是：

- CLI 解析：`src/qw3_cli.cpp`；
- 默认 all-on 和 legacy profile 映射：`src/qwen_native_backend.cpp`；
- 有效能力检查及 `[kvmem-opt-status]` 日志：`src/qwen_executor.cpp`；
- 选择差分、reuse 和 raw-refresh plan：`src/kvmem_store.cpp`。

如果硬件或 tier 配置无法满足一个默认开启的优化，当前非 legacy 路径会明确报错，
或者在 context 全部驻留 GPU 时记录 `not-applicable`，不会静默把 all-on 降级成
另一种实现。

### 8.2 四组实验共同保留的公共优化

以下机制已经实现并在当前普通 all-on 路径中使用，但没有被当作第四、第五个论文
消融组。核心四组表也共同保留它们。

| 机制 | 当前状态 | 效率作用 | 为什么不由三个开关控制 |
|---|---|---|---|
| Bounded GPU KV page pool | tiering 生效时默认 | 让 active K/V 随 `budget + generation reserve` 有界，而不是随逻辑 history 增长 | KVMem 可扩展性的基本资源约束 |
| Page-table working-set assembly | 默认 | 通过重排 physical page IDs 组装 compact window，不把全部 selected K/V 再复制到一个 dense cache；只处理必须加载或换位的 pages | KVMem attention 的基本数据布局 |
| Step/chunk-level scheduling | 默认 | 在 semantic/pressure boundary 执行选择，在 prefill chunk 边界维护状态，而不是每 token 运行完整 memory scheduler | 避免选择和 I/O 调度成为 token-level 开销 |
| Headroom-aware prefill chunking | bounded pool 生效时默认 | 在页池剩余空间内尽量保持最高 2048-token matmul 宽度，只在下一 chunk 无法安全落页时缩小；修复早期进入 spill 后退化成 16/32-token tiny batches 的约 7× prefill cliff | bounded pool 的执行安全与吞吐基础 |
| Immutable raw-K + 单 GPU working K | 默认 | CPU 保存 position-free K authority；GPU 只保存一份活动 K，去掉额外 working-K 镜像，并避免冷 block 多次有损旋转 | 正确性和显存布局基础 |
| V-only lower-tier records | immutable 模式默认 | standard-attention K 从 raw-K 重建，CPU/NVMe spill record 主要保存 V，约减半普通 spill bytes | 与 immutable authority 的存储格式绑定 |
| Demand-allocated raw-K | 默认 | 按实际 ingest 的 chunks 分配 CPU raw-K，而不是按整个 `--ctx` 预分配 | 主机内存可扩展性基础 |
| Sparse/lazy CPU V slabs | 默认 | CPU spill slots 按约 64 MiB pageable slabs 按需分配，并与 raw-K 共用严格 `--kvmem-cpu-gb` 预算 | 防止每 block 分配和 host RAM 失控 |
| Bounded reusable transfer slabs | 默认 | stage-in/out 共用有界 GPU staging，host 使用固定数量 pinned slabs，避免按整个 selection 创建临时 tensor | 保证性能 pipeline 不重新制造显存峰值 |
| Persistent CPU workers | tier pipeline 生效时默认 | 复用 gather/scatter worker，消除每次 reselection 创建线程的开销 | 三个高层组共享的执行基础设施 |
| RoPE sin/cos lookup table | CUDA immutable 路径默认 | 预计算模型位置范围内的 FP32 sin/cos，避免 remap kernel 反复执行 `powf/sincosf` | assembly 的公共数值实现 |
| Bounded in-place re-RoPE | 当前默认最多 8 次 | 驻留 block 小幅换位时复用 working K；超过次数/位移阈值或 cold stage-in 时从 raw-K 刷新 | 同时涉及数值正确性，不能由传输消融隐式改变 |
| MTP bounded sibling pool/local positions | MTP+tiering 时默认 | MTP KV 与主模型一起有界，并使用 compact-window 位置，避免完整逻辑 ctx 的 MTP cache | 长上下文 MTP 的资源和位置正确性约束 |
| Split prepare/finish reselection | 默认 | 将 plan/issue 与 wait/assembly 分开，使 NVMe read/H2D 能与独立的 MTP prefix rebuild 等计算重叠 | 调度接口基础 |
| Incremental mean-K capture | query-conditioned mean-K 默认 | 在每个 prefill block 的 K 刚生成时构建 position-free mean，避免之后重新 stage-in 全历史 K 来建索引 | full-history retrieval 的计算/I/O 基础 |
| Scalable mean-K scorer | 自动 dispatch | `<=8192` pages 用 fused CTA；更大输入用 exact tiled one-dot，突破单 CTA shared-memory 上限 | 检索语义正确性和长上下文可扩展性要求 |
| FP16 mean-K index | 生产默认 | active KV 即使是 FP8，mean-K 仍用 IEEE FP16；相对 FP32 减半索引显存且避免 FP8 二次量化改变排名 | retrieval 表示的统一基线 |
| FP16/bounded query capture | mean-K CUDA 默认 | 长 query 放 pageable host backing，GPU 只保留默认 256-token scoring stage 和两个小 bounce buffers | 防止真实 query 长度使显存线性增长 |
| FlashInfer prefill-plan reuse | 默认 | 相同 prefill shape 的 16 个 normal-attention layers 共用一次 plan；典型为 1 miss + 15 hits | 通用 attention scheduler 优化 |
| MTP prefix KV-only fast path | MTP 默认 | accepted-token prefix prime 只追加 MTP K/V，跳过结果不会被读取的 attention/residual/FFN，将每 token 的历史依赖从 \(O(\mathrm{ctx})\) 降到 \(O(1)\) | qw3/MTP 公共执行优化，非三个 tiering 组之一 |
| Cold host allocation reuse | 当前 Opt3 公共基础设施默认 | 独立长请求之间保留无效但有界的 host slab 拓扑，避免重复 free/fault/trim | 请求级 allocator 优化，不改变一次 reselection 算法 |
| Capacity validation and page-cache bounding | 默认 | 启动时检查 CPU+NVMe spill 容量；buffered NVMe I/O 后 range writeback/`DONTNEED`，避免内核 page cache 复制整个 SSD arena | 防止性能方案在长请求上退化为 OOM |

其中 scalable scorer 的当前生产大输入路径与早期实现不同。早期 tiled scorer 为取得
online-softmax 分母需要两遍 dot；现在的大输入默认路径先把每个 \(Q\cdot\bar K\)
logit 写入有界 workspace，再执行归一化和累加，因此每个 dot 只计算一次。旧的
two-dot 路径仍保留为测试/A-B baseline。

### 8.3 显存和运行 profile 优化

这些优化已经实现，但是否开启由实验 profile 决定，而不是 KVMem 三个性能开关。

| Profile 项 | 当前建议/状态 | 已测效果或约束 |
|---|---|---|
| FP8 active K/V | 通过 `--kv-dtype fp8` 选择；mean-K/query 仍为 FP16 | 200K+32K、chunk 2048 的单样本峰值约从 47.1 GiB 降到 39.9 GiB；需要数据集级精度验证 |
| Prefill chunk 2048 | 当前共享 GPU/准确率脚本显式设置 | 相比被 MTP 隐式覆盖为 4096，主 scratch 及 staging 约减少 1.2--1.25 GiB |
| MTP 复用主 prefill scratch | 已实现，串行安全复用 | chunk 2048 约减少 0.8 GiB；旧 4096 宽度下约 1.6 GiB |
| FlashInfer workspace 192 MiB | 当前单请求脚本设置；代码通用默认仍为 512 MiB | 相对 512 MiB 减少 320 MiB；不足时重试 non-split plan |
| 200K 而非 224K context budget | 仅显存受限 profile | FP16 active KV 约少 1.59 GiB，但已观察到 retrieval accuracy 下降，不是正式准确率默认 |
| FP8 更大 prefill chunk | 可选性能点，不是当前共享 GPU 默认 | 8192 相比 FP8/2048 prefill 快约 7.26%，但使用更多 scratch、降低显存余量 |

完整显存推导、48 GiB profile 和对应准确率控制记录在
`docs/kvmem_gpu_memory_optimization.md`。

### 8.4 SSD/NVMe 路径已经实现到什么程度

当前 SSD 路径不是纯设计稿，已经实现：

- 独立 copy stream 和后台 host workers；
- chunk 级 proactive write-through；
- CPU inclusive admission 与 SSD persistence 并行；
- CPU hit 的 packed H2D 与 SSD miss 的批量 positional reads 共存；
- 相邻 file slots/buffer spans 合并为较少的 `pread`/`pwrite`；
- SSD stage-in 可与后续 raw-K/计算阶段重叠；
- page-cache range writeback 和 `POSIX_FADV_DONTNEED`；
- 有界 read/write slab pools 和 queue depth。

但底层 `NvmeKvTier` 目前仍使用 buffered positional `pread`/`pwrite`。异步性来自
后台 worker 和上层流水线，而不是 kernel-native async I/O。以下方案尚未实现：

- `io_uring`；
- `O_DIRECT`；
- GPUDirect Storage；
- 自动硬件校准和跨层预测预取。

因此 SSD benchmark 可以证明当前 worker/coalescing pipeline 的效果，不能被表述成
已经完成了 direct-I/O 或 GDS。

### 8.5 已实现但默认关闭或未推广的性能路径

| 路径 | 状态 | 原因 |
|---|---|---|
| Query-first-pass speculative prefetch | 实现保留，默认 off | 固定样本只命中 213/410，额外传输 197 blocks；ordinary stage-in 已被 assembly 隐藏，没有改善 TTFT |
| Incremental mutable-MTP assembly | 实验开关保留，默认 off | temperature=0 A/B 改变生成轨迹；默认从 immutable raw authority 重建 MTP K |
| FP16 32-remap threshold | 已否决 | 虽明显降低 assembly，但确定性样本由正确变错误 |
| Native fused paged FP8 attention | 未合入 | 当前 FlashInfer 架构/shape 不支持所需 Blackwell paged FP8 实例；现路径直接读取 E4M3 并在消费 kernel 转换 |
| Direct-delta RoPE kernel | 未推广 | table+pipeline 后 GPU RoPE 已不是主要瓶颈，引入不同数值路径收益不足 |

Query replay 本身默认开启，但它是**准确率修复**：retrieval 后重新计算 query hidden
state。它通常增加计算量，不应写成效率优化。相同地，canonical raw-K rebuild 是
正确性对照，不是最快生产路径。

### 8.6 实现与专项证据索引

| 主题 | 主要实现 | 专项记录 |
|---|---|---|
| Stage-out/writeback | `src/qwen_executor.cpp`, `include/qw3/nvme_kv_tier.hpp` | `docs/kvmem_cpu_proactive_writeback_benchmark_20260725.md`, `docs/kvmem_ssd_writeback_benchmark_20260724.md` |
| Stage-in/packed transfer | `src/qwen_executor.cpp`, `src/kernels_cuda.cu`, `include/qw3/device_backend.hpp` | `docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md` |
| Assembly/re-RoPE | `src/qwen_executor.cpp`, `src/kernels_cuda.cu` | `docs/kvmem_assembly_rerope_optimization_benchmark_20260724.md` |
| Retrieval scoring | `src/qwen_executor.cpp`, `src/kernels_cuda.cu` | `tests/kvmem_softmax_pages.cu`, `docs/kvmem_implementation_notes.md` |
| FP8/chunk/plan cache | `src/flashinfer_prefill_adapter.cu`, `src/kernels_cuda.cu` | `docs/kvmem_fp8_performance_benchmark_20260725.md` |
| GPU/host memory | `src/qwen_executor.cpp`, `include/qw3/pinned_kv_tier.hpp` | `docs/kvmem_gpu_memory_optimization.md` |
| SSD architecture status | `include/qw3/nvme_kv_tier.hpp`, `src/qwen_executor.cpp` | `docs/kvmem_nvme_ssd_architecture.md`, `docs/kvmem_ssd_complete_design.md` |

## 9. 原始结果索引

核心四组汇总：

```text
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_canonical_four_20260726_performance_summary.json
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_canonical_four_20260726_performance_summary.md
```

四个 cell：

```text
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_coldkfix_v3_20260726_all-off/
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_canonical_ab_mid_20260726_proactive-only/
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_canonical_ab_mid_20260726_proactive-plus-reuse/
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_kvmem_coldkfix_v3_20260726_all-on/
```

re-RoPE accuracy/performance controls：

```text
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_canonical_rawk_accuracy9_20260726/
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_rerope_fp16_n32_accuracy9_20260726/
/data/chaidi/kvmem_eval/results/
  agentlongbench_512k_rerope_fp8_n8_accuracy9_20260726/
```

相关详细文档：

- `docs/section_43.tex`：三个优化的论文设计描述；
- `docs/kvmem_gpu_memory_optimization.md`：显存构成和 48 GiB profile；
- `docs/kvmem_cpu_proactive_writeback_benchmark_20260725.md`：CPU-only
  proactive writeback；
- `docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md`：packed
  D2H/H2D 和 CPU gather；
- `docs/kvmem_assembly_rerope_optimization_benchmark_20260724.md`：RoPE
  table 和双 buffer raw-K assembly；
- `docs/kvmem_ssd_writeback_benchmark_20260724.md`：CPU+SSD mixed pipeline；
- `docs/kvmem_fp8_performance_benchmark_20260725.md`：FP8 和 prefill-chunk
  性能控制实验；
- `docs/kvmem_ssd_complete_design.md`：SSD/NVMe tier 设计。

## 10. 结果限制与下一步

当前核心消融具有以下限制：

1. 只有一个 512K sample，尚未报告多样本均值、方差和置信区间；
2. 四组顺序执行，没有交错随机化，可能受到温度、频率或系统背景负载影响；
3. GPU memory 以 1 秒采样，可能漏掉极短峰值；
4. CPU-only 结果不能直接代表 NVMe/SSD stage-in；
5. canonical raw-K 表格不是当前 bounded 8-remap 默认的最终吞吐结果。

论文正式报告前，建议：

1. 固定 GPU clocks/persistence mode，并记录功耗和时钟；
2. 至少选择 10 个不同任务类型的 512K 样本；
3. 每个 cell 重复 3 次，交错执行并报告 mean、standard deviation 和 p95；
4. 分别报告 CPU-only 和 NVMe-backed 两组；
5. 对当前 FP16/8-remap 默认重新执行四组消融，同时保留 canonical raw-K
   作为准确性参考。
