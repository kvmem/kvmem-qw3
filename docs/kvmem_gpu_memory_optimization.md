# KVMem GPU 显存优化设计与实测记录

更新时间：2026-07-26

本文集中记录 `qw3` 当前为 KVMem 长上下文推理完成的 GPU 显存优化，包括：

- 每项优化解决的具体冗余；
- 当前代码中的实现位置；
- 理论或实测节省量；
- 对 CPU 内存、吞吐和准确率的影响；
- 48 GiB 显存目标下推荐使用的配置；
- 仍会随逻辑上下文长度增长、尚未解决的显存问题。

本文只把真正降低 GPU 峰值或限制 GPU 分配上界的改动称为“显存优化”。
CPU/NVMe 分层、异步 stage-out、预取和 assembly pipeline 如果只降低 I/O
延迟，不会被重复计算成显存收益。

## 1. 当前结论

当前最主要的显存收益来自以下四类改动：

1. **有界 GPU KV page pool**：GPU KV 容量由活动预算决定，不再随 512K、1M
   等完整逻辑上下文线性增长。
2. **immutable raw-K 去除重复 working-K**：raw-K 作为 CPU 上的位置无关权威
   副本，GPU 只保留一份活动 K/V，去除了旧实现的额外完整 GPU K 镜像。
3. **FP8 active KV**：只把 attention K/V 压缩到 FP8；mean-K 索引和 retrieval
   query 仍保持 FP16，打分累积仍为 FP32。
4. **限制并复用临时计算缓冲**：prefill chunk 固定为 2048，MTP prefix 复用
   主模型 scratch，FlashInfer workspace 从实验配置中的 512 MiB 收紧到 192 MiB。

在同一约 517K-token AgentLongBench 样本、200K KVMem context budget、32K
generation reserve 下，已有如下实测：

| KV dtype | Prefill chunk | GPU 峰值 |
|---|---:|---:|
| FP16 | 2048 | 约 47.1 GiB |
| FP8 | 2048 | 约 39.9 GiB |
| FP8 | 4096 | 约 41.0 GiB |
| FP8 | 8192 | 约 43.3 GiB |

在准确率优先的 224K context budget + 32K generation reserve 配置中，FP8、
chunk 2048 的近期 512K 实验显存约为 40--41 GiB。较早的 chunk 8192 对照峰值
约为 43--44 GiB，仍低于 48 GiB。

这些节省数字不能直接全部相加。不同改动可能作用于同一组缓冲，CUDA allocator
还会保留曾经申请过的高水位内存，因此最终峰值必须以同一测试样本、同一进程的
实测为准。

## 2. GPU 显存组成

KVMem 模式下，可将 GPU 峰值近似拆成：

```text
M_gpu_peak
  ≈ M_weights
  + M_active_kv
  + M_retrieval_index
  + M_prefill_scratch
  + M_attention_workspace
  + M_transfer_staging
  + M_cuda_runtime
```

各部分的含义如下。

| 部分 | 主要决定因素 | 是否随完整逻辑上下文增长 |
|---|---|---|
| 模型权重 | 模型大小和权重量化格式 | 否 |
| active K/V | KVMem budget、generation reserve、KV dtype | 否，受活动预算约束 |
| mean-K index | `--ctx`、block size、索引 dtype | 是 |
| prefill/MTP scratch | prefill chunk、hidden/FFN/QKV 尺寸 | 不随完整 prompt 增长，但随 chunk 增长 |
| FlashInfer/workspace | kernel 配置和并发规模 | 通常固定上限 |
| stage-in/out staging | transfer slab 数量与大小 | 固定上限 |
| CUDA runtime | CUDA context、modules、cuBLAS/allocator 高水位 | 近似固定，但具有高水位行为 |

当前 Qwen3.6-27B Q8_0 模型权重本身约占 27 GiB。KVMem 的显存优化不能消除这
部分；它们主要控制权重之外的 K/V、索引、scratch 和 workspace。

### 2.1 Standard-attention KV 的数量级

Qwen3.6-27B 有 16 个 standard-attention layers。按当前模型的 KV head 配置，
standard layers 的 FP16 K/V 约为：

```text
K/V bytes per token
  = standard_layers
  × n_kv_heads
  × head_dim
  × 2(K and V)
  × 2 bytes
  ≈ 64 KiB/token
```

MTP 自身还有一套较小的活动 K/V。因此，活动容量从 256K tokens 降低 24K
tokens 时，完整 FP16 GPU KV pool 的实测/估算节省约为 1.59 GiB。

### 2.2 Mean-K 索引的数量级

当前 mean-K 每个 block、每个 standard-attention layer 保存一条 FP16 mean key：

```text
mean-K bytes
  = n_blocks
  × standard_layers
  × n_kv_heads
  × head_dim
  × 2 bytes
```

在 block size = 32 时，每个逻辑 token 对应的 mean-K 索引约为 1 KiB：

| 配置的逻辑 `--ctx` | FP16 mean-K 约占 |
|---:|---:|
| 256K | 256 MiB |
| 640K | 640 MiB |
| 1M | 1 GiB |
| 1.25M | 1.25 GiB |
| 10M | 约 10 GiB |

因此 active K/V 已经有界，但 mean-K 索引仍然是未来 10M 上下文中的显存扩展
瓶颈之一。

## 3. 已实现的显存优化

### 3.1 有界 GPU KV page pool

#### 原问题

如果按完整逻辑上下文直接在 GPU 上分配 KV，GPU 占用会随 prompt 长度线性增长。
对于 512K、1M 或更长上下文，这一方案无法在单卡上运行。

#### 当前实现

KVMem 为 GPU 创建固定容量的 physical page pool：

- GPU 只保留当前活动窗口的 K/V；
- 活动窗口容量由 `KVMem context budget + generation reserve` 决定；
- 未选中的历史 blocks 保存到 CPU 或 NVMe；
- `kv_pages_` 把逻辑 token pages 映射到 bounded physical pages；
- reselection 只把新选中的 blocks stage-in 到活动页池。

因此，下面两条 512K 与 1M 请求如果使用相同的 224K + 32K 活动预算，其 GPU
active KV 容量相同：

```text
512K logical source  -> 224K selected + 32K generation
1M logical source    -> 224K selected + 32K generation
```

逻辑上下文增长会增加 CPU/NVMe 占用和 mean-K index，但不会继续线性扩大 GPU
active KV pool。

#### GPU pool 的安全边界

`--kvmem-gpu-memory-ratio` 用于限制 page pool 可使用的 GPU 比例。分配器在
估算页池大小时还会为以下内容留出空间：

- prefill scratch；
- FlashInfer workspace；
- Q/O packing；
- immutable raw-K materialization staging；
- CUDA runtime 和 allocator 高水位。

当前实验常用：

```text
--kvmem-gpu-memory-ratio 0.51
```

该值不是“整个进程只能占 GPU 的 51%”，而是 KVMem page pool 的规划参数。
模型权重、索引、scratch 和 CUDA workspace 仍在 page pool 之外。

实现说明见：

- `src/qwen_executor.cpp`
- `docs/kvmem_implementation_notes.md` 的 “GPU Bounded Page Pool”

### 3.2 Immutable raw-K：去掉额外 GPU working-K 镜像

#### 原问题

selected blocks 在不同 reselection 窗口中会被映射到不同 compact positions。
如果直接在同一份 K 上反复执行 re-RoPE：

```text
position A -> position B -> position C -> ...
```

FP16/FP8 量化误差会累计。早期修复为避免不可逆漂移，曾在 GPU 上额外维护完整
working-K/authority，从而显著增加显存。

#### 当前布局

现在使用位置无关的 immutable raw-K：

```text
CPU:
  pre-RoPE raw-K authority

GPU:
  当前活动窗口的一份 working K/V

CPU/NVMe standard KV spill:
  immutable 模式下主要保存 V；
  K 可由 CPU raw-K 在目标 compact position 一次性构建
```

新的 K 计算流程为：

```text
CPU raw-K
  -> bounded H2D staging
  -> GPU scatter
  -> 对目标 compact position 执行一次 RoPE
  -> active working K
```

这带来两个收益：

1. 避免同一 K 在多个位置之间反复旋转产生的累计精度漂移；
2. 去掉旧方案额外的完整 GPU working-K 镜像。

对于 FP16、256K 活动容量，standard layers 的额外完整 K 镜像约为 **8 GiB**，
这是目前最大的单项 KVMem 专用显存优化之一。

#### 代价与边界

这一改动是显存向 CPU/SSD 分层的迁移，不是系统总内存凭空消失：

- Qwen3.6-27B FP16 raw-K 每 256K 真实上下文约占 8 GiB CPU 内存；
- 默认兼容路径仍按需分配完整 CPU raw-K；
- `--kvmem-raw-k-nvme` 将 SSD 设为完整 raw-K backing，CPU chunk
  变为受 `--kvmem-cpu-gb` 限制的缓存；
- V 和其他 tiered KV 可以进入 NVMe；
- cold selected raw-K 按物理 block 范围直接读入双缓冲 staging，不加载完整
  2048-token chunk。

因此，10M 上下文不再要求数百 GiB CPU raw-K；代价是 SSD authority 容量和
cold-block 读取延迟。后续仍需用 io_uring/固定 buffer 和预测预取进一步隐藏 I/O。

相关实现与证据：

- `src/qwen_executor.cpp` 中 immutable raw-K capture/materialization；
- `docs/kvmem_known_issues.md` 的 KVMI-011；
- `tests/rope_remap_drift.cu` 的位置重映射误差诊断。

### 3.3 FP8 active KV

#### 当前策略

通过：

```text
--kv-dtype fp8
```

将 active attention K/V 从 FP16 改为 FP8。理论上 K/V 存储减半，但总进程显存
不会减半，因为模型权重、mean-K index、scratch 和 workspace 不随 KV dtype
等比例下降。

#### 哪些数据使用 FP8

当前生产配置中：

| 数据/计算 | dtype |
|---|---|
| Standard-attention active K/V | FP8 |
| MTP active K/V | FP8 |
| Mean-K retrieval index | FP16 |
| Retrieval query capture | FP16 |
| Mean-K builder/scorer accumulation | FP32 |
| 模型权重 | GGUF Q8_0，与 KV dtype 独立 |

也就是说，FP8 只作用于占用最大的 active K/V。retrieval index/query 没有跟随
KV 变成 FP8，避免把 retrieval 精度和 KV attention 精度两个变量混在一起。

#### 实测节省

在相同的 200K context + 32K generation、chunk 2048 配置下：

```text
FP16 peak: 约 47.1 GiB
FP8  peak: 约 39.9 GiB
差值:      约 7.2 GiB
```

#### 准确率结论

早期 FP8 全量实验同时把 KVMem budget 从 224K 降到 200K，导致结果看起来明显
下降。后续控制变量显示，主要下降来自 retrieval budget 缩小，而不是 FP8 本身：

- 224K FP8 的敏感样本恢复到 224K FP16 对照结果；
- AgentLongBench 512K 的合并正式结果中，FP8 与既有 FP16 结果均为 53/100；
- FP8 仍是有损格式，正式新数据集仍应做 accuracy validation。

详细数据见 `docs/kvmem_fp8_performance_benchmark_20260725.md`。

### 3.4 Prefill chunk 固定为 2048

#### 原问题

早期 `forward_n_tokens` 会按整个 prompt 分配以下 batch scratch：

- residual/hidden；
- norm；
- FFN gate/up/mid；
- Q/K/V projections；
- attention output；
- concat/projection staging。

这些 tensor 大多是 FP32，并随 batch tokens 线性增长。在 64K prompt 下，仅
per-prompt batch scratch 就可能超过 30 GiB。

#### 当前实现

prefill 被切成固定大小的 chunks：

```text
--prefill-chunk 2048
```

scratch capacity 只按 2048 tokens 分配，完整 prompt 再长也复用同一组缓冲。

另一个已修复的问题是：MTP prefix 路径曾无条件设置内部 chunk=4096，覆盖命令行
显式提供的 2048。现在只有用户未显式指定 chunk 时，MTP 才能使用内部 override；
当前准确率实验会真正保持 2048。

关键实现：

- `src/qwen_executor.cpp::forward_n_tokens`
- `src/qwen_native_backend.cpp` 的 MTP prefix chunk 选择

#### 节省量

修复隐式 4096 -> 2048 后，理论上减少：

| 项目 | 约节省 |
|---|---:|
| 主模型 batch scratch | 985 MiB |
| MMQ activation staging | 153 MiB |
| Immutable raw-K capture staging | 68 MiB |
| FlashInfer Q/O staging | 48 MiB |
| 合计 | 约 1.2--1.25 GiB |

同一 FP8 真实样本的整体峰值进一步验证：

| Chunk | GPU 峰值 |
|---:|---:|
| 2048 | 39.9 GiB |
| 4096 | 41.0 GiB |
| 8192 | 43.3 GiB |

4096/8192 能提高少量 prefill throughput，但会明显压缩与另一实验共享 GPU 时的
显存余量。因此当前准确率测试固定使用 2048。

### 3.5 MTP prefix 复用主模型 scratch

#### 原问题

主模型处理完一个 prefill chunk 后，MTP prefix 会串行执行。旧实现仍为 MTP
常驻维护第二套 capacity-sized batch buffers，包括：

- input/hidden/norm；
- concat；
- Q/K/V；
- FFN gate/up/mid；
- attention output。

实际上 MTP 开始时，除主模型最终 `h_batch` 外，其余主模型中间张量都已失效。

#### 当前实现

MTP prefix 复用已失效的主模型 scratch：

```text
MTP h_input  <- target norm buffer
MTP hidden   <- target attention-output buffer
MTP concat   <- target projection buffer
MTP Q/K/V    <- target Q/K/V buffers
```

计算仍然严格串行，不会覆盖 MTP 仍需要的 `h_batch`。

#### 节省量

- 当前 chunk=2048：约 **0.8 GiB**；
- 如果旧路径实际按 chunk=4096 分配：约 **1.6 GiB**。

实现位置：`src/qwen_executor.cpp` 的 `prime_mtp_prefix_from_last_batch` 路径。

### 3.6 FlashInfer workspace 从 512 MiB 收紧到 192 MiB

FlashInfer prefill split-KV workspace 的通用代码默认上限仍是 512 MiB，可通过：

```text
QW3_FLASHINFER_PREFILL_WORKSPACE_MIB
```

配置。当前单请求 KVMem 准确率脚本设置：

```text
QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192
```

相对 512 MiB 减少 **320 MiB** 常驻 workspace。

真实长样本曾测得约 138 MiB 的 workspace 需求，64 MiB 不足，因此选择 192 MiB
保留安全余量。若 split plan 无法在 workspace 中执行，当前路径会尝试安全的
non-split fallback，而不是直接让请求失败。

需要注意：

- 192 MiB 是当前单请求测试 profile，不是所有并发服务的通用默认值；
- 提高并发 batch 后可能需要重新评估；
- 配置入口位于 `src/kernels_cuda.cu`；
- 当前实验脚本位于 `scripts/kvmem_eval/`。

### 3.7 Mean-K 索引保持 FP16

Mean-K 生产索引现在固定使用 FP16，即使 active KV 使用 FP8。builder 和 scorer
仍使用 FP32 accumulation。

与 FP32 索引相比，FP16 将索引显存减半：

| `--ctx` | FP16 | 若使用FP32 | FP16节省 |
|---:|---:|---:|---:|
| 256K | 256 MiB | 512 MiB | 256 MiB |
| 640K | 640 MiB | 1.25 GiB | 640 MiB |
| 1.25M | 1.25 GiB | 2.5 GiB | 1.25 GiB |

此前曾实验让 mean-K 跟随 FP8 KV 使用 FP8，但这会把 retrieval 精度变化和
attention KV 精度变化混合。目前已经恢复 FP16，作为显存和准确率之间的稳定折中。

实现位置：`src/qwen_executor.cpp` 中 `g_kbar_multi_` 的分配与 capture。

### 3.8 Retrieval query 的 FP16 与 bounded streaming

早期 query capture 使用完整 GPU FP32 tensor。取消旧的 512-token query 截断后，
如果 user query 很长，完整 query tensor 会随长度线性增长。

当前 mean-K query 路径改为：

- query capture 默认 FP16；
- query 不超过 scoring chunk 时，直接保留在 GPU；
- 长 query 使用 pageable CPU backing；
- GPU 只保留默认 256 tokens 的 scoring stage；
- 两个小的 pinned bounce buffers 用于异步 D2H/H2D。

默认 query score chunk：

```text
QW3_KVMEM_QUERY_SCORE_CHUNK=256
```

对于 Qwen3.6-27B，256-token FP16 GPU query stage 约为 48 MiB。无论实际 query
是 1K、8K 还是更长，GPU capture 不再按完整 query 线性增长。

普通 AgentLongBench 最终问题通常很短，因此这一优化在当前样本上节省有限；它的
主要价值是让“按真实 user query 长度 retrieval”仍保持 bounded GPU memory。

### 3.9 有界、复用的传输 staging

CPU/NVMe 优化需要 GPU gather/scatter staging。为防止这些性能优化重新制造显存
峰值，当前实现使用：

- 一份可复用的约 64 MiB GPU staging slab，stage-in/out 共用；
- 两份约 64 MiB pinned host slabs 形成双 buffer；
- raw-K transfer 按固定 block cap 分批；
- 不为每次 reselection 按全部 stage-in/out 数据量分配临时 GPU tensor。

这项优化的主要作用是**限制上界**，而不是产生数 GiB 的净节省。某些 assembly
pipeline A/B 中，额外 staging 反而使 GPU 峰值增加约 346 MiB，但换取了更好的
I/O/计算重叠。因此 staging 必须计入显存预算，不能被描述为“零开销”。

### 3.10 降低 KVMem budget：有效但有准确率代价

将：

```text
224K context + 32K generation
```

改为：

```text
200K context + 32K generation
```

可减少 24K active tokens：

- FP16 GPU K/V 约减少 1.59 GiB；
- FP8 GPU K/V 约减少 0.8 GiB；
- selected blocks 从 7168 降到 6400。

真实 517K AgentLongBench 样本中，FP16 200K/32K/chunk2048 曾测得：

- 约 47.1 GiB 峰值；
- 另一次采样为 47,114 MiB；
- 能满足单进程低于 48 GiB 的目标。

但是后续 accuracy control 证明，200K 相比 224K 会损失关键 retrieval blocks，
并造成明显准确率下降。因此：

> 200K 是显存受限 profile，不是当前正式准确率 profile。

正式准确率测试优先保留 224K budget，通过 FP8 KV 和 chunk 2048 获得显存余量。

## 4. 继承自 qw3 基础推理系统的显存优化

下面两项不是 KVMem 专用改动，但会直接决定 KVMem 实验的总显存基线。

### 4.1 Q8_0 模型权重不创建 FP16 权重镜像

当前模型使用 Qwen3.6-27B Q8_0 GGUF，权重在 GPU 上保持 Q8。默认 MMQ 路径直接
消费量化权重，不长期创建完整 FP16 weight mirror。

如果为全部27B权重额外维护FP16镜像，单这一项就会增加数十GiB显存，KVMem的
224K活动窗口无法在当前96GB GPU上与第二个实验安全共存。

### 4.2 MMQ 路径避免大规模 FP16 dequant scratch

当前 prefill matmul 默认使用 MMQ v8/v7，不再通过 HGEMM 为大 batch 创建大规模
FP16 dequantized-weight scratch。历史开发记录中，该路径避免了约 3 GiB 的批量
临时占用。

这两项属于 `qw3` 模型执行器基线，不能在 KVMem 优化汇总中再次宣称为新增收益，
但解释总显存时必须包含。

## 5. 哪些改动不是显存优化

### 5.1 Opt1/2/3 的主要目标是 I/O 延迟

目前合并后的性能优化包括：

- proactive stage-out；
- hierarchical CPU reuse/热块缓存；
- 批量 D2H/H2D；
- 双 buffer；
- SSD 异步 writeback；
- packed raw-K rematerialization；
- stage-in 与 assembly 重叠。

它们主要减少：

- pressure point 的同步 stage-out；
- reselection 的 CPU/SSD stage-in；
- 小块 PCIe 传输；
- CPU gather 和 GPU scatter/re-RoPE 延迟。

除 bounded staging 外，它们不会显著减少 GPU 常驻容量。有些 pipeline 为了重叠
计算与 I/O 还会小幅增加显存。

### 5.2 CPU lazy slabs、cold reset 和 `malloc_trim`

以下改动减少的是 host RAM 或 pinned memory：

- immutable raw-K 按需分配 pageable chunks；
- sparse CPU spill slots 使用 lazy slabs；
- cold reset 释放 raw-K validity/storage；
- 请求结束后 `malloc_trim(0)`；
- 复用有限数量的 cold host buffers。

这些改动对于连续运行多个超长样本非常重要，但不应计入 GPU 显存节省。

### 5.3 NVMe cache、预取和 io_uring

NVMe 分层决定 CPU 内存压力和 I/O 延迟，不直接降低已经有界的 GPU active KV。
只有当它改变 GPU staging 的分配方式时，才会对 GPU 峰值产生小幅影响。

### 5.4 Incremental assembly

Incremental assembly 的目标是减少 re-RoPE 和搬运计算，不是减少活动 KV 容量。
此外，MTP mutable-K 的增量位置旋转会累积 FP16/FP8 rounding error，目前默认
禁用该实验路径，正式配置从 immutable raw authority 重建 MTP K。

## 6. 当前推荐配置

### 6.1 准确率优先且低于 48 GiB

当前 AgentLongBench 512K/1M 正式测试推荐：

```text
--kv-dtype fp8
--ctx <按数据集设置>
--kvmem
--kvmem-block-tokens 32
--kvmem-budget 229376
--kvmem-gen-budget 32768
--kvmem-sink-blocks 8
--kvmem-recent-blocks 0
--kvmem-method retrieval
--kvmem-retrieval-method mean-k
--kvmem-update-mode step
--kvmem-query-conditioned
--kvmem-immutable-k
--kvmem-gpu-memory-ratio 0.51
--prefill-chunk 2048
--native-mtp-speculate
--mtp-chain 4
```

环境变量：

```text
QW3_KVMEM_RECOMPUTE_QUERY=1
QW3_KVMEM_IMMUTABLE_SOURCE_K=1
QW3_FLASHINFER_PREFILL_WORKSPACE_MIB=192
```

这套配置的关键含义：

- 224K retrieval context 保持已验证的准确率；
- 32K generation reserve；
- active K/V 使用 FP8；
- mean-K/query 保持 FP16；
- prefill scratch 按 2048 固定；
- 不使用 DeltaNet rebuilt-state；
- CPU tier 优先，只有 host capacity 不足时才启用 NVMe。

### 6.2 FP16 KV、强制控制在约 48 GiB

如果必须使用 FP16 KV：

```text
--kv-dtype fp16
--kvmem-budget 204800
--kvmem-gen-budget 32768
--prefill-chunk 2048
```

已测峰值约为 47.1 GiB。但这是 200K retrieval budget，准确率可能低于224K正式
配置，因此只适合作为性能/精度控制变量或显存强约束环境。

### 6.3 不推荐作为共享 GPU 默认值

下列设置虽可能提高少量吞吐，但会显著降低显存余量：

```text
--prefill-chunk 4096
--prefill-chunk 8192
```

特别是运行两个 qw3 server 时，CUDA context、模型权重和 allocator 高水位会使
两个“单独看似低于一半”的进程仍可能共同 OOM。

## 7. 仍未解决的显存扩展问题

### 7.1 Mean-K index 仍按逻辑 `--ctx` 常驻 GPU

active KV 已按 KVMem budget 有界，但 mean-K index stride 按完整 context capacity
分配。block size 32 下，FP16 索引约为 1 KiB/token：

```text
1M  ctx -> 约 1 GiB
10M ctx -> 约 10 GiB
```

10M 级别需要进一步实现：

- CPU-resident mean-K；
- 分层/粗到细索引；
- GPU 分块 streaming scorer；
- 只把当前打分 tile 搬到 GPU；
- 热块索引常驻 GPU、冷块索引放 CPU/NVMe。

### 7.2 Immutable raw-K 的 SSD authority

raw-K 约为 32 KiB/token：

```text
256K -> 约 8 GiB CPU
1M   -> 约 32 GiB CPU
10M  -> 约 305 GiB CPU
```

默认关闭 `--kvmem-raw-k-nvme` 时，raw-K 仍随完整历史占用 CPU。开启后：

- admission 从总 NVMe budget 先保留完整 raw-K arena；
- CPU 成为有界 raw-K/V cache；
- 完整对齐 chunk 直接 D2H 到两个固定 pinned record slot；
- main/MTP GPU raw-capture tensor 都固定为两组，轮回复用时通过 stream
  event 建立依赖，避免下一 chunk 覆盖尚未完成 D2H 的源；
- 每批只记录 CUDA transfer fence，不在每个 chunk 后同步 host；后台写线程先
  等 fence，CPU cache copy 在 slot 回收前才发布；
- 默认每个 slot 合并最多三个连续 FP8+MTP record，由后台线程执行一次
  大块 `pwrite` 和一次 range writeback；
- staging slot 复用、prefix checkpoint、truncate/reset 或 in-flight chunk
  冷淘汰时才等待；dirty partial chunk 仍在淘汰前同步落盘；
- retrieval 后只读取 selected raw-K block；
- 固定 pinned/device staging 与 GPU scatter/RoPE 继续双缓冲。

10M FP8 + local-position MTP 的 raw-K authority 约 162.13 GiB，V authority
也约 162.13 GiB；因此 360 GiB NVMe 配额可留出约 35 GiB 余量。

### 7.3 CUDA allocator 和 runtime 高水位

即使逻辑 tensor 已释放，CUDA allocator、cuBLAS、FlashInfer modules 和驱动
workspace 也可能保留历史高水位。测试显存时应：

- 用独立冷启动进程；
- 对相同样本从server启动前开始采样；
- 区分 `nvidia-smi` process memory 与框架统计 tensor bytes；
- 不用运行过大 chunk 后的同一进程测较小 chunk；
- 记录峰值，而不只记录请求结束后的占用。

### 7.4 并发服务仍需要全局显存仲裁

`--kvmem-gpu-memory-ratio` 只控制单进程内部 KVMem page pool，无法感知另一
qw3 server 随后会分配多少模型scratch。两个服务共用GPU时，需要外部启动器根据：

- 两份模型权重；
- 两份 CUDA context/workspace；
- 两个进程各自的 active KV；
- 最大同时 prefill chunk；

做全局 admission control。不能简单认为两个 `gpu_memory_ratio=0.5` 的进程必然
能够安全共存。

## 8. 测量与复现原则

显存优化必须使用相同数据和相同请求流程做控制变量。建议每次记录：

```text
model
logical ctx
KV dtype
KVMem context budget
generation reserve
block size
prefill chunk
mean-K/query dtype
FlashInfer workspace
CPU/NVMe tier capacity
optimization groups
prompt tokens
selected blocks
peak process GPU memory
prefill throughput
TTFT
TPOT/decode throughput
answer/evaluation result
```

推荐的显存对照矩阵：

| 对照 | 只改变 |
|---|---|
| FP16 vs FP8 | active KV dtype |
| chunk 2048 vs 4096 | prefill scratch capacity |
| 224K vs 200K | selected context budget |
| immutable on vs legacy | raw-K authority与GPU重复K |
| workspace 512 vs 192 | FlashInfer workspace |

不能把同时改变 KV dtype、budget、chunk 和 sampling 参数的两次全量实验直接归因
于某一项显存优化。

## 9. 相关资料

- `docs/kvmem_fp8_performance_benchmark_20260725.md`
- `docs/kvmem_cpu_transfer_optimization_benchmark_20260724.md`
- `docs/kvmem_assembly_rerope_optimization_benchmark_20260724.md`
- `docs/kvmem_known_issues.md`
- `docs/kvmem_implementation_notes.md`
- `docs/kvmem_ssd_complete_design.md`
- `scripts/kvmem_eval/run_agentlongbench_512k_normal100_k224k_immutable.sh`
- `scripts/kvmem_eval/run_agentlongbench_perf_ab.sh`
