# KVMem Semantic Stage-in 优化与验证（2026-08-01）

本文记录 semantic reselection 的 stage-in 优化。目标是在**不修改 CPU cache
热度定义**的前提下，减少 selected KV 从 CPU/NVMe 恢复到 GPU 的关键路径延迟。

本轮实现建立在以下提交之上：

- `89fbef4`：checkpoint profiling 与独立优化开关；
- `afd0808`：守恒的 reselection telemetry；
- `da9f3cb`：NUMA-aware stage-in、assembly overlap 与安全预取。

## 1. 性能口径

一次 semantic reselection 的互斥关键路径按下式统计：

```text
semantic total
  = retrieval scoring
  + exposed stage-out submit
  + materialization critical path
```

其中 materialization 同时包含 stage-in 与 assembly/re-RoPE。二者已并行执行，
所以日志里的 `stage_in_wall_ms` 和 `assemble_ms` 是嵌套指标，不能相加得到
semantic total。`accounting_error_ms` 用于检查上述互斥分段守恒。

实验使用真实 Qwen3.6-27B-Q8_0 推理，而不是 I/O microbenchmark：

| 参数 | 值 |
|---|---|
| 历史长度 | 96K tokens |
| KVMem context / generation reserve | 32K / 32K |
| KV dtype | FP16 |
| Physical/retrieval block | 512 tokens |
| Retrieval | key-direction-adaptive，16 subblocks |
| Index | CPU，64 MiB staging |
| Prefill chunk | 2,048 tokens |
| MTP | enabled，chain 4 |
| Query mode | frozen checkpoint，4 次 query |
| Stage-out / stage-in / pack | all on |

CPU-only 测试使用 8 GiB CPU tier；NVMe 测试将 CPU tier 限制为 4 GiB，强制
semantic query 从 NVMe 恢复约 0.65--0.71 GiB selected KV。

## 2. NUMA-aware CPU stage-in

`--kvmem-numa-policy auto` 根据 GPU PCI BDF 的 sysfs NUMA 信息选择本地 CPU，
并将 CPU gather/packing worker 固定到对应 CPU set。当前测试机 GPU 位于
`0000:61:00.0`，对应 NUMA node 0，CPU set 为 `0-15,32-47`。

96K CPU-only frozen-query 的稳定第二次 query：

| NUMA policy | Score | CPU gather | Stage-in wall | Assembly | Semantic total |
|---|---:|---:|---:|---:|---:|
| off | 32.772 ms | 88.686 ms | 91.336 ms | 70.724 ms | 124.402 ms |
| auto | 32.697 ms | 79.217 ms | 81.831 ms | 71.768 ms | 114.781 ms |
| 改善 | -0.2% | **-10.7%** | **-10.4%** | +1.5% | **-7.7%** |

该实现不是针对当前机器硬编码：`auto` 会在每台机器启动时重新解析 GPU 的 PCI
拓扑和 CPU set；无法解析时安全回退为不绑定。也可使用 `off` 或 `node:N` 做控制实验。

## 3. NVMe packed stage-in

### 3.1 原瓶颈

旧 NVMe 路径已经合并磁盘读取，但读取完成后仍按 block、layer、K/V 发起大量小
H2D copy。典型 semantic query 读取 39 个 NVMe blocks（0.647 GiB）时：

- `nvme_h2d_enqueue_ms = 55.750`；
- materialization critical path 为 428.295 ms；
- semantic total 为 461.431 ms。

因此，磁盘 batch 已经连续，并不代表 host-to-GPU 阶段也已经连续。

### 3.2 实现

CPU 与 NVMe 路径现在共用 packed stage-in：

1. lower tier 将多个 selected blocks 放入连续 pinned slab；
2. 一次大块 H2D 复制到 GPU staging；
3. GPU 根据 source/destination page index scatter 到各 layer 的目标 KV pages；
4. 每批用 transfer fence 保护 staging slab 生命周期；
5. assembly/re-RoPE 与 lower-tier I/O 继续重叠。

NVMe batch 新增持久化的 page mapping 与 transfer fence，移除了读取完成后的
per-block `kvmem_copy_block_from_host` 循环。CPU 路径也重构为调用同一映射函数，
避免两套布局逻辑发生漂移。

### 3.3 QD=2 控制结果

稳定 query 1--3：

| NVMe stage-in | Q1 | Q2 | Q3 | Mean semantic |
|---|---:|---:|---:|---:|
| 旧 per-block H2D，QD=2 | 458.480 ms | 491.533 ms | 461.431 ms | 470.481 ms |
| Packed H2D/scatter，QD=2 | 447.018 ms | 468.628 ms | 442.246 ms | 452.631 ms |
| 改善 | 2.5% | 4.7% | 4.2% | **3.8%** |

在对应 Q3 上，H2D enqueue 从 55.750 ms 降到 0.840 ms（**-98.5%**）。总体收益
小于 enqueue 降幅，因为此时真正的关键路径已经转移到 NVMe random read。

## 4. NVMe read queue depth

新增环境变量：

```text
QW3_KVMEM_READ_QUEUE_DEPTH
```

默认请求深度为 4，可设范围为 1--8；实际深度还会被已分配 read slab 数量限制：

```text
effective QD = min(configured QD, provisioned read slabs)
```

因此只有两个 slab 的配置仍为 QD=2，不会为了追求并发额外扩大常驻内存；混合
CPU+SSD 配置已有四个 slab 时可直接使用 QD=4。

稳定 query 1--3：

| 路径 | Q1 | Q2 | Q3 | Mean semantic | 相对旧路径 |
|---|---:|---:|---:|---:|---:|
| 旧 per-block H2D，QD=2 | 458.480 ms | 491.533 ms | 461.431 ms | 470.481 ms | baseline |
| Packed，QD=2 | 447.018 ms | 468.628 ms | 442.246 ms | 452.631 ms | -3.8% |
| Packed，QD=4 | 382.063 ms | 388.309 ms | 359.523 ms | **376.632 ms** | **-20.0%** |

QD=4 相对 packed QD=2 降低 16.8%。Q3 的互斥分解为：

| 阶段 | 时间 |
|---|---:|
| Retrieval scoring | 32.666 ms |
| Exposed stage-out submit | 0.401 ms |
| Materialization critical path | 326.387 ms |
| Semantic total | 359.454 ms |
| Accounting error | 0.000 ms |

该次 materialization 中，39 个 NVMe blocks（0.647 GiB）的累计 read service time
为 1,108.483 ms，但 796.815 ms 被队列并发和其他工作隐藏；暴露的 NVMe wait 为
311.668 ms。说明累计 I/O service time不能直接作为 query 的用户可见等待时间。

## 5. 内存控制实验

将全局 I/O slab 从 128 MiB 增大到 256 MiB，materialization 只从约 67.1 ms 降到
65.4 ms（约 2.5%），host RSS 却从约 40.6 GiB 增至 43.0 GiB（约 +2.3 GiB）。
因此默认 slab 保持 128 MiB。提高 QD 优先复用已存在 slab，而不是扩大每个 slab。

packed QD=2 与 QD=4 的真实 profile 中：

- peak GPU process memory 均约 37.4 GiB；
- host RSS 均约 38.3 GiB；
- 未观察到 QD=4 额外常驻内存增长。

## 6. 正确性与回归

- NUMA on/off、packed on/off 和 QD=2/4 对应 query 的 `selected_hash` 一致；
- CPU-only packed refactor 完成 66K history 的 prefill、semantic reselection、query
  replay 和 decode；
- NVMe packed QD=4 完成 96K history 与 4 次 frozen query；
- 完整构建成功；
- `ctest --test-dir build --output-on-failure`：15/15 tests passed。

## 7. 结论与后续边界

本轮没有修改 CPU cache 热度定义，也没有通过缓存更多 blocks 获得收益。改进来自：

1. CPU gather 在 GPU-local NUMA node 执行；
2. NVMe 后半段从小 H2D copy 改为 packed H2D + GPU scatter；
3. 使用已有 read slabs 提升 NVMe 请求并发。

在强制 NVMe miss 的 semantic query 上，最终稳定均值从 470.481 ms 降到
376.632 ms（**-20.0%**）。剩余关键瓶颈是 NVMe random read 的暴露等待，而不是
H2D enqueue。下一步若继续优化，应该优先测试异步 direct I/O、block layout
locality 和跨 query 的预测预取；CPU cache 热度策略应在“同一 checkpoint 多次真实
query”实验中单独评估，不与本轮传输优化混合。

## 8. 原始产物

所有产物已持久化到：

```text
/data/chaidi/kvmem_eval/results/kvmem_semantic_stagein_20260801/
```

| 实验 | JSON | Log |
|---|---|---|
| NUMA auto CPU-only | `qw3_kvmem_numa_auto_96k_fixed2.json` | `qw3_kvmem_session_bnm8iget.log` |
| NUMA off CPU-only | `qw3_kvmem_numa_off_96k_fixed2.json` | `qw3_kvmem_session_v4f54a98.log` |
| NVMe legacy QD=2 | `qw3_kvmem_nvme_96k_q4.json` | `qw3_kvmem_session_o_lz46de.log` |
| NVMe packed QD=2 | `qw3_kvmem_nvme_packed_96k_q4.json` | `qw3_kvmem_session_7bu4ot4j.log` |
| NVMe packed QD=4 | `qw3_kvmem_nvme_packed_qd4_96k_q4.json` | `qw3_kvmem_session_afw4is_5.log` |
| CPU refactor smoke | `qw3_kvmem_cpu_refactor_66k_q2.json` | `qw3_kvmem_session_ice1lae4.log` |
| 128 MiB slab control | `qw3_kvmem_slab128_auto_96k_q6.json` | `qw3_kvmem_session_69o3jakp.log` |
| 256 MiB slab control | `qw3_kvmem_slab256_auto_96k_q6.json` | `qw3_kvmem_session_j4fjj02k.log` |
