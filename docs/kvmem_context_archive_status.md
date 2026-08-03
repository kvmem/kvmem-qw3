# KVMem Context Archive — 实现状态

Date: 2026-08-01  
Design: [`kvmem_context_archive_design.md`](kvmem_context_archive_design.md)  
Working tree: uncommitted; archive work is layered on top of commit `c826e67`.

本文记录当前代码实际完成了什么、做过哪些验证，以及仍未完成的工作。磁盘格式和设计边界以 design 文档为准；本文不把“代码存在”当作“已经验证”。

---

## 1. 当前结论

以下主链路已经打通：

```text
build → incremental process-crash commit → seal
      → read-only attach / prefix truncate
      → residual replay → policy-specific index rebuild
      → bounded base-window materialization
      → multi-question frozen branching
```

在 Qwen3.6-27B-Q8_0 / FP8 archive 上，已经完成
12K/32K/1M/2M/10M 端到端验证、真实 SIGKILL 后续建、CoW overlay、非
ladder 截断、branch isolation、truncate parity、retrieval-policy A/B、冻结归档
Serve API，以及完整 GGUF SHA-256 模型身份。

当前 correctness 主链路和 10M 容量链路已经闭合。尚未完成的是进一步隐藏/缓存
retrieval-index rebuild，以及断电级 durability。

---

## 2. 已实现内容

### 2.1 Archive format 与 CLI

| 路径 | 作用 |
|---|---|
| `include/qw3/kvmem_archive.hpp`, `src/kvmem_archive.cpp` | `manifest.json`、raw-K/V arena、token stream、validity bitmap、ladder state、resume/seal |
| `src/kvmem_archive_cli.hpp` | `archive build/query/info` 配置与入口 |
| `tests/kvmem_archive_test.cpp` | format、ladder、durable/direct-map、read-only CoW、token rollback 单测 |
| `docs/kvmem_context_archive_design.md` | on-disk contract 与设计理由 |

CLI 已支持：

- `qw3 archive build`
- `qw3 archive query`
- `qw3 archive info`
- 任意 block-aligned `--archive-tokens N`；从最近 ladder 恢复并 replay residual
- 一个进程中通过多个 `--archive-question` 从同一 frozen base 提问

### 2.2 Durable/direct-mapped NVMe tier

- `durable`：保留目录项，已有文件不 truncate，可跨进程恢复。
- `direct_mapped`：`slot == block/chunk id`，无需持久化 LRU/slot table。
- `read_only`：sealed archive 以 `O_RDONLY` 打开。
- attach 不读取全量 KV；只恢复 ladder state、流式重建 index，并按需 stage-in selected blocks。
- GPU page table attach 时安装 absent logical pages，不为全历史分配 GPU physical pages。

### 2.3 Read-only CoW overlay

sealed base 永远不修改。对 residual/query/decode 所产生的写入：

- 使用 archive 外部、unlink 后的 sparse overlay；
- overlay 与 base 使用相同 slot offset；
- 首次 partial-slot write 会先完整 copy-up base slot，再 patch 指定 range；
- slot 状态为 `base → initializing → valid`，并发 reader 不会读到半复制数据；
- batch write 经过逐 slot copy-up，避免 main/MTP 其中一段被 sparse zero 覆盖。

单测覆盖 partial write 的 prefix/suffix 保留、相邻 base/overlay batch read、full-slot write、ephemeral unlink，以及 base byte 不变。

### 2.4 Ladder、truncate 与 residual replay

- ladder state 保存 hidden、MTP prefix hidden、48 层 recurrent/conv state。
- attach 到任意 block-aligned `N`：恢复 `p <= N` 的最近 ladder，再 teacher-force `[p,N)`。
- residual 前先从 archive 构建 ladder 前缀 index，并物化 deterministic sink+recent window；否则 residual 的第一段 attention 会面对 all-absent page table。
- residual 后从 immutable raw-K 对 `[0,N)` 做 canonical bulk index rebuild，避免 incremental boundary merge 与 fresh build 的 reduction 分块不同。
- query base 捕获前同样建立明确的 bounded selection。

### 2.5 Frozen branch isolation

每个问题开始前执行：

1. restore 相同 recurrent/hidden base state；
2. truncate 掉上一条 branch suffix；
3. 恢复同一个 base selection；
4. 对 base selection 强制从 position-free raw-K 重建 working K；
5. 在此后捕获 query-replay boundary。

这避免第二个问题继承第一个问题重新烘焙后的 K/window frame。

### 2.6 Resumable build

端到端恢复已实现，而不只是 bitmap helper：

- build-mode archive 允许从 durable ladder 调用 attach/restore；
- 恢复前分块校验输入 token prefix，拒绝把不同 corpus 接到旧 archive；
- crash 后若 `tokens.bin` 超前于最后 manifest commit，先 truncate 回 resume ladder；
- 从 raw-K 重建已有前缀 index；
- 恢复 canonical pressure window 和 API continuation checkpoint；
- writer 在每个 ladder commit 前也强制构建同一个 raw-authority canonical window，保证 live continuation 与 restart continuation 使用相同 attention frame。

### 2.7 MTP physical layout、resume parity 与兼容性

- 新 archive format version 为 3；v2 引入正确的 MTP physical layout，v3 增加完整 GGUF SHA-256。
- `mtp_archived=true` 时，manifest 显式记录 `mtp_chunk_bytes`，`v_block_bytes` 包含主层和 MTP 层。
- 早期 v1 archive 实际写了 MTP 段但 manifest 漏记；attach 仅在主 layout 完全匹配、且 raw/V 文件大小能证明完整 MTP physical stride 时兼容，其他 mismatch 仍拒绝。
- 修复 cold restore 没有先分配 `mtp_prefix_h_`、从而静默漏恢复 `MtpPrefixHidden` 的问题；真实 SIGKILL/resume 现在连 MTP raw-K/V 也与 uninterrupted build 全文件 byte-identical。

### 2.8 Retrieval policy 与 payload 解耦

Archive 不保存 mean-K/subblock/adaptive index。attach 时从 raw-K 按当前 CLI policy 重建，因此同一 payload 已验证可切换：

- mean-K
- contiguous sub-block mean-K（4 × 32 tokens）
- key-direction adaptive prototypes

同时修正 scorer 审计日志：contiguous subblock 不再错误打印为 `mean-k`，而是 `sub-block-mean-k-max|sum`。

### 2.9 Frozen archive Serve API

`qw3 serve --kvmem-archive DIR` 提供专用只读归档服务：

- server 启动时只 attach 一次、只 rebuild 一次 index；
- 每个 HTTP request 从相同 frozen base state/selection 开始；
- request prompt 只作为新增 query，不需要再次发送归档 prefix；
- branch suffix、working selection 和 re-RoPE frame 不会污染下一请求；
- 当前 v1 API 明确拒绝 mutable session、local-cache save/load、transcript replay 和 semantic-expansion 等未定义组合。

### 2.10 Model content identity

- v3 manifest 的 `layout.model_sha256` 是完整 GGUF 文件的 SHA-256，而非 basename/file size 代理。
- 摘要缓存由 canonical path + device/inode + size + mtime + ctime 校验；缓存默认位于 `~/.cache/qw3/model-digests`，也可由 `QW3_MODEL_DIGEST_CACHE_DIR` 指定。
- 构建环境存在 OpenSSL 时使用其优化的 SHA-256 实现；否则使用内置 portable 实现。
- v1/v2 没有摘要，仍通过显式 legacy 分支兼容；新 v3 archive 必须有 64 个十六进制字符的 digest。

### 2.11 Bulk index rebuild I/O

1M 基线的 block-major raw-K index rebuild 会按 layer 读取：每个 32 MiB
chunk 被拆成 256 次 128 KiB 的跨距 `pread`，因此没有获得 NVMe 顺序带宽。
当前已改为：

1. 每个 raw-K chunk 做一次连续读取；
2. 在 pinned host buffer 中按 layer gather；
3. 保持原 H2D、FP8→FP32 widen 和 prototype kernel 不变。

12K correctness 回归的 mean-K selected hash 仍为
`789f877201f59702`，与优化前一致。日志新增 `read_ms/gather_ms/other_ms`
分解。2M 的 1,703,936-token SIGKILL/resume 前缀重建耗时 34.40 秒，
归一化约 20.2 秒/1M；同一 1M archive 的严格前后 A/B 为
36.704→22.924 秒（提升 37.5%），其中读取仍占 21.473 秒。`filefrag`
进一步确认旧 archive 的 raw-K/V 文件因交错增量写入
形成大量 extent，因此新 build 会对两个 direct-mapped arena 执行
`posix_fallocate`，尽早检查容量并减少物理碎片。该 preallocation 对不支持
它的文件系统会显式告警并退回 sparse growth，不影响 attach 或普通 KVMem。

---

## 3. 实测结果

环境：

- Model: `models/Qwen3.6-27B-Q8_0.gguf`
- `QW3_Q8_BF16_MAIN=0`
- Archive KV: FP8 e4m3，immutable raw-K
- Physical block: 128 tokens；raw-K chunk: 2048 tokens
- MTP chain: 4（archive CLI 默认）

### 3.1 构建、CoW 与截断

| 场景 | 结果 |
|---|---|
| CTest | **16/16 passed** |
| 32K build/seal | Pass；`/home/chaidi/kca/t32k` |
| 12K v2 build/seal | Pass；`/home/chaidi/kca/t12k_v2`，约 708 MiB（含两个 ladder state） |
| v1 32K attach | Pass；先验证 physical MTP stride，再兼容 attach |
| residual truncate @ 12,288 | Pass：8K ladder + 4K replay |
| residual truncate @ 20,480 | Pass：base archive 六个文件 hash 前后不变 |
| CoW partial/full-slot unit tests | Pass |

### 3.2 Truncate parity

同一 synthetic corpus：

- 独立 build 到 12,288；
- 从 32K archive 截到 12,288（8K ladder + 4K replay）。

两者最终：

- selected hash 都为 `f61cf88a005dfa9e`；
- greedy answer 逐字节相同。

此外，fresh 12K archive 与原 32K archive 的前 12K payload：主 raw-K、V、token stream 和 8K ladder state 均做过 byte parity。

### 3.3 Branch isolation

在 20,480-token residual attach 上，同一问题连续执行两次：

- selected hash 都为 `568d71bb01154e4c`；
- 40-token greedy output 逐字节相同；
- q0/q1 wall time 分别约 0.720/0.713 秒。

修复前两次 selected hash 分别为 `25d4800412d73b6f` / `c2a0fea478b5b7b6`，说明该测试真实捕获了 branch 污染，而不是偶然输出一致。

### 3.4 Policy A/B

同一个 `/home/chaidi/kca/t12k_v2`，同一 query `What does Record 500 say?`：

| Runtime policy | Index rebuild | selected hash | scorer fallback |
|---|---:|---|---:|
| mean-K | 约 0.30 s | `789f877201f59702` | 0 |
| sub-block mean-K, 4 × 32, max | 约 0.30 s | `cbaab43e3f324cc9` | 0 |
| key-direction adaptive, gains 0.10/0.06 | 约 0.45 s | `729ec82f5c6b8a01` | 0 |

三个 selection 不同，证明 attach 确实使用 runtime policy 重建 index，而不是沿用 build-time mean-K。

### 3.5 SIGKILL / resume parity

测试流程：12K、ladder=4K；在日志确认 8K manifest commit 后 `SIGKILL`，随后同目录恢复到 12K。

- 中断后 manifest：unsealed，tokens=8192，blocks=64，chunks=4。
- 恢复识别 8K ladder，已有 index rebuild 约 0.24 s；只 ingest 剩余 4K，约 3.0 s 后 seal。
- token stream、最终 recurrent/conv/hidden state、16 个主 attention 层 raw-K/V 与同参数 uninterrupted build 逐字节一致。
- 两份 archive 的 selected hash 都为 `48d6e8901781b9d3`；40-token greedy output 与 MTP acceptance（27/52）一致。

最初仅 resume 边界后的第一个 MTP token（token 8192）的 K/V 行不同。根因是 cold restore 时 `mtp_prefix_h_` 尚未分配，persisted `MtpPrefixHidden` 被静默跳过。修复后重新执行真实 SIGKILL，resumed 与 uninterrupted archive 的：

- `rawk.bin` 完整 SHA-256 相同；
- `v.bin` 完整 SHA-256 相同；
- 最终 state 完整 SHA-256 相同；
- 因而主模型和 MTP physical payload 都达到 byte parity。

### 3.6 1M build/attach/query

`/home/chaidi/kca/t1m_20260801_b`：

| 指标 | 结果 |
|---|---:|
| Tokens / blocks / chunks | 1,048,576 / 8,192 / 512 |
| Ladder points | 8（stride=131,072） |
| Build wall time | 623.760 s（约 10.4 min） |
| Archive disk usage | 约 36 GiB |
| Attach metadata | 0.127 s |
| Runtime mean-K index rebuild（旧 layer-strided 基线） | 36.704 s |
| Runtime mean-K index rebuild（contiguous chunk） | 22.924 s（read 21.473 / gather 0.688 / other 0.764） |
| Initial 1,600-block materialization | 2.585 s |
| Final-query mean-K scoring | 13.794 ms，`fallback=0` |
| Final semantic reselection | 3.365 s；其中 materialization/assembly 为主 |
| 16-token query total | 3.793 s |

这表明 1M correctness/容量链路可用。contiguous-chunk read 将 index 固定成本降低 37.5%，但 read 仍占 93.7%，后续优化重点已从 GPU reduction 明确转为 SSD layout、并行读取/预取或 derived sidecar。

### 3.7 Serve branch isolation

12K frozen archive 上执行 A→A 和 A→B→A：

- A 两次 semantic selected hash 均为 `23311a2ce348049e`；
- A 的回答逐字相同；
- 中间 B 使用不同 query/selection，不影响最后一次 A；
- index 只在 server 启动时 rebuild 一次；HTTP 四次均返回 200。

### 3.8 Model digest 冷/热启动

使用独立的空 digest-cache 目录，对同一个 25+ GiB GGUF 和 4K v3 archive
各执行一次 cold/warm attach：

| 场景 | 进程 wall time | 说明 |
|---|---:|---|
| OpenSSL cold digest | 约 49 s | 包含约 22 s 模型加载和首次完整 GGUF 扫描 |
| Stat-validated cache hit | 25.37 s | 不再读取完整 GGUF，只校验 path/stat 元数据 |

两次均通过 v3 digest 校验并成功 attach。由二者差值估算，本机 OpenSSL
首次完整摘要扫描约 23 s；旧 portable cold 流程曾约为 124 s，因此 portable
实现只作为无 OpenSSL 环境的兼容 fallback，不应作为本机性能基线。

### 3.9 2M build / SIGKILL-resume / attach / query

`/home/chaidi/kca/t2m_20260801_f`：

| 指标 | 结果 |
|---|---:|
| Tokens / blocks / chunks | 2,097,152 / 16,384 / 1,024 |
| Ladder points | 16（stride=131,072） |
| Archive disk usage | 约 70.3 GiB |
| SIGKILL point | 1,703,936 tokens（13/16 ladder） |
| Resume prefix index | 34.396 s（read 32.067 / gather 1.131 / other 1.198） |
| Resume suffix ingest | 393,216 tokens，251.459 s 后 seal |
| Full attach metadata | 0.121 s |
| Full 2M index rebuild | 42.652 s（read 39.781 / gather 1.402 / other 1.469） |
| Final semantic reselection | 4.055 s；`selected_hash=3e6f6e55e49e93f1` |
| 16-token question wall | 4.460 s |

最终问题命中 `Record 500: the planner Rigel dispatched ...`；scorer 无 fallback。
这份 2M 在 arena preallocation 修复前开始构建，因此其大量 extent 和
42.7 秒 index 是保守基线，不代表新建低碎片 archive 的最终带宽。

### 3.10 Arena preallocation 与并发 attach

- 新 4K v3 smoke 对 raw-K/V 各预分配 272 MiB，3.084 秒 build/seal；
  `filefrag` 分别为 8/2 个 extent，证明预分配路径生效。
- 两个独立进程并发 attach 同一 12K sealed archive，均成功完成 index、
  semantic reselection 和 decode，退出码均为 0；未观察到写冲突或 base 污染。

### 3.11 10M build / attach / 多问题语义重选

真实 BEAM-10M conversation 1（19,895 messages、20 questions）构建为：

`/home/chaidi/kca/beam10m_c1_9998336_b128_20260802`

| 指标 | 结果 |
|---|---:|
| Tokens / blocks / chunks | 9,998,336 / 78,112 / 4,882 |
| Ladder points | 10（约每 1,048,576 tokens） |
| Build wall time | 6,291.614 s（104.86 min） |
| Archive physical usage | 328 GiB |
| raw-K / V logical bytes | 175,227,535,360 / 175,220,850,688 |
| 每个完整 1M segment | 首段 617.98 s，后续约 663.6–664.1 s |
| Runtime index rebuild | 226.9 s（约 208.2 s read / 9.8 s gather / 8.9 s other） |
| 20-question scorer fallback | 0 |
| selected-hash parity | 四个优化 arm 以及修复前后 all-on 全部逐问题一致 |

manifest 已 sealed；valid chunk/block bitmap、10 个 ladder state、raw-K/V 文件尺寸均
完成校验。MTP 对所有超过 256K 的 logical position 使用 compact-window position，未把
超长原始位置送入 MTP RoPE。

### 3.12 10M semantic reselection 优化消融

同一 sealed archive、同一组 20 questions，每个 arm 只 attach/index 一次；表中不包含
固定的 index rebuild。非重叠定义为：

```text
semantic KVMem total
= score/select + exposed stage-out submit + materialization critical path
+ query replay
```

初始四臂结果：

| Arm | Semantic total mean | 稳态 mean（q1–q19） | 稳态加速（vs all-off） |
|---|---:|---:|---:|
| all-off | 9,717.781 ms | 9,370.277 ms | 1.000× |
| pack-only | 4,728.401 ms | 4,602.743 ms | 2.036× |
| pack+stage-out | 6,332.687 ms | 6,323.894 ms | 1.482× |
| all-on（修复前） | 6,230.831 ms | 6,225.234 ms | 1.505× |

该结果暴露出 archive attach 的 stage-in 缺口：sealed SSD record 被读入 pinned slab 并
H2D 后立即丢弃，`optimize_stage_in` 从未把 clean V record admission 到 CPU cache。
64 GiB CPU 配额实际只有约 34 MiB raw-K，`cpu_spill=0`，所以每个问题仍从 NVMe 读取约
3 GiB。

修复后，SSD 仍是 durable authority；同一 read-only slab 在 H2D 期间并行复制到
heat-aware CPU cache。完整 20 问结果：

| 指标 | 修复后 all-on |
|---|---:|
| Score/select | 1,266.803 ms |
| Exposed stage-out submit | 6.089 ms |
| Materialization critical path | 3,061.797 ms |
| Query replay | 88.592 ms |
| Semantic total mean | 4,423.280 ms |
| 稳态 mean / median / p95 | 4,321.381 / 4,245.981 / 5,256.084 ms |
| 平均 incoming CPU hit rate | 85.106% |
| 后 5 问 CPU hit rate | 96.4%–99.9% |
| 20 问 CPU V cache 最终占用 | 约 12.2 GiB |
| 稳态提升 vs 修复前 all-on | 30.6% |
| 稳态提升 vs all-off | 2.169× |

新增 CPU admission 平均复制 0.445 GiB / 75.722 ms，每个 slab 的 host copy 与同一
slab 的 H2D 重叠。20 个 selected hash 与原四臂结果逐条一致，scorer fallback=0，最大
非重叠 closure error 为 0.001 ms。

由于新增 admission 只在 `stage-in=on` 时生效，前三个 arm 的原始结果仍是严格可比
control。以修复后的 all-on 替换旧 all-on 后，最终完整分解为：

| Arm | Score/select | Stage-out submit | Materialize critical | Reselect total | Query replay | Semantic total | Stage-in diagnostic | Assembly diagnostic | CPU hit | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all-off | 1,627.810 | 1,816.189 | 6,187.838 | 9,631.837 | 85.945 | 9,717.781 | 1,565.714 | 5,390.141 | 86.61% | 0 |
| pack-only | 1,247.100 | 317.888 | 3,075.055 | 4,640.042 | 88.359 | 4,728.401 | 862.521 | 3,074.959 | 86.61% | 0 |
| pack+stage-out | 1,256.425 | 8.002 | 4,977.359 | 6,241.786 | 90.900 | 6,332.687 | 4,964.136 | 4,977.177 | 0.06% | 0 |
| all-on（fixed） | 1,266.803 | 6.089 | 3,061.797 | 4,334.688 | 88.592 | **4,423.280** | 879.592 | 3,061.696 | 85.11% | 0 |

单位均为每次 semantic operation 的 20-query mean（CPU hit/fallback 除外）。最终
all-on 比 pack-only 进一步降低 6.45%，说明 stage-out 的 clean backing 只有和
stage-in CPU admission 配对后才形成可解释的正收益；修复前只打开 stage-out 会把
exclusive CPU working set 退化为重复 SSD read。

原始结果：

- `results/kvmem_archive_beam10m_c1/semantic_ablation_20260802_1152_v2/`
- `results/kvmem_archive_beam10m_c1/semantic_all_on_cpu_admit_full_20260802/`

---

## 4. 当前完成度

| 项目 | 状态 |
|---|---|
| On-disk format + v3 manifest | Done |
| Durable/direct-mapped raw-K/V | Done |
| Ladder capture/restore | Done |
| Read-only attach + absent page table | Done |
| Bulk index rebuild | Done（12K/32K/1M/2M 实测） |
| Archive arena preallocation | Done + 4K extent smoke verified |
| CoW overlay | Done + unit/E2E verified |
| Ladder-aligned truncate | Done |
| Residual truncate | Done |
| Frozen multi-question branch | Done |
| Runtime retrieval-policy A/B | Done |
| SIGKILL/resume | Done（主模型 + MTP 全 payload parity） |
| FP8 scorer end-to-end | Done；上述 A/B 全部 fallback=0 |
| MTP physical payload byte parity | Done |
| OpenAI Serve API | Done + A→B→A smoke verified |
| True GGUF content hash | Done（v3 SHA-256 + stat-validated cache） |
| 1M scale | Done |
| 2M scale | Done（含真实 SIGKILL/resume + attach/query） |
| 10M scale | Done（9,998,336 tokens；build + attach + 20-query ablation） |
| Archive SSD→CPU heat-cache admission | Done；20-query CPU hit/perf verified |
| Existing process-local cache fingerprint split | **Not done**；不阻塞 archive A/B |

---

## 5. 当前使用方式

```bash
# Build
QW3_Q8_BF16_MAIN=0 ./build/qw3 archive build \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --kvmem-archive /path/to/archive \
  --ctx 40960 --archive-tokens 32768 --archive-ladder-tokens 8192 \
  --kvmem-budget 8192 --kvmem-gen-budget 4096 --kvmem-cpu-gb 8 \
  --prefill-chunk 2048

# Inspect without loading the model
./build/qw3 archive info --kvmem-archive /path/to/archive

# Attach/truncate/query; repeat --archive-question for frozen branches
QW3_Q8_BF16_MAIN=0 QW3_KVMEM_TRACE=1 ./build/qw3 archive query \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --kvmem-archive /path/to/archive \
  --ctx 40960 --archive-tokens 20480 \
  --kvmem-budget 8192 --kvmem-gen-budget 4096 --kvmem-cpu-gb 8 \
  --prefill-chunk 2048 --kvmem-retrieval-method mean-k \
  -n 40 --archive-question "..." --archive-question "..."

# Dedicated frozen archive HTTP server.  The archive prefix is attached once;
# clients send only each new query.
QW3_Q8_BF16_MAIN=0 ./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 18081 \
  --kvmem-archive /path/to/archive --archive-tokens 32768 \
  --ctx 65536 --kvmem-budget 8192 --kvmem-gen-budget 4096 \
  --kvmem-cpu-gb 8 --kvmem-gpu-memory-ratio 0.5 \
  --prefill-chunk 2048 -n 40

curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-Q8_0.gguf","messages":[{"role":"user","content":"..."}],"temperature":0,"max_tokens":40}'
```

当前硬约束：

1. Archive 只支持 FP8 + immutable raw-K。
2. Q8_0 模型在本机需 `QW3_Q8_BF16_MAIN=0`。
3. `--ctx` 必须覆盖 archive prefix + query/decode reserve。
4. Archive physical `block_tokens`、page size、model layout、MTP presence 必须匹配；budget/retrieval/index placement 不需要匹配。

---

## 6. 剩余任务（优先级）

1. **Index rebuild 继续优化**：评估 double-buffer/并行 pread、I/O 与
   H2D/reduction overlap，或可选 policy-keyed derived-index sidecar。10M attach 的
   226.9 s 是一次性固定成本，不计入多问题 query，但仍是冷启动瓶颈。
2. **大规模 preallocation A/B**：4K extent smoke 已通过；10M 新 archive 可用，但
   raw-K/V 仍分别有 906/1,255 extents，需要和显式未预分配同规模构建做严格 A/B。
3. **并发 read amplification**：双进程 correctness smoke 已通过；仍需在 1M+
   archive 上测 page-cache/DONTNEED 下的 aggregate bandwidth。
4. **Power-loss durability（可选）**：当前已验证 SIGKILL/进程重启；若需要承诺主机
   掉电恢复，还需在发布 manifest 前对 raw-K/V、tokens、state、bitmap 和目录执行完整
   fsync 顺序。
5. **Existing process-local cache fingerprint split（可选）**：不阻塞 archive A/B。

---

## 7. 已知风险

- KVMI-012（normal-attention selected window 与累计 DeltaNet history 语义不完全一致）不会被 archive 自动修复。
- Overlay scratch 位于 `--kvmem-nvme-dir` 或 `/tmp`，不在 sealed archive 内；生产部署需给它配置足够且快速的本地盘。
- v1/v2 archive 没有模型内容摘要，只能继续使用旧的 size/layout 校验；需要强身份保证时应重新 build 为 v3。
- direct-mapped arena 的 capacity 按 `--ctx` 预留；超长实验应先检查 sparse-file logical size 与真实可用空间。
- 多进程 attach 共享 kernel page cache；启用 `DONTNEED` 时仍需测试并发 read amplification。
- 当前 crash/resume 保证针对进程崩溃/SIGKILL；尚未把 manifest commit 定义成断电后仍成立的 fsync durability barrier。

---

## 8. Working tree 范围

Archive 主体新增/修改包括：

```text
CMakeLists.txt
include/qw3/device_backend.hpp
include/qw3/kvmem_archive.hpp
include/qw3/kvmem_store.hpp
include/qw3/nvme_kv_tier.hpp
include/qw3/qw3.hpp
src/kernels_cuda.cu
src/kvmem_archive.cpp
src/kvmem_archive_cli.hpp
src/kvmem_store.cpp
src/qw3_cli.cpp
src/qwen_executor.cpp
src/qwen_executor.hpp
src/qwen_native_backend.cpp
tests/kvmem_archive_test.cpp
tests/kvmem_store_test.cpp
```

本文件只描述 archive 相关改动；worktree 中已有的 build 目录、profiling 日志和其他用户文件不属于该功能。
