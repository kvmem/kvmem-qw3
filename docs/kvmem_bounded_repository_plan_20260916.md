# KVMem 有界后台管理实施与验收

状态：2026-09-17 实现与本轮验收完成；功能默认关闭，须显式启用。

最终结论与参数、全部验证依据见
`docs/kvmem_bounded_repository_validation_20260917.md`，使用说明见
`docs/kvmem_bounded_repository.md`。此处后半部分保留分阶段工作记录，
其“待完成”等措辞描述当时状态，以最终验收报告为准。

已验证固定 8K 仓库累计处理 128K 历史，CPU/NVMe 物理槽位复用、64 位历史、
MTP、完整 query replay、实际 Claude Code 与真实视觉 HTTP 连续请求。
当前 host 回归 25/25 通过，重点 sanitizer 回归 3/3 通过。
受控三次 off/shadow/delete 对照未显示明显吞吐退化；不能承诺零延迟开销。

真实文档质量消融已完成：32 题，512K 仓库 off/shadow 均 20/32，
64K 仓库 delete 为 12/32，存在明确的容量/遗忘代价。
全部 1,035 请求成功，首请求之外 1,029 HIT/0 MISS。
这是独立 4B 在线保留消融，不是正式 MemoryAgentBench 成绩，不混入 9B/27B。

## 目标与边界

将累计历史与有界的 KV / metadata / index / page 映射分离，在固定 CPU 或
CPU+NVMe 预算内循环回收冷历史，累计追加量至少达到容量的 10 倍。保留原有
guided query、selection、query replay 和 MTP 计算。支持 Claude Code 连续请求，
包括有界尾部重写与 checkpoint 恢复。GC 本身只在 CPU 管理资源，不发起 CUDA
计算、传输、同步或设备释放；实际服务延迟必须测量，不能用 CPU-only 代替证据。

有限仓库允许遗忘；不承诺有限存储无损保留无限历史，也不承诺任意突发输入下
绝不等待。客户端每轮完整 transcript 的网络量仍随历史增加。KV 有界、服务端
消息处理有界和客户端传输量必须分别报告。

现有 gc v1 的规则、保护条件、NVMe allocator generation 与空洞掩码可复用。
不得通过取消长度检查、保留无限 tombstone 或关闭 MTP/replay 宣称目标完成。

## 实施前现场

- 基线提交：2321313。已有修改：qwen_executor.cpp、qwen_native_backend.cpp、
  LongMemEval 文档及评分脚本；保留，不回滚、不混入本任务的验证结论。
- 正在运行 9B AgentLongBench，campaign PID 234867，冻结 qw3 PID 234875，
  reader PID 235559，端口 18124。新代码使用独立构建目录，不能覆盖冻结程序。
- 30 分钟自动检测仍保持取消。此目标不是重新启用定时任务的授权。
- 本任务不使用新的子代理。

## 里程碑（全部完成方可关闭目标）

1. [x] **有界资源身份与生命周期（独立 CPU 基础模块）。** 固定数量槽位；64 位历史位置和单调身份；
   handle=(slot,generation)；准备/存活/退休/实际释放分离；引用跨线程安全；
   旧完成事件不能释放或发布新一代内容。释放未完成不能进入 ready pool。
2. [x] **资源池与检索接入。** main/MTP raw-K/V、CPU cache、NVMe、Mean-K / Adaptive
   index、GPU page 映射均按物理容量分配。槽位不代表历史顺序；新目录、索引及
   scorer 结果属于同一版本。所有 selector、all-fit、pressure 路径正确处理空洞。
3. [x] **规则 GC 与资源储备。** 复用已有规则，不训练模型；按各资源池的
   used+reserved+retired 水位提前回收，限制后台扫描时间与内存流量；前台只领
   已准备资源，不代替后台 GC。明确容量不足、保护集合过大及事件丢失行为。
4. [x] **位置、checkpoint 与 MTP。** 历史位置不受物理 ctx 上限绑定；局部窗口
   位置仍满足模型限制；main/DeltaNet/MTP 及尾部状态一致。checkpoint 引用有界，
   不复活删除块；普通追加不因旧历史存在空洞而全量 prefill。
5. [x] **Claude Code 服务接入。** 保持原协议，前缀验证与历史表示有界；尾部
   改写恢复及超出回滚保留范围的行为明确。检查文本与多模态、请求取消和异常。
6. [x] **验收。** 下列功能、质量、性能全部有可追溯产物；记录残余限制。

## 先行基础模块契约

新增 CPU-only KvMemRepository 管理有界身份与资源租约。固定 slot 是资源束的
地址，而不是历史 block id。单前台生产者预留/发布新块，单后台回收者在资源
释放完成后归还 ready pool；其它读者通过 lease 保持 payload 生命周期。

基础模块不会自动接管现有 serving，不能将独立单测通过描述成端到端支持。
后续接入必须覆盖 CPU/NVMe/index/MTP 的完整资源束；异步 I/O 必须持有 lease，
generation 检查不能代替等待读写完成。已开始部分释放的资源不能再被救回。

## 验证矩阵

- CPU：小池循环至少 100 倍容量、位置跨 32 位边界、旧 handle、延迟 I/O、
  checkpoint 引用、取消、部分释放重试、全保护无 victim、尾部重写、资源耗尽。
- 并发：前台发布/选入与后台退休/复用交错；ASan/UBSan，TSan 能运行时执行。
- 存储：真实 NvmeKvTier 和 CPU payload 的新旧字节核对；统计存活、退休、
  在途与空闲资源，未释放物理槽位不得计为空闲。
- 检索：同一有效目录的 CPU 参考与 GPU 结果一致；删除块不进入 softmax 分母；
  单/多 prototype 以及 MTP 配套资源一致。
- 延续：累计输入 >=10x 物理容量；RSS、metadata、index、NVMe 槽位/文件保持
  平台；常规 append 不因 GC 全量 cold prefill，guidance/replay 不被省略。
- 性能：固定执行轨迹下 off/shadow/delete 先隔离管理开销；记录 TTFT/TBT
  p50/p95/p99、decode token/s、CPU tick、ready 水位、分配等待与 GC miss。
  真实轨迹另报任务得分、后续复用、额外 prefill 时间，不能仅用删除后访问次数
  下降证明策略正确。至少 3 次受控重复，报告波动，不预设“零开销”的结论。
- GPU 性能实验等现有 GPU 推理结束后串行执行，不并跑污染测量。

## 工作记录（历史状态）

- 2026-09-16：创建目标，核对当前 v1 限制与活动实验；开始有界仓库基础模块。
- 2026-09-16：新增 `include/qw3/kvmem_repository.hpp` / `src/kvmem_repository.cpp`。
  预分配 metadata 与 SPSC ready queue；lease 获取与退休共享原子状态字，
  先取得引用再检查 generation；资源部分释放后不可 rescue。取消的未发布写入
  与正常退休区分，不能通过旧 I/O 引用将未提交数据重新发布。
- 测试：4 槽位 / 4096 次追加，跨 32 位历史位置；4 槽位 / 400 轮真实 NVMe
  多资源束复用；8 槽位 / 20000 次追加与 3 个读者、1 个回收线程交错。
  并发测试重复 20 轮通过。CPU core 构建及 GC/repository/NVMe 三项回归通过。
  ASan/UBSan/LeakSanitizer 通过；TSan 在执行测试前因 `unexpected memory mapping`
  退出，不能视为通过。产物位于 `benchmark/kvmem_bounded_gc_20260916/`。
- 下一步是将新 handle 接入 KvMemStore、CPU/NVMe 资源及索引目录，再接后台规则与
  reserve admission。**当前 serving 尚未使用新仓库；未解除 ctx、未支持 GC+MTP，
  未完成真实长任务验收。** 不得把本阶段的 CPU 测试等同于目标完成。
- 2026-09-16 后续进展：KvMemStore 新增显式有界仓库构造路径，metadata 固定
  按 slot 容量分配。历史位置/身份与物理地址分离；top-k、quota、pressure、
  all-fit、窗口重映射使用历史顺序，语义组支持 64 位历史区间并拒绝删除空洞。
  scorer 返回、selection 与 tier/IO 元数据更新必须携带 generation；GPU
  驻留、scoring frame、旧窗口和 checkpoint 可持有 repository lease。
  新增 `retire_if_unpinned`，规则 GC 不能退休正在评分/执行/I/O 引用的块。
- 新验证：实际 KvMemStore 在 8 个槽位上执行 2048 轮追加/打分/选择/重映射/
  冷块回收，metadata 容量保持不变；回收 2040 次。覆盖低编号槽位装入最新
  内容、旧分数/写回/选择拒绝、checkpoint/在途窗口保护、跨 32 位区间、
  缺历史的语义组不被部分选择。4 项定向回归及 Store ASan/UBSan/LSan 通过。
  这仍是 CPU 存储/选择集成测试，不是实际模型的 256 倍容量运行证据。

## 下一轮接入注意事项

- 当前 `KvMemStore(cfg, repository)` 只有测试调用；executor 尚未构造此路径。
  不要先加 CLI 开关放行半完成的 serving。里程碑 2 仍未完成。
- 有界模式的 `block_count()` 是物理目录槽位数，含空洞；`orig_pos_start` 是
  `slot * block_tokens` 的有界物理寻址；历史排序/请求 spans 必须改读
  `history_pos_start` / `history_span_blocks()`。legacy 模式语义不变。
- 下一项先审计 executor 的写 KV page 地址、raw-K chunk/valid bits 以及
  index capture 地址，避免只改 store 后将累计 `position_` 当物理 slot 使用。
  不能为了地址不连续把原有 prefill 任意退化成逐 block 小批次。
- Mean-K 固定 stride 应按 slot 容量，scorer 必须保留 frame leases 到选择安装，
  counts 在 softmax 前排除空洞。Adaptive prototype arena 仍是 append-only，
  需要有界可复用的 prototype 资源，不能只清 metadata counts。
- 非完整尾块的追加、版本化 truncate、checkpoint restore、MTP 状态及 backend
  长度限制仍待接入；新 store 对旧式 unversioned append/truncate/score/select
  显式拒绝，不能把这些临时护栏当成最终功能边界。

## 2026-09-16 地址与批量追加进展

- Store 新增有界历史目录。每个物理槽预分配一个树节点，复用时移动该节点；
  目录最多为仓库容量，不保留累计历史的 tombstone 列表。区间查找复杂度为
  O(log(capacity) + 涉及块数)，chronological 遍历不再每次排序整个目录。
- `repository_storage_read` 返回历史到物理地址的 spans 以及对应 leases，
  未写完、被删除或已复用的历史地址明确失败，不能命中新一代内容。
- `prepare_repository_append` 一次预留整个普通 forward 所需的块，ready pool
  容量检查为 O(1)。不会在前台扫描 GC。新块在写完并 commit 前保持 Writing；
  部分尾块的有效长度直到 commit 才增加。`commit(accepted_tokens)` 支持仅发布
  接受的前缀，取消的后缀经后台释放，IO lease 未结束不能复用。
- 注意：Repository Record 的 `token_count` 是不可变的预留 extent；Store 的
  `n_tokens` 才是已提交有效长度。索引、checkpoint 不得把 extent 当成有效 KV。
- `kvmem_storage_page_spans` 在 CPU 上生成可用于一次批量前向的页别名，包含
  首尾非整页；历史位置支持超过 32 位，物理/窗口位置受各自容量限制。没有
  将不连续物理地址退化成逐 block prefill。当前仅在 CPU 测试中消费该计划。
- CPU 验证：8 槽位，4096 轮 1..113 tokens 批量追加，从 2^34 历史位置开始，
  每轮核对物理 payload 和窗口页寻址，目录不超过 8 项；另覆盖部分 MTP 接受、
  取消、迟到 IO、ready 不足不产生半次追加、尾块续写与被删除地址拒绝。
  4 项定向回归及 ASan/UBSan/LSan 通过。真实服务尚未启用，因此没有新增的
  GPU 性能或模型质量结论。

### Executor 接入的确切落点

1. `ensure_kv_pages` / `prepare_mtp_prefix_pages` 和另外四处直接 MTP ensure
   需要消费一次批量 append 的 spans；`kvmem_extend_window_for_decode_n` 及 MTP
   版本需要消费页别名，不能再使用 `true_page = history_pos / page_size`。
2. 当前 `ensure_logical_page_ranges_resident` 在 host vector 扩大时会分配中间
   所有 GPU 页。有界模式应在配置阶段预分配固定 `pages=-1, owned=false` 元数据
   及设备页表，再让 range ensure 只分配触及的页。不能直接对稀疏高槽位调用
   旧 `ensure_pages`，否则会把前面的空洞全部分配出来。
3. 标准 prefill 每个 chunk 在 raw-K flush 后调用 `kvmem_register_until`；
   必须将这里改为提交该 chunk 的 append 事务。MTP verify 和 provisional query
   需要延后到接受/回滚决定后提交，不能混用普通 prefill 的提交时点。
4. `kvmem_flush_raw_k_capture`、Mean-K capture 的 dense D2H 可保留；CPU scatter
   目的地改为 span 的有界地址。GPU remap 的 `orig_pos_start` 可沿用物理地址，
   但 baked-position 修正中反向遍历并按累计位置 break 的代码必须替换。
5. 此轮未改 executor/backend；原有未提交修改仍字节相同。没有开放半完成的
   CLI 开关。里程碑 2、3、4、5、6 均未完成；不得将上述 host 验证当成服务验收。

## 2026-09-16 有界后台规则线程进展

- 新增 `KvMemRepositoryGc`，并接入 `KvMemStore` 的块发布、有效尾块提交、tier/
  I/O 状态变化、请求 step、评分与 working-set 选择。后台不读取 Store 的
  vector/map，也不持有 DeviceBackend；资源释放通过 generation-bound 回调。
- 复用 `kvmem_gc_score`，保留 recency/frequency、近三次相对 Mean-K 分数、
  低分连续观察、选择阈值邻近保护、candidate/grace 规则。缺评分事件打断低分
  连续证据；旧 generation 的分数拒绝。全相同分数不能被当作语义低分证据。
- ready 低水位触发补充，达到高水位停止；每 tick 扫描和释放数量有上限。
  已退休但释放未完成的块不进入 ready；同时计入待释放数量，防止 allocator
  忙时无限扩大退休集合。毫秒级 tick 预算是协作式检查；score 排序、回调耗时
  均计入 max_tick_ns，不承诺实时调度上界或服务零开销。
- GPU/IO、sink/recent、mandatory、checkpoint/read leases 均保护块。
  所有 main/MTP raw-K/V/index 就绪标记必须由资源所有者在实际完成后发布；
  CPU-only authority 同样可以被回收，不要求一定存在 NVMe。后台以原子快照读
  metadata，并在持有独占 lease 的退休 CAS 后复查 publication 版本，发生竞争
  就 rescue；没有把 generation 检查代替 IO 生命周期。
- Off 不删除 live 内容，仍清理取消的未发布写入；Shadow 维护虚拟空闲水位、
  不实际删除，并统计候选之后的使用；Delete 实际退休和重用。三者的端到端
  质量/性能比较尚未进行。
- 验证：真实后台线程驱动 Store，CPU 16 槽位持续写入 512 块；实际 NvmeKvTier
  三个 main raw-K / V / MTP 资源池以 8 槽位写入 256 块，逐次核对数据与
  allocation generation。二者均为 32 倍物理容量的 host/storage 验证。
  另覆盖保护集合、低分优先、score gap、旧事件、partial cleanup/exception
  重试、取消时延迟 IO、off/shadow 不删除。5 项定向回归与 ASan/UBSan/LSan
  通过。该结果不代表真实模型已在 32 倍容量上运行。
- 上述真实后台 CPU/NVMe 测试另外重复 10 轮全部通过，日志见
  `benchmark/kvmem_bounded_gc_20260916/background_gc_repetitions.log`。
- NVMe 并发测试曾错误地要求 `try_release_allocation` 一次返回 true；真实
  allocator 锁竞争时 false 是正常非阻塞返回。测试改为有期限地重试，并在
  重试成功后读回新一代数据。后台本来就保留 Releasing、下一轮再试。
- executor 仍未构造 bounded Store，资源回调尚未绑定服务中的 CPU/raw-K/索引/
  NVMe/MTP 实例。因此里程碑 3 的策略线程已实现，完整资源接入和里程碑验收
  仍未完成。下一轮必须推进 executor 接入，不能以独立测试代替模型验证。

## Executor 地址接入与评分目录（后续实施）

- `QwenExecutor::KvPageTable` 新增固定物理目录：先初始化空洞，仅为触及的
  槽位分配 GPU 页。主 KV 和 MTP 的 ensure / window extension 消费 Store 的
  批量历史→物理 spans。MTP prefix prime 使用批次局部页表，保留一次完整
  batch 的原有计算，不按块拆分 prefill。legacy 分支继续使用原来的页表。
- main/MTP raw-K 保留批量 D2H，CPU scatter 写物理槽地址；CPU Mean-K capture
  同样在 D2H 完成后按 generation-bound spans 写入固定槽位。索引维护
  `(slot,generation,valid_tokens)`，部分均值合并要求精确匹配旧有效长度。
- 新增评分 frame：持有全部当前块的 leases 到 selection 安装，按历史顺序
  屏蔽 sink/recent，并通过 `counts=0` 在 softmax **之前**屏蔽删除空洞。
  Mean-K 返回分数和 selection 安装走带 generation 的 Store API。索引不能
  用旧一代数据冒充复用槽位的新数据，也不能用整块均值冒充较短前缀均值。
- 本轮索引接入覆盖 CPU plain Mean-K；Adaptive、Fixed4/sub-block、semantic
  group 及 GPU index 的 bounded 映射尚未完成，已有明确拒绝路径。尚未打开
  runtime 构造入口，不能将这些暂时护栏当成最终支持边界。
- Append 可增量扩展预留（MTP draft/verify 跨块），提供持 lease 的 prepared
  write view。部分接受时先由前台释放完全落在拒绝后缀的 main/MTP GPU 页，
  保留最后一个部分接受页，再将取消槽位交给 CPU 回收；迟到 IO 未完成时
  不进入 ready。完整接受的普通 batch 跳过这段拒绝后缀清理。
- 新测试直接调用生产 `KvPageTable`（使用记录型 DeviceBackend，不执行
  attention）：从 2^34 的历史位置在 4 个仓库槽上追加 1,024 块，分别核对
  main/MTP 的物理字节和局部页映射。另覆盖跨不连续槽位的非整页 batch、
  部分接受后的页归还、迟到 IO、读写/索引旧 generation、按历史顺序的掩码、
  删除空洞、不同 token 覆盖，以及 legacy 分配行为对照。
- 当前全量 host build 和 24 项 host CTest 通过。Store 的 ASan/UBSan/LSan
  覆盖 append extension / score frame；最终日志见 progress.json 所列产物。
  这些测试没有执行模型 attention，**不构成 GPU 数值、真实 serving、质量
  或 TTFT/TBT 的验收结果**。
- 实施前 executor 的 7 行诊断修改仍逐字保留，其他三个已有修改文件的
  SHA256 不变。由于修改了相邻条件，原补丁不能直接 reverse-check；核验的是
  原新增完整片段仍存在，而非错误宣称 executor 文件仍字节相同。

### 下一步必须解决的运行时缺口

1. 在 CPU/raw-K/V/index/NVMe 资源所有者之间发布完整 resource bundle，再绑定
   `KvMemRepositoryGc` 回调和 bounded Store 构造。固定 arena 允许保留已分配
   的物理内存，但复用前必须清旧 generation 的 valid/backed/index 状态。
   回调不得直接读写 executor 的非原子 vector/map，更不能执行 GPU 释放。
2. 将 stage-out/prefetch/persistence 的完成事件绑定 handle 和真实 IO lease；
   目前大量 `set_block_tier` / `set_block_io_in_flight` 仍走旧参数形式。
   异常/取消必须先完成普通前台 GPU 清理，再放弃 Writing 预留。
3. 完成 snapshot leases、版本化 rewind / truncate，处理部分尾块、拒绝前缀
   索引重建和 replay 后缀。`reset_state()` 目前仍重建 legacy Store，不能启用
   bounded 服务后静默丢失仓库模式；`restore_state()` 的历史 page-count 截断
   也必须改为实际 physical ranges。不得仅靠跳过 truncate 来通过测试。
4. 分离 executor/backend 的 64 位累计历史、32 位物理地址和局部 RoPE 坐标；
   MTP prefix length、raw decode cursor、query/index cursors、请求检查和
   Claude Code 前缀状态仍有历史容量/uint32 假设。`--ctx` 尚未解除累计上限。
5. 完成 Adaptive 有界 arena 和掩码路径，随后才开放配置并运行真实模型
   >10x 容量与 off/shadow/delete 的质量、cold-prefill、TTFT/TBT 验收。

现有 9B campaign 未被改动；最近检查仍为 `waiting_protocol_preparation`，原
reader/server 已退出。GPU 实验前需重新核实当前所有权。30 分钟检测 timer 仍
为 disabled，本目标不重新启用它。

## 2026-09-17 资源生命周期接入进展

- 新增内部 `repository_blocks` 构造配置（CLI 尚不暴露）。executor 构造
  bounded Store、固定主/MTP 页目录及 CPU GC；reset 重建同一容量的仓库，
  析构/重新配置在 tier 销毁前停止 worker。尚未用真实模型验证此入口。
- 生产资源回调只持有可选 NVMe raw/V allocator，按 allocation generation
  非阻塞归还；main/MTP 共用的完整 record 一起释放。CPU raw/V/index arena
  保留有界分配，ready 交接后由前台重置旧一代元数据。
- 新槽领取时清 CPU tier 映射与访问热度、raw valid/backed、索引 stamp 和
  raw GPU cache 的 CPU 映射，检查主/MTP 旧 GPU 页已由正常前台流程归还。
  空 CPU slab 可保留并继续计入预算，避免每次复用重新分配。
- V 的后台写入、proactive D2H 和 prefetch 持有实际 repository leases；
  完成回调使用捕获的 generation 与有效 token 覆盖校验。前台 cache victim
  单独 pin；已经退休的缓存条目不再写回/复活。raw chunk 写回尚未完成同样
  的接入，因此 **不能据此启用 NVMe bounded serving**。
- 允许已提交块的版本化 tier/index 完成通知与 prepared append 重叠；历史
  目录、长度、selection 等结构修改仍拒绝。新增测试检查旧批完成不会发布
  下一批预留 token，以及读 lease 释放前不能回收。
- prefetch 异常/reset 路径先停止 GC，等 CPU future 和普通前台 copy stream
  完成后才销毁缓冲和 lease；device drain 失败会保留成员状态。proactive/
  packed D2H 的所有异常与取消边界仍需继续审计，不能宣称完整错误恢复。
- 生产回调新增 2 槽/128 次取消写入验证（包括没有 V 分配的部分资源束）；
  CPU-only/可选 raw tier 路径和旧热度清理测试通过。最终 24/24 host CTest、
  Store 与 GC 的 ASan/UBSan/LSan 均通过，日志为 `resource_binding_*`、
  `resource_store_asan*`、`resource_gc_asan*`。尚无真实 serving/GPU 数值与
  延迟、质量、cold-prefill 的新增验收证据。
- 其余前置缺口依旧存在：raw-K 写回不能将物理 chunk 内退休空洞重新分配；
  checkpoint/rewind/truncate、部分尾块有效覆盖、64 位累计位置、backend
  长度检查、Adaptive arena 和 Claude Code 前缀表示均需完成。
  内部构造接入不是解除 `--ctx` 累计上限，目标继续保持进行中。
- 原有三个非 executor 修改文件的 hash 不变，executor 原诊断仍保留。
  PID 234867 仍存在，原 reader/server 已退出，检查时 GPU compute 列表为空；
  本轮没有重启实验或已取消的 monitor timer。

## 2026-09-17 Raw-K 持久化与回退接入

- Store 增加按物理区间读取的版本化快照，包含通过 reservation 持有的
  Writing 块，跳过退休空洞。raw-K/main-MTP 的同步持久化、缓存重载、异步
  packed writeback 和读取均绑定实际有效区间及 payload lease。
- bounded 写回只打包发生变化且 main/MTP 捕获齐全的块；不因一个槽位复用
  重写整个 chunk，也不在完成通知中用旧 chunk 覆盖新预留槽。该路径已经
  使用 CPU 权威副本，因此移除了这段写回中的额外 GPU transfer fence。
  CPU 打包失败会撤销本条 staging 记录，保留同批之前的有效记录和租约。
- 生产 executor 存储测试（记录页表的 backend，不执行模型数值计算）在
  4 个物理槽上完成 128 轮 main/MTP raw NVMe 复用，核对实际字节和槽数；
  覆盖退休空洞、迟到完成、脏块单独写回、同步 eviction、缺失 CPU 副本
  的异常恢复，以及退休后的原始字节不可读取。
- 增加固定 32 槽 checkpoint 有效性登记；长期 checkpoint 持有 working-set
  租约，MTP 临时快照只登记有效性并要求同一 selection generation，避免
  每次 speculative cycle 扫描/复制整套块租约。回退使之后的 checkpoint
  永久失效，即使新分支再次增长到同一 token 位置也不能恢复旧状态。
- 增加 foreground rewind 事务：先完成 I/O，再释放被丢弃的 main/MTP GPU
  页、清有效 token 和索引 stamp，最后退休对应资源束供 CPU worker 回收。
  保留部分尾块的有效前缀；其旧 Mean-K 被明确失效，不能继续当作正确 seed。
  executor 存储测试覆盖超过物理容量后的回退、部分页保留和新分支追加。
- 主状态恢复、recurrent 恢复、query replay 和 checkpoint rebase 增加历史
  分支有效性检查；query replay 的历史排序使用历史坐标而非物理槽号。
  MTP per-row checkpoint 额外绑定 append serial 与 selection generation；
  拒绝的物理页由最终 accepted-prefix commit 释放，不再按累计位置裁目录。
- **仍未完成服务验收。** 部分尾块 Mean-K seed 的 raw-K 重建仍须接入；
  finalize-content-index 等路径的历史/物理寻址、所有 I/O 取消异常、64 位
  executor/backend 累计位置、模型局部 RoPE 边界、Adaptive/其它索引布局及
  Claude Code 有界前缀仍待完成。不能开放 CLI 或声称无限持续 serving 已可用。
- 验证产物：`checkpoint_binding_build.log`、`checkpoint_binding_ctest.log`、
  `checkpoint_asan_build.log`、`checkpoint_asan_ctest.log`。模型质量、TTFT/TBT、
  cold-prefill 和真实模型累计 10 倍容量实验仍是待验收项。

## 2026-09-17 CUDA 索引与真实模型持续追加验证

- 部分尾块的 Mean-K 可从保留的 raw K 重建。只对 generation/有效长度失配的
  块运行现有 widen/归约 kernel，使用固定 tile 的 scratch；恢复前缀均值后再
  合并新增行。不重新 prefill 被保留的历史。物理空洞不作为历史内容参与评分。
- CPU index authority 重分配会失效其覆盖证书；较短的部分前缀不再被错误地
  当作已有完整均值。CPU-demoted prefix checkpoint 保留仓库有效性对象。
- 独立 CUDA 数值测试覆盖 FP32 raw/FP32 index 和 FP8 raw/FP16 index，64 轮
  rewind/append/reuse；累计 4096 tokens、128-token 物理容量，核对实际归约
  值、部分尾块合并、CPU index 重建和 checkpoint 作废。它只测试索引 kernel。
- 新增显式模型探针 `qw3-kvmem-repository-model-probe`，不加入普通 CTest。
  使用真实 4B Q8 权重、FP8 KV、CPU plain Mean-K、8192-token 物理容量、
  1024-token 选择预算、512-token 生成储备。每轮 128 个合成 teacher-forced
  tokens；检查真实前向结果、位置、query/index 就绪、固定目录及资源占用。
  这不是数据集质量实验，也不是 HTTP serving 或 latency 对比。
- 首次真实前向暴露窗口 attention 热度仍走无版本接口，已改成捕获原窗口
  generation 后校验整帧再累计；旧 generation 或重复块不能部分污染新分数。
- MTP 前缀位置不再被物理 ctx 裁短，地址仍经有界物理页映射检查；MTP draft
  会补齐刚完成 priming、尚未加入其窗口的连续尾部，避免与主窗口出现 page gap。
- 私有 guided-query 回滚只取消未提交尾部，不清空已发布的 query Q 与保留的
  content index。多轮 MTP verify 可在同一未提交事务内继续，checkpoint 必须
  匹配 append serial/selection generation，且不超过当前有效执行位置；
  registered position 始终取已提交历史边界。
- 压力测试曾在约 4.8～6 倍容量出现 ready 耗尽，失败 CSV/log 均保留。补上：
  低水位前的候选观察、按最大 forward 批次计算默认储备，以及评分租约释放后的
  CPU 唤醒，避免固定轮询反复错过短暂释放窗口。前台通知后立即继续，不等待 GC。
  这些改动的独立性能贡献尚未做消融，不能把所有失败都归因于使用频率。
- 有限容量的保留优先级与资源安全保护分开：先选普通冷块；完整扫描后仍不足，
  才允许按分数使用 level-3 高频/近期高分候选，单独计数。GPU/IO/reader/
  checkpoint/mandatory/anchor 保护仍严格执行。单测覆盖全热历史与必留块；
  质量代价仍待真实数据集评估。
- 已完成真实模型 102400-token（12.5 倍容量）前向+选择+回收；带 MTP 的
  紧密追加重复运行也达到同一长度。guided-MTP 模式执行 198 次、每次 8 个
  query tokens 的生成/接受拒绝/回滚，再消费 Q 做重选。固定目录 256 槽，
  index 4 MiB；MTP 模式稳定后的外部 KV 约 200 MiB、GPU KV 18 MiB。
  `reclaimed` 包含未提交私有分支的取消，不能全部算作策略删除。
- 产物位于 `benchmark/kvmem_bounded_gc_20260916/`，详见 progress.json 与
  model_probe*_validation.json。构建/部分 CPU 测试与功能探针并行过，
  不从这些 CSV 宣称正式吞吐/TTFT 性能。GC CPU tick 也不能代替服务延迟测量。

**后续仍必做：** 完整 query replay 与实际请求续接；64 位 executor/backend
历史坐标和模型局部位置边界；有界 Adaptive/fixed prototype arena；Claude Code
历史表示及请求取消/所有在途 I/O 错误恢复；独立 serving >=10 倍容量、普通追加
无 GC 引起的 cold prefill，以及 off/shadow/delete 的质量和延迟对比。内部入口
仍不暴露 CLI，原 9B 冻结实验及已取消的 monitor timer 未修改。目标未完成。

## 2026-09-17 完整 query replay 与真实 MTP decode 验证

- 修复历史尾部 pin、request mandatory span 和 bounded recent pin 的寻址。
  这些范围使用历史坐标，不能将复用槽位编号当成时间顺序。request 必保块取得
  generation-bound lease，替换失败保留旧 pins；清空、reset、reconfigure 释放。
  单测覆盖低物理编号的新历史、预算去重、已删除区间和 mandatory 生命周期。
- 首次完整 replay 测试在下一轮出现 `attention accumulator lost its window
  identities`。原因是截去重放尾部时只缩短 block ID 元数据，未更新累计分数的
  generation 列表和 representative-K 的 head stride。现在截短前 drain，截短后
  重建该窗口的代表 K 与 accumulator，继续保留完整计算。失败日志是
  `model_probe_replay_attempt1.log`，没有抹去失败轨迹。
- checkpoint admission 区分“当前均值可直接评分”与“部分尾块可从保留的 raw-K
  修复”。较早的部分 checkpoint 不再因当前均值覆盖更长尾部而被误拒绝；缺 raw-K
  仍拒绝。CUDA oracle 覆盖 FP32、FP8 的判断、真实修复和后续增量合并。
- 最终冻结测试使用 Qwen3.5-4B Q8_0/MTP、FP8 KV、CPU plain Mean-K、无 NVMe，
  physical ctx=8192、block=32、selection/prefill=1024、generation reserve=512、
  CPU tier budget=1 GiB。800 轮，每轮 128 synthetic 输入 + 32 个实际 target
  接受的 decode tokens，共 128000 tokens，即物理容量的 15.625 倍。
- 完成 198 次 guided Q、211 次 query replay（含 13 次相同 context 下重复 replay），
  25600 个提交的 decode tokens；14848 个 decode 周期接受 10752 个 draft tokens、
  拒绝 44496 个候选行。guided 诊断忽略 stop markers，固定长 decode 忽略 EOS，
  因而这些是功能压力数据，不是自然 agent 轨迹或任务质量结果。
- 13 次重复恢复逐字节核对 main hidden、MTP prefix hidden、全部 recurrent/conv
  状态、main/MTP raw-K、重放尾部 GPU K/V，并核对下一 token argmax/logit；全部一致。
  这证明相同边界和最终 context 的重复计算一致，不证明删除策略不损失任务能力。
- 第 200 轮后：physical metadata=256 slots，CPU KV=209682432 B，索引=4194304 B，
  GPU KV=19464192 B，RSS 峰值计数为 6427484–6427500 KiB，ready=20–25。GPU KV
  不包括权重、激活、checkpoint 等；GC reclaimed 含私有/重放后缀取消资源，不能
  将其全部记作策略删除量。后台 tick 不等价于服务额外 latency；本轮不做性能结论。
- 回归：Host CTest 24/24、CUDA index/prefix 2/2、ASan/UBSan/LSan 3/3，通过。
  Source lock、源码 zip、CSV、日志和验证 JSON 均在本任务 artifact 目录。原有未
  提交修改保持，冻结 9B 实验没有变更，已取消的 monitor timer 未启用。
- 下一步仍需检查 pressure selector 在候选产生到 pin 之间的 GC 竞争及异常收尾，
  完成 executor/backend 历史位置 64 位化、Claude Code 服务接入和有界前缀表示、
  Adaptive/fixed index、真实 NVMe/多模态/取消请求，以及 off/shadow/delete 的质量
  与受控 TTFT/TBT/吞吐对比。当前没有发布面向正式服务的开关，目标仍未完成。

## 2026-09-17 压力选择并发与 64 位历史验证

- 压力选择现在在同一批存活块租约内选取候选，并把选中块的租约保留到工作集
  安装完成。此前仅返回物理 ID，后台可能在选择后、安装前回收候选。新增真实
  CPU 线程交错测试，在这个间隙尝试删除：只允许未选中块退休；临时租约释放
  后由已安装工作集继续保护，清空工作集后才允许回收。
- Executor 的累计位置、main/MTP prefix、snapshot、MTP verify checkpoint、
  raw-K decode 捕获、query span/重叠计算、Mean-K 捕获历史 block ordinal、
  mandatory/oracle spans 及尾部 pin 改用 uint64。物理槽位、GPU 页和紧凑窗口
  仍保持容量内的本地索引。新增 pin 乘法溢出和本地 kernel 位置范围检查。
- 修正 MTP prefix prime 对历史位置的误用：已计算 target 行通过仓库 spans
  定位，不把它们当作新的绝对地址追加；KV-only 写入映射到有界批次页表。
  MTP checkpoint 完整性检查只遍历租约保护的当前代块，并区分历史范围和
  物理地址。累积位置不会进入局部 RoPE。
- 新功能验证使用同一冻结源码和 CUDA 程序：origin=0、origin=2^40 各 800 轮；
  每轮 128 个输入 tokens + 32 个实际接受的生成 tokens，总计每项 128000，
  为物理 8192 容量的 15.625 倍。每项执行 198 次 guided query、211 次 replay，
  并进行 13 次重复 replay 字节核对，覆盖 hidden、recurrent/conv、main/MTP
  raw K、GPU K/V 及下一个 token 的 argmax/logit。
- 两项在第 200 轮后 CPU backing 固定为 209682432 B，索引固定为 4194304 B，
  GPU KV 固定为 19464192 B。高起点项的 RSS high-water 有不到 1 MiB 的小幅
  上升；不能把有界 payload 等同于整个进程 RSS 从第一轮起严格不变。详细
  ready/RSS 区间见报告。
- 验证：CPU 全量 CTest 24/24；GPU index 与真实 prefix storage 往返 2/2；
  Store/页表/GC 的 ASan+UBSan+LSan 3/3。Prefix storage 另外核对 origin=0 和
  2^40 时的全部历史元数据，不允许 CPU/GPU 冷热迁移截断位置。
- 本轮是功能测试。高起点部分过程与 CPU 编译/内存检查准备重叠，不用于报告
  性能差异。GC reclaimed 包括取消的 speculation/query/replay 预留；合成
  tokens 也不提供 agent 准确率结论。
- 产物：`history64_source_lock.json`、`history64_sources.zip`、
  `history64_model_validation.json`、`history64_serving_followup_audit.json`。
  原有用户修改保留：两个实验文档/评分脚本 hash 不变，原 executor 诊断和
  backend bootstrap 修正逻辑仍在，涉及位置的日志格式同步改为 64 位。
- 下一项必须推进服务层接入：`append_base`、`KvmemReuse` 和 prefix helper
  仍为 32 位；generate 仍拒绝累计 prompt > ctx，并按累计长度限制输出；
  `kvmem_warm_log_` 和视觉历史副本仍累计增长。不能单独移除 ctx 检查。
  此外 semantic-chunk 的 all-fit 判断不能再把包含空洞的物理 block_count
  视作可用历史量。上述位置见 followup audit，避免下一轮重复已完成长测。
- 对外 CLI/HTTP 尚未启用；Adaptive/fixed prototype、真实 NVMe/多模态、异常
  与取消、Claude Code 持续请求、off/shadow/delete 的质量和受控延迟仍未验收。
  冻结 9B campaign 未改动，已取消的检测 timer 仍为 disabled。

- HTTP 接入回归修复：storage-only 回合已有有效短尾 P 时保留已组装窗口，避免结束时
  再次 pressure assembly 令其过期；post-selection checkpoint 不得落入 scoring query
  内部，live-suffix replay 起点与下一轮可复用 P 分离。全部 semantic/query 计算保留。
  新增 CLI experimental repository 选项；HTTP 逻辑 token 上限明确为 UINT32_MAX，
  物理 ctx 独立；serialized text only 护栏保留，多模态待验证。

## 2026-09-17 HTTP、NVMe、受控延迟与多模态进展

- CPU 和 NVMe 的 native Anthropic HTTP 合成轨迹各完成 640 请求、87115 tokens，
  为 8192-token 物理容量的 10.63 倍。两项各 639 次前缀命中、0 次常规 miss，
  119 次 private guided query、158 次完整 replay。尾部改写复用成功；旧前缀
  改写与取消后续请求安全冷重建。`http_cpu_validation.json`、`http_nvme_validation.json`。
- NVMe 跑在实际本机 NVMe 文件系统，main/MTP raw K 和 value backing 均有固定预算。
  CPU backing 75497472 B、index 4194304 B；NVMe used 135954432–144506880 B，
  不随累计历史增长。功能测试存在 CPU 构建重叠，不拿其 wall time 比较 CPU/NVMe。
- 受控 native callback timing：三轮 off/shadow/delete、每项 86 请求、32 输出 tokens。
  每次 delete run 实际删除 22 个冷块，所有模式均 0 prefix miss。三轮中位数
  decode 为 300.19/298.63/299.95 token/s，request effective 为
  186.73/185.37/186.19 token/s。delete TTFT p50/p99 为 37.459/320.930 ms，
  off 为 37.051/320.062 ms。见 `timing_validation.json` 及各次 CSV/配置/源码锁。
  这不是大容量/27B/多模态性能结论，也不证明严格零影响；MTP callbacks 的成批
  输出令 TBT p50 接近零，不能把它倒数当作吞吐。后续多模态修复后的源码需补测。
- 多模态紧凑位置缓存改用固定物理 arena，选中块的源坐标仍从历史位置读取；
  slot generation 变化和 rewind 仅清理对应区间。普通 bounded 请求结束后释放
  executor 持有的完整输入视觉副本，保留有界窗口/checkpoint 元数据。
- 新长测发现 MTP window-frame 局部 slot 被误当成源历史坐标；gdb 已定位并修复。
  generated-text priming 使用 target 提供的标量位置，避免再次推进视觉 tail。
  保留 `mrope_model_long.log` 和 `mrope_model_debug1.log`，不抹去失败。
- 修复后真实 4B Q8/FP8/MTP 模型用非均匀三轴位置和合成视觉向量完成 800 轮、
  128000 tokens、211 replay、25600 decode tokens；13 次 repeated replay 比较
  main/MTP hidden、recurrent/conv、raw K、GPU KV 及下一 token/logit，均相同。
  CPU backing/index/GPU KV 固定，视觉 compact arena 为 8192 行。测试端完整
  输入和 embedding 向量仍增长，RSS 高水位不能解释为全部服务内存恒定。
  见 `mrope_model_validation.json`、`mrope_fixed_source_lock.json`。
- CPU CTest 25/25；视觉页表/地址 ASan+UBSan+LSan 1/1。真实图片 HTTP 正在验收。
  永久 pin 所有旧图在有限容量下必然超出必保预算；相应失败证据保留。使用现有
  `QW3_KVMEM_RETRIEVABLE_VISUAL_HISTORY=1` 验证旧图可检索、当前图受保护的策略。
- 质量微基准早期工具轨迹把 lookup 后的参考动作写成 Read，模型跟随该模式输出
  tool_use，所有方法一致出现非答案。pilot/v2 证据保留但不作为质量验收；后续
  统一改为只读文本观察与瞬时 lookup，仍须真实轨迹的任务质量验证。

## 2026-09-17 索引布局、实际客户端与边界修复

- 实际 Claude Code 2.1.148 连续会话完成 192 轮；固定 8K 仓库处理 95,647
  历史 tokens，191 HIT/0 MISS，后台退休 2,758 个已发布冷块。协议压力测试
  回答的是固定 marker，不将其作为官方 agent 任务准确率。
- 修复短首消息 header 在 block 0 的情况：无非零 pre-query P 时，仅在 Q
  完整结束后仍有可用 aligned 边界时捕获该边界；从不强行恢复 >LCP 状态。
- Adaptive 从 append-only packed rows 改为每物理槽固定最大 prototype slice；
  保留原 1/2/4 原型选择与 max 评分，实际 counts 仍按算法决定。raw-K 修复
  覆盖部分块及回滚，版本 stamp 在全部层完成后发布。Fixed4 同样从完整 raw
  prefix 重建非对齐部分块；子块 Mean-K 保留增量 merge。
- 新槽及部分尾块未填充行清零，并以实际有效 prototype counts 参加 masked
  softmax。GPU 数值测试覆盖一/四 prototype、整 tile 空洞、全空及极大脏 logits；
  CPU 参考一致。Mean-K raw 修复/回滚测试覆盖 FP32/FP8、ns=1/4、64 次槽位循环。
- 三种模式 executor 长测从 2^40 的模拟历史起点追加 128K tokens（15.625 倍
  仓库），各 800 轮、211 replay、13 个字节 oracle、25,600 接受的 decode tokens。
  CPU KV 与 index 固定；测试 binary 版本与 partial-tail 后续修复分开记录。
- B128 的 HTTP 测试另发现两次正常续接 cold prefill。原因是 live header 的
  aligned replay 起点位于 Q 内，当前 generation 没有合法 P。修复为这种
  情况下从 Q 前的对齐边界完整 replay，保留更完整的计算，没有略去 replay。
  冻结复测三种模式各 256 请求，全部 255 HIT/0 MISS；显式旧前缀改写/取消
  后安全 cold rebuild 单独记录。
- 新 host 回归 25/25、ASan/UBSan/LSan 定向 3/3 通过。原始 LongMemEval 文档
  和评分脚本哈希不变；未重启已取消的 30 分钟定时器。
- 最终文档集实验固定 Accurate_Retrieval 的 rows 0/1 各前 16 题，以官方
  substring exact-match 评分。off/shadow 使用 524K 容量，delete 使用 64K；
  活动窗口均 16K、生成预算 8K、block32、Q8/FP8/MTP4。这是质量消融，不用
  不同物理容量的时间差冒充 GC 管理开销。两个启动前参数/容量错误均保留，
  发生在任何回答生成之前，不计作模型失败。实际产物目录见 progress.json。
