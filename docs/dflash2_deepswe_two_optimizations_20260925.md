# DeepSWE 两项优化与验证队列

用户要求先优化 KVMem 等待工具闭合时的串行段，以及长历史 target verification，
然后测试。主线保持 Qwen3.8-27B Q8_0 目标权重、FP8 目标 KV，单请求优先。
主目录 `/home/chaidi/qw3` 及正在运行的服务未修改。

当前状态：两项候选已接入源码，独立 CUDA 构建通过，完整 CPU CTest **38/38**
通过。9 月 26 日 03:38（北京时间）自动队列已取得 GPU 并执行第一项检查，但
测试程序误用接口导致该项失败，后续阶段未执行。测试修正、编译通过后已重排
`validation02`；前六项 GPU 检查均通过，但 10:15 刷新回归又被无效测试参数
中止。13:23 核对后修正参数，以 `build07` / `validation04` 恢复，核验并
复用六项已通过记录；13:35:55 的 4K 刷新回归全部通过。128K 回归于
14:01:47 因内部刷新策略对照不一致而失败，吞吐阶段没有启动。15:18 核对后
已开始独立的重复性及长 attention 分块边界诊断。15:38 时，分块修复候选
的八项算子检查已通过，正进入 128K 策略复测；仍无新 DeepSWE 吞吐。详见文末。

15:50 核对发现该复测进程已消失、日志未正常闭合，不能据旧 running 状态
认为仍在执行。已审计标记中断，并改由独立 systemd 用户服务运行
`partition_fix_validation02`，从策略复测恢复；详见最新记录。
不能将“已编译”解释为“已经提速”，也不能将验证调度优化称为长历史 kernel 瓶颈
已全部解决。两项开关默认均关闭，测试明确设置开关，未修改生产默认配置。

## 1. 等待安全闭合点期间继续 DFlash 投机

涉及 `src/dflash_deferred_refresh.hpp`、`src/qwen_native_backend.cpp`。
开关：`QW3_DFLASH_KVMEM_DEFERRED_REFRESH_SPECULATE=1`。

- 保留原有第一次刷新阈值；达到阈值后，如果工具调用或代码围栏仍未闭合，允许
  继续提出草稿，直到原串行解码策略的第一个安全闭合点。
- 闭合候选必须参与 p/q 验证。完整接受时把它保留为 pending token，恢复到它
  之前的状态，不提交额外 bonus，下一轮按原安全边界刷新。
- 紧急容量预留仍为 512 token；草稿加 bonus 不跨越紧急边界。未知容量、已经
  安全、最大刷新次数用尽等情况保留相应约束。
- 新增 `deferred_batches`、`refresh_boundaries`、`zero_draft_steps/s`、
  `bridge_steps/s`，避免旧 `plain_s=0` 隐藏零草稿路径。串行计时与 prefix
  部分重叠，日志明确标记，分析不能重复相加。

CPU 测试包含记录中的 2,534-token 阈值、分片 XML/代码围栏、安全与紧急边界、
三个精确的闭合 token 目标边际概率，以及跳过闭合候选验证会产生偏差的负对照。
GPU probe 覆盖 greedy 开关对照、CPU/GPU sampling、取消、恢复和私有查询。
这些是范围明确的正确性检查，不是对任意输入的穷尽证明。

## 2. 减少验证阶段的重复调度和元数据传输

涉及 `src/flashinfer_decode_plan.hpp`、`src/kernels_cuda.cu`、
`src/flashinfer_decode_adapter.cu`。
开关：`QW3_FLASHINFER_VERIFY_PLAN_CACHE=1`。

原来每个 attention 层都重新计算相同的 split/page 调度布局，并上传相同的
metadata，包括稍后会被设备页表展开覆盖的零值区域。新路径：

- 同一轮、相同因果长度和分块参数的 2–8 行验证共享 host plan 与已上传的
  GPU metadata；只上传实际调度数据，省去页表占位零值。
- 物理页号仍在**每次调用**从当前设备页表重新展开。缓存键不把指针相同当成
  内容相同，因此重选、回滚、跨请求复用不能依赖过期的物理页号。
- 其他 attention API 使用共享 metadata workspace、缓冲区重分配、CUDA graph
  replay 都会使上传缓存失效；graph capture 期间不使用此缓存。
- kernel/device 的 occupancy 容量按线程和设备缓存，原分块大小及 reduction
  算术保持不变；不切到另一个 attention 算法，不改变 Q8 量化或 FP8 KV。

它针对确定存在的重复调度开销。约 38 ms 的验证日志时间还包含 Q8 投影、
attention、DeltaNet 等执行，当前没有证据说明本项能消除其中多少耗时。
实际收益必须由下面的 GPU 对照确定。

CPU 检查覆盖页边界、部分页、split 合并布局、回滚、开关变化和 int32 越界。
新增 GPU probe 包含同一页表指针内容变化、另一个 API 覆盖共享 metadata、
宽度/位置变化、回滚、输出 guard，以及连续 16 层复用；比较结果要求逐值一致。
1024、65536、131072 历史分别执行。暖 KV 算子时间只作诊断，不替代端到端吞吐。

## 构建及已完成检查

独立构建：`benchmark/dflash2/deepswe_two_optimizations_build02/`。
通过 CPU 编译/链接生成 CLI、刷新回归、attention cache、serial parity、target
parity probe；没有在编译阶段初始化 GPU。

- 候选 CLI SHA256：`c21707fb91aa74a2325641576f2bf6b37e2596a717e16311ff2de5529cc2e8fc`
- 候选 core SHA256：`3fe5854fe6778e78b31be8a9bbcd534431d505a5bac2b9cc64d46de8f5704352`
- 原 build17 CLI 保持 `a33a95b302ad5b41df24d2665011efd19bca1b4d91212af181544c9620bf4061`
- 原 build17 core 保持 `2f712307bd1b7aee803adc8f1fae6c8d3d7a3638716f2409e271d6b49681a132`

`build01` 因复制的 pragma-once 头文件与原目录的间接包含冲突而编译失败，保留
错误日志；修正为一致的头文件快照后 `build02` 通过。没有把失败记录覆盖掉。
CPU 完整构建目录为 `build-dflash-twoopt-host`，CTest 38/38 通过（5.70 秒）。
配置、构建、CTest 日志已复制到验证队列目录，`git diff --check` 通过。

## GPU 队列及冻结对照

执行器：`benchmark/dflash2/run_deepswe_two_optimizations.py`。
运行目录：`benchmark/dflash2/deepswe_two_optimizations_validation01/`。
本次启动 PID 959863；进程是否仍在及 `manifest.json` 必须一起检查，不能仅从
旧 manifest 的 waiting/running 字段断言仍在运行。

独立原型队列 `deferred_refresh_regression02` 在尚未取得实验锁、没有 GPU 子进程
时被本轮替换，状态保存为 `superseded_before_gpu_start`。新队列使用正式源码的
集成候选执行回归，避免将旧原型的通过结果当成新候选的验证。

取得 `long_running_gpu.lock` 后，各 worker 启动前等待 GPU 连续空闲 60 秒：

1. 三种历史长度的 cache 正确性/算子诊断；原 attention 的 serial parity。
2. 4K KVMem 和 128K 历史的完整 target 层残差及词表 logits 一致性。
3. 4K、128K 刷新回归，各 28 个响应，必须实际覆盖延期投机、安全闭合、紧急
   刷新和取消分支。CPU/GPU sampling 与 greedy 对照保留原条件。
4. 从既有冻结任务中各取第一个 >=128K 请求：Quill 131,310、Effect 132,075、
   Dynamo 132,233 token。五组：原 build17、新候选双关、只开刷新、只开 plan、
   两项均开。原 build17/双关、双关/仅 plan、仅刷新/均开必须保持生成内容、
   终止和 token 数一致；忽略服务分配的 response/tool-call ID。
5. 全部 **45 个冻结请求**：双关 seed1000、两项均开 seed1000、两项均开 seed1001。
   保留原温度 1、top-k 20、top-p .95、自然 EOS、4096 输出上限和所有短输出/截断。

真实重放统一 **K128K/G64K**，prefill budget 131072、刷新阈值 61440，其他
策略沿用原冻结配置。不能将这些结果与旧 K64K/G32K 直接相减作为两项优化净收益。
分析时还需核对运行时有效窗口，不能把资源限制引起的缩窗当成 K128K 验证通过。

整卡上限仍为 **48,000,000,000 字节**，NVML 请求 10ms 采样并记录实际间隙；
出现显存、其他 GPU 计算进程或正确性错误时，仅终止本队列自己的 worker，保存
失败记录并停止后续阶段。采样峰值不等于 allocator 瞬时峰值。

重放只测推理，不执行工具；本队列不声称官方 DeepSWE 任务已通过，也不会仅靠
CPU/GPU 算子检查认定整个长任务目标完成。当前没有新吞吐、提速比例或显存结果。

## 2026-09-26 自动执行检查及恢复

`deepswe_two_optimizations_validation01/manifest.json` 已记录
`gpu_executed=true`、`status=failed`。03:38 第一项 `plan_cache_1024` 启动，
worker PID 1234645 返回 1，日志为：

```
paged attention fallback only supports identity page tables
```

原因在 `tests/flashinfer_verify_plan_cache.cu`：为了覆盖“其他 API 改写共享
metadata 后原缓存失效”，测试把 FP8 KV 和非连续页表传给 host-page API；
这个既有接口只有 FP16 KV 的 FlashInfer 实现，FP8 会进入只接受 identity
页表的 fallback。失败发生在关闭 plan cache 的参考路径第三个步骤，尚未
执行启用缓存的对照。它不能证明优化后的数值有错，也不能证明优化正确。
按照失败即停止的策略，64K/128K、target parity、刷新回归及全部 DeepSWE
重放都未执行。原队列进程已退出，失败目录原样保留。

修复将该次辅助调用改用独立的 9 页 FP16 合成 KV，主测试的 FP8 KV 和设备
页表完全保留；没有增加 BF16 目标权重支持，也没有修改推理代码。仍检查
9 次状态转换、页表内容变化、输出 guard、16 层连续复用及逐值完全一致。
每个步骤新增输出便于定位下一次失败，没有删除缓存覆盖检查或放宽误差。

新增构建 `deepswe_two_optimizations_build04`：验证源文件、编译参数与原构建
一致后复用 `build02` 的 core/CLI，重新编译四个 GPU probe；两个 CPU 策略
测试通过。core/CLI SHA256 与前述 `build02` 完全相同。增量构建的来源及
哈希存于 manifest 的 `reused_core`。中间 `build03` 的测试日志前缀拼接
编译错误也保留，修正 `std::string` 类型后 `build04` 通过。

新队列 `deepswe_two_optimizations_validation02` 使用 `build04` 从第一项
重新验证，仍遵守共享锁、连续空闲 60 秒、48 GB 整卡采样上限及失败停止。
恢复时 GPU 被 CacheBlend 实验占用，未打断其工作。配置、冻结输入与原验证
计划保持相同：Q8_0 + FP8 KV、单请求 K128K/G64K。GPU 修正后的正确性和
性能结果仍待实际执行。

### 10:11 执行进度（北京时间）

再次核对 `validation02`：runner PID 1247567 存活，前三项
`plan_cache_1024/65536/131072`、`attention_serial_parity` 和
`target_kvmem_parity` 均通过。缓存测试保持逐值一致；KVMem 检查的所有层
residual 和 logits 均 bit-exact。五项资源观察均没有记录违规或采样器错误。
这些只是正确性与算子检查，不是 DeepSWE 端到端吞吐结果。

10:11:27 确认第六项 `target_long_parity` 正在 GPU 上执行，worker PID
1318950；没有重复启动队列。后续仍自动执行刷新回归和冻结请求 A/B 重放。

### 13:23 核对及刷新回归恢复（北京时间）

`validation02` 最终在 10:15:03 失败退出。第六项 `target_long_parity` 已经
通过：128K 完整模型 serial/verifier 的逐层 residual 和词表 logits 均
bit-exact。刷新回归的 `greedy_off` 前六个请求完成，第七个
`after-private-query` 被参数校验拒绝：

```
KVMem mid-decode steady trigger requires a positive first-epoch trigger
```

该用例原本要检查私有查询后恢复并关闭刷新，首轮 trigger=0，却被测试统一
设置了 steady trigger=64。修正为 `item.trigger ? 64 : 0`；框架的参数校验
和推理代码均未改动。失败发生于关闭延期投机的参考分支，因此尚无延期投机
开关对照结果。该阶段整卡采样峰值 **42,946,330,624 B（42.95 GB）**，
无采样器错误或资源违规；这不代表 K128K/G64K 的完整 DeepSWE 显存已验收。

局部 attention 算子测量（每次 320 调用，3 次交替测量取中位耗时）：

| 历史 | cache 关 | cache 开 | 耗时下降 |
|---|---:|---:|---:|
| 1K | 9.83284 ms | 9.18056 ms | 6.63% |
| 64K | 183.283 ms | 179.228 ms | 2.21% |
| 128K | 341.736 ms | 337.399 ms | 1.27% |

这些是暖 KV 的算子数据；不能换算成完整模型 decode 提速或 DeepSWE 吞吐。

`build06` 仅更新 `deferred_probe`，CLI、core 及其他 probe 均与 `build04`
逐字节一致。新增 `--reuse-passed-from` 在恢复前核验旧 frozen plan、模型
哈希、二进制、参数、环境、资源状态及每份输出文件哈希，只允许复用前六项
已通过检查，并明确记录 `reused_passed`，不宣称本轮重新执行。
`validation03` 从 `deferred_4096` 接续；原失败日志和结果全部保留。

第一次修复只清零 steady trigger，`validation03` 随后在同一恢复用例被
下一项参数校验拒绝：trigger、refresh-count、query-token limits 必须同时
为正或同时关闭。最终 `build07` 对 trigger=0 的用例同时清零 steady、
max-refreshes、query-max-tokens 和 epoch-limit，并让 metadata 记录有效
参数；其他六个用例保持原参数。`validation04` 已重新进入 4K 回归。
两次配置错误均发生在 `greedy_off`，不可计为优化开关的数值失败。

### 13:35:55 的 4K 刷新回归结果（北京时间）

`validation04/deferred_4096/summary.json`：28 个响应全部完成，14 组 token
和刷新/query/selection 策略对照一致。greedy 的优化开关对照、采样请求的
CPU/GPU sampler 对照均通过，callback 字节与响应一致。三个开启优化的
变体均实际覆盖 accepted safe boundary、emergency-open-XML、延期投机
期间取消；并非只生成文本而没有触发目标分支。

- greedy-on：271 个延期投机 batch、1 个已接受安全边界。
- sampling CPU/GPU-on：各 265 个延期投机 batch、各 1 个已接受安全边界。
- greedy XML：零草稿步骤 550 → 1，累计计时 12.435 s → 0.023 s。
- greedy 双代码块：零草稿步骤 647 → 1，累计计时 14.683 s → 0.023 s。

相应生成输出逐 token 一致。这说明第一项优化在合成工具/代码用例确实减少
串行等待，但这些短刷新阈值的计时不能作为 DeepSWE 性能结果。

该完整 4K 回归整卡采样峰值 **42,988,273,664 B（42.99 GB）**，资源观察
无违规或错误。测试配置是 K64K/G32K，尚不构成真实长任务 K128K/G64K 的
48 GB 验收。10ms 请求采样的最大实际间隙为 47.65ms，瞬时峰值仍可能漏采。

runner PID 1413850 存活，下一项为 `deferred_131072`；之后保持原计划执行
三任务诊断与 45 请求重放。当前可支持的结论是：attention cache 算子收益
较小，延期投机明显消除了合成用例的串行段；端到端收益仍待重放验证。

### 15:18 核对：128K 输出一致，但内部策略回归失败

`validation04/deferred_131072` 的 worker 完成全部 28 个请求并返回 0，
14 组对照的输出 token 与 callback 均一致；随后分析器发现内部策略差异，
14:01:47 将该阶段及整个流水线标记为 failed。不能把 worker 返回 0 当成
完整回归通过。这次不是参数配置错误，也没有放宽原策略检查。

发现三项 greedy-off / greedy-on 差异：

- `two-fences`：刷新位置与私有查询文本相同，KV 选块哈希不同。
- `xml`、`after-cancel-xml`：均在 decoded=562 的相同安全边界刷新，
  私有查询由 44 token 变成 45 token，句末措辞不同，选块哈希也不同。
- sampling CPU/GPU 两组的策略对照没有发现差异。

这说明当前用例最终文本相同，并不足以证明后续检索上下文保持一致。
可能涉及累计数值差异或状态处理，需要隔离验证；尚未确定原因。
该阶段整卡采样峰值 **43,546,116,096 B（43.55 GB）**，无资源违规。
它仍是 K64K/G32K 的 128K 合成历史回归，不是 K128K/G64K DeepSWE 验收。
整个 `validation04` runner 已退出，后续 A/B 和 45 请求重放均未开始。

本轮准备并执行两个独立诊断，保留原失败记录、保持 core/CLI 不变：

1. `deferred_policy_diagnostic01` / `build08`：原前两个请求及顺序保持不变，
   执行 greedy-off、相同配置 greedy-off-repeat、greedy-on，先判断原路径
   的重复性。只测六个请求，明确不替代完整回归。
2. `long_attention_partition_diagnostic01` / `build09`：FP8 KV，64K/128K
   长度附近按生产 planner 找到分块大小变化点，分别测试边界前和跨边界的
   8 行 batch 与逐 token decode，cache 开关都测。源码中 chunk size
   依赖长度，因此它是候选数值因素；未获得 GPU 结果前不视为已定位根因。

两个诊断均使用共享 GPU 锁和 48 GB 资源观察，按顺序执行。

### 15:32 定位结果及修复候选

重复性诊断完成：关闭优化重复两次，两个用例的 token 和内部策略均一致；
开启后 `two-fences` 的选块差异再次出现，外部 token 仍一致。
这组对照支持差异随延期投机执行路径稳定复现，而非这两次参考运行随机波动。

长 attention 的八组检查中，四组边界前对照通过，四组跨界对照失败，cache
关闭/开启都相同：

- base=65,609：串行第一行 chunk=349，批量末行 chunk=350。
- base=131,221：串行第一行 chunk=698，批量末行 chunk=699。

原实现将批量最后一行的 chunk size 应用于所有因果行，跨界时改变了较早行
的归约分组，FP8 attention 输出已不能逐值对齐 serial。该算子层数值缺陷已
复现；是否足以解释全部私有查询/选块差异，仍需修复后的完整回归证明。

`src/kernels_cuda.cu` 新候选在 chunk-size 变化点拆分批次，连续同 chunk 的
行继续批量执行，并保留原始 Q/O stride 与完整设备页表。没有更改 FP8 数据
或逐 token decoder 的分块规则。新增 long-serial-parity CUDA 测试纳入 CTest。
`build10` 独立编译该候选；尚未获得修复后的 GPU 或性能结果，不宣布问题解决。

### 15:38 分块修复的算子验证通过

`build10` 编译、链接及 CPU 策略检查通过。
`partition_fix_validation01_attention` 在相同八组用例上全部通过，
`long_serial_failures=0`；原先四个失败场景现在与逐 token decode 逐值相同。
cache 开关及 64K/128K 两种长度均覆盖。这证明本次修复消除了已复现的
算子跨界差异，尚不能直接证明所有检索策略差异均已解决。

新增串行验证执行器 `run_partition_fix_validation.py`，当前运行
`partition_fix_validation01`：等待独立构建完成后，先跑 attention 诊断，
再跑 off/off-repeat/on 两用例对照；只有 token 与全部策略比较都通过，
才进入新的完整回归及 DeepSWE 重放。每阶段保留独立目录和状态，失败停止。
当前已自动进入 `partition_fix_validation01_policy`，后续完整运行 ID 为
`partition_fix_validation01_full`。这次 core 发生变化，完整阶段会重新验证
基础检查，不复用旧 core 的通过记录；模型权重、FP8 KV、显存上限不变。

### 15:50 中断核对与独立后台恢复

发现 `partition_fix_validation01` 的父执行器、policy 执行器及 GPU worker
均已不存在。policy 只有首个请求的开始日志，没有完整响应；resource gzip
未正常结束。最后写入时间分别约为 15:40:29（资源）和 15:40:39（worker
日志），没有正常退出码。查询此时段 kernel journal 未找到 OOM/Xid 记录，
但这些证据不足以判定终止原因，记录为 unknown，不算正确性通过或数值失败。

两个未完成 manifest 已标记 `interrupted_process_missing`，原始 manifest
另存为 `manifest.before_interruption_audit.json`；未修改日志或修补截断的
gzip。已完成的 attention 结果及其 manifest 原样保留。

恢复服务：`qw3-dflash-partition-fix-validation02.service`，由当前用户的
systemd manager 启动。已确认主进程位于该服务独立 cgroup。运行目录为
`partition_fix_validation02`，仍使用 `build10`，主执行器 PID 1486403。
增加恢复参数，在核对 attention 二进制、命令、环境、资源记录及全部输出
哈希后复用此前成功结果；policy 重新完整运行，不拼接未完成的请求。

后续路径为 `partition_fix_validation02_policy` 和
`partition_fix_validation02_full`。共享 GPU 锁、连续空闲 60 秒、48 GB 整卡
资源观察与失败停止规则不变。策略复测仍未完成，未进入 DeepSWE 吞吐阶段。

### 16:05 策略复测通过与每五分钟后台检查

`partition_fix_validation02_policy` 六个请求全部完成，token 对照、
off/off-repeat 和 off/on 的 prose、two-fences 策略对照全部相同。
完整回归已自动接续，当前在等待 GPU 连续空闲；尚无新 DeepSWE 吞吐。
策略阶段整卡采样峰值为 43,546,116,096 字节，仍是诊断配置，不能替代
真实 K128K/G64K 长任务的显存验收。

按用户要求新增独立、只读的 `check_validation_progress.py`，由
`qw3-dflash-validation02-monitor.timer` 每五分钟整点触发。16:05 首次定时
执行成功，下一次为 16:10；与启动前的手动检查合计两次。检查测试服务、
各层 manifest、实际进程存活、完成请求数、业务日志进展与整卡 GPU 状态。
排除资源采样器和等待观察时间戳，避免把监测日志增长误当作测试进展。
20 分钟无业务进展且 GPU 空闲，或 60 分钟无业务进展，标记需要检查；
等待 GPU/实验锁单独显示。八项状态分类检查通过。

检查结果保存至 `partition_fix_validation02_monitor/latest.md`、
`latest.json` 与追加式 `checks.jsonl`，有状态变化才写 journal；测试链
结束后自动停止此 timer。原测试执行器与其已冻结依赖均未修改。
该定时器不依赖当前工具会话，但仅服务于本次主机运行；当前会话未提供
Codex 自动化/对话唤醒接口，因此没有聊天主动推送。异常只记录，不自动
重启 GPU 作业或修改通过标准。
