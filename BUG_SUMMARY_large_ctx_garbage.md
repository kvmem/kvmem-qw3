# 大 ctx 直接生成乱码问题 — 已定位并修复（最终版）

> 本文档随调查推进持续更新。**本版为最终结论：根因已定位、修复已落地并验证。**
> 之前所有"逐算子定位 / recurrent_batch 怀疑 / vmed kernel bug"的中间判断均被推翻，
> 下方「调查弯路」单列存档，请勿再据此推理。

## 问题现象

命令 `./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf -p "..." -n 256` 在默认
`--ctx 262144` 下输出乱码（`<think> </think>` 空循环、或语义连贯但完全跑题的文本）。
乱码不是随机字节，而是**合法 token 串**——说明 hidden state 被错误/陈旧显存污染后
仍能解码成合法 token。`serve` 服务不报错，小 ctx 也正常。

## 根因（已确认）

**默认流（null stream）上的 `cudaMemset` 与非阻塞 `exec_stream_` 竞争。**

1. `exec_stream_` 用 `cudaStreamNonBlocking` 创建（kernels_cuda.cu `begin()`），
   因此它**不会**与默认/null 流隐式同步。
2. `CudaTensor` 构造函数用 `cudaMemset(ptr,0,bytes)` 在**默认流**上做零初始化
   （kernels_cuda.cu 约 905 行；`make_q8_kv` 的 scale memset 同理）。
3. KV 页索引张量 `tensor_i32(max_pages,...)` 是**前向过程中按需惰性分配**的
   （qwen_executor.cpp `KvPageTable::ensure_pages`，约 319 行）。`max_pages` 随
   `--ctx` 线性增大，所以在 ctx=262144 下这块 memset 很大，且会与正在 `exec_stream_`
   上运行的 vmed/MMQ matmul **重叠执行** → 激活值被清零/污染 → 乱码。

这解释了全部观测：
- **为何随 ctx 触发**：小 ctx 下该缓冲区小且分配早，不与前向重叠；大 ctx 下分配
  发生在前向中途且耗时长，才会重叠。
- **为何 device-sync（mode 2）能修、exec_stream-sync（mode 1）不能修**：device-sync
  会排空 null 流，mode 1 只排空 exec_stream_（错的流）。这正是**跨流竞争**的指纹。
- **为何 "sync vmed" 也能修**：那是在最频繁算子后插入全局 barrier，纯属串行化掩盖，
  不代表 vmed 有 bug。vmed 数学一直正确：相同输入在 ctx≤131072 下完全连贯
  （rope 对固定 prompt 与 ctx 无关，layer-0 激活逐字节一致）。
- **决定性验证**：把 `exec_stream_` 改成**阻塞流**（去掉 `cudaStreamNonBlocking`，
  使其与 null 流隐式排序）→ ctx=262144 乱码消失。锁定元凶为 null 流上的算子。

## 修复（已落地）

让构造函数的零初始化在 `exec_stream_` 上执行，从而与所有消费者严格保序：

- 新增文件作用域 `g_tensor_init_stream`，在 `begin()` 中设为 `exec_stream_`；
- `CudaTensor` 构造函数：`g_tensor_init_stream` 非空时用
  `cudaMemsetAsync(ptr,0,bytes,g_tensor_init_stream)`，否则回退到同步的 null 流
  `cudaMemset`（仅 `begin()` 之前的启动期分配会走这条，无并发前向，安全）；
- `make_q8_kv` 的 scale memset 同样改造。

**不涉及 HGEMM、不引入全局串行化。** 性能反而略好（去掉了一次同步的 host 阻塞，
换成异步 memset）。实测吞吐无回归（85-token prompt：prefill 832 tok/s）。

## 一并清理

- 删除 `qwen_executor.cpp` 里的 `chunk_size=8` 临时 band-aid
  （`kv_ctx_size_>=98304 && total<=64`）；
- 删除 `kernels_cuda.cu` 里全部诊断探针：`QW3_DEBUG_SYNC_EVERY` / `QW3_DEBUG_SYNC_OP`
  / `g_debug_sync_*`，以及临时的 `QW3_EXEC_STREAM_BLOCKING` 验证开关。`launch_status`
  恢复为仅检查 `cudaGetLastError()`。

## 验证结果（全部连贯正确）

| 配置 | 结果 |
|------|------|
| 原始复现命令（默认 chunk/ctx，31-token） | ✅ 连贯 |
| 85-token，默认 ctx，chunk=2048 | ✅ |
| 85-token，ctx=262144，chunk=32 / 48 / 64 | ✅（原来全乱码） |
| 85-token，ctx=4096，chunk=64（小 ctx 回归检查） | ✅ |
| ~600-word 长 prompt，默认 chunk=2048（满 batch 走 v8） | ✅ |

## serve 为何不报错、直接生成才报错（用户提问）

`serve` 在 warmup / 首个请求时就把整个页索引缓冲区分配完，进入稳态后前向过程中
**不再有张量被构造** → 没有 null 流 memset 与 exec 工作重叠。而直接 generate 是全新
executor，惰性 KV 缓冲区在**正在生成的这条 prompt 的 prefill 过程中**首次分配，恰好
与 vmed/MMQ 重叠 → 竞争触发乱码。

核心问题一句话：**非阻塞 exec_stream_ + 默认流上的张量零初始化 memset，二者无保序，
当大 ctx 把惰性分配推到前向中途时发生数据竞争。**

## 用户硬约束（已满足）

- 未使用 HGEMM 修复（HGEMM 在 chunked prefill 下性能差，是控制显存的关键约束）；
- 修复在 host 编排（流保序）内完成，未引入全局串行化、未切到 cuBLAS HGEMM。

---

## 调查弯路（存档，勿再据此推理）

- ❌ 「元凶是 vmed matmul kernel」——vmed 数学正确，相同输入小 ctx 全对；"sync vmed
  能修"只是全局 barrier 串行化掩盖。
- ❌ 「元凶是 recurrent_batch / 某个批处理算子」——根因与算子无关，是流保序问题。
- ❌ 「FlashInfer 是元凶」——non-paged（不走 FI）同样乱码。
- ❌ 「matmul 全部排除」/「逐算子同步无法修复」——这些是被竞争掩盖效应误导的中间
  判断；真正可靠的信号是 **mode-2 device-sync 修复而 mode-1 exec_stream-sync 不修复**，
  指向跨流（null 流）竞争。
- ⚠️ racecheck/memcheck/initcheck 全 0 错误——sanitizer 会串行化执行，恰好掩盖时序
  竞争，"干净"反而印证了并发/保序 bug 而非静态内存错误。
