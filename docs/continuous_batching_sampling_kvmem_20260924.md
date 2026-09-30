# Continuous batching：采样与 KVMem 窗口接入

日期：2026-09-24。当前阶段为源码实现和 CPU 验证；GPU 正确性、吞吐测试及
profiling 等待正在运行的大海捞针实验完成后再安排。未更新运行中的服务。

## 开关与范围

新路径暂时显式启用，完成 GPU 验证前不改默认值：

```sh
QW3_CONTINUOUS_BATCHING_SAMPLING=1
QW3_CONTINUOUS_BATCHING_KVMEM_BODY=1
```

- `SAMPLING`：允许需要 logits 的请求进入普通 ragged prefill、body/LM-head
  batch，以及 ragged MTP verifier。包括 temperature > 0 和带惩罚的贪心请求。
- `KVMEM_BODY`：允许已有连续批处理路径上的 KVMem 请求进入窗口感知的 body
  batch。各行可有不同的窗口长度、页表和 RoPE 坐标。
- bounded repository、命名多会话、每请求存储配额及全局 CPU/NVMe 仲裁尚未
  接入；原有拒绝这些组合的入口仍保留。本次不代表完整 bounded KVMem 并发服务。
- 私有 query probe、attention trace、多模态输入状态和非 FP32 激活等不满足
  新 body 路径条件时，保留逐请求执行。普通 KVMem prefill 保留完整 executor
  路径，以保留 query、索引等前置处理。DeltaNet kernel 本次没有改动。

## 实现

1. 批量 LM-head 输出一次性复制到 CPU，再按请求拆分。普通生成取该请求的一行；
   MTP 取全部验证行及 bonus 行。每请求继续使用自己的 RNG、采样参数和 penalties，
   复用原采样及 point-mass speculative acceptance 算法。
2. 混合采样/贪心 batch 可以使用 ragged verifier；只有需要分布的请求持有
   host logits。layered verifier 仍只用于不需要 logits 的请求。
3. 新采样路径下保留验证前快照，供 ragged 失败后恢复再走 single verifier。
   验证输出数量先检查，再访问结果；缺失分布不会降级为贪心。
4. ragged verify 将每请求真实 target hidden rows 写回 executor，避免 MTP
   prefix 重建读取上一次 prefill 的 h_batch。同步记录本次 forward 的位置、
   raw K 和 checkpoint selection generation。
5. KVMem body batch 组合每行的窗口页表、窗口长度、追加位置；RoPE 位置单独
   传入，兼容非压缩位置实验。合批前检查物理 K/V arena 相同。
6. executor hooks 保留 decode raw-K capture、内容索引捕获及窗口尾部推进。
   未支持的 bounded 生命周期不会绕过原限制。

采样 verifier 默认允许两条请求合批；
`QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS` 仍可显式覆盖阈值。
原贪心默认阈值保留 16。trace 增加
`native continuous_mtp_verify: mode=... sampling=... requests=... logits_rows=...`，
用于核对实际运行路径，不能仅凭 scheduler batch 推断计算已经合批。

## 已进行的验证

- 独立 CPU Release 构建：`/tmp/qw3-cb-sampling-build-20260924`。
- CPU CTest 28/28 通过，含新增 logits 行归属、不同长度验证结果、窗口页表、
  大历史位置、边界及非法元数据测试。
- benchmark 参数传递测试通过：模拟 HTTP，检查串行和并发请求都发送指定的
  temperature/top-p/top-k/seed；测试不启动网络服务或模型。
- 使用现有 CUDA 构建的宏和头文件，对修改的两个 C++ 实现执行编译语法检查；
  未编译运行新的 CUDA kernel，也未启动 GPU 进程。

以上不能替代真实模型上的 logits、DeltaNet 状态、KV 内容与恢复正确性验证。
本轮修改前已有大量工作区改动；仅本轮变更的备份位于
`/tmp/qw3-cb-sampling-before-2e_523kj`。

## GPU 空闲后需要验证

先运行正确性检查，再 profiling：

- 并发 1/2/4/8，temperature=0/1，MTP 开/关；普通 KV 与非 bounded KVMem。
- 不同 prompt 长度、不同有效窗口长度，跨页/跨块追加；窗口重选后继续生成。
- 混合贪心和采样请求；penalties、固定 seed、提前 EOS、取消和错误退出。
- MTP 全接受/部分接受/拒绝时，核对 logits 行、KV/DeltaNet 恢复和后续 prefix。
- 检查 trace 的 body/ragged 实际命中率，记录整个 batch 的有效输出吞吐、
  prefill/decode 时间、TTFT、每请求延迟、显存和 CPU 内存。
- 不使用不同请求 decode 时间的相加值作为全局吞吐分母。

基准脚本现在支持 `--temperature 1.0 --top-p 0.95 --top-k 20 --seed 1000`，
并在结果 `config.sampling` 中记录这些参数。旧默认 temperature=0 保留。
