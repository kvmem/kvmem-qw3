# 单请求 KVMem：逐 token 对照与冻结 4K 工作负载

继续使用 Q8_0 主模型、E4M3 FP8 target KV、官方 BF16 草稿权重；没有新增 BF16
主模型或 target KV。主运行时与阶段 7 相同，本轮只改测试观测和执行工具。
CLI SHA-256：`0286b569927ab2f124f89f0e16d4c6f9e43e003891d6fa4334e6e7b92c8e0c98`。
启用 SwiGLU reuse、down streaming/int32、fanout mode=2、GPU target sampling，
DFlash 最多七草稿；原生 MTP 按矩阵保留 chain=4。continuous batching 关闭。

## 逐 token 观测

原生生成 API 每个已输出 token 回调一次 `decode_one(id)`。对实际词表全部
248,320 个 ID 检查：原始字节片段均非空且互不相同。测试辅助类对每个加载的
词表再次检查，再按单个原始片段反查唯一 ID；全词表 round trip 通过，空回调拒绝。
这里没有对拼接文本重新分词，也不适用于会合并片段的 HTTP transport。

重建后的生命周期探针在 4K guided 和 long 两套测试中通过 **34 个路线步骤**，
主回答 ID 序列与无投机路线一致。另行比较完整私有生成查询、刷新位置及计数。
旧的 128K 长历史测试也已完成 **16 个步骤**；它使用旧观测器，只证明响应文本和
回调数量一致，不能追认为逐 ID 证据。旧前缀分叉、取消和恢复仍需约 98 秒的重放。
128K 正确性运行期间有 host token-map 审计工作，墙钟仅作诊断。

## 冻结 4K C1 实测

读取原有 `kvmem_matrix_v2.json` 的 `full_4k_c1_cold`，保留原始输入、顺序、
查询区间和种子。16 条正式请求加独立 ordinal=32 预热，8K 工作窗口、1024
生成预留、2 GiB CPU tier，全部历史可容纳。禁用 prefix cache，每条请求从
完整 4K 历史构建；日志确认没有 prefix-cache HIT。低于工作窗口时，语义检索
按原实现成为无操作，日志中的 window-profile 回退不代表测到了稀疏检索。

greedy 的 MTP、DFlash 分别 **17/17** 响应（含预热）与 none 的输出 token IDs
完全一致。评分严格从最后一条 `FINAL:` 行抽取有序 NOTE 标记，答案/事实标签
没有传入推理 worker。采样使用 T=1、top-k=20、top-p=.95、top-k-first，EOS 生效，
最多 256 token，不强制生成到长度上限。全部失败、截断和回答均保留。

| 模式 | 路线 | 正确率 | 解码中位数 tok/s | 含 prefill 请求吞吐 tok/s | TTFT 中位数 秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy | none | 16/16 | 44.64 | 24.90 | 1.719 |
| greedy | mtp | 16/16 | 114.92 | 37.04 | 1.731 |
| greedy | dflash | 16/16 | 122.84 | 38.47 | 1.737 |
| sampling | none | 16/16 | 44.29 | 25.05 | 1.722 |
| sampling | mtp | 16/16 | 111.98 | 37.63 | 1.742 |
| sampling | dflash | 16/16 | 123.16 | 38.67 | 1.733 |

解码速度为每条请求 `(输出数−1)/(末次−首次 callback 时间)` 的中位数；原始
逐 token ID 和 steady-clock 时间保存。DFlash 同批 token 可连续回调，这记录
native API 可用时间，而非独立 GPU kernel 完成时刻。端到端吞吐按总输出数除以
总请求墙钟，包含 prefill、查询处理和 generate 返回前清理，不包含模型加载。

这组短合成检索请求适合验证状态和正确性，不是公开长推理任务的替代。
每路线各一个进程、固定顺序，且尚缺 allocator high-water、完整资源细分和
≤20ms NVML 峰值观测；**表中为诊断性能，未达到正式性能验收资格**。
完整 LongMemEval、32K 稀疏/128K 有界 C1 矩阵和独立交替重复仍待完成。
公开参考实现资格也仍未解决，不能宣称已经接近或超过公开同口径结果。

源码与原始证据索引：`benchmark/dflash2/q8_c1_optimization_stage09.json`。
目标保持 active，多并发开发继续后置；没有集成到主工作区或替换服务。
