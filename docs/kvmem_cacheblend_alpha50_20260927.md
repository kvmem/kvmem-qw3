# 10% 重算后，以 50% 新 KV + 50% 旧 KV 更新

本次新增 `QW3_KVMEM_BLEND_ALPHA`。默认值 1 保留原来的全替换行为；设置为 0.5 时，选中的 10% token 仍按原策略完成全部重计算，然后分别更新 K 与 V：

```
K = 0.5 * K_recomputed + 0.5 * K_before_recompute
V = 0.5 * V_recomputed + 0.5 * V_before_recompute
```

旧 K 来自窗口完成位置调整后的 GPU 备份，与新 K 在相同的 RoPE 坐标中。混合发生在所有重算段完成之后，不在段间或层间提前混合，避免改变后续重算的输入。选块比例、排序、重算顺序、GDN 清零一次后跨段推进及 Query 边界状态恢复都沿用原实现；没有混合 GDN recurrent/conv 状态。

混合直接使用现有的紧凑旧 K/V 备份。只增加一个物理行索引表，每个注意力层发射一次 CUDA kernel 同时处理 K 和 V；在 FP32 中计算后转换回缓存 dtype，支持 FP32、FP16、BF16、FP8 E4M3。原始 CPU KV、检索索引及不可变 raw K 保持原样；之后仍通过原有 restore 路径恢复 GPU 工作窗口。alpha=0 是恢复旧 KV 的数值端点检查，仍会执行重算，不代表性能上的关闭重算。

## 验证与队列

实验目录：`/data/chaidi/qw3-experiments/cacheblend_20260925/alpha50_20260927`。

使用独立的 source/build/bin，当前运行中的冻结二进制不受影响。后台队列等待 `external_full180_20260927` 全部 180 例结束并释放服务进程后，依次执行：

1. 审计并更新当前全替换组的完整双指标结果。
2. CUDA 数值检查：四种 dtype、稀疏行、不整齐宽度，alpha=0/0.25/0.5/1，检查未选位置和旧备份未变化。
3. 模型检查：FP16、FP8 的 alpha=1 与 0.5，以及 FP8 的 alpha=0；与独立 CPU 量化参考逐字节比较。原有 source KV、因果遮罩、GDN/conv、hidden state、页表及坐标检查同时运行。
4. 四个不同任务的 alpha=1 服务复测，检查重构后的二进制完整响应和实际请求与当前组相同。
5. 上述验证全部通过后，运行 alpha=0.5 的完整 180 例。失败则记录错误并停止，避免将未通过验证的结果混入正式组。

正式组继续使用用户指定的冻结 token 输入、Qwen3.8-27B Q8_0、FP8 KV、MTP4、guided query、32K history/prefill budget、重算比例 0.1、thinking budget=4096、输出上限=8192。每例重新启动服务。64K、128K、256K 各 60 例，四类任务各 15 例。没有增加无重算基线。

队列状态为 `queue-state.json`，正式运行状态为 `full180/state.json`。完成后自动保存 `paired-comparison.md/json` 和 `paired-summary.csv`，分别统计完整数字覆盖率与答案集合全对率，并保存名称对应的待复核项。自动数值评分不冒充已经完成的人工语义复核。

这里比较的是**同样本的 10% 重算全替换与 10% 重算 50/50 更新**。此前历史无重算组的数据不同，仍保留在上一轮报告中作独立参考。
