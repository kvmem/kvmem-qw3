# 有界 KVMem 后台管理（实验性）

此路径允许累计历史超过 `--ctx`，同时循环复用固定数量的 KV 槽位。
它会遗忘被删除的历史；有限容量不能无损保存无限历史。
默认配置 `kvmem_repository_blocks=0` 不启用此路径，也不会改变已有实验。

## 启用与容量

同时设置 `--kvmem-repository-blocks N` 和 `--ctx N*block_tokens`。
这里 `ctx` 是物理仓库容量，选择预算与生成预算仍分别由原参数指定。
CPU/NVMe tier 必须足以保存仓库的 main/MTP 权威副本；启动时检查容量。
仓库应留出 working set、保留 checkpoint、在途 I/O 和下一批写入的空间，
不能将全部物理槽位都分给活动窗口。索引、模型权重和临时工作区须另外核算。

以下是功能测试用的 4B、8K 小仓库配置，不是 27B 生产容量建议：

```sh
qw3 serve --model models/Qwen3.5-4B-Q8_0-with-MTP.gguf \
  --ctx 8192 --prefill-chunk 128 --paged-kv --kv-page-size 16 \
  --kv-dtype fp8 --mtp-chain 4 \
  --kvmem --kvmem-query-conditioned \
  --kvmem-repository-blocks 256 --kvmem-repository-gc delete \
  --kvmem-block-tokens 32 --kvmem-budget 1024 --kvmem-gen-budget 1024 \
  --kvmem-sink-blocks 1 --kvmem-recent-blocks 4 \
  --kvmem-index-placement cpu --kvmem-cpu-gb 1 \
  --kvmem-guided-reselect both --kvmem-guided-query-tokens 64 \
  --kvmem-middecode-trigger-tokens 512
```

`off` 不删除已发布历史；`shadow` 记录候选但不实际删除；`delete` 才执行回收。
三者仍会清理取消的未发布写入。off/shadow 容量耗尽不能自动转为 delete。
不设置 NVMe 参数即使用 CPU；原 NVMe tier 配置也可用于有界仓库。

当前支持 CPU 索引的 Mean-K、SubBlockMeanK、Fixed4 和 Adaptive，后两者以及
多子块使用 max 归约；sum、per-token/DeltaNet、GPU 索引、archive/语义组扩张、
continuous batching 和显式 named sessions 尚未接入，程序明确拒绝这些组合。
Claude Code 使用普通完整 transcript 请求与隐式 prefix cache。
视觉输入支持原有视觉路径；若允许回收旧图像，应显式采用
`QW3_KVMEM_RETRIEVABLE_VISUAL_HISTORY=1`。默认 pin-all 图像政策下，受保护图像
超过选择预算会拒绝请求，不会暗中丢弃图像或放宽预算。

## 生命周期与正确性

历史位置为 64 位，物理地址由 `(slot, generation)` 决定。每个槽位只存储一个
当前版本的块，并依次经过预留、发布、退休、释放、重新就绪。主 KV、MTP KV、
raw K、Mean-K/prototype 索引及 page 映射都使用有界地址。Adaptive 每个槽位
预留最大 prototype 区域，实际参与评分的数量仍由原算法决定。

读请求、working set、checkpoint 和在途 I/O 持有 lease。后台不能删除有引用
的资源；异步操作完成、全部 CPU/NVMe 资源释放后才归还 ready pool。generation
检查防止旧 I/O 完成事件或旧分数污染新一代槽位。部分回收失败会保留待释放
状态并重试，不把它计为空闲；前台只领取 ready 槽位，不在分配路径同步运行 GC。

后台复用规则评分：最近使用、选择频率、最近几轮相对检索分数、持续低分证据
及候选宽限期。GPU 驻留、在途 I/O、checkpoint、sink/recent 与强制块都受到
硬保护。低水位启动补充，高水位停止；压力下允许降低软优先级门槛，但不绕过
硬保护。长请求可按新增历史量推进宽限期，避免没有新请求轮次就永远无法回收。

这里的分数是保留优先级，不是经过校准的“未来使用概率”，也不保证全局最优
删除。尤其在先导入整篇文档、最后才提问的负载中，此前的访问频率和检索分数
不能提前暴露问题所需的证据。删除后的质量代价必须随具体容量一起报告。

评分持有同一代目录快照，已删除、未发布和未填充 prototype 在 softmax 前被
排除。回滚会永久失效其后的 checkpoint，不复活已删除块。持续追加保留有界
前缀摘要、精确尾部和有限 checkpoint；恢复仍要求 `checkpoint <= LCP`，
semantic replay 还要求 block 对齐及 main/MTP/recurrent 状态一致。

ready 不足、受保护集合过大或权威 KV 缺失会明确报错；不会强行删除受保护块。
早于保留尾部的历史改写、取消后无安全状态可恢复时，可能执行安全冷重建。
这些情况与正常 append 被 GC 意外破坏所造成的 cold prefill 分开统计。

## 资源与延迟边界

GC worker 只访问 CPU 元数据与 CPU/NVMe allocator，不调用 CUDA、设备同步或
GPU free。通常由原前台计算路径完成索引构建、KV 搬运和窗口切换。CPU 线程仍
可能争用内存带宽或调度资源，因此不能据此声称“零延迟开销”。

固定的是持久 KV、索引、目录、checkpoint 引用和保留前缀身份的容量。
Claude Code 仍发送完整 transcript，网络输入、解析、tokenization 与临时视觉
源数据仍随历史增长；HTTP token 计数目前限制为 UINT32_MAX。因此不能把该
功能描述为整个 HTTP 服务进程使用恒定内存，或协议支持数学意义上的无限输入。

## 验证入口

所有实验均在 `benchmark/kvmem_bounded_gc_20260916/`，源文件、二进制哈希、
失败现场、修复记录与原始日志随各验证阶段冻结。实施状态见同目录 `progress.json`。

- `live_claude_fixed_validation.json`：实际 Claude Code 2.1.148，192 轮、95,647
  历史 tokens、8K 仓库，191 HIT/0 MISS；删除 2,758 个已发布冷块。
- `http_visual_validation.json`：实际视觉编码与 640 次 HTTP 请求，89,755
  历史 tokens、639 HIT/0 MISS；另测尾部改写、旧前缀改写和取消。
- `http_nvme_validation.json`：真实 NVMe 连续请求及物理槽位复用。
- `mrope_model_validation.json`：800 轮、128K 历史，完整 guided query、replay、
  MTP 接受/拒绝；重复 replay 的状态字节与下一 token/logit 核对。
- `prototype_model_validation.json`：各 prototype 布局从 2^40 模拟历史起点追加
  128K；并非实际计算了 2^40 tokens。部分尾块与 softmax 空洞有独立数值测试。
- `timing_final_mrope_boundary/summary.json`：同一固定轨迹 off/shadow/delete，
  各重复三次，报告 TTFT、TBT 与吞吐。该结果针对 4B 的小仓库测试配置。
- `quality_micro_validation.json`：合成保留测试，三种模式各 9/16、原始答案相同；
  不能据此推广到真实任务准确率。
- `dataset_retention_validation.json`：两条真实文档、32 题，off/shadow 各 20/32，
  64K 仓库 delete 为 12/32；全部请求正常续接，明确存在删除后的质量损失。

最终验收报告见 `kvmem_bounded_repository_validation_20260917.md`；
完整实施历史见 `kvmem_bounded_repository_plan_20260916.md`。
