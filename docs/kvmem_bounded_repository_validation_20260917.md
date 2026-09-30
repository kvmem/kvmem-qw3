# 有界 KVMem 后台管理验收结果

2026-09-17：实现和本轮验收完成，功能仍须显式启用，默认关闭。
操作与支持范围见 [使用说明](kvmem_bounded_repository.md)。
全部原始产物位于 `benchmark/kvmem_bounded_gc_20260916/`。

## 实现内容

- 累计历史位置使用 64 位；main/MTP KV、raw K、索引、目录与 GPU page 映射
  使用固定数量的物理槽位。槽位通过 generation 区分不同历史块，可反复复用。
- CPU 后台线程根据近期使用、衰减频率、相对检索得分与宽限期选择冷块；提前
  补充 ready pool。线程不调用 CUDA、GPU 同步或 GPU free。
- working set、checkpoint、强制块和在途 I/O 持有 lease。全部资源释放完成才
  归还槽位；旧异步事件不能发布或释放新一代资源。部分失败留待后台重试。
- 保留原 guided query、selection、完整 query replay 和 MTP。修复了首次 query
  从 block 0 开始，以及大 block 的对齐边界落入 query 内时缺少安全 checkpoint
  的问题。实现有界尾部身份验证、尾部改写和取消后的保守恢复。
- 支持 CPU Mean-K、SubBlockMeanK、Fixed4、Adaptive/max，以及 CPU 或 CPU+NVMe
  backing、普通 Claude Code transcript 和已有视觉输入路径。

## 功能与资源验证

测试模型为 Qwen3.5-4B Q8_0，FP8 KV，MTP=4；未将这些结果冒充 9B/27B 验收。

| 验证 | 结果 | 产物 |
|---|---|---|
| 当前 host 回归 | 25/25 通过 | `config_guard_host_ctest.log` |
| ASan/UBSan/LSan 重点回归 | 3/3 通过 | `prototype_asan_ctest.log` |
| FP32/FP8 索引数值 | 单/四子块、各 64 次回退复用通过 | `prototype_serving_attempt1/qw3-kvmem-repository-index.log` |
| softmax 删除空洞 | 空块、未填充 prototype、全空 tile 通过 | `prototype_serving_attempt1/qw3-kvmem-softmax-pages.log` |
| 模型状态正确性 | 8K 仓库累计 128K，800 轮；211 次 replay、13 次状态字节与预测核对 | `mrope_model_validation.json` |
| 64 位历史 | 从 2^40 模拟起点追加 128K；各 prototype 布局通过 | `prototype_model_validation.json` |
| CPU HTTP | 640 请求、87,115 历史 tokens；639 HIT / 0 MISS | `http_cpu_validation.json` |
| NVMe HTTP | 640 请求、87,115 历史 tokens；639 HIT / 0 MISS | `http_nvme_validation.json` |
| 真实视觉编码 HTTP | 640 请求、40 张图、89,755 tokens；639 HIT / 0 MISS | `http_visual_validation.json` |
| 实际 Claude Code 2.1.148 | 192 轮、95,647 tokens；191 HIT / 0 MISS，回收 2,758 冷块 | `live_claude_fixed_validation.json` |
| 三种 prototype HTTP | 各 256 请求，均 255 HIT / 0 MISS | `prototype_serving_validation.json` |

以上 HIT/MISS 排除首请求的正常冷启动。另行测试保留尾部改写、早期前缀改写、
客户端取消和后续请求：保留尾部可复用；超出恢复范围时安全冷重建，未强制恢复
大于 LCP 或未对齐的状态。模拟 2^40 起点不代表实际计算了 2^40 tokens。

8K 模型压力测试达到容量的 15.625 倍，稳定阶段 main/MTP CPU 存储为
209,682,432 字节，Mean-K 索引 4,194,304 字节，GPU KV 19,464,192 字节。
HTTP 测试保留前缀身份为固定 327,680 字节。NVMe 使用真实 allocator 和文件，
反复复用物理槽位；并非仅标记删除。

TSan 在当前运行环境中报 `unexpected memory mapping`，不能运行；未计为通过。
各阶段冻结了源码与二进制，版本差异保留在 source_lock 和验证文件中。

## CPU 管理开销

固定轨迹、相同 8K 仓库、1K 选择预算、1K 生成预算、block=32。每种模式运行
3 次，轮换顺序；每次 86 请求、每请求 32 输出 tokens，排除前 8 次暖机。
测量时没有其它 GPU 推理或 CPU 编译。下面是各运行统计值的中位数。
计时采用原生 backend 回调口径，不包括 Claude Code 的历史构造、网络和工具
执行耗时，不能当作完整 agent step 的延迟。

| 模式 | Decode token/s | 请求有效 token/s | TTFT p50 / p95 / p99 (ms) | TBT p95 / p99 (ms) |
|---|---:|---:|---:|---:|
| off | 299.52 | 186.33 | 37.45 / 161.71 / 316.59 | 9.204 / 9.408 |
| shadow | 299.48 | 185.76 | 37.38 / 164.26 / 317.07 | 9.229 / 9.664 |
| delete | 300.05 | 186.89 | 36.98 / 162.42 / 321.10 | 9.150 / 9.384 |

每次 delete 实际删除 22 个已发布冷块，三种模式都没有中途 MISS。
本次测试没有显示明显吞吐退化，delete 的 p99 TTFT 高约 4.5 ms；三次重复不能
据此证明零开销或加速。测量比较同一有界引擎的 GC 模式，不是旧引擎与新引擎
的性能对比。MTP 回调会成批到达，TBT 不能简单取倒数当作模型吞吐。

产物为 `timing_final_mrope_boundary/summary.json`，源码冻结在
`live_boundary_source_lock.json`。此测量早于首次 query 边界、额外 prototype、
大 block replay 边界和配置检查的后续补丁；不能声称测过这些新增路径的耗时。

## 实际文档保留质量

使用 MemoryAgentBench Accurate_Retrieval 的原始第 0、1 条文档及各前 16 个
问题，原提示模板和官方 `substring_exact_match` 评分器。官方评分仓库固定为
`455306dcabc3842526eb83cd4e225e5d486c5c5d`。
这是一项独立的 4B 在线保留消融，不是正式 MemoryAgentBench archive 协议，
不替换或合并既有 9B/27B 成绩。

文档按原 chunk 流式读入，然后连续追加问答，保留模型实际回复；不把标准答案
写入后续提示，也不为答错而重试。所有模式相同的 16K 选择预算、8K 生成预算、
block=32、greedy、关闭 thinking、每题最多 128 输出 tokens。
off/shadow 仓库为 512K，以保留全文；delete 仓库为 64K。

| 原始来源 | off | shadow | delete（64K） |
|---|---:|---:|---:|
| ruler_qa1_197K | 15/16 | 15/16 | 9/16 |
| ruler_qa2_421K | 5/16 | 5/16 | 3/16 |
| 合计 | 20/32（62.50%） | 20/32（62.50%） | 12/32（37.50%） |

off/shadow 的 32 条回答逐字一致。delete 相对 off 有 9 道由对变错、1 道由错
变对，净少答对 8 道。全部 1,035 次请求成功，六个运行各自首请求之外共有
1,029 HIT、0 MISS；不存在借助冷重建重新读入已删除证据的情况。
全部 96 条问答均正常 end_turn，没有输出上限截断。

delete 两条轨迹最终历史分别为 220,746 / 481,640 tokens，删除 5,023 /
13,148 个已发布冷块。超过 64K 后 CPU KV 保持 1,274,904,576 字节、索引
33,554,432 字节、前缀身份 2,621,440 字节、物理槽位 2,048 个。

该实验说明固定容量可以持续运行，也暴露了激进删除的质量损失。问题在文档
导入后才出现，此前的使用频率不能充分预测答题证据。规则分数不是未来使用
概率的校准估计，不能保证最优删除。两条文档、32 题也不足以代表整个数据集。
不同仓库容量和实际回复导致的后续轨迹差异均须保留，不能用于声称等存储或
等输入速度优势。不能将这个 64K 容量直接作为正式长记忆实验的默认配置。

原始答案、日志与参数见 `dataset_retention_append_v2/`；离线核算脚本为
`summarize_dataset_retention.py`，结果为 `dataset_retention_validation.json`。
前面的启动配置故障及不适用的独立问答包装均保留为诊断，不混入本表。

## 使用边界与交付

- 默认不开启；新入口为 `--kvmem-repository-blocks N` 与
  `--kvmem-repository-gc off|shadow|delete`，要求 `--ctx=N*block_tokens`。
- 当前是串行在线 serving。显式 named sessions、continuous batching、archive、
  semantic groups、GPU index、sum 等未接入组合明确拒绝，不静默降级。
- 固定容量有损遗忘。ready 耗尽或硬保护集合过大时明确失败，不强删活动资源。
- 固定的是持久 KV/索引/目录等；完整 transcript 的传输、解析与临时输入仍随
  历史增长，HTTP token 计数还有 UINT32_MAX 上限。整个服务进程并非恒定内存。
- 本轮没有重新启用定时检测，也没有覆盖原冻结实验二进制或重跑 9B/27B。
  原有不相关修改保留。代码未提交，构建与验收产物可独立审查。
