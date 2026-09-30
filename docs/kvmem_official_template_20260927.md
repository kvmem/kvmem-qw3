# 新样本的 RULER 官方任务模板实验

用户要求先运行本组，再恢复 50/50 KV 混合组。

产物：`/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927`。

## 模板与输入

- 官方来源：NVIDIA/RULER，提交 `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`；下载源码与哈希保存在 `official-source/`。
- 采用 `scripts/data/synthetic/constants.py` 的 NIAH 任务正文，按 `niah.py` 调整单数问题和多名称连接方式。
- 去掉原输入的 xhigh system 消息，使用 Qwen user/assistant 聊天标记。保留原 assistant `<think>` 前缀；不使用额外答案预填。
- 不重新生成海量正文：所有正文 token、目标名称、数字及证据 token 保持不变。输入文件为新副本，原样本未修改。
- 原 64K 档实际为 65493–65495 tokens；128K 档为 131029–131031；256K 档为 262101–262103。不同任务的单复数和多名称连接方式造成 1–2 token 差别。
- 全部证据绝对位置前移 42 tokens。块对齐因此改变，不能将成绩变化完全归因于提示词语义。
- 每档四类、每类 15 例，合计 180 例。`template-audit.json` 保存正文、证据一致性与官方文本匹配核验。

## 推理设置与评分

沿用本批原始 10% 重算组的冻结二进制、模型、FP8 KV、MTP4、guided query、32K 检索/prefill budget、16K generation reserve、thinking=4096、max output=8192、温度/top-p/top-k/seed。

重算 KV 直接替换旧 KV。本组不是 50/50 更新，也没有补跑无重算基线。每例独立服务进程，先盲 ingest，再 query replay，保存实际请求和响应。

这是“同一正文改用官方任务模板”的实验，保留已有推理配置；不是完整复现官方数据生成、答案预填或 128-token 输出配置。

分别统计完整数字答案覆盖率和整题数字集合全对率，排除更长数字的子串命中。多名称对应关系与完整语义复核另列。`compare_templates.py` 会与同一正文的原提示词 10% 全替换组逐例比较。

## 执行顺序

`campaign.py` 串行执行：

1. 官方模板全 180 例。
2. 请求、输入、重算日志审计，以及模板成绩比较。
3. 恢复原样本 50/50 混合组：保留已完成的 2 例，从被中断的第 3 例重新开始。该组仍使用原 xhigh 输入，以保持与原直接替换组的配对关系。
4. 混合组审计及配对比较。

进度分别见官方组 `queue-state.json`、`state.json` 和 `alpha50_20260927/full180/state.json`。任何执行或审计错误都会中止接续并保存错误。暂停混合组的记录保存在 `alpha50_20260927/priority-switch-20260927/`。
