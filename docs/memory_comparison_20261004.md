# Qwen3.8-27B 完整 memory 对照：2026-10-04 建立

用户授权本机补充测试：对比重算前后、完整或尽可能完整的上下文、Compact、Compact＋RAG，并增加 AgentLongBench 样本。实验继续沿用历史样本、提示词、评分，以及输入≤1,048,576的筛选。

实验根：`/home/chaidi/qw3/benchmark/qw3_memory_comparison_20261004`。机器可读设置为 `plan.json`，实际输入身份为 `cohort.json`，冻结推理身份为 `identity.json`。

## 样本与顺序

| 数据集 | 每组题数 | 范围 |
|---|---:|---|
| LongMemEval-S | 102 | 历史固定子集，六类各17题；不换成S500同ID记录 |
| MemoryAgentBench | 1116 | 历史29个完整context，输入>256K且≤1M，三个保留类别 |
| AgentLongBench ≤256K档 | 250 | 冻结canonical样本，八种任务 |
| AgentLongBench 512K档 | 100 | 从此前完成20题扩大到全部100题 |
| AgentLongBench 1M档 | 50 | 从此前完成1题扩大到全部50题 |
| 合计 | 1618 | ALB共400题，比此前271题增加129题 |

按数据集串行运行：LME全部方法 → MBA全部方法 → ALB三个长度档。一个独立模型服务由引擎自动重置/复用请求状态；不逐题重启。更换需要不同启动参数的方法才重启。本次不启动30分钟定时器。

## 六组

1. `kvmem0`：36,864历史KV，重算比例0；保留Query Replay和DFlash。
2. `kvmem10`：同一二进制、同一设置，原位替换重算比例0.1。该组重新运行，避免不同二进制/数值路径混入主要A/B。
3. `dual10`：补偿双KV，33,184原位＋至多3,680同位置视图，attention中各重复视图减ln2。经原结果SHA、原审计、权重、二进制和启动设置核验，复用已经完成的LME102＋MBA1116；新增ALB400。该组同时改变基础预算和attention数值路径，不能解释成仅替换策略一个变量。
4. `native256`：关闭KVMem，完整KV容量262,144，保留DFlash。若完整输入＋原最大输出＋1≤容量，则完整保留原输入。放不下的题采用固定首1,312＋尾部窗口，保留问题和输出预算，逐题记录保留区间和比例。**窗口题单独报告，不能声称它们是完整或接近完整上下文。** 原生完整子集另作所有方法的同题配对。
5. `compact`：同一27B生成不含当前问题/参考答案/检索片段的全历史checkpoint摘要，无原文尾部。最终回答输入≤36,864。
6. `compact_rag`：复用第5组完全相同的摘要，再对原始完整历史检索。Jina CPU，1024-token块、128重叠，按分数选到最终36,864输入预算，再按原文顺序排列。

共9,708个结果位置，其中1,218个双KV答案核验后复用，计划新增8,490次最终回答；Compact摘要请求另计。首例预检使用正式清单中的真实样本，保存答案并在正式队列中跳过，不根据答对与否重试。

原生上下文超长方案在开始前询问了用户；未收到更改时按以上256K完整/窗口方案执行。不把超出模型原生长度的1M RoPE外推实验混称原生完整上下文。

## 不变的模型和协议

目标：`/home/chaidi/qw3-dflash2/models/Qwen3.8-27B-Q8_0.gguf`，SHA256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`。

共同使用FP8 KV、DFlash2（7 draft tokens），采样temperature0.6/top_p0.95/top_k20/min_p0/repetition_penalty1/presence_penalty0/seed20260722。LME与ALB thinking开启，预算4096，max_output32768；MBA关闭thinking，保留各family原输出上限。模型是Qwen3.8-27B，不能用旧Qwen3.6或9B的答案替代。

KVMem的logical context=2,097,152、block32、sink1312、recent10496、generation reserve32768、mean-k、Query Replay、immutable source KV等沿用此前配置。GPU与用户另一服务共享，使用此前已验证的managed只读权重方式；没有停止或改动另一服务。耗时仅为共享GPU条件下观察值，不是严格延迟A/B。

Compact采用冻结历史 `compaction_prompt`。单轮完整渲染输入≤232,000，生成≤24,576，关闭thinking、temperature0。持续摘要直到全部原始历史被读取，按字符区间记录无缺口覆盖；各轮只包含上一份摘要和下一段历史。空摘要/length截断是模型方法失败，保留单次生成，不为正确性重试。此为本次统一的新基线参数，并非旧实验的逐参数复现。

## 分词预检修复

首次离线核验发现MBA的一份相同文本：HF为470,066 tokens，历史GGUF服务为470,009。文本SHA完全相同；当时尚无模型回答请求。现场、脚本、HF长度清单和缓存保存在 `attempts/20261005_hf_token_preflight/`。

现在正式原输入、Query token边界、Compact最终输入及RAG最终预算均用GGUF服务实际分词。HF仅作摘要片段长度搜索的估计，最终再以GGUF计数收缩到上限；不会改变历史token hash来绕过核验。新129题在回答前重新做实际GGUF长度检查，若超限则全部组统一排除。

## 执行和恢复

四个本机systemd用户服务：

- `qw3-memory-comparison-server-20261004.service`：只拥有实验端口18234。
- `qw3-memory-comparison-validate-20261004.service`：协议、实际首例六组预检；通过后启动正式队列与评分。
- `qw3-memory-comparison-campaign-20261004.service`：串行推理队列。
- `qw3-memory-comparison-judge-20261004.service`：独立DeepSeek评分与报告，不启动推理。

服务文件位于实验根 `units/`，以 `systemctl --user link` 安装，不创建timer。当前阶段看 `queue-state.json`、`preflight/status.json`、每组 `results/<arm>/status.json`，实际进程和命令另行核验。服务存活不代表某个答案已完成，完成计数以 `cases/<suite>/<id>/result.json` 为准。

恢复先检查已完成的组/样本、原推理身份和实际进程。控制器从完成结果跳过；已有response只恢复解析，不再次请求模型。未完成传输先归档再重试，不挑选正确答案。源码、启动设置、权重和输入hash检查失败时停止，不能覆盖旧identity或伪造hash。

已有双KV结果只通过来源引用和衍生记录加入本实验，不修改原目录。DeepSeek凭据沿用用户已授权的历史judge进程，在内存中读取，不写入文件/日志。

## 评分和交付

- LME：历史DeepSeek Flash官方题型prompt，disabled thinking、temperature0、max_tokens16；保留12例校准与答案SHA绑定。
- MBA：冻结官方local指标，LongMemEval/InfBench特殊评分沿用DeepSeek V4 Pro和原上限。API/裁判预算失败保持pending，不能计成模型答错。排除ReDial后只有三个类别，综合headline包含EM、judge及摘要F1，不能称四类别官方总准确率。
- ALB：冻结canonical evaluator；分别报告250/100/50档、八种任务的可用子类、平均score及整题correct比例。列表/F1与整题全对分开说明。
- 同题比较核验原文本、实际原token SHA和Query SHA；原始输入相同不意味着各种压缩方法实际送入的token相同。

输出：根目录 `comparison.md`、`comparison.json`、`paired-results.csv`、`audit.json`；每组 `report.json` 与原始请求/回答/trace。精确完整上下文同题子集单独放在 `comparison.json/native_exact_full_subsets`。只有六组均覆盖最终清单、评分无pending、审计通过才生成 `complete.json`。不自动归档用户会话。
