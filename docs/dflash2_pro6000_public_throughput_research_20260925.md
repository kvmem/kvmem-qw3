# RTX PRO 6000 上 Qwen3.8-27B 的公开吞吐调研

调研日期：2026-09-25。按用户追加要求，以 **Q8_0 目标权重** 为主；FP8
权重及 NVFP4 权重另列，均不能改标成 Q8_0。没有修改目标精度、主工作区或既有
GPU 测试队列，也没有下载/运行外部推理框架。

结论：Q8 单卡短上下文多并发已有约 1,200 tok/s 的公开厂商自测；其中最高的
投机开启披露为 C32 1,285 tok/s。现有证据不足以评选同条件的“最优框架＋最优
MTP 方法”，更不能证明 48GB 内 K128K/G64K 的 SWE 长任务可达此速度。

## Q8 投机路径：较高的公开披露

Paddock 开发者在原帖追加 0.1.2 的自测，模型标为 Qwen3.8-27B Q8：

| 并发 | 投机关闭 | 投机开启 | 开启/关闭 |
| --- | ---: | ---: | ---: |
| 1 | 46.8 | 202 | 4.32x |
| 8 | 320 | 822 | 2.57x |
| 32 | 1,005 | 1,285 | 1.28x |

单位为输出 tok/s。发布者将版本描述为含 MTP 与其 DFlash 草稿实现，但没有将
这些数字拆成各个算法和草稿 checkpoint 的独立测量，不能确定最佳草稿/算法。
另一个同作者帖子的后续说明给出 max-ctx=8192、max-batch=32、FP8 E4M3 KV、
128 输入/128 输出、32 客户端、32 请求预热后 192 请求的 aiperf 测试配置。
属于开发者披露，尚未在本机独立复现；未见对应完整 SWE 正确性及长任务性能数据。

来源：[开发者原帖](https://www.reddit.com/r/LocalLLM/comments/1vspb61/new_free_inference_engine_runs_qwen363827b_on_rtx/)、
[补充测试参数](https://www.reddit.com/r/Qwen_AI/comments/1vsnl5s/qwen3627b_at_32_concurrent_clients_697_ms_ttft/)。

## 有详细配置的 Q8_0 对照板：没有开启投机

Paddock 官方 2026-08-22/23 测试明确所有框架都关闭投机。Q8_0 组让 Paddock 与
llama.cpp 读相同 GGUF，不能将页面主折线图的 FP8 数据误算成 Q8_0。
其交互图表中 Q8_0 的独立数值为：

| 输入 / 输出 tokens | C1 | C8 | C32 |
| --- | ---: | ---: | ---: |
| 128 / 128 | 47.9 | 318.4 | 984.9 |
| 128 / 2048 | 48.5 | 350.7 | 1,168.5 |
| 1024 / 1024 | 48.2 | 340.0 | 页面该分组未列 |

硬件为 RTX PRO 6000 SM120、96GB、600W；Paddock Q8_0 使用 FP8 KV 和 F32 GDN
状态，max-ctx=8192、max-batch=32，页面记录峰值预留显存 **80.7GB**。不同框架
显存设置不同；页面说明该值是预留量，不能代替逐阶段有效占用。该实验不满足本任务
48,000,000,000 字节上限，且只是短输入/固定输出测试。

来源：[厂商完整对照板](https://truespar.com/paddock/benchmarks/qwen38-27b)。
本地从页面 Astro 图表 props 解码出的 Q8 独立行保存在
`benchmark/dflash2/pro6000_public_research20260925_01/paddock_q8_extracted.json`，
原 HTML 与哈希保留。未把 FP8 曲线、输入+输出之和或模型文件大小当作 Q8 decode。

## 最接近的真实长任务记录：FP8 权重，非 Q8_0

jpezzulli 的冻结 SGLang/DFlash2 27B 记录是相对完整的真实 agent 参照。注意其
仓库当前首页还包含 Flash-Next 的后续结果，不能把这些不同模型的更高速度移入27B。

补充审计：冻结发布 `68e05c793c656816553c0bcdc9093e945b22bde1` 的
`PROVENANCE.md` 和 `evidence/models/checkpoints.json` 明确目标为
`orcarouter/Qwen3.8-27B-Uncensored-FP8` 衍生模型，**此前答复称“官方 FP8
权重”不准确，现予纠正**。公开的是日常 agent 使用的脱敏服务日志，没有原始任务
说明或对话，不能认定为 DeepSWE。其草稿权重及 config SHA256 与本地 DFlash2
文件一致；目标权重、KV 管理和任务不同。冻结原始证据已保存在研究目录的
`frozen_27b_agentic/`。进一步归因见 `dflash2_deepswe_cause_diagnosis_20260925.md`。

- 27B FP8 + DFlash2 的受控短输入：C1 108.75 tok/s，C4 总输出 / 组耗时
  390.23 tok/s；各请求136-token prompt、1024-token output。
- 真实 agent 全部124请求、85,156输出、最长350,195输入：token加权有效decode
  102.57 tok/s；剔除7个重叠请求后的117请求为125.36 tok/s。
- 无重叠样本的100–150K为130.36 tok/s（14请求）；150–262K为116.10
  （只有2请求、605输出）；340–360K为101.69（23请求）。

这些是公开真实使用日志，非统一的 SWE-bench/DeepSWE 官方任务吞吐排行。有效decode
分母排除初始prefill，但保留之后的等待/重算等，与理想GPU kernel纯执行时间不同。
既不能把中位数131.31解释为全样本加权持续吞吐，也不能把C4短输入390.23推到128K。

来源：[冻结27B结果与指标定义](https://github.com/jpezzulli/sglang-rtxpro6000/blob/68e05c793c656816553c0bcdc9093e945b22bde1/RESULTS.md)、
[目标及草稿来源](https://github.com/jpezzulli/sglang-rtxpro6000/blob/68e05c793c656816553c0bcdc9093e945b22bde1/PROVENANCE.md#checkpoints)。

## NVFP4 和多卡记录的排除边界

fakoli 的 SGLang/Inferact NVFP4 + DFlash2 K12 在单卡 RTX PRO 6000 Max-Q 上，
4K/C8/100请求的固定持续输出测试为764.3 tok/s；两卡独立副本为1401.8–1423.4。
该记录明确不是 SWE 验证，且更长82K/C8请求延迟很高。它不属于Q8结果，也不是
Server Edition上限。来源：[原始实验及证据入口](https://github.com/fakoli/anvil-serving/blob/main/docs/findings/2026-09-04-qwen38-27b-pro6000-possibility-plan.md)。

Helix 的八卡7882.6 tok/s、loswald的四副本约5194 tok/s等不能直接称为“单卡实测”，
即使除以卡数也只是每卡平均贡献，不是独立单卡测量；其主目标量化也为NVFP4。
本报告不给出由多卡结果除算得到的单卡最高值。

## 对本目标的含义

目前能说“Q8短上下文多并发具备千tok/s级公开参照”，不能说已经证明某框架是
Q8+FP8 KV+48GB+K128K/G64K+真实SWE上的最快实现。尤其缺少同任务、相同量化、
相同上下文策略、完整任务正确率和同步总输出/墙钟计时的跨框架对照。

对照框架候选应包括Q8_0原生加载的Paddock；这份SGLang+FP8衍生模型/DFlash2作为另一条
8位权重参考线保留目标模型及量化差异。当前48GB长任务目标仍以本机固定重放及实际agent结果
验收，不因公开短上下文高并发纪录而改变K/G、精度或显存边界。

研究快照目录：`benchmark/dflash2/pro6000_public_research20260925_01/`。
保存公开网页/源码、抓取URL、SHA256和图表抽取结果；抓取失败也保留错误，不冒充已读。
