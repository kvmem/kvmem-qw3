# Memory eval ≤1M 交接：本机已停止，迁移3090并新增无重算对照

更新时间：2026-09-28 16:41停止后快照。**本文取代此前完整1818问迁移方案；按本次筛选清单继续，不能直接启动旧campaign跑完整队列。**

## 1. 用户的最新要求与范围

- 停止当前本机实验，交给另一个任务继续。
- 去除输入超过1M的样本，两组采用完全相同的筛选。
- 迁往SUFE-3090；继续10%重算组的未完成样本，同时新增关闭重算对照。
- 保留本次Qwen3.8-27B IQ3_S、MTP3、36,864 KV预算及其他推理配置，只对齐历史样本、提示词和评分。

本文明确 **1M = 1,048,576 tokens**，筛选的是完整输入（含chat framing、问题和assistant prefix），不是输入+最大输出，也不是数据集文件夹的名义长度。保留恰好1M的输入。仍保留逻辑context=2,097,152，以容纳输出reserve，不要把`-c`改成1M。

已完成题使用实际当前模型prompt_tokens；未运行题先用冻结长度记录筛选。当前预筛只排除ReDial的200问；接手时须在实际render/tokenize后再次检查每个未运行输入，若发现新的超限题，整题从两组移除、更新清单和分母，**不能截断正文**。筛选须在模型回答和评分前完成。

## 2. 本机停止状态：已经执行

2026-09-28 16:41:52已向本实验控制器发送SIGTERM，控制器清理自己的server；同时终止等待全队列完成的评分进程。随后检查确认三个PID均已退出、GPU compute进程列表为空：

| 进程 | 原PID | 当前状态 |
|---|---:|---|
| campaign.py | 2982410 | 已退出 |
| llama-kvmem-server | 2983010 | 已退出 |
| grading_queue.py | 2942386 | 已退出 |

没有启动任何替代推理、没有恢复旧队列、没有在3090部署或运行实验。

中断样本：`memoryagentbench/011_Accurate_Retrieval_r021_longmemeval_s_q0025`，当时输入373,534 tokens。目录只有request_meta.json，没有response.json或result.json，属于**用户主动中断的在途请求**，不计模型答错；在剩余清单中，恢复后正常完成一次。中断现场已复制归档。

停止证据和清单根目录：

```text
/data/chaidi/kvmem_eval/results/cacheblend_memory_pro6000_20260928/handoff_le1m_20260928/
```

其中：`stop-record.json`、`stop-verification.json`、`before-stop-*`、`stopped-*`、`interrupted_attempt/`。原始结果保留，未重写答案。

**评分凭据现状已变化**：旧waiter已退出，它内存中的DeepSeek密钥不再可用；文档、脚本和归档均不保存密钥。接手需要环境变量或本机终端隐藏输入。不要沿用旧文档中“waiter仍在等待”的描述。

## 3. 哪些要跑、哪些已经跑完

### 新的双组清单

| 实验 | 筛选后每组总量 | 10%组已完成 | 10%组待运行 | 无重算组待运行 |
|---|---:|---:|---:|---:|
| LongMemEval-S 历史子集 |102|**102**|**0**|**102**|
| MemoryAgentBench ≤1M子集 |1116|**865**|**251**|**1116**|
| AgentLongBench ≤256K |250|1|249|250|
| AgentLongBench 512K |100|1|99|100|
| AgentLongBench 1M档 |50|1|49|50|
| **合计** |**1618**|**970**|**648**|**1618**|

剩余新增推理总工作量：**648 + 1618 = 2266问**。10%组不重跑已完成970问；关闭重算组是新组，不能复用10%组答案。所有已完成结果来自原PRO6000机器。

机器可读清单：

- `selection-manifest.json`：1618条selected、200条excluded、长度来源、完成状态、结果路径及SHA。
- `blend10-pending.jsonl`：648条待续跑。
- `blend0-all.jsonl`：1618条完整对照组。
- `excluded.jsonl`：200条排除项。

### 明确排除的部分

排除MAB上下文 `030_Test_Time_Learning_r000_recsys_redial_full` 的全部200问。历史archive约1,665,281 tokens，明显超过门槛；其10%组尚未运行，因此没有需要移出统计的已完成答案。

ALB名为“1M档”的50题**保留**：历史prompt计数为1,009,875～1,037,975；已完成一题的当前完整输入为1,030,722。不能因为文件夹名为1M就把整档删掉。剩余每题仍须当前模型完整render复核。

### MAB内部的进度

| 来源/类别 | 筛选后题数 | 10%已完成 | 待运行 |
|---|---:|---:|---:|
| RULER QA2 421K / Accurate_Retrieval |100|100|0|
| EventQA / Accurate_Retrieval |500|500|0|
| longmemeval_s* / Accurate_Retrieval |300|265|35|
| Fact Consolidation / Conflict_Resolution |200|0|200|
| InfBench summary / Long_Range_Understanding |16|0|16|
| ReDial / Test_Time_Learning |**排除200**|0|不运行|

MAB内的longmemeval_s* 300问与独立LongMemEval-S历史102题不是同一组，不能混合计数或套用同一个实验完成标志。

## 4. 哪些已经评分、哪些还没做

### 已完成的评分工作

- 独立LME102：Codex逐题语义复核暂计 **84/102=82.35%**，包括4个边界判对；这不是DeepSeek评分。
- MAB官方本地scorer已在16:32快照运行：EventQA **409/500=81.80%**；RULER **49/100=49.00%**。这600问之后答案未重跑。该指标是官方substring exact match。
- ALB三个首样本：canonical evaluator均0/1，总0/3；每档只有一题，不代表总体。

评分产物：

```text
原结果根/partial_reviews/matched102_20260928T072104Z/
原结果根/partial_scores/20260928T083239Z/
```

### 尚未进行

1. LME102的正式DeepSeek Flash judge（未发送正式请求，只有preflight及离线评分逻辑测试）。
2. MAB已完成的265道longmemeval_s*题的正式DeepSeek Pro judge。
3. 未推理MAB的200道fact、16道InfBench以及35道longmemeval_s*的最终评分。
4. 剩余397道ALB推理及评分。
5. **关闭重算组所有1618问的推理与评分**。
6. 3090的工具链验证、源码/模型/数据迁移、SM86编译、显存/主存验证。
7. 新筛选范围的正式双组配对比较。

本机现在保持停止，交接任务不应自动恢复完整旧1818问队列。

## 5. 固定的引擎、模型和参数

当前是 **kvmem-llama-cacheblend（llama.cpp实现）**，不是qw3。虽然使用 `/home/chaidi/qw3/.venv/bin/python` 运行脚本，模型推理由llama-kvmem-server执行。

```text
源码：/home/chaidi/kvmem-freetoken/kvmem-llama-cacheblend
控制器：/home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-pro6000-20260928
原结果根：/data/chaidi/kvmem_eval/results/cacheblend_memory_pro6000_20260928
模型：/data/chaidi/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
模型SHA256：58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f
```

两组只计划改变 `--kvmem-blend-ratio`：blend10=0.1，blend0=0。

| 配置 | 两组共同值 |
|---|---|
| KVMem |开启，method=retrieval|
| history budget / generation reserve |36864 / 32768|
| block / sink / recent |32 / 1312 / 10496|
| logical context |2097152|
| main KV / MTP KV |q8_0 / f16|
| MTP |draft-mtp，3个draft tokens|
| batch / ubatch |1024 / 1024|
| query replay / policy / max |auto / user / 32768，显式query span|
| MTP state |replay|
| blend alpha / state / neighbors |1 / reset-restore / 0|
| CPU参数 / GPU层 / threads |kvmem-cpu-gb=0 / ngl99 / t8、tb8|
| sampling |temp.6、top_p.95、top_k20、min_p0、repeat_penalty1、presence_penalty0、seed20260722|
| LME与ALB |thinking开启，thinking预算4096，max_output32768|
| MAB |thinking关闭，按原family输出上限|
| env |KVMEM_BENCHMARK_API=1、KVMEM_TRACE=1、CUDA_VISIBLE_DEVICES=0|

MAB输出上限原字典：`ruler50, eventqa40, longmemeval50, fact10, infbench1200, detective2000, icl20, recsys300`。保留原request()行为。

完整原server命令与二进制/库SHA见原根 `identity.json`，也已复制到本次交接目录 `stopped-identity.json`。路径/端口可适配远端；不要换模型量化或降低KV预算。off组用ratio=0，不能以alpha=0替代，不关闭KVMem、Query Replay、MTP。

## 6. 样本与提示词身份

- LME使用 `inputs/longmemeval_matched102.jsonl`，来源 `/home/chaidi/qw3/selected_12_samples.jsonl`，SHA `7bdc816736e4505d16d0fa196415ee32cb63314e9f64b7341a50ba76f4ec888c`，六类各17，无abstention。保持原始完整记录和顺序，不重抽样。
- 使用 `historical_protocol.py` 从冻结历史reader提取的SYSTEM/history/question文字。不要回到S500的旧提示词或按同ID替换为S500内容。
- 原S500的39个结果已在 `revisions/matched102_20260928/s500_longmemeval_cases` 归档，不属于本次102题。
- ALB用 `inputs/agentlongbench/{le256k,512k,1m}/{samples.jsonl,manifest.jsonl}`，原canonical prompt与query_byte_span，按stable_sample_id配对。
- MAB用 `inputs/memoryagentbench/<row>/` 下已冻结的五种准备文件，system+单user正文/问题，raw_question span；保留相同context的原始问题顺序及prefix checkpoint/rollback策略。
- 原始输入目录可以保留全部旧文件，**实际job和评分key集合必须以新selection-manifest为准**；不能因ReDial文件仍存在就把200问重新加进来。
- 不把gold或方法标签放入模型请求；不改tokenizer或静默截断。

## 7. SUFE-3090部署情况

用户配置：

```sshconfig
Host SUFE-3090
    HostName 10.1.139.69
    User dell
    IdentityFile /Users/dichai/.ssh/id_rsa.txt
    Port 22
```

当前Linux无该Mac私钥路径，但已经成功通过已有认证直接运行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes -p 22 dell@10.1.139.69 'hostname; nvidia-smi'
```

已读到：`dell-Precision-3680`，3090显存24GiB、系统内存约62GiB、磁盘当时可用160GiB、驱动570.86.10。检查时GPU空闲。**尚未传文件/建新目录/编译/跑验证。**

建议独立目录 `/home/dell/cacheblend-memory-le1m-ab-20260928/`。不要覆盖服务器既有kvmem实验。PRO6000的SM120a/CUDA13二进制不能直接假定可用于3090；在兼容工具链下用同源代码为SM86构建。源码有工作树改动，原快照 `handoff_3090_ab_20260928/` 保存脚本和diff，但diff不含全部未跟踪文件，迁移需冻结完整必要源码/子模块。

当前实测约40KiB CPU KV+index/token，1,048,576 tokens估算约39.96GiB；还需checkpoint、模型映射、输入、临时buffer。≤1M较合理，但不等于已通过3090峰值内存验证。预检失败先保存日志，不修改输入/预算来掩盖失败。

## 8. 接手执行步骤与必要代码改动

1. 确认原机三个进程仍已停止；不要重复启动旧queue。核验本次清单970条完成结果的hash，筛选无重复键、648条on待跑、1618条off、200条excluded。
2. 冻结源码、权重、inputs及当前控制器；在3090新目录准备依赖、同源SM86构建、验证模型和数据SHA。只传实验必要文件，不传SSH私钥或API密钥。
3. 参数化runner的路径、port、OUT、ratio；分别设 `blend10/`、`blend0/` 输出。10%组使用复用清单引用或带来源复制970个完成结果；off组从空结果开始。
4. `prepare()/jobs()/summarize()`及scorer当前有1818、1316、30contexts等全量断言，需改为依据新manifest：总1618，MAB1116问/29contexts，其他数量不变。保持原输入hash并另存筛选manifest hash。原inputs含ReDial也不能重新加入job。
5. 实际完整render/tokenize后强制 `prompt_tokens <= 1048576`。输出上限仍32768或family原值；总context仍2097152。预筛未跑题使用旧长度，实际发现超限须在两组清单一致删除并更新分母。
6. 日志验证按arm区分：on长输入必须有原有效`KVMEM_BLEND ratio=.1 alpha1 neighbors0 recurrent=reset-restore`；off有效ratio0、无实际重算。不能为off运行而删除全部trace校验。
7. 新硬件/二进制/脚本建立新的identity，保留旧identity及复用来源；不能改旧hash绕过resume。每个结果记录host、arm、binary/source/model/input身份。
8. 先续完on的648问，再跑off的1618问；也可按suite交替，但单3090不同时加载两个服务，不跨arm复用KV。MAB重启时第一条待运行问题冷启动，后续按原context顺序进行，计时记录冷暖状态。
9. 当前中断题仅有request_meta，可在归档证据基础上重新发起；若其他目录有response无result，先恢复解析，不直接重生成。完整错误答案仍保留，不按正确性重试。
10. 评分可在原机CPU/网络环境进行，推理在3090；各arm独立快照、锁和judge缓存。维护同步与来源清单，不恢复原已停止的全量judge waiter。

不要单纯修改历史counter使其看似完成。验收以选定 `(suite,id)` 集合、完整结果、待裁判集合为准。

## 9. 评分与对照报告

### LongMemEval正式judge

历史DeepSeek Flash：`deepseek-flash`、thinking disabled、temperature0、max_tokens16、streamfalse；官方每题型prompt，12例校准，严格正常stop yes/no。只发送question/reference/final answer，不发送reasoning和arm名。异常留pending，不能计错或改用字符串命中。

原机已准备入口：

```bash
/home/chaidi/qw3/.venv/bin/python /home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-pro6000-20260928/judge_longmemeval_now.py --prompt-key
```

它可直接judge当前on组102题。off组需适配输出路径、同样协议。密钥使用环境变量或终端隐藏输入，不保存到文件。无需再向用户确认“是否使用DeepSeek”，已经授权；缺少凭据时如实报告，不用Codex复核代替正式评分。

### MAB：筛选后的指标必须重新命名

- 本地官方指标保持不变；longmemeval子集/InfBench特殊评分沿用DeepSeek V4 Pro和原judge脚本。
- **排除ReDial后，本子集不再含Test_Time_Learning，不能报告原四类别MAB总分。**
- 正式报告三个保留类别及各source，注明“MemoryAgentBench >256K且≤1M子集：29contexts、1116问”。
- 如报告综合值，可另列明确标注的“三个已覆盖类别等权宏平均”，沿用各category内source等权；不将缺失类别填0，不冒充原四类官方总分。
- 与历史结果比较也必须筛到完全相同1116个question key再汇总，不能和历史全1316问总分直接相减。
- 当前 `compare_memoryagentbench_methods.py` 会要求原selected key集合；接手应由新manifest建立目标集合并校验29contexts/1116问，而非继续使用1316断言。

### ALB与硬件配对

ALB用冻结canonical evaluator，分别报告250/100/50档及任务子类，不把3个首样本结果外推。

970个已完成on结果是PRO6000生成，后续on和全部off计划3090生成。输出全1618问配对时标注设备来源；另列两个arm都在3090生成的共同子集（默认648问）以辅助比较。跨设备结果不能宣称纯单变量，PRO6000与3090延迟也不能直接相减作为重算开销。不得为取得更好答案重跑旧on；若要严格同3090完整AB，额外970问on重跑需单独明确工作量。

## 10. 交付与完成判定

- `migration-manifest.json`、过滤manifest及来源hash。
- 每arm1618个选定样本的完整答案或显式模型失败；基础设施失败和API评分失败单独列pending。
- LME102正式judge；MAB1116对应指标及三类别标签；ALB400题canonical评分。
- `paired-results.csv`：suite/id、输入hash、arm、设备、模型/build、答案hash、score、score delta、冷暖cache及可比计时。
- `comparison.md`：分benchmark/子类型比较，明确原始模型配置、≤1M选择标准、丢弃ReDial200题、硬件混合与judge信息。
- on970旧结果hash保持不变；无跨arm答案复用；没有重复键或遗漏。
- 不自动重建周期巡检，不自动归档用户会话。只在当前接手任务中按授权推进。

## 11. 直接粘贴给新任务

> 请先读 `/home/chaidi/qw3/docs/memory_eval_le1m_handoff_20260928.md`。本机实验已停止。把kvmem-llama-cacheblend的≤1M输入评测迁到SUFE-3090（dell@10.1.139.69）。以 `handoff_le1m_20260928/selection-manifest.json` 为准：保留1618问，排除ReDial200问；1M定义为1048576输入tokens，保持logical context2M和原输出上限。当前on组已完成970问，续跑648问；新建ratio0组完整跑1618问。保留Qwen3.8-27B IQ3_S、MTP3、36864 KV及原推理参数，仅ratio为0.1/0。按历史输入/提示/评分协议，先对已完成LME102补DeepSeek正式judge。旧waiter已停止且不保存密钥。准确记录迁移、结果来源和硬件，保留原结果，MAB报告三类≤1M子集而非原四类总分。完成推理、正式评分和配对比较。
